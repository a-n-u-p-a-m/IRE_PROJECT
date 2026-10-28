"""
predictors.py - Fit and apply effect predictors.

Predicts E_rerank, E_gate and Delta_t from conventional features extracted by
features.py.  Three targets, kept strictly separate (never combined into one
regression or feature-engineered together).

Design decisions:

  * Fitted on TRAINING queries only.  Never peek at validation or test splits
    during fitting.  Query IDs come from data/query_split.json.

  * Features and targets are in separate tables (shared/schema.py rule).  The
    join is performed here, not in the feature or effect modules.

  * Three baselines from shared/baselines.py must beat before the predictor
    is considered to contribute.

  * The headline comparison is within-stratum (shared/metrics.py
    evaluate_within_strata), because pooled statistics reward membership
    detection for free.

Usage::

    from experiments_chaitanya.predictors import EffectPredictor
    predictor = EffectPredictor(effect_type="e_rerank")
    predictor.fit(train_features, train_effects)
    predictions = predictor.predict(test_features)
    fidelity = predictor.evaluate(test_features, test_effects)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shared.baselines import attach_all
from shared.metrics import evaluate_within_strata, macro_fidelity
from shared.schema import DELTA_T, E_GATE, E_RERANK

logger = logging.getLogger(__name__)


# Feature names used in each predictor (by effect type)
_FEATURE_SETS: dict[str, list[str]] = {
    E_RERANK: [
        "stage1_rank", "stage1_score", "stage1_rank_norm",
        "stage1_percentile", "stage2_score", "stage2_score_norm",
        "rank_drop", "is_survivor",
    ],
    E_GATE: [
        # E_gate targets non-survivors (gate failures), so stage1 rank
        # relative to k_c is the dominant signal.  We keep all features.
        "stage1_rank", "stage1_score", "stage1_rank_norm",
        "stage1_percentile", "stage2_score", "stage2_score_norm",
        "is_survivor",
    ],
    DELTA_T: [
        "stage1_rank", "stage1_score", "stage1_rank_norm",
        "stage1_percentile", "stage2_score", "stage2_score_norm",
        "is_survivor",
    ],
}


def _pivot_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long-form feature table to wide format: one row per (query, doc).

    Columns: query_id, doc_id, <feature_name_1>, <feature_name_2>, ...
    """
    wide = feature_df.pivot_table(
        index=["query_id", "doc_id"],
        columns="feature_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def _join_features_to_effects(
    feature_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    effect_type: str,
) -> pd.DataFrame:
    """
    Join feature (wide) to effect table on (query_id, doc_id).

    Only keeps rows for the given effect_type and drops rows with missing
    values in either table.
    """
    wide = _pivot_features(feature_df)
    effects = effect_df[effect_df["effect_type"] == effect_type].copy()
    merged = effects.merge(wide, on=["query_id", "doc_id"], how="left")
    return merged


class _NumpyRidge:
    """Pure NumPy Ridge regression fallback when scikit-learn is not installed."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.mean_x: np.ndarray | None = None
        self.std_x: np.ndarray | None = None
        self.mean_y: float = 0.0
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyRidge":
        self.mean_x = np.mean(X, axis=0)
        self.std_x = np.std(X, axis=0)
        self.std_x = np.where(self.std_x < 1e-8, 1.0, self.std_x)
        self.mean_y = float(np.mean(y))

        X_std = (X - self.mean_x) / self.std_x
        X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
        y_cent = y - self.mean_y

        n_features = X_std.shape[1]
        I = np.eye(n_features)
        A = X_std.T @ X_std + self.alpha * I
        b = X_std.T @ y_cent
        self.w = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_std = (X - self.mean_x) / self.std_x
        X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
        return self.mean_y + X_std @ self.w


class EffectPredictor:
    """
    Predicts the magnitude of one effect type from conventional features.

    Uses a Ridge regression for interpretability and stability.  Tree-based
    models (RandomForest, GradientBoosting) are optionally available via the
    `model_type` parameter for comparison in Week 4.

    Parameters
    ----------
    effect_type:
        One of 'e_rerank', 'e_gate', 'delta_t'.
    model_type:
        'ridge' (default) or 'rf' (random forest).
    alpha:
        Regularisation strength for Ridge.
    """

    def __init__(
        self,
        effect_type: str,
        model_type: str = "ridge",
        alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        if effect_type not in (E_RERANK, E_GATE, DELTA_T):
            raise ValueError(
                f"effect_type must be one of {(E_RERANK, E_GATE, DELTA_T)}, "
                f"got {effect_type!r}"
            )
        self.effect_type = effect_type
        self.model_type = model_type
        self.alpha = alpha
        self.random_state = random_state
        self._model = None
        self._feature_names: list[str] = []
        self._is_fitted = False

    # ── fitting ──────────────────────────────────────────────────────

    def fit(
        self,
        train_features: pd.DataFrame,
        train_effects: pd.DataFrame,
    ) -> "EffectPredictor":
        """
        Fit the predictor on training-split features and effects.

        Both tables must contain only training-query rows; validation and test
        query IDs must be excluded BEFORE calling this method.
        """
        merged = _join_features_to_effects(
            train_features, train_effects, self.effect_type
        )
        wanted = _FEATURE_SETS[self.effect_type]
        available = [f for f in wanted if f in merged.columns]
        if not available:
            raise ValueError(
                f"No features available for {self.effect_type}. "
                "Check that features were extracted for the same system and k_c."
            )

        X = merged[available].to_numpy(dtype=float)
        y = merged["effect"].to_numpy(dtype=float)

        # Drop rows with any NaN
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        if len(y) == 0:
            raise ValueError(
                f"No valid training rows for {self.effect_type} after "
                "dropping NaN. Check feature and effect table alignment."
            )

        try:
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            has_sklearn = True
        except ImportError:
            has_sklearn = False

        if has_sklearn:
            if self.model_type == "ridge":
                estimator = Ridge(alpha=self.alpha, random_state=self.random_state)
            elif self.model_type == "rf":
                estimator = RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state, n_jobs=-1
                )
            else:
                raise ValueError(f"model_type must be 'ridge' or 'rf', got {self.model_type!r}")

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", estimator),
            ])
            pipe.fit(X, y)
            self._model = pipe
        else:
            if self.model_type == "ridge":
                model = _NumpyRidge(alpha=self.alpha)
                model.fit(X, y)
                self._model = model
            else:
                raise ImportError(
                    f"scikit-learn is required for model_type={self.model_type!r}"
                )

        self._feature_names = available
        self._is_fitted = True

        logger.info(
            "fitted %s predictor for %s on %d rows, %d features: %s",
            self.model_type, self.effect_type, len(y),
            len(available), available,
        )
        return self

    # ── prediction ───────────────────────────────────────────────────

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a frame with (query_id, doc_id, pred_<effect_type>) columns.

        Documents with missing features get NaN predictions (they are excluded
        from fidelity metrics rather than silently receiving a wrong number).
        """
        if not self._is_fitted:
            raise RuntimeError("call fit() before predict()")

        wide = _pivot_features(feature_df)
        available = [f for f in self._feature_names if f in wide.columns]
        X = wide[available].to_numpy(dtype=float)

        pred = np.full(len(wide), float("nan"))
        mask = np.isfinite(X).all(axis=1)
        if mask.any():
            pred[mask] = self._model.predict(X[mask])

        col = f"pred_{self.effect_type}"
        result = wide[["query_id", "doc_id"]].copy()
        result[col] = pred
        return result

    # ── evaluation ───────────────────────────────────────────────────

    def evaluate(
        self,
        test_features: pd.DataFrame,
        test_effects: pd.DataFrame,
        n_iter: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
        top_m: int = 20,
    ) -> pd.DataFrame:
        """
        Evaluate fidelity on held-out (test) queries, within stratum.

        Returns a per-stratum fidelity table (from metrics.evaluate_within_strata).
        """
        if not self._is_fitted:
            raise RuntimeError("call fit() before evaluate()")

        pred_df = self.predict(test_features)
        col = f"pred_{self.effect_type}"

        merged = _join_features_to_effects(
            test_features, test_effects, self.effect_type
        )
        merged = merged.merge(
            pred_df[["query_id", "doc_id", col]],
            on=["query_id", "doc_id"],
            how="left",
        )

        fidelity = evaluate_within_strata(
            merged,
            predicted_column=col,
            effect_type=self.effect_type,
            n_iter=n_iter,
            alpha=alpha,
            seed=seed,
            top_m=top_m,
        )
        return fidelity


# ──────────────────────────────────────────────────────────────────────
# Convenience: fit all three predictors
# ──────────────────────────────────────────────────────────────────────

def fit_all_predictors(
    train_features: pd.DataFrame,
    train_effects: pd.DataFrame,
    model_type: str = "ridge",
    alpha: float = 1.0,
    random_state: int = 42,
) -> dict[str, EffectPredictor]:
    """
    Fit one predictor per effect type on training data.

    Returns {effect_type: EffectPredictor}.
    """
    predictors: dict[str, EffectPredictor] = {}
    for et in (E_RERANK, E_GATE, DELTA_T):
        try:
            p = EffectPredictor(et, model_type=model_type, alpha=alpha,
                                random_state=random_state)
            p.fit(train_features, train_effects)
            predictors[et] = p
        except Exception as exc:
            logger.warning("predictor for %s failed: %s", et, exc)
    return predictors


def evaluate_all_predictors(
    predictors: dict[str, EffectPredictor],
    test_features: pd.DataFrame,
    test_effects: pd.DataFrame,
    train_effects: pd.DataFrame,
    n_iter: int = 1000,
    seed: int = 42,
    top_m: int = 20,
) -> pd.DataFrame:
    """
    Evaluate all predictors and baselines together.

    Returns a combined fidelity table with per-stratum rows for each
    (effect_type, predictor) combination, including the three baselines.
    """
    parts: list[pd.DataFrame] = []

    for effect_type, predictor in predictors.items():
        # Baseline columns
        eff_block = test_effects[test_effects["effect_type"] == effect_type].copy()
        train_block = train_effects[train_effects["effect_type"] == effect_type].copy()
        eff_with_baselines = attach_all(eff_block, effect_type, train=train_block)

        for baseline_col in ("pred_zero", "pred_membership", "pred_stratum_mean"):
            if baseline_col not in eff_with_baselines.columns:
                continue
            fid = evaluate_within_strata(
                eff_with_baselines,
                predicted_column=baseline_col,
                effect_type=effect_type,
                n_iter=n_iter,
                seed=seed,
                top_m=top_m,
            )
            parts.append(fid)

        # Actual predictor
        fid = predictor.evaluate(
            test_features, test_effects, n_iter=n_iter, seed=seed, top_m=top_m
        )
        parts.append(fid)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


__all__ = [
    "EffectPredictor",
    "fit_all_predictors",
    "evaluate_all_predictors",
]
