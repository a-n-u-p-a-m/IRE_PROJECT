"""
test_real.py - The real-data path, on a synthetic cascade written out as the
four real artefacts.

The synthetic path builds `QueryInputs` in memory; the real path reads them
from the universe, the Stage-1 tables, the qrels and the Stage-2 cache. Fed the
same cascade, the two must produce identical effect tables, so any difference
is the loader's fault.

    python -m pytest experiments_shubham/tests/test_real.py
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from experiments_shubham import real
from experiments_shubham.engine import (
    EffectSettings,
    run_budget,
    run_worlds,
    synthetic_frame,
    synthetic_inputs,
)
from shared.allocation import RANDOM, allocate
from shared.cache import Stage2Cache
from shared.ordering import order_by
from shared.provenance import read_manifest
from shared.schema import save
from shared.universe import freeze

KEY = ["query_id", "doc_id", "effect_type", "t", "seed"]


@pytest.fixture
def cascade(small_cfg):
    return synthetic_inputs(synthetic_frame(small_cfg, "strong_stages"))


def _run_table(system, scores, ranks=None):
    rows = []
    for q, s in scores.items():
        order = order_by(list(s), s)
        for r, d in enumerate(order, start=1):
            rows.append({"query_id": q, "doc_id": d, "system": system,
                         "rank": r if ranks is None else ranks[q][d],
                         "score": s[d], "checkpoint": "toy"})
    return pd.DataFrame(rows)


def _published(cfg, queries):
    """The allocations exactly as experiments_anupam.allocations writes them."""
    rows = []
    for q in queries:
        order = order_by(list(q.universe), q.s1)
        for k_c in cfg.candidate_depths:
            for t in cfg.t_grid:
                for policy in cfg.outsider_policies:
                    seeds = cfg.budget_seeds if policy == RANDOM else cfg.budget_seeds[:1]
                    for seed in seeds:
                        alloc = allocate(order[:k_c], order[k_c:], cfg.budget_for(k_c),
                                         t, policy, s1=q.s1,
                                         alternative_rank=q.alternative_rank, seed=seed)
                        rows += [{"query_id": q.query_id, "stage1_system": "bm25",
                                  "k_c": k_c, "t": t, "policy": policy, "seed": seed,
                                  "doc_id": d, "promoted_rank": i}
                                 for i, d in enumerate(alloc["selected"], start=1)]
    return pd.DataFrame(rows)


@pytest.fixture
def artefacts(tmp_path, small_cfg, cascade):
    """The cascade as universe, Stage-1 tables, qrels, cache and allocations."""
    universe = pd.DataFrame([{"query_id": q.query_id, "doc_id": d, "sources": "bm25,dpr"}
                             for q in cascade for d in q.universe])
    freeze(universe, tmp_path / "universe.parquet", small_cfg.fingerprint())

    save(_run_table("bm25", {q.query_id: q.s1 for q in cascade}),
         tmp_path / "stage1_over_universe_bm25.parquet", "run")
    alt = {q.query_id: q.alternative_rank for q in cascade}
    save(_run_table("dpr", {q: {d: -float(r) for d, r in a.items()} for q, a in alt.items()},
                    alt), tmp_path / "stage1_over_universe_dpr.parquet", "run")

    save(pd.DataFrame([{"query_id": q.query_id, "doc_id": d, "relevance": g, "judged": True}
                       for q in cascade for d, g in q.relevance.items()]),
         tmp_path / "qrels.parquet", "qrels")

    cache = Stage2Cache(tmp_path / "stage2.sqlite")
    for q in cascade:
        cache.put_many(small_cfg.reranker_model, [(q.query_id, d, s) for d, s in q.s2.items()])

    _published(small_cfg, cascade).to_parquet(tmp_path / "allocation_promoted.parquet")
    return tmp_path


def _load(cfg, root, cache=None):
    return real.load_queries(cfg, "bm25", cache or Stage2Cache(root / "stage2.sqlite"),
                             universe_path=root / "universe.parquet", tables_dir=root,
                             qrels_path=root / "qrels.parquet")


def _argv(cfg_path, root):
    return ["--system", "bm25", "--config", str(cfg_path),
            "--cache", str(root / "stage2.sqlite"),
            "--universe", str(root / "universe.parquet"), "--tables-dir", str(root),
            "--qrels", str(root / "qrels.parquet"),
            "--published-allocations", str(root / "allocation_promoted.parquet"),
            "--out-dir", str(root / "effects")]


def _same(a, b):
    a = a.sort_values(KEY).reset_index(drop=True)
    b = b.sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_real_path_reproduces_the_synthetic_path(small_cfg, cascade, artefacts):
    loaded = _load(small_cfg, artefacts)
    assert [q.query_id for q in loaded] == sorted(q.query_id for q in cascade)
    st = EffectSettings.from_config(small_cfg, "bm25")
    k_c = small_cfg.candidate_depths[0]
    for condensed in (False, True):
        _same(run_worlds(loaded, k_c, st, condensed=condensed),
              run_worlds(cascade, k_c, st, condensed=condensed))
    for policy in small_cfg.outsider_policies:
        _same(run_budget(loaded, k_c, k_c, small_cfg.t_grid, policy,
                         small_cfg.budget_seeds, st),
              run_budget(cascade, k_c, k_c, small_cfg.t_grid, policy,
                         small_cfg.budget_seeds, st))


def test_unjudged_documents_carry_no_relevance(small_cfg, cascade, artefacts):
    q = cascade[0]
    dropped = sorted(q.relevance)[:5]
    qrels = pd.read_parquet(artefacts / "qrels.parquet")
    qrels = qrels[~((qrels["query_id"] == q.query_id) & qrels["doc_id"].isin(dropped))]
    save(qrels, artefacts / "qrels.parquet", "qrels")
    loaded = {x.query_id: x for x in _load(small_cfg, artefacts)}[q.query_id]
    assert not set(dropped) & set(loaded.relevance)
    assert not set(dropped) & set(loaded.judged)
    assert len(loaded.judged) == len(q.relevance) - 5


def test_refuses_an_incomplete_cache(small_cfg, cascade, artefacts):
    with sqlite3.connect(artefacts / "stage2.sqlite") as con:
        con.execute("DELETE FROM stage2 WHERE query_id=? AND doc_id=?",
                    (cascade[0].query_id, cascade[0].universe[-1]))
    with pytest.raises(RuntimeError, match="missing 1 scores"):
        _load(small_cfg, artefacts)


def test_refuses_another_rerankers_scores(small_cfg, artefacts):
    with pytest.raises(RuntimeError, match="missing"):
        real.load_queries(small_cfg, "bm25", Stage2Cache(artefacts / "stage2.sqlite"),
                          model="some/other-reranker",
                          universe_path=artefacts / "universe.parquet",
                          tables_dir=artefacts, qrels_path=artefacts / "qrels.parquet")


@pytest.mark.parametrize("system", ["bm25", "dpr"])
def test_refuses_a_stage1_table_that_does_not_cover_the_universe(small_cfg, artefacts,
                                                                 system):
    path = artefacts / f"stage1_over_universe_{system}.parquet"
    table = pd.read_parquet(path)
    q = table["query_id"].iloc[0]
    last = table[table["query_id"] == q]["rank"].idxmax()   # ranks stay contiguous
    save(table.drop(index=last), path, "run")
    with pytest.raises(real.RealInputError, match="does not cover exactly U_q"):
        _load(small_cfg, artefacts)


def test_alternative_is_the_other_system(small_cfg):
    assert real.alternative_system(small_cfg, "bm25") == "dpr"
    assert real.alternative_system(small_cfg, "dpr") == "bm25"
    with pytest.raises(real.RealInputError):
        real.alternative_system(small_cfg, "splade")


def test_cache_digest_ignores_insertion_order(small_cfg, cascade, tmp_path):
    model = small_cfg.reranker_model
    rows = [(q.query_id, d, s) for q in cascade[:2] for d, s in q.s2.items()]
    a, b = Stage2Cache(tmp_path / "a.sqlite"), Stage2Cache(tmp_path / "b.sqlite")
    a.put_many(model, rows)
    b.put_many(model, rows[::-1])
    assert real.cache_digest(a, model) == real.cache_digest(b, model)
    b.put_many(model, [(rows[0][0], rows[0][1], rows[0][2] + 1e-9)])
    assert real.cache_digest(a, model) != real.cache_digest(b, model)


def test_main_writes_the_grid_and_matches_the_published_allocations(
        small_cfg, small_cfg_path, artefacts):
    assert real.main(_argv(small_cfg_path, artefacts)) == 0
    manifest = read_manifest(artefacts / "effects" / "bm25_manifest.json")
    check = manifest["allocation_check"]
    n_alloc = (len(small_cfg.candidate_depths) * 4 * (len(small_cfg.t_grid) - 1)
               * (len(small_cfg.outsider_policies) - 1 + len(small_cfg.budget_seeds)))
    assert check["ok"] and check["n_mismatched"] == 0
    assert check["n_allocations"] == n_alloc
    assert manifest["source"] == "real"
    assert manifest["alternative_system"] == "dpr"
    assert manifest["stage2_model"] == small_cfg.reranker_model
    assert manifest["stage2_rows"] == sum(1 for _ in pd.read_parquet(
        artefacts / "universe.parquet").itertuples())
    assert manifest["condensed_modes"] == [False, True]


@pytest.mark.parametrize("policy", ["boundary", "alternative", "random"])
def test_main_fails_when_world_d_disagrees_with_the_published_allocations(
        small_cfg_path, artefacts, policy):
    path = artefacts / "allocation_promoted.parquet"
    pub = pd.read_parquet(path)
    i = pub.index[pub["policy"] == policy][0]
    pub.loc[i, "doc_id"] = "not-a-promoted-doc"
    pub.to_parquet(path)
    assert real.main(_argv(small_cfg_path, artefacts)) == 1
    check = read_manifest(artefacts / "effects" / "bm25_manifest.json")["allocation_check"]
    assert not check["ok"] and check["n_mismatched"] == 1
    assert check["examples"][0]["policy"] == policy


def test_main_refuses_a_missing_cache_without_creating_one(small_cfg_path, artefacts):
    (artefacts / "stage2.sqlite").unlink()
    with pytest.raises(SystemExit):
        real.main(_argv(small_cfg_path, artefacts))
    assert not (artefacts / "stage2.sqlite").exists()


def test_main_refuses_allocations_from_another_freeze(small_cfg_path, artefacts):
    (artefacts / "allocation_manifest.json").write_text('{"config_fingerprint": "000000000000"}')
    with pytest.raises(SystemExit):
        real.main(_argv(small_cfg_path, artefacts))
