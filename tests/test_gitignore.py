"""
test_gitignore.py - What leaves the repo, and what must not.

`.gitignore` carries a rule: anything large and regenerable is ignored, and
anything that makes a number TRACEABLE is kept however small. That rule was
quietly broken for every run manifest, and nothing caught it, because a
gitignore mistake fails silently - the file is simply never offered for
commit, and `git status` stays clean.

The trap is the one the file warns about at the top, one level further down.
`effects/*` excludes the `effects/real` DIRECTORY entry; git then never
descends into it, so `!effects/*manifest*.json` cannot match anything inside.
Both runners write into exactly such a subdirectory:

    real.py      -> effects/real/            (or effects/real_preview/)
    analysis.py  -> analysis/real_<system>_kc<k_c>/

so every manifest for the headline numbers was unreachable while the rule
claimed otherwise. The fix needs three lines per directory, not one: descend
into the subdirectory, exclude its contents again, then rescue the manifests.
Getting only the first two right un-ignores the parquets instead, which is the
opposite failure and just as silent.

These tests assert both directions. A path that should be committable and a
path that must never be are different mistakes, and the first fix for one of
them broke the other.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _is_ignored(path: str) -> bool:
    """
    True when git would refuse to track `path`.

    Uses the exit code, not the output: `git check-ignore -v` also prints
    NEGATION patterns it matched, so a path rescued by a `!` rule looks
    identical to an ignored one if you read stdout. That is how the first
    version of this check reported the bug backwards.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=REPO, capture_output=True,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO / ".git").exists(),
    reason="needs a git checkout to evaluate .gitignore",
)


# Small, and each one makes a number traceable back to the run that produced it.
MUST_BE_TRACKABLE = [
    "effects/real/real_manifest.json",
    "effects/real_preview/real_manifest.json",
    "analysis/real_bm25_kc100/analysis_manifest.json",
    "analysis/real_dpr_kc1000/analysis_manifest.json",
    "analysis/real_preview_bm25_kc100/analysis_manifest.json",
    "analysis/feasibility_manifest.json",
    "analysis/h7_power_manifest.json",
    "cache/stage2_manifest.json",
    "cache/stage2_shard_anupam_manifest.json",
    "cache/stage2_shard_shubham_manifest.json",
    "data/universe_manifest.json",
    "data/query_split.json",
    "runs/bm25_manifest.json",
    "runs/bm25.parquet",          # run tables ARE committed, deliberately
]

# Large, regenerable from the committed run tables plus the frozen universe.
MUST_STAY_IGNORED = [
    "effects/real/bm25_stage_kc100.parquet",
    "effects/real/dpr_budget_kc500_random.parquet",
    "effects/real_preview/bm25_stage_kc100_condensed.parquet",
    "analysis/real_bm25_kc100/h7.csv",
    "analysis/real_bm25_kc100/unjudged_at_k.csv",
    "analysis/real_bm25_kc100/additivity.png",
    "analysis/stage1_floor_bm25.parquet",
    "analysis/allocation_promoted.parquet",
    "analysis/allocation_audit.parquet",
    "cache/stage2_shard_anupam.sqlite",
    "data/universe.parquet",
    "data/universe.frozen",
    "data/qrels.parquet",
]


@pytest.mark.parametrize("path", MUST_BE_TRACKABLE)
def test_traceability_artefacts_can_be_committed(path):
    assert not _is_ignored(path), (
        f"{path} is ignored, so the numbers it provenances cannot be "
        "committed. .gitignore's own rule is that anything making a number "
        "traceable is kept however small."
    )


@pytest.mark.parametrize("path", MUST_STAY_IGNORED)
def test_bulk_artefacts_stay_out_of_the_repo(path):
    assert _is_ignored(path), (
        f"{path} is trackable. It is large and regenerable, and committing it "
        "would put hundreds of MB of derived data in git history."
    )


def test_the_subdirectory_rescue_needs_all_three_lines():
    """
    Pins the shape of the fix, not just its outcome.

    `!effects/*/` alone lets git descend AND un-ignores everything inside.
    Without the `effects/*/*` line that follows it, the parquets become
    committable - which `test_bulk_artefacts_stay_out_of_the_repo` would catch,
    but only if someone remembers why the middle line is there. This states it.
    """
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    for top in ("effects", "analysis"):
        for line in (f"!{top}/*/", f"{top}/*/*", f"!{top}/*/*manifest*.json"):
            assert line in text.splitlines(), (
                f"missing {line!r} from .gitignore. All three lines are "
                f"required for {top}/<run>/ manifests to be committable while "
                "their bulk outputs stay ignored."
            )


def test_a_manifest_is_rescued_at_both_depths():
    """
    Manifests land directly in the directory (feasibility) and one level down
    (every real run). Both must work; only the second was ever broken.
    """
    assert not _is_ignored("analysis/feasibility_manifest.json")
    assert not _is_ignored("analysis/real_bm25_kc100/analysis_manifest.json")
