"""
make_kaggle_notebook.py - Derive the Kaggle fill notebook from the Colab one.

    python -m experiments_shubham.make_kaggle_notebook

The two notebooks must differ ONLY in platform plumbing. Generating one from
the other is how that stays true: every scientific cell - the universe
rebuild, the shard plan, the reranker setup, the shuffle-invariance check, the
fill loop, the completeness verification and the manifest - is copied
byte-identically, and this script asserts it afterwards. Hand-maintaining two
notebooks would let them drift, and a drift in the fill loop would mean the
three shards were not produced by the same procedure.

Two cells are replaced:

  * cell 2, which clones into /content and names a Drive directory;
  * cell 8, which mounts Drive to restore a partial shard.

On Kaggle, /kaggle/working persists for the session and is preserved as the
notebook's output, so it plays Drive's role with no mount. Everything
downstream refers to `SHARD` and `saved`, so nothing else needs to change.

ONE CELL IS ADDED, the cross-platform canary. See its own commentary.
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path("experiments_shubham/cache_fill_colab.ipynb")
DST = Path("experiments_shubham/cache_fill_kaggle.ipynb")

# Cells copied verbatim; the fill procedure lives here and must not diverge.
# 7 is excluded only because its heading names Drive, which Kaggle has not.
SCIENTIFIC_CELLS = (1, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14)

CELL_7 = """\
## 4. Resume a partial shard, if there is one

`/kaggle/working/cache/` survives within a session, so a notebook that died
mid-fill resumes from whatever it had written. Run All again; the cache is
keyed on `(query_id, doc_id, model)` and already-scored pairs are skipped.
"""

CELL_0 = """\
# Stage-2 cache fill: one shard (Week 3) — Kaggle

Generated from `cache_fill_colab.ipynb` by
`experiments_shubham/make_kaggle_notebook.py`. Only the two platform cells
differ; every scientific cell is byte-identical, which the generator asserts.

**Before running:** Settings → Accelerator → **GPU T4 x2** (one is used), and
Settings → **Internet: On** (the clone, pip and the ir_datasets download all
need it). Then Run All. The shard is written to `/kaggle/working/cache/`,
which survives the session and is downloadable from the Output tab.

If the session dies, Run All again: the cache is keyed on
`(query_id, doc_id, model)` and the fill resumes from whatever is already in
`/kaggle/working/cache/`.
"""

CELL_2 = """\
OWNER = 'anupam'           # anupam | shubham | chaitanya
BRANCH = 'main'

# /kaggle/working persists for the session and is kept as notebook output, so
# it does the job Drive does on Colab - with no mount and no authorisation.
DRIVE_DIR = '/kaggle/working/cache'

!rm -rf /kaggle/working/IRE_PROJECT
!git clone -q -b {BRANCH} https://github.com/a-n-u-p-a-m/IRE_PROJECT.git /kaggle/working/IRE_PROJECT
%cd /kaggle/working/IRE_PROJECT
!git log --oneline -1
!pip install -q ir_datasets sentence-transformers
!python -m experiments_shubham.fetch_trec_dl
"""

CELL_8 = """\
import os, shutil

# No drive.mount: DRIVE_DIR is already a persistent local path on Kaggle.
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs('cache', exist_ok=True)
SHARD = f'cache/stage2_shard_{OWNER}.sqlite'
saved = f'{DRIVE_DIR}/stage2_shard_{OWNER}.sqlite'
if os.path.exists(saved) and not os.path.exists(SHARD):
    shutil.copy(saved, SHARD)
    print('restored', saved)
"""

CANARY_MD = """\
## 8. Cross-machine canary — post this digest

The three shards are disjoint, so no pair is ever scored twice and a numerical
difference between the machines that produced them is **silent**: each shard's
digest covers only its own pairs, and `merge_from` unions them without ever
comparing like with like.

Shard 2 was filled under `torch 2.11.0+cu130` on Python 3.13.16. Kaggle will
differ, and float32 matmul is not bit-reproducible across cuBLAS versions.

`shared/canary.py` scores eight frozen literal pairs — no corpus, no universe,
no filled shard, milliseconds. **Post `CANARY_DIGEST` in the group.** All three
must match. If they differ, `compare_canaries` says whether the merge is still
sound; a difference in *ordering* means it is not.
"""

CANARY_CODE = """\
from shared.canary import canary_digest, canary_scores

CANARY_SCORES = canary_scores(reranker)
CANARY_DIGEST = canary_digest(reranker)

for d, s in CANARY_SCORES.items():
    print(f'{d}  {s: .6f}')
print('\\nCANARY_DIGEST', CANARY_DIGEST, '  <- post this')
print('torch', torch.__version__, '| gpu', torch.cuda.get_device_name(0))
"""


def build() -> tuple[int, int]:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def md(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {},
                "source": text.splitlines(keepends=True)}

    def code(text: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": text.splitlines(keepends=True)}

    out = []
    for i, cell in enumerate(cells):
        if i == 0:
            out.append(md(CELL_0))
        elif i == 2:
            out.append(code(CELL_2))
        elif i == 7:
            out.append(md(CELL_7))
        elif i == 8:
            out.append(code(CELL_8))
        else:
            out.append(json.loads(json.dumps(cell)))   # verbatim copy

    # The canary goes after the manifest cell: by then `universe`, `fetcher`,
    # `reranker` and `query_texts` all exist, and the shard is already saved,
    # so a failure here cannot cost the fill.
    out.append(md(CANARY_MD))
    out.append(code(CANARY_CODE))

    nb["cells"] = out
    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    # Assert the invariant this script exists to maintain.
    built = json.loads(DST.read_text(encoding="utf-8"))["cells"]
    for i in SCIENTIFIC_CELLS:
        if "".join(built[i]["source"]) != "".join(cells[i]["source"]):
            raise SystemExit(
                f"cell {i} differs between the Colab and Kaggle notebooks. "
                "Only platform cells 0, 2 and 8 may differ; a change in any "
                "other cell means the three shards would not be produced by "
                "the same procedure."
            )
    return len(cells), len(out)


def main() -> int:
    before, after = build()
    print(f"wrote {DST}: {before} cells in, {after} out "
          f"(2 platform cells replaced, 1 header, 2 appended for the canary)")
    print(f"verbatim: cells {SCIENTIFIC_CELLS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
