"""
dpr.py — Dense bi-encoder Stage 1 retriever (cosDPR-distil over MS MARCO).

Model:  castorini/cosdpr-distil — DPR architecture (BERT-base, shared
        query/passage tower: [CLS] → linear → L2-normalise) trained on
        MS MARCO.  Encoding mirrors Pyserini's CosDprEncoder, so results are
        comparable with Pyserini's published numbers (TREC-DL19 nDCG@10
        0.7250, DL20 0.7025, flat search).

Index:  No ANN index.  Passage embeddings are stored as float16 shards
        (emb_00000.npy + ids_00000.txt, 100k passages each, ~13.5 GB for the
        8.8M-passage corpus) and searched *exactly* by streaming the shards.
        For our ~100 evaluation queries this costs one pass over the shards,
        and it avoids the ~27 GB of RAM an in-memory flat index would need.

Usage:
    # 1) Encode the corpus (GPU).  Resumable: rerun after a timeout and
    #    finished shards are skipped.
    python -m retrievers.dpr encode --index-dir /path/to/dpr_index

    # 2) Check Stage 1 against the published baselines (no reranker)
    python -m retrievers.dpr verify --index-dir /path/to/dpr_index

    # 3) Full pipeline (index_dir from config/dpr.yaml or $DPR_INDEX_DIR)
    python -m experiments.run_pipeline --retriever dpr \\
        --retriever-config config/dpr.yaml --query-set dl19
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "castorini/cosdpr-distil"
EMBEDDING_DIM = 768
PASSAGE_MAX_LENGTH = 256      # Pyserini's default for passage encoding
DEFAULT_SHARD_SIZE = 100_000

# Pyserini 2CR, "cosDPR-distil: Faiss flat, PyTorch"
PUBLISHED_BASELINES = {
    "dl19": {"ndcg@10": 0.7250, "recall@1000": 0.8201},
    "dl20": {"ndcg@10": 0.7025, "recall@1000": 0.8533},
}


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────

class Encoder(Protocol):
    def encode(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        ...


class CosDPREncoder:
    """
    cosDPR-distil encoder: BERT [CLS] → linear → L2 normalise.

    The same tower encodes queries and passages (the checkpoint's query and
    passage heads are tied).  Returns float32 arrays of shape (n, 768).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        fp16: bool = False,
    ) -> None:
        from transformers import BertConfig, BertModel, BertTokenizerFast

        self.device = device or _default_device()
        self.fp16 = fp16 and self.device == "cuda"

        state_dict = torch.load(
            self._resolve(model_name, "pytorch_model.bin"), map_location="cpu"
        )

        # strict=True: fail loudly if the checkpoint layout ever changes
        self.bert = BertModel(BertConfig.from_pretrained(model_name),
                              add_pooling_layer=False)
        self.bert.load_state_dict({
            k[len("bert."):]: v for k, v in state_dict.items()
            if k.startswith("bert.")
            and not k.startswith("bert.pooler.")
            and k != "bert.embeddings.position_ids"
        }, strict=True)

        self.linear = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.linear.load_state_dict({
            "weight": state_dict["linear.weight"],
            "bias": state_dict["linear.bias"],
        })

        self.bert.to(self.device).eval()
        self.linear.to(self.device).eval()
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    @staticmethod
    def _resolve(model_name: str, filename: str) -> str:
        local = Path(model_name) / filename
        if local.exists():
            return str(local)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(model_name, filename)

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        """Encode texts.  max_length=None means BERT's limit (512), as
        Pyserini does for queries."""
        inputs = self.tokenizer(
            list(texts),
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",
        ).to(self.device)

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            cls = self.bert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).last_hidden_state[:, 0, :]
            emb = self.linear(cls)

        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        return emb.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────
# Corpus encoding  (resumable float16 shards)
# ──────────────────────────────────────────────────────────────────────

def _read_collection(path: Path) -> Iterable[tuple[str, str]]:
    """Yield (pid, text) from a two-column TSV."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            yield pid, text


def _check_meta(index_dir: Path, meta: dict) -> None:
    """Write index metadata, or verify it matches when resuming."""
    path = index_dir / "meta.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != meta:
            raise ValueError(
                f"{index_dir} was built with different settings "
                f"({existing}) than requested ({meta}); use a new index dir."
            )
    else:
        path.write_text(json.dumps(meta, indent=2))


def encode_corpus(
    index_dir: str | Path,
    collection_path: str | Path | None = None,
    encoder: Encoder | None = None,
    model_name: str = DEFAULT_MODEL,
    shard_size: int = DEFAULT_SHARD_SIZE,
    batch_size: int = 256,
    max_length: int = PASSAGE_MAX_LENGTH,
    device: str | None = None,
    fp16: bool = True,
) -> Path:
    """
    Encode the corpus into float16 shards under *index_dir*.

    Shard i holds collection rows [i*shard_size, (i+1)*shard_size).  A shard
    counts as done once emb_{i}.npy exists (written last, atomically), so an
    interrupted run can simply be restarted.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    _check_meta(index_dir, {
        "model_name": model_name,
        "max_length": max_length,
        "shard_size": shard_size,
        "dim": EMBEDDING_DIM,
        "dtype": "float16",
    })

    if collection_path is None:
        from shared.data import _ensure_collection_tsv
        collection_path = _ensure_collection_tsv()

    rows = _read_collection(Path(collection_path))
    n_done = n_new = 0
    start = time.perf_counter()

    for shard_no in itertools.count():
        batch_rows = list(itertools.islice(rows, shard_size))
        if not batch_rows:
            break

        emb_path = index_dir / f"emb_{shard_no:05d}.npy"
        if emb_path.exists():
            n_done += len(batch_rows)
            continue

        if encoder is None:
            encoder = CosDPREncoder(model_name, device=device, fp16=fp16)

        ids = [pid for pid, _ in batch_rows]
        texts = [text for _, text in batch_rows]

        # Encode longest-first so each batch pads to similar lengths
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        emb = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float16)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            emb[idx] = encoder.encode([texts[i] for i in idx], max_length=max_length)

        ids_tmp = index_dir / f"ids_{shard_no:05d}.txt.tmp"
        ids_tmp.write_text("\n".join(ids) + "\n")
        os.replace(ids_tmp, index_dir / f"ids_{shard_no:05d}.txt")

        emb_tmp = index_dir / f"emb_{shard_no:05d}.tmp.npy"
        np.save(emb_tmp, emb)
        os.replace(emb_tmp, emb_path)

        n_new += len(batch_rows)
        rate = n_new / (time.perf_counter() - start)
        logger.info(
            "Shard %d done: %d passages total (%d this run, %.0f passages/s)",
            shard_no, n_done + n_new, n_new, rate,
        )

    logger.info("Corpus encoded: %d passages in %s", n_done + n_new, index_dir)
    return index_dir


# ──────────────────────────────────────────────────────────────────────
# Exact search over shards
# ──────────────────────────────────────────────────────────────────────

def _shard_numbers(index_dir: Path) -> list[int]:
    nums = sorted(int(p.stem.split("_")[1]) for p in index_dir.glob("emb_*.npy")
                  if not p.stem.endswith(".tmp"))
    if not nums:
        raise FileNotFoundError(f"No embedding shards found in {index_dir}")
    if nums != list(range(len(nums))):
        raise ValueError(
            f"{index_dir} has missing shards {sorted(set(range(nums[-1] + 1)) - set(nums))}; "
            "finish encoding first."
        )
    return nums


def exact_search(
    query_emb: np.ndarray,
    index_dir: str | Path,
    k: int,
    device: str | None = None,
    query_block: int = 512,
) -> tuple[list[list[str]], np.ndarray]:
    """
    Exact maximum-inner-product search of *query_emb* (nq × dim) over all
    shards.  Returns (doc_ids per query, scores of shape nq × k), sorted by
    score descending.
    """
    index_dir = Path(index_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.from_numpy(np.asarray(query_emb, dtype=np.float32)).to(device)
    nq = q.shape[0]

    top_scores = torch.full((nq, 0), float("-inf"), device=device)
    top_rows = torch.zeros((nq, 0), dtype=torch.long, device=device)
    shard_ids: list[list[str]] = []
    offset = 0

    for shard_no in _shard_numbers(index_dir):
        ids = (index_dir / f"ids_{shard_no:05d}.txt").read_text().split("\n")[:-1]
        emb = np.load(index_dir / f"emb_{shard_no:05d}.npy", mmap_mode="r")
        if len(ids) != emb.shape[0]:
            raise ValueError(f"Shard {shard_no}: {len(ids)} ids vs {emb.shape[0]} vectors")
        shard_ids.append(ids)
        e = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)

        new_scores, new_rows = [], []
        for b in range(0, nq, query_block):
            s = q[b:b + query_block] @ e.T
            vals, idx = torch.topk(s, min(k, e.shape[0]), dim=1)
            new_scores.append(vals)
            new_rows.append(idx + offset)

        scores = torch.cat([top_scores, torch.cat(new_scores)], dim=1)
        rows = torch.cat([top_rows, torch.cat(new_rows)], dim=1)
        top_scores, pos = torch.topk(scores, min(k, scores.shape[1]), dim=1)
        top_rows = torch.gather(rows, 1, pos)
        offset += e.shape[0]

    # Global row → doc id
    starts = np.cumsum([0] + [len(ids) for ids in shard_ids])
    rows_np = top_rows.cpu().numpy()
    shard_of = np.searchsorted(starts, rows_np, side="right") - 1
    doc_ids = [
        [shard_ids[s][r - starts[s]] for s, r in zip(shard_of[i], rows_np[i])]
        for i in range(nq)
    ]
    return doc_ids, top_scores.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────
# Retriever (the interface experiments/run_pipeline.py expects)
# ──────────────────────────────────────────────────────────────────────

class DPRRetriever:
    """
    Stage 1 dense retriever.

    Config (config/dpr.yaml, under `retriever:`): index_dir, model_name,
    query_max_length, device.  $DPR_INDEX_DIR overrides index_dir, which is
    handy on Colab where the index lives on Google Drive.
    """

    def __init__(self, config: dict, encoder: Encoder | None = None) -> None:
        cfg = config.get("retriever", config)
        self.index_dir = Path(os.environ.get("DPR_INDEX_DIR") or cfg["index_dir"])
        self.query_max_length = cfg.get("query_max_length")
        self.device = cfg.get("device")
        self.encoder = encoder or CosDPREncoder(
            cfg.get("model_name", DEFAULT_MODEL), device=self.device
        )
        _shard_numbers(self.index_dir)  # fail fast if the index is missing

    def batch_search(
        self, queries: dict[str, str], k: int = 1000
    ) -> dict[str, list[tuple[str, float]]]:
        """Search all queries in one pass over the shards."""
        qids = list(queries)
        q_emb = self.encoder.encode(
            [queries[q] for q in qids], max_length=self.query_max_length
        )
        doc_ids, scores = exact_search(q_emb, self.index_dir, k, device=self.device)
        return {
            qid: list(zip(doc_ids[i], scores[i].tolist()))
            for i, qid in enumerate(qids)
        }

    def search(self, query: str, k: int = 1000) -> list[tuple[str, float]]:
        """Single query (one full pass over the shards; prefer batch_search)."""
        return self.batch_search({"_": query}, k)["_"]


# ──────────────────────────────────────────────────────────────────────
# Baseline verification (Stage 1 only)
# ──────────────────────────────────────────────────────────────────────

def recall_at_k(ranked: list[str], judged: dict[str, int], k: int = 1000,
                min_rel: int = 2) -> float:
    """Recall@k with relevance >= min_rel (trec_eval -l 2, as Pyserini
    reports for TREC-DL)."""
    relevant = {d for d, r in judged.items() if r >= min_rel}
    if not relevant:
        return float("nan")
    return len(relevant & set(ranked[:k])) / len(relevant)


def verify(index_dir: str | Path, query_sets=("dl19", "dl20"),
           model_name: str = DEFAULT_MODEL, device: str | None = None) -> dict:
    """Stage-1 nDCG@10 and Recall@1000 vs Pyserini's published numbers."""
    from shared.data import load_qrels, load_queries
    from shared.evaluation import ndcg

    retriever = DPRRetriever({"index_dir": str(index_dir), "model_name": model_name,
                              "device": device})
    report = {}
    for qs in query_sets:
        queries, qrels = load_queries(qs), load_qrels(qs)
        results = retriever.batch_search(queries, k=1000)
        ndcgs, recalls = [], []
        for qid, hits in results.items():
            ranked = [d for d, _ in hits]
            rels = np.array([qrels[qid].get(d, 0) for d in ranked], dtype=float)
            ndcgs.append(ndcg(rels, k=10, ideal_relevances=np.array(list(qrels[qid].values()))))
            recalls.append(recall_at_k(ranked, qrels[qid]))
        got = {"ndcg@10": float(np.mean(ndcgs)), "recall@1000": float(np.nanmean(recalls))}
        report[qs] = got
        for metric, value in got.items():
            target = PUBLISHED_BASELINES[qs][metric]
            ok = abs(value - target) <= 0.02
            logger.info("%s %-12s %.4f  (published %.4f)  %s", qs, metric, value,
                        target, "OK" if ok else "OUTSIDE ±0.02 — debug before experiments")
    return report


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from shared.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="cosDPR-distil corpus encoding and checks")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode the corpus into float16 shards (resumable)")
    enc.add_argument("--index-dir", required=True)
    enc.add_argument("--collection", default=None,
                     help="TSV of (pid, passage); default: MS MARCO via shared.data")
    enc.add_argument("--model", default=DEFAULT_MODEL)
    enc.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    enc.add_argument("--batch-size", type=int, default=256)
    enc.add_argument("--max-length", type=int, default=PASSAGE_MAX_LENGTH)
    enc.add_argument("--device", default=None)
    enc.add_argument("--no-fp16", action="store_true", help="Disable fp16 autocast on CUDA")

    ver = sub.add_parser("verify", help="Stage-1 nDCG@10 / R@1000 vs published baselines")
    ver.add_argument("--index-dir", required=True)
    ver.add_argument("--query-sets", nargs="+", default=["dl19", "dl20"])
    ver.add_argument("--model", default=DEFAULT_MODEL)
    ver.add_argument("--device", default=None)

    args = parser.parse_args()
    if args.command == "encode":
        encode_corpus(args.index_dir, args.collection, model_name=args.model,
                      shard_size=args.shard_size, batch_size=args.batch_size,
                      max_length=args.max_length, device=args.device,
                      fp16=not args.no_fp16)
    else:
        verify(args.index_dir, args.query_sets, args.model, args.device)


if __name__ == "__main__":
    main()
