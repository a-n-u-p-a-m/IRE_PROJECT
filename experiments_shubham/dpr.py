"""
dpr.py - Dense Stage 1: cosDPR-distil with exact search over float16 shards.

Model   castorini/cosdpr-distil: DPR architecture (BERT-base, shared query and
        passage tower, [CLS] -> linear -> L2-normalise), trained on MS MARCO.
        The encoder mirrors Pyserini's CosDprEncoder and was checked against
        Castorini's ONNX export (max |diff| 5e-7 on queries and passages).

Index   89 float16 shards of 100k passages (emb_XXXXX.npy + ids_XXXXX.txt,
        ~13.5 GB) built from the full MS MARCO passage collection. Search is
        exact: one pass over all shards, no ANN approximation. Shards can be
        read from a local directory or streamed one at a time from an rclone
        remote, so the index never has to fit on local disk.

Ranking follows shared.ordering: score descending, ties broken by ascending
doc_id. Candidates are over-fetched by a margin so a tie at the depth cut-off
can never be resolved arbitrarily; if the margin is not enough, the search
raises instead of guessing.

Index build (one-off; done in Oct 2026):
    python -m experiments_shubham.dpr encode --collection collection.tsv --index-dir DIR
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "castorini/cosdpr-distil"
EMBEDDING_DIM = 768
PASSAGE_MAX_LENGTH = 256          # Pyserini's default for passage encoding
DEFAULT_SHARD_SIZE = 100_000
TIE_MARGIN = 64                   # extra candidates fetched to resolve boundary ties


# torch is imported inside the functions that use it, so the module (and the
# run_dpr manifest code) imports on machines without torch.

def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────

class CosDPREncoder:
    """BERT [CLS] -> linear -> L2 normalise; one tower for queries and passages."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 fp16: bool = False) -> None:
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast

        self.model_name = model_name
        self.device = device or _default_device()
        self.fp16 = fp16 and self.device == "cuda"
        weights_path = self._resolve(model_name, "pytorch_model.bin")
        state = torch.load(weights_path, map_location="cpu")

        # strict=True: fail loudly if the checkpoint layout ever changes
        self.bert = BertModel(BertConfig.from_pretrained(model_name), add_pooling_layer=False)
        self.bert.load_state_dict({
            k[len("bert."):]: v for k, v in state.items()
            if k.startswith("bert.") and not k.startswith("bert.pooler.")
            and k != "bert.embeddings.position_ids"
        }, strict=True)
        self.linear = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.linear.load_state_dict({"weight": state["linear.weight"], "bias": state["linear.bias"]})
        self.bert.to(self.device).eval()
        self.linear.to(self.device).eval()
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        # The snapshot hash pins the exact weights in the run's checkpoint column
        self.revision = Path(weights_path).parent.name

    @staticmethod
    def _resolve(model_name: str, filename: str) -> str:
        local = Path(model_name) / filename
        if local.exists():
            return str(local)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(model_name, filename)

    def encode(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        """float32 (n, 768). max_length=None means BERT's 512, as Pyserini does for queries."""
        import torch

        inputs = self.tokenizer(list(texts), max_length=max_length, padding="longest",
                                truncation=True, return_token_type_ids=False,
                                return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            cls = self.bert(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"]).last_hidden_state[:, 0, :]
            emb = self.linear(cls)
        return torch.nn.functional.normalize(emb.float(), p=2, dim=1).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────
# Shard sources
# ──────────────────────────────────────────────────────────────────────

class LocalShards:
    """Shards in a local directory."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.meta = json.loads((self.index_dir / "meta.json").read_text())
        self.numbers = _contiguous(sorted(
            int(p.stem.split("_")[1]) for p in self.index_dir.glob("emb_*.npy")
            if not p.stem.endswith(".tmp")))
        self.description = str(self.index_dir)

    def __iter__(self) -> Iterator[tuple[list[str], np.ndarray]]:
        for n in self.numbers:
            yield _read_shard(self.index_dir, n)


class RcloneShards:
    """Shards streamed from an rclone remote, one or two at a time on local disk."""

    def __init__(self, remote: str, prefetch: int = 2) -> None:
        self.remote = remote.rstrip("/")
        self.prefetch = prefetch
        self.meta = json.loads(self._run(["cat", f"{self.remote}/meta.json"]))
        names = self._run(["lsf", self.remote, "--include", "emb_*.npy"]).split()
        self.numbers = _contiguous(sorted(int(n.split("_")[1].split(".")[0]) for n in names))
        self.description = self.remote

    @staticmethod
    def _run(args: list[str]) -> str:
        out = subprocess.run(["rclone", *args], capture_output=True, text=True, check=True)
        return out.stdout

    def _fetch(self, n: int, tmp: Path) -> Path:
        shard_dir = tmp / f"{n:05d}"
        shard_dir.mkdir()
        for name in (f"ids_{n:05d}.txt", f"emb_{n:05d}.npy"):
            subprocess.run(["rclone", "copyto", f"{self.remote}/{name}", str(shard_dir / name)],
                           check=True, capture_output=True)
        return shard_dir

    def __iter__(self) -> Iterator[tuple[list[str], np.ndarray]]:
        with tempfile.TemporaryDirectory(prefix="dpr_shards_") as tmp, \
                ThreadPoolExecutor(max_workers=self.prefetch) as pool:
            futures = {n: pool.submit(self._fetch, n, Path(tmp))
                       for n in self.numbers[:self.prefetch]}
            for i, n in enumerate(self.numbers):
                shard_dir = futures.pop(n).result()
                nxt = i + self.prefetch
                if nxt < len(self.numbers):
                    futures[self.numbers[nxt]] = pool.submit(self._fetch, self.numbers[nxt], Path(tmp))
                yield _read_shard(shard_dir, n, mmap=False)
                shutil.rmtree(shard_dir)


def _contiguous(numbers: list[int]) -> list[int]:
    if not numbers:
        raise FileNotFoundError("no embedding shards found")
    if numbers != list(range(len(numbers))):
        missing = sorted(set(range(numbers[-1] + 1)) - set(numbers))
        raise ValueError(f"index has missing shards {missing}; finish encoding first")
    return numbers


def _read_shard(directory: Path, n: int, mmap: bool = True) -> tuple[list[str], np.ndarray]:
    ids = (directory / f"ids_{n:05d}.txt").read_text().split("\n")[:-1]
    emb = np.load(directory / f"emb_{n:05d}.npy", mmap_mode="r" if mmap else None)
    if len(ids) != emb.shape[0]:
        raise ValueError(f"shard {n}: {len(ids)} ids vs {emb.shape[0]} vectors")
    return ids, emb


def open_shards(location: str) -> LocalShards | RcloneShards:
    """A local directory, or an rclone remote such as gdrive:IRE_PROJECT/dpr_index_cosdpr_distil."""
    if Path(location).is_dir():
        return LocalShards(location)
    if ":" in location:
        return RcloneShards(location)
    raise FileNotFoundError(f"{location} is neither a local directory nor an rclone remote")


# ──────────────────────────────────────────────────────────────────────
# Exact search
# ──────────────────────────────────────────────────────────────────────

def exact_search(query_emb: np.ndarray, shards, k: int, margin: int = TIE_MARGIN,
                 device: str = "cpu") -> list[list[tuple[str, float]]]:
    """
    Exact top-k by inner product for every query, ordered by (-score, doc_id).

    Over-fetches k + margin candidates so documents tied with the k-th score
    are all seen; raises if the margin was not enough to prove that.
    """
    import torch

    q = torch.from_numpy(np.asarray(query_emb, dtype=np.float32)).to(device)
    nq, fetch = q.shape[0], k + margin
    best_scores = torch.full((nq, 0), float("-inf"), device=device)
    best_ids = np.empty((nq, 0), dtype=object)

    for ids, emb in shards:
        e = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
        scores = q @ e.T
        vals, idx = torch.topk(scores, min(fetch, e.shape[0]), dim=1)
        shard_ids = np.asarray(ids, dtype=object)[idx.cpu().numpy()]
        merged_scores = torch.cat([best_scores, vals], dim=1)
        merged_ids = np.concatenate([best_ids, shard_ids], axis=1)
        best_scores, pos = torch.topk(merged_scores, min(fetch, merged_scores.shape[1]), dim=1)
        best_ids = np.take_along_axis(merged_ids, pos.cpu().numpy(), axis=1)

    results = []
    scores_np = best_scores.cpu().numpy().astype(np.float64)
    for i in range(nq):
        ranked = sorted(zip(best_ids[i].tolist(), scores_np[i].tolist()),
                        key=lambda t: (-t[1], t[0]))
        if len(ranked) > k and ranked[-1][1] >= ranked[k - 1][1]:
            raise ValueError(
                f"query {i}: more than {margin} documents tie with the depth-{k} score; "
                "raise the margin so the cut-off is resolved by doc_id, not by chance")
        results.append(ranked[:k])
    return results


# ──────────────────────────────────────────────────────────────────────
# Corpus encoding (how the existing index was built)
# ──────────────────────────────────────────────────────────────────────

def encode_corpus(index_dir: str | Path, collection: str | Path, model_name: str = DEFAULT_MODEL,
                  shard_size: int = DEFAULT_SHARD_SIZE, batch_size: int = 256,
                  max_length: int = PASSAGE_MAX_LENGTH, device: str | None = None,
                  fp16: bool = True, encoder=None) -> Path:
    """
    Encode a (pid TAB text) collection into float16 shards. Resumable: a shard
    counts as done once its emb file exists (written last, atomically).
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model_name": model_name, "max_length": max_length, "shard_size": shard_size,
            "dim": EMBEDDING_DIM, "dtype": "float16"}
    meta_path = index_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError(f"{index_dir} was built with different settings; use a new directory")
    meta_path.write_text(json.dumps(meta, indent=2))

    with open(collection, encoding="utf-8") as f:
        rows = (line.rstrip("\n").split("\t", 1) for line in f)
        start, n_new = time.perf_counter(), 0
        for shard_no in itertools.count():
            batch = list(itertools.islice(rows, shard_size))
            if not batch:
                break
            emb_path = index_dir / f"emb_{shard_no:05d}.npy"
            if emb_path.exists():
                continue
            encoder = encoder or CosDPREncoder(model_name, device=device, fp16=fp16)
            texts = [t for _, t in batch]
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
            emb = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float16)
            for b in range(0, len(order), batch_size):
                idx = order[b:b + batch_size]
                emb[idx] = encoder.encode([texts[i] for i in idx], max_length=max_length)
            ids_tmp = index_dir / f"ids_{shard_no:05d}.txt.tmp"
            ids_tmp.write_text("".join(f"{pid}\n" for pid, _ in batch))
            os.replace(ids_tmp, index_dir / f"ids_{shard_no:05d}.txt")
            emb_tmp = index_dir / f"emb_{shard_no:05d}.tmp.npy"
            np.save(emb_tmp, emb)
            os.replace(emb_tmp, emb_path)
            n_new += len(batch)
            logger.info("shard %d done (%.0f passages/s)", shard_no,
                        n_new / (time.perf_counter() - start))
    return index_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="cosDPR-distil index tools")
    sub = parser.add_subparsers(dest="command", required=True)
    enc = sub.add_parser("encode", help="encode a collection into float16 shards (resumable)")
    enc.add_argument("--collection", required=True, help="TSV of pid TAB passage")
    enc.add_argument("--index-dir", required=True)
    enc.add_argument("--batch-size", type=int, default=256)
    enc.add_argument("--device", default=None)
    args = parser.parse_args()
    encode_corpus(args.index_dir, args.collection, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
