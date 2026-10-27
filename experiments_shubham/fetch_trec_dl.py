"""
fetch_trec_dl.py - Write data/queries.tsv and data/qrels.txt for TREC-DL 2019 + 2020.

Downloads the official passage-ranking test queries and NIST judgments, keeps
only the judged queries (43 for 2019, 54 for 2020 = 97), and writes them in the
formats data/README.md expects:

    data/queries.tsv   query_id<TAB>query_text
    data/qrels.txt     TREC format: query_id iteration doc_id relevance

Both files are gitignored; every member regenerates identical copies with:

    python -m experiments_shubham.fetch_trec_dl
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

SOURCES = {
    "dl19": {
        "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
        "qrels": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
        "expected_queries": 43,
    },
    "dl20": {
        "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
        "qrels": "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
        "expected_queries": 54,
    },
}


def _get(url: str) -> bytes:
    # trec.nist.gov returns 403 for urllib's default User-Agent
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"})) as resp:
        return resp.read()


def fetch(out_dir: Path) -> tuple[Path, Path]:
    queries: dict[str, str] = {}
    qrel_lines: list[str] = []

    for name, src in SOURCES.items():
        qrels_text = _get(src["qrels"]).decode("utf-8")
        judged = set()
        for line in qrels_text.splitlines():
            parts = line.split()
            if len(parts) == 4:
                judged.add(parts[0])
                qrel_lines.append(" ".join(parts))

        raw = gzip.GzipFile(fileobj=io.BytesIO(_get(src["queries"]))).read().decode("utf-8")
        kept = 0
        for line in raw.splitlines():
            qid, text = line.split("\t", 1)
            if qid in judged:
                if qid in queries:
                    raise ValueError(f"query id {qid} appears in more than one year")
                queries[qid] = text.strip()
                kept += 1
        if kept != src["expected_queries"]:
            raise ValueError(f"{name}: expected {src['expected_queries']} judged queries, got {kept}")
        logger.info("%s: %d judged queries", name, kept)

    out_dir.mkdir(parents=True, exist_ok=True)
    q_path, r_path = out_dir / "queries.tsv", out_dir / "qrels.txt"
    q_path.write_text("".join(f"{q}\t{t}\n" for q, t in queries.items()), encoding="utf-8")
    r_path.write_text("\n".join(qrel_lines) + "\n", encoding="utf-8")
    logger.info("wrote %d queries to %s and %d judgments to %s",
                len(queries), q_path, len(qrel_lines), r_path)
    return q_path, r_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out-dir", default="data")
    fetch(Path(parser.parse_args().out_dir))


if __name__ == "__main__":
    main()
