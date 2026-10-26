# data/

Inputs and the frozen universe. Everything except this file is gitignored.

| file | produced by | notes |
|---|---|---|
| `queries.tsv` | you | `query_id<TAB>query_text` |
| `qrels.txt` | the collection | TREC format: `query_id iteration doc_id relevance` |
| `qrels.parquet` | M1 step 1.1.2 | validated; unjudged rows carry relevance=NA |
| `universe.parquet` | M1 step 3.5 | the frozen U_q |
| `universe.frozen` | M1 step 3.6 | the config fingerprint sidecar |
| `reference_subset.txt` | M1 step 3.8 | queries receiving the full no-gate world |

Converting raw qrels once:

```python
from shared.qrels import read_trec_qrels
from shared.schema import save
save(read_trec_qrels("data/qrels.txt"), "data/qrels.parquet", "qrels")
```

`universe.parquet` and `universe.frozen` travel together. The loader compares
the sidecar against the current fingerprint and refuses to proceed when they
disagree, because effects computed against different universes are not
comparable. Do not regenerate the universe partway through a series; if you
must, every effect table has to be recomputed.

`reference_subset.txt` is produced by the frozen sampling rule in the config,
never by hand and never by looking at effect magnitudes. Its size comes from
the H7 power simulation (ROADMAP step 1.2.8).
