PYTHON ?= python
CONFIG ?= config/base.yaml

.PHONY: help test test-all smoke smoke-week1 verify-bm25 feasibility lint freeze clean

help:
	@echo "test         run the shared suite (no GPU, no index, no Java)"
	@echo "test-all     shared plus every member's own suite"
	@echo "smoke        run the shared toy cascade end to end"
	@echo "smoke-week1  drive queries -> split -> runs -> frozen universe, no retriever"
	@echo "verify-bm25  reproduce published BM25 effectiveness, or refuse"
	@echo "feasibility  the Week-2 gate: leverage, judgment density, gate-failure supply"
	@echo "lint         ruff over shared/ and tests/"
	@echo "freeze       print the current config fingerprint"

test:
	$(PYTHON) -m pytest tests

# experiments_shubham/tests/test_dpr.py needs torch, which the CPU-only
# environments do not have; it is skipped here rather than failing collection.
test-all:
	$(PYTHON) -m pytest tests experiments_anupam experiments_shubham \
		--ignore=experiments_shubham/tests/test_dpr.py

smoke:
	$(PYTHON) -c "from shared.fixtures import smoke_test; print('ok', smoke_test())"

smoke-week1:
	$(PYTHON) -m experiments_anupam.smoke_pipeline

# Needs the real runs and qrels, so it is not part of `test-all`.
verify-bm25:
	$(PYTHON) -m experiments_anupam.verify_bm25

feasibility:
	$(PYTHON) -m experiments_anupam.feasibility_report

lint:
	ruff check shared tests

freeze:
	@$(PYTHON) -c "from shared.config import load_config; \
	print(load_config('$(CONFIG)').fingerprint())"

# Deliberately does NOT remove cache/ or runs/: those are the expensive
# artefacts, and the frozen universe must never be casually regenerated.
clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
