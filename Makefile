PYTHON ?= python
CONFIG ?= config/base.yaml

.PHONY: help test test-all smoke smoke-week1 lint freeze clean

help:
	@echo "test         run the shared suite (no GPU, no index, no Java)"
	@echo "test-all     shared plus every member's own suite"
	@echo "smoke        run the shared toy cascade end to end"
	@echo "smoke-week1  drive queries -> split -> runs -> frozen universe, no retriever"
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
