PYTHON ?= python
CONFIG ?= config/base.yaml

.PHONY: help test smoke lint freeze clean

help:
	@echo "test    run the full suite (no GPU, no index, no Java)"
	@echo "smoke   run the shared toy cascade end to end"
	@echo "lint    ruff over shared/ and tests/"
	@echo "freeze  print the current config fingerprint"

test:
	$(PYTHON) -m pytest tests

smoke:
	$(PYTHON) -c "from shared.fixtures import smoke_test; print('ok', smoke_test())"

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
