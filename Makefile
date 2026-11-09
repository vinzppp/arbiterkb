PY ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
export ARBITER_DEBUG ?= 1

.PHONY: test quality selftest check eval

test:
	$(PY) -m pytest -q tests

quality:
	$(PY) scripts/quality.py

selftest:
	$(PY) scripts/run.py --selftest

check: test quality selftest

eval:
	$(PY) scripts/run.py $(ARGS)
