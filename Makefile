PY ?= python3
export ARBITER_DEBUG ?= 1

.PHONY: test quality selftest check eval notebook

test:
	$(PY) -m pytest -q tests

quality:
	$(PY) scripts/quality.py

selftest:
	$(PY) scripts/run.py --selftest

check: test quality selftest

eval:
	$(PY) scripts/run.py $(ARGS)

notebook:
	$(PY) scripts/make_notebook.py
	$(PY) -m jupyter nbconvert --to notebook --execute notebooks/m0_walkthrough.ipynb --output-dir executed
