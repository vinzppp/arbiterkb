# Reproducibility

## Offline inspection and checks

Follow the README environment setup, then run `python -m pytest -q tests` and `python scripts/quality.py`. These use synthetic fixtures, make no model calls and do not read confirmation episode logs. Open the midpoint notebook to view its saved output; re-execution is not required.

The confirmation report and machine-readable aggregate are available directly in this repository. `scripts/confirmation_report.py` implements the analysis but requires the original per-cell result records to regenerate it. Those raw records are retained outside this publication; aggregate-only inspection cannot independently reconstruct paired confidence intervals. Do not infer that passing unit tests reproduces the experiment.

## Full experiment environment

The frozen manifest records the original host, model, effort, limits and external source revisions. The imported `results/robomme_common/pod_env/` files record Python dependencies and source commits. `stress/Dockerfile.robomme-cpu` is a diagnostic fallback, not the original GPU execution image. External MEMI/RoboMME/ManiSkill and their datasets are not bundled. No simulator environment or fresh dependency installation was tested during publication cleanup.

The manifest chain and MEMI prompt splits in `results/memory_or_policy/g0/memi_split/` are unchanged. Preserve hashes and the read ledger. A new source commit cannot masquerade as the original clean execution commit: provenance checks intentionally enforce the recorded source identity. No experiment, refreeze or paid API call is part of the quick start.

## History

Prototype results already present in the GitHub base remain intact. User-owned literature notes, export ZIPs, PDFs, datasets, raw episode logs and model-response caches were not added. Original working trees and untracked material remain untouched by this publication preparation.
