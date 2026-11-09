# ArbiterKB — robot memory research

This project studies whether robot-task performance comes from memory structure or task instructions. ArbiterKB implements a belief-bearing knowledge base with evidence, retraction, event history and controller views. The later “Memory or policy?” study compares four memories × two instruction conditions, plus stock MEMI, using RoboMME’s fixed `multi_choice` executor.

## Status and scope

The confirmation study completed on 2026-11-03: **250 paired task/episode blocks, 2,250 attempts**, with no infrastructure errors or dropped pairs. The aggregate report supports the instruction main effect (+19.6 percentage points, 95% CI [16.1, 23.1]); the preregistered half-gain portability claim was falsified. Memory equivalence is **not established**. MEMI fields can encode action advice, so their advantage is not a pure representation effect.

This publication branch consolidates the prototype and completed study code, frozen manifests and aggregate evidence. Start with [the confirmation report](paper/CONFIRMATION_MEMORY_OR_POLICY_REPORT_ZH.md) and [evidence index](docs/RESEARCH_INDEX.md). Raw episode logs and external datasets are intentionally not distributed here; full statistical recomputation requires those retained research artifacts. Older handoffs describe historical plans, not current run instructions.

## Start here

- [Executed midpoint notebook](paper/notebook/arbiterkb_midpoint.ipynb): the earlier offline prototype walkthrough, **not** the later confirmation experiment. Open it without re-executing to inspect saved outputs.
- [Research goals and historical claims](docs/GOALS.md), then [evidence and history index](docs/RESEARCH_INDEX.md).
- Core implementation: [KB](arbiter_kb/kb.py), [beliefs](arbiter_kb/beliefs.py), [views](arbiter_kb/views.py), [controller](arbiter_kb/controller.py). Supporting unit tests are in `tests/`.

## Repository map

| Location | Contents |
|---|---|
| `arbiter_kb/` | Prototype memory, rules, sources, baselines and controller |
| `tests/` | Local unit and integration checks |
| `scripts/`, `stress/`, `tools/` | Earlier evaluation/reporting, pod runners and conversion tools |
| `results/latest/`, `figures/` | Earlier aggregate results and figures |
| `data/`, `results/runs/`, `results/llm_cache/` | Local ignored datasets, runs and cached responses; not supplied by a clone |
| `paper/` | Confirmation report and earlier executed midpoint notebook |
| `docs/`, `reviews/` | Goals, historical handoffs and review provenance |
| `notebooks/`, `executed/` | Earlier M0/M1 walkthroughs retained from the GitHub base |

## Minimal local setup and smoke check

Use Python 3.10 or newer (the code uses modern type syntax and `sys.stdlib_module_names`). From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
python scripts/quality.py
```

These checks require no API key or simulator. Dependencies are lower-bounded, not a locked environment; a fresh installation has not been verified by this cleanup. `make check` additionally runs synthetic evaluations and may write outputs; it is not the minimal smoke check. Full RoboMME reproduction requires external simulator/MEMI dependencies and the frozen protocol. See [reproducibility](docs/REPRODUCIBILITY.md). Never regenerate or refreeze study parameters merely to browse results.

## Completed-study code

[Common harness](stress/common_harness_pod.py), [factorial memories](arbiter_kb/mop_memory.py), [confirmation analysis](scripts/confirmation_report.py), and [locked manifest](experiments/memory_or_policy_confirmation_v1.json) are included with their supporting modules and tests.

## History and preservation

[DEVELOPMENT.md](DEVELOPMENT.md) is the prototype specification; [NOTES.md](NOTES.md) preserves dated decisions and preregistration amendments. [HANDOFF.md](HANDOFF.md) is historical. [Cleanup manifest](docs/CLEANUP_2026-11-09.md) records the reversible housekeeping and verification results.
