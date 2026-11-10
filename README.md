# ArbiterKB

ArbiterKB is a structured memory module for robot manipulation. It builds a knowledge base from perception and robot state, records the evidence behind its beliefs, and renders a view for a language-model controller. The completed **Memory or policy?** study asks how much task-specific instructions help when the executor stays fixed and the controller is given different forms of memory.

This research branch contains the experiment history. The [study publication branch](https://github.com/vinzppp/arbiterkb/tree/codex/research-publication) is the maintained starting point for readers.

## Start here

| If you want to… | Open |
| --- | --- |
| Read the completed experiment and its limitations | [Confirmation report](paper/CONFIRMATION_MEMORY_OR_POLICY_REPORT_ZH.md) |
| Inspect the results or the frozen protocol | [Evidence index](https://github.com/vinzppp/arbiterkb/blob/codex/research-publication/docs/RESEARCH_INDEX.md), [aggregate JSON](results/robomme_common/memory_or_policy_confirmation/confirmation_20261102T170540Z/confirmation_report.json), [frozen manifest](experiments/memory_or_policy_confirmation_v1.json) |
| See the memory module work on saved examples | [Executed midpoint notebook](https://github.com/vinzppp/arbiterkb/blob/codex/research-publication/paper/notebook/arbiterkb_midpoint.ipynb) |
| Understand the environment and which files are needed to reproduce the analysis | [Reproducibility guide](https://github.com/vinzppp/arbiterkb/blob/codex/research-publication/docs/REPRODUCIBILITY.md) |

The midpoint notebook covers the earlier offline prototype. The confirmation report covers the later closed-loop RoboMME comparison.

## How ArbiterKB works

1. **Observe.** The observation bridge converts simulator or perception records into named objects and events.
2. **Update memory.** The KB records object state and relations, including containment hypotheses when an object is hidden. Beliefs carry supporting evidence, and later observations can revise them.
3. **Present context.** The controller view turns relevant facts, uncertainty, and event history into text that the model can use to choose its next action.
4. **Execute and score.** In the confirmation study, the shared RoboMME `multi_choice` executor carries out the selected action. The study harness records the outcome, timeouts, and infrastructure retries.

| Component | Where to look |
| --- | --- |
| Beliefs, evidence, and memory updates | [KB](arbiter_kb/kb.py) and [belief types](arbiter_kb/beliefs.py) |
| How memory reaches the model | [Views](arbiter_kb/views.py) and [controller](arbiter_kb/controller.py) |
| Memory variants used in the completed study | [Memory implementations](arbiter_kb/mop_memory.py) |
| Execution, pairing, and retries | [Common harness](stress/common_harness_pod.py) |
| Analysis and report generation | [Confirmation analysis](scripts/confirmation_report.py) |

## What the completed study found

The experiment compared four memory variants with generic and matched task instructions, plus stock MEMI. It used GPT-6 Luna and a common executor across 250 paired task–episode blocks, for **2,250 scored attempts**. The five tasks were RouteStick, SwingXtimes, VideoUnmaskSwap, StopCube, and VideoRepick.

| Memory | Generic instructions | Matched task instructions |
| --- | ---: | ---: |
| Full History | 30.4% | 49.2% |
| Event Ledger | 41.6% | 47.2% |
| Learned Memory Fields | 37.6% | 80.0% |
| Relational KB | 22.4% | 34.0% |

Stock MEMI achieved **85.2%**. Across the four memories, matched instructions improved success by **19.6 percentage points** on average (95% CI **16.1–23.1**). Transferring those instructions to Full History recovered about **34%** of the stock MEMI advantage over generic Full History, below the preregistered half-gain threshold. The evidence does not establish equivalence among the memory variants. MEMI's learned fields can also carry action guidance, which limits attribution to memory structure alone. See the [confirmation report](paper/CONFIRMATION_MEMORY_OR_POLICY_REPORT_ZH.md) for task-level counts and uncertainty.

## Run the offline checks

Use Python 3.10 or newer. To start from the branch with the completed study:

```sh
git clone --branch codex/research-publication https://github.com/vinzppp/arbiterkb.git
cd arbiterkb
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q -rs tests
python scripts/quality.py
```

The standard tests run on a CPU without model calls. The MEMI integration tests require the pinned external MEMI environment. The G perception test requires the saved perception files under `data/g_perceived/`. See the [reproducibility guide](https://github.com/vinzppp/arbiterkb/blob/codex/research-publication/docs/REPRODUCIBILITY.md) for those dependencies.

## Inspect or reproduce the experiment

- **Inspect the results.** Use the saved [aggregate JSON](results/robomme_common/memory_or_policy_confirmation/confirmation_20261102T170540Z/confirmation_report.json) and [confirmation report](paper/CONFIRMATION_MEMORY_OR_POLICY_REPORT_ZH.md).
- **Recompute the paired statistics.** Use the frozen manifest, the analysis script, and the original per-attempt records. Contact the maintainer for those records. The published aggregate cannot reconstruct the paired intervals by itself.
- **Rerun the closed loop.** Use the [recorded environment and source revisions](https://github.com/vinzppp/arbiterkb/blob/codex/research-publication/docs/RESEARCH_INDEX.md), including the pinned MEMI and RoboMME/ManiSkill code, simulator data, and access to the recorded model. The manifest records the original run configuration and provenance.
