# ArbiterKB under perception error: a comparative audit against MEMI and belief repair

**Research branch:** `research/perception-belief-robomme`  
**Audit date:** 2026-10-30 (America/Los_Angeles)  
**ArbiterKB comparison code:** `d77be1c`  
**MEMI release:** `bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0`

## Abstract

This report asks the narrowest comparison that the available evidence can answer without new simulator or API access.
First, it independently recomputes MEMI's released RoboMME closed-loop results from all public test logs. Second, it
compares a frozen repair ladder on ArbiterKB's held-out perceived VideoUnmaskSwap traces: raw deterministic tracking,
causal identity re-linking, re-linking plus a detector-only soft belief repair, and the full ArbiterKB. The two panels
are not pooled because MEMI uses simulator-derived semantic observations and RoboMME's privileged low-level executor,
whereas the ArbiterKB panel uses noisy perception and scores memory answers offline.

MEMI's release reproduces exactly (`ALL MATCH`) and shows large, paired closed-loop gains on the 800-episode test set
for every released policy. On the 90-episode perceived holdout (235 cube queries), ArbiterKB does **not** improve
top-1 accuracy over the simple repair baselines and does not improve AURC over detector-only soft repair. Its Brier
score is lower than deterministic re-linking, but the difference against soft repair is uncertain. The strongest
defensible conclusion is therefore negative: the present rule-rich KB has not established an advantage over a much
simpler belief-repair pipeline under perception noise. A direct MEMI-versus-ArbiterKB closed-loop claim remains
unmeasured.

## 1. Scope and provenance

The requested branch was not present in any local ref, worktree, reflog, Git bundle, or visible archive. The auditable
starting point was ArbiterKB commit `069beb7` plus the untracked midpoint paper archive. This research branch was
created from that commit; the source checkout and its `main` branch were never modified.

The local source assets used here are:

- 90 held-out G perception traces (episode suffixes 10--19 in each setting), content-hashed in
  `results/latest/paper_comparison/summary.json`;
- the frozen parameter hash `fe4e94112a74`;
- the training-selected `min_3d/0.02` repair setting, whose choice file is also content-hashed;
- MEMI's public repository and released logs at `bb6c37c`;
- the existing RoboMME closed-loop logs, used only as clearly labelled exploratory context.

No model API was called. No credential was read, printed, copied, or written.

## 2. Methods

### 2.1 MEMI release reproduction

For each of five released policies and three conditions, every test episode is parsed directly from MEMI's release:
800 episodes per condition, covering 16 tasks with 50 episodes each. `python -m memi.report` was also run independently
and ended in `ALL MATCH`. Success differences between MEMI and FullHist use a two-sided exact McNemar test paired by
task and episode. Wilson intervals are descriptive marginal intervals.

MEMI-Plan receives text observations produced from simulator semantic segmentation and uses RoboMME's ground-truth
low-level controller. MEMI-Act receives images and outputs waypoints. Its optimized memory schema and policy prompt are
task-specific and learned from demonstrations without weight updates.

### 2.2 Perceived repair ladder

The local held-out panel contains 90 VideoUnmaskSwap perception traces and 235 scored cube-to-container answers.
Parameters and the soft-repair scale were selected before this read on the train half. The four systems are:

1. **raw Bound:** deterministic cover following on raw tracker identities;
2. **relink Bound:** the same rule after causal Hungarian identity re-linking with smoothed velocity;
3. **relink Soft:** re-linking plus the frozen near-pass detector's continuous probability haircut, with no KB;
4. **ArbiterKB:** re-linking, cover beliefs, exclusivity, swap mixing, evidence support, and the full belief update path.

Metrics are top-1 accuracy, silent-failure rate at the frozen confidence threshold 0.8, coverage at that threshold,
area under the risk-coverage curve (AURC; lower is better), multiclass Brier score, and ECE. Pairwise differences use
5,000 paired bootstrap draws at the episode level. Accuracy also uses an exact paired McNemar test. Confidence
intervals on individual rates are Wilson intervals; because cube rows within an episode are correlated, the paired
episode bootstrap is the primary uncertainty analysis for method differences.

This is a post-hoc repair-ladder analysis of a previously read holdout, not a new preregistered test. No parameter was
changed after inspecting it.

## 3. Results

### 3.1 MEMI's released closed-loop results reproduce

| policy | FullHist | Init | MEMI | exact McNemar, MEMI vs FullHist |
|---|---:|---:|---:|---:|
| GPT-6 Astra-Plan | 730/800 (91.3%) | 739/800 (92.4%) | 795/800 (99.4%) | `p=5.06e-17` |
| GPT-6 Luna-Plan | 503/800 (62.9%) | 533/800 (66.6%) | 683/800 (85.4%) | `p=1.41e-31` |
| Claude Fable-Plan | 687/800 (85.9%) | 709/800 (88.6%) | 777/800 (97.1%) | `p=6.01e-19` |
| Claude Haiku-Plan | 464/800 (58.0%) | 591/800 (73.9%) | 726/800 (90.8%) | `p=1.35e-63` |
| GPT-6 Astra-Act | 682/800 (85.3%) | 706/800 (88.3%) | 776/800 (97.0%) | `p=4.25e-18` |

On VideoUnmaskSwap alone, MEMI versus FullHist is 49/50 versus 50/50 for Astra-Plan (`p=1.0`), 41/50 versus
17/50 for Luna-Plan (`p=3.03e-6`), 49/50 versus 48/50 for Fable-Plan (`p=1.0`), 48/50 versus 11/50 for
Haiku-Plan (`p=1.46e-11`), and 48/50 versus 45/50 for Astra-Act (`p=0.453`). The broad MEMI result is strong, but
on this one task it is model-dependent and is not a universal gain over FullHist.

MEMI also does not establish a compression result by itself. Median Plan input tokens per episode are 8,044 versus
8,095 for Astra, 9,240 versus 6,417 for Luna, 11,368 versus 10,298 for Fable, and 12,287 versus 8,051 for Haiku
(MEMI versus FullHist). For Astra-Act, median logged cost is $0.428 versus $0.158. MEMI buys success, not consistently
lower inference cost.

### 3.2 Held-out perceived belief repair

| system | accuracy (Wilson 95%) | silent failure | coverage | AURC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| raw Bound | 80.9% [75.3, 85.4] | 19.1% | 100.0% | 0.1915 | 0.4043 | 0.1915 |
| relink Bound | 82.1% [76.7, 86.5] | 17.9% | 100.0% | 0.1787 | 0.3787 | 0.1787 |
| relink Soft | 82.1% [76.7, 86.5] | 16.6% | 93.6% | **0.1386** | 0.3377 | 0.1578 |
| ArbiterKB | 81.3% [75.8, 85.7] | **15.3%** | 91.5% | 0.1435 | **0.3201** | **0.1414** |

Pairwise episode-bootstrap differences are reported as ArbiterKB minus baseline:

| baseline | accuracy difference | AURC difference | Brier difference | McNemar p |
|---|---:|---:|---:|---:|
| raw Bound | +0.004 [-0.070, +0.081] | -0.048 [-0.122, +0.024] | -0.084 [-0.218, +0.042] | 1.000 |
| relink Bound | -0.009 [-0.032, +0.013] | -0.035 [-0.098, +0.030] | -0.059 [-0.104, -0.018] | 0.727 |
| relink Soft | -0.009 [-0.032, +0.013] | +0.005 [-0.048, +0.059] | -0.018 [-0.052, +0.013] | 0.727 |

ArbiterKB's lower silent-failure count partly comes from answering fewer cases above threshold. AURC removes that
single-threshold advantage, and the full KB is statistically indistinguishable from soft repair on AURC. The Brier
gain over deterministic re-linking is the clearest positive signal, but it is post-hoc and one of several comparisons;
it does not survive as a distinct advantage over the soft baseline.

### 3.3 Existing ArbiterKB closed-loop logs are exploratory only

On VideoUnmaskSwap validation episodes 0--9 with GPT-6 Astra, the existing waypoint runs were H 9/10, R 6/10, and
K 8/10. K used the now-known gripper-0 self-memory bug. A later two-pick K run was 9/20 and used the same bug. These
small validation results are useful failure analysis, but they are not included in the primary comparison and cannot
be pooled with fixed-code results.

## 4. What can and cannot be concluded

Supported conclusions:

1. The MEMI release is internally reproducible and demonstrates that task-optimized structured memory can greatly
   improve closed-loop RoboMME success, especially for weaker planners, under MEMI's observation/execution setup.
2. Causal identity re-linking is a useful low-complexity repair, but its accuracy gain over raw tracking is small on
   this holdout.
3. Continuous uncertainty repair improves selective risk relative to deterministic memory.
4. The current full ArbiterKB does not beat detector-only soft repair on accuracy or AURC. Its best signal is better
   probability scoring than deterministic re-linking, not better decisions.

Unsupported conclusions:

- ArbiterKB beats MEMI. No common-observation, common-model, common-executor closed-loop experiment was run.
- ArbiterKB improves RoboMME closed-loop success over simple belief repair. The required fixed-code test run does not
  exist.
- ArbiterKB's old claim-3 advantage remains valid after the current code and stronger repair ladder. The new audit
  supersedes that optimistic reading for paper claims.
- A learned task-specific memory is worse or better than a generic KB under noisy perception. That interaction is
  precisely the missing experiment.

## 5. Blocker audit

The local machine cannot run the required closed loop. All five simulator preflight checks fail before environment
creation because `robomme` is not installed; the host is macOS with no NVIDIA device. MEMI's own instructions require
an NVIDIA GPU for rollouts. No attached pod or GPU worktree was available. Installing only the Python package would
not remove the Linux/CUDA/Vulkan and perception-model requirements, so no synthetic success numbers were substituted.

The exact failure traces are preserved in
`results/latest/paper_comparison/local_environment_audit.json`. The direct experiment that would close this gap is
specified in `paper/DIRECT_CLOSED_LOOP_PROTOCOL.md`.

## 6. Final assessment

The most defensible paper today is a careful negative/diagnostic result about the perception-to-belief bottleneck,
not a claim that ArbiterKB outperforms existing memory systems. MEMI shows that memory design matters when observation
and execution are strong. ArbiterKB shows that, once identities are noisy, a sophisticated belief graph can fail to
outperform a compact detector-only repair. That narrows the research target: the next contribution must improve the
actual identity evidence or demonstrate that calibrated uncertainty changes closed-loop actions, rather than adding
more symbolic structure around the same corrupted tracks.

## 7. Reproduction

From this branch, with the ignored `data` path pointing at the local source data and a MEMI checkout at `bb6c37c`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make check
.venv/bin/python scripts/paper_compare.py --memi-root /path/to/memi --split test
```

The machine-readable result, every data SHA-256, exact paired tests, MEMI verification digest, and source commits are
in `results/latest/paper_comparison/summary.json`.
