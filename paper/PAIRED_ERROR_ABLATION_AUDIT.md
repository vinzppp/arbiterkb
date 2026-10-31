# Why the full ArbiterKB does not beat soft belief repair

**Branch:** `research/perception-belief-robomme`
**Frozen analysis commit:** `392de75`
**Parameter hash:** `fe4e94112a74`
**Status:** post-hoc diagnostic audit; no test-set tuning

## Abstract

We audit the negative result that full ArbiterKB does not outperform causal identity re-linking plus a detector-only
soft belief repair on held-out perceived VideoUnmaskSwap traces. The audit aligns every cube query, separates top-1
prediction flips from probability-only changes, runs targeted rule ablations on both train and the already-read test
split, resamples by episode, and corrects all reported Brier scores to include ArbiterKB's explicit
`None/elsewhere` outcome.

On test (90 episodes, 235 queries), full ArbiterKB scores 81.28% accuracy, 0.1435 AURC, and 0.3370 Brier; soft repair
scores 82.13%, 0.1386, and 0.3377. The paired differences are respectively -0.85 percentage points
(95% episode-bootstrap CI -3.17 to +1.35; exact McNemar `p=0.727`), +0.0049 AURC
(-0.0479 to +0.0597), and -0.0007 Brier (-0.0422 to +0.0404). None is distinguishable from zero, with or without
Holm correction. Train shows the same lack of an overall advantage.

The failure is compositional rather than a universal failure of belief updates. R1 replaces soft repair's strong
re-linked Bound anchor with a cover reconstruction and loses one net top-1 on test. R6 materially improves Brier
inside the KB but loses one additional net top-1. R5 changes only four final test distributions, fixes no prediction,
and makes two already-wrong `None` beliefs more extreme. R4 has exactly zero effect on either split. Most rows receive
small probability changes; a few large, opposing repairs determine the mean. The full rule stack therefore adds no
stable information beyond the soft detector and sometimes amplifies weak cover evidence.

## 1. Experimental design

The unit presented to each system is the same causally re-linked perception trace. Each final Q1 query asks which
container hides a cube. The compared systems are:

- **relink Bound:** deterministic answer from the re-linked tracker;
- **relink Soft:** the Bound answer plus R6's continuous near-pass mixing, without a KB;
- **R1 only:** ArbiterKB cover reconstruction without R5 or R6;
- **full ArbiterKB:** R1 cover kernel, R5 same-frame exclusivity, and R6 swap mixing;
- **no R4, no R5, no R6:** one-rule removals from the full system;
- **point R1:** an almost point-mass R1 (`sigma=0.001`, no `None` floors), retaining the other rules.

The targeted variants diagnose mechanisms; they were not selected as replacement models. The test split had already
been read, so every test result is explicitly post-hoc and is recorded in the append-only ledger.

Uncertainty is evaluated with 5,000 episode-cluster bootstrap draws. Accuracy additionally uses the two-sided exact
McNemar test. AURC and Brier use two-sided paired randomization tests that swap whole episodes. For each split, Holm
correction controls family-wise error across eight contrasts and three metrics (24 tests). Rows within an episode are
never treated as independent.

## 2. Corrected aggregate results

| split / system | accuracy | AURC ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| train / relink Bound | 83.40% | 0.1660 | 0.3382 | 0.1660 |
| train / relink Soft | 83.40% | **0.1214** | 0.3144 | 0.1425 |
| train / full ArbiterKB | 83.40% | 0.1319 | **0.3013** | **0.1071** |
| test / relink Bound | 82.13% | 0.1787 | 0.3787 | 0.1787 |
| test / relink Soft | **82.13%** | **0.1386** | 0.3377 | 0.1578 |
| test / full ArbiterKB | 81.28% | 0.1435 | **0.3370** | **0.1414** |

Full versus soft is inconclusive on every primary metric:

| split | accuracy difference | AURC difference | Brier difference |
|---|---:|---:|---:|
| train | 0.0000 [-0.0204, +0.0207] | +0.0105 [-0.0305, +0.0503] | -0.0131 [-0.0556, +0.0296] |
| test | -0.0085 [-0.0317, +0.0135] | +0.0049 [-0.0479, +0.0597] | -0.0007 [-0.0422, +0.0404] |

All brackets are 95% episode-bootstrap intervals. Test exact McNemar is `p=0.7266`; the episode-randomization
`p`-values for AURC and Brier are `0.8874` and `0.9750`. All three Holm-adjusted values are `1.0`.

### Brier audit correction

The previous comparison computed Brier only over container bins. ArbiterKB can also assign probability to
`None/elsewhere`, so omitting that class failed to penalize a confident unknown prediction. For example, a wrong
`p(None)=0.99` prediction contributed about 1 rather than about 2. The corrected outcome space is every container plus
the explicit `None` class for every system. This changes ArbiterKB test Brier from 0.3201 to 0.3370 and its difference
from soft repair from -0.0177 to -0.0007. Accuracy, AURC, ECE, and every prediction are unchanged.

## 3. Paired error anatomy

Full and soft agree on 223/235 test predictions. Among the 12 prediction disagreements:

- full alone is correct on 3;
- soft alone is correct on 5;
- both are correct on 2 set-valued-truth queries but select different accepted containers;
- both are wrong on 2.

Thus the accuracy gap comes from only eight correctness-discordant rows, with no statistical evidence of a stable
direction. The cases nevertheless identify three repeatable mechanisms.

### 3.1 The initial anchor differs

R1-only and relink Bound differ on 10 test predictions. R1 wins 3 correctness flips and loses 4, a net loss of one.
Five important R1 failures occur when no container is accepted inside `r_cover`: R1 uses the nearest visible
container versus `None` at an even 0.5/0.5 prior, while Bound retains the tracked cover identity. One further query
has no final `Inside` distribution and becomes `{None: 1}`. These are not errors that more downstream symbolic rules
can repair reliably; the two pipelines begin from different evidence anchors.

### 3.2 R6 improves probabilities but not top-1

Removing R6 raises test Brier by +0.0380 (95% CI +0.0162 to +0.0662). Its episode-randomization `p=0.0004` and
Holm-adjusted `p=0.0096` are the strongest ablation result. The direction repeats on train: +0.0180
(+0.0068 to +0.0322), Holm `p=0.0230`. R6 is therefore useful calibration machinery inside the KB.

However, full versus R1-only loses one net top-1 on test and three on train. On test, R6 changes five final top-1
answers; four remain wrong and one turns a correct 0.5 tie into a wrong `None` argmax after small mass is mixed to a
third container. This reconciles the apparently conflicting findings: R6 improves the distribution on average while
not improving, and occasionally harming, the discrete decision.

### 3.3 R5 is sparse and brittle

Removing R5 changes only four final test distributions and no correctness outcomes. Its point estimates improve AURC
from 0.1435 to 0.1171 and Brier from 0.3370 to 0.3328, but the changes are concentrated in two episodes; an
episode-level randomization test has no power to distinguish this pattern (`p=0.50` for Brier, Holm `p=1.0`). In the
two affected wrong rows, R5 approximately doubles an existing `None` mass: 0.50 becomes 0.99. The exclusivity
constraint is logically plausible, but the current implementation treats ambiguous R1 marginals as independent and
renormalizes them, amplifying the shared weak evidence. It supplies no new observation.

### 3.4 R4 and a sharper R1 do not rescue the stack

Removing R4 changes no final probability, prediction, or metric on either split. This is a coverage result for these
traces, not evidence that reveal reasoning is useless in other tasks.

Making R1 nearly point-valued raises test accuracy from 81.28% to 81.70% but worsens AURC to 0.1590 and Brier to
0.3478. The Brier difference is +0.0108 (+0.0006 to +0.0229), but it does not survive the 24-test Holm family. The
cover kernel is therefore not the single cause; removing uncertainty trades calibration for one top-1 tie.

## 4. Why the full KB does not win

The evidence supports the following causal account:

1. **No new information enters the full stack.** Soft and full use the same re-linked tracks and near-pass detector.
   R5 adds a constraint, not an observation; R4 is inactive.
2. **R1 discards a stronger point anchor in rare miss cases.** Its explicit `None` hypothesis is principled, but the
   0.5 far-cover fallback is weakly identified and loses one net answer relative to Bound.
3. **R6's benefit is primarily probabilistic.** It significantly improves Brier but does not improve accuracy; its
   mass mixing can change argmax behavior around ties.
4. **R5 can compound correlated uncertainty.** The same-frame R1 marginals share scene and detector errors. Treating
   them as independent before enforcing exclusivity overstates the evidence and produces high-confidence `None`.
5. **Small penalties are offset by a few large wins.** On test, full has worse row-level Brier than soft on 183/235
   rows, largely because of its 0.01 `None` floor, yet a few large repairs almost exactly cancel them in the mean.
   This heavy-tailed cancellation is not a robust aggregate advantage.

The negative result is therefore unsurprising in retrospect: symbolic structure reorganizes the same fallible
evidence, while the simple baseline preserves a strong tracker-derived answer and only softens it when the detector
directly signals risk.

## 5. Defensible conclusions

Supported:

- Full ArbiterKB has not beaten soft belief repair on this held-out perceived benchmark.
- R6 improves probability quality within ArbiterKB, with a replicated and multiplicity-corrected Brier effect.
- The full-stack deficit is driven by rare R1 anchor failures, one R6 top-1 loss, and sparse R5 overconfidence—not by
  a broad collapse across all queries.
- `None/elsewhere` must be scored as a common outcome class; the earlier Brier advantage was overstated.

Not supported:

- that no KB can outperform soft repair;
- that removing R5 is a validated improvement (the apparent AURC gain is based on two test episodes);
- that these offline answer metrics imply a closed-loop success difference;
- any direct ArbiterKB-versus-MEMI performance ranking.

## 6. Limitations and next experiment

This audit is post-hoc, uses 90 test episodes, and evaluates final beliefs rather than robot success. Set-valued truth
is represented by uniform target mass across accepted containers for Brier. The 24-test Holm family is deliberately
conservative, while the 5,000-draw Monte Carlo randomization resolution is finite. The result applies to the current
perceived VideoUnmaskSwap traces and current relinker; it does not cover tasks where R4 fires or a KB supplies genuinely
new evidence.

The next defensible experiment is not another test-set ablation. It is the frozen direct closed-loop protocol in
`paper/DIRECT_CLOSED_LOOP_PROTOCOL.md`, plus a train-only design change that either (a) keeps Bound as R1's prior,
(b) models shared R1/R5 evidence jointly rather than as independent marginals, or (c) separates action selection from
`None` tie-breaking. A fresh seed set must evaluate any such change.

## 7. Reproduction

```bash
make check
PYTHONPATH=scripts:. .venv/bin/python scripts/paper_error_analysis.py --resamples 5000
```

The complete paired rows, rule-update signatures, data hashes, bootstrap intervals, randomization tests, exact
McNemar tests, and Holm adjustments are in `results/latest/paper_error_analysis/summary.json`. The test read is
append-only in `results/test_reads.log`.
