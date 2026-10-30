# NOTES — engineering log

**How to read this log (2026-10-29).** One section per phase or test, in time order; the newest is at the bottom. A
test's section has its rule (fixed and committed before the run), then its results and readings. Later sections
correct earlier ones; when they do, they say so. The ones that supersede others:
- "Corrected tables after the code review" supersedes every perceived-QA number before it (Q6 truth, "swapped" flag).
- "Claim 3, corrected read and attribution" supersedes every claim-3 AURC before it (tie handling).
- "Is the counting win ArbiterKB's, or the counter's?" closes counting as a test of the KB (RULE = K).
- "Closed loop: the 2610.17350 harness ported, pod runbook" and its smoke-test notes are the current work.

A `results/latest/<name>` cited here that is no longer there is in `archive/results/<name>` (superseded; see
`archive/README.md`). The short current state is in `HANDOFF.md`.

## M0 — walking skeleton (2026-10-28)

### Verified while building
- `sg_pilot.py` @ e2e06f9 imports only numpy + stdlib at top level. Functions copied into `arbiter_kb/sg_compat.py`; `tests/test_sg_compat.py` matched the original (`apply_rule` frozen/bound, `wilson`) when run with `SG_PILOT_PATH` set.
- Synthetic swaps reproduce RoboMME's two-lane path: the two containers are 0.14 m apart at mid-swap (`test_swap_path_endpoints_and_bulge`).
- Measured in synthetic data on TRAIN seeds 0–39, not assumed (`results/m0_synthetic_stats.json`, `scripts/measure_synthetic.py`):
  - container speed during a 50-frame swap peaks at 0.0055–0.0063 m/frame (median 0.0058);
  - the minimum image distance between two passing containers is 22–36 px (median 27);
  - the `id_swap` corruption makes a label jump 0.116–0.138 m in one frame, while the largest true per-frame step is 0.0063 m (see the goal review).

### Deviations from DEVELOPMENT.md (and why)
1. **`v_move` 0.005 → 0.001 m/frame.** The measured peak speed is just above 0.005, so the old value counted a container as "moving" only near mid-swap and split one pass into two near-pass events.
2. **Correctness is judged physically.** `evaluate.physical_truth`: the label whose believed position is nearest the true container at t* (≤ 5 cm). Under `id_swap` the labels lie. Scoring by id marked Bound "correct" while it grasped the wrong container (grasp error 10 cm). Rows keep both `truth` (physical) and `truth_id`.
3. **`MemorySystem.pose(obj)` added** to the protocol, because evaluation needs it for 2. This was added before the first M0 freeze; `CONTRACT_VERSION` stays 1, and `test_contracts` covers it.
4. **R6 adds `Event("swap", conf=1.0)`.** The physical motion is certain; identity uncertainty lives in the `Inside` distributions (§3.5 said conf = 1 − w).
5. **Retract policy details.** A belief left with no support becomes `source="prior"`, which keeps invariant I4. Retracting distribution-level support moves half of the mass to `None`.
6. **Op dataclasses moved to `arbiter_kb/ops.py`** (re-exported by `kb.py`), because `kb.py` exceeded 250 lines (E1).
7. **Trace filenames** are `<condition>_<system>_<ep>.jsonl`. Observed poses are not logged as ops (the `obs` field holds them); inferred poses are logged only when they change by more than 1 mm.
8. **`KB.snapshot()` only**; `restore()` is deferred until a debugging session needs it (YAGNI). `text_view` lives in `views.py`.
9. **`sg_compat` also carries** `active_targets` and `apply_rule` (needed by the replay source and check (g)). `target_error` was removed in M0.1 (no callers).

### Measured (synthetic, 30 test episodes, M0 defaults, untuned)
Clean input (`results/latest/synthetic_test_n30_p1.0/summary.csv`):

| system | accuracy | silent failure | mean conf when right | Brier | ECE |
|---|---|---|---|---|---|
| ArbiterKB | 1.00 | 0.00 | 0.94 | 0.019 | 0.059 |
| Bound | 1.00 | 0.00 | 1.00 | 0 | 0 |
| Frozen | 0.53 | 0.47 (moved 0/14 right, back-at-start 16/16) | 1.00 | 0.93 | 0.47 |

`id_swap`, p = 1.0 (20 of 30 episodes could be corrupted: the target's container swapped):

| system | silent failure on corrupted episodes | mean conf when wrong |
|---|---|---|
| Bound | 1.00 | 1.00 |
| ArbiterKB | 0.90 | 0.91 |

At p = 0.5, only 5 episodes were corrupted (`results/latest/synthetic_test_n30_p0.5/summary.csv`): ArbiterKB 0.8 vs Bound 1.0 silent on those 5.

Median ArbiterKB step: see `median_step_ms` in each `results/latest/*/meta.json` (0.1–0.4 ms depending on the machine).

### What these numbers do and do not show (for the goal reviewer)
- M0 proves that the pipeline runs end to end and that the measurement is honest. **It is not evidence for claim 1.** The corruption is injected at the closest approach of a swap, the same place R6 mixes. That is circular by construction (docs/GOALS.md, "Known traps").
- With the untuned defaults the belief layer barely helps (0.90 vs 1.00 silent). Passes stay 22–36 px apart (21.6 px minimum), so with `d_px` = 30, w = 0.5·(1 − d/30) ≤ 0.15, and confidence in the wrong label stays around 0.9 > τ_hi.
- Clean-input cost: ArbiterKB's confidence when right drops to 0.94, because R6 mixes on every detected pass.
- M1 tuning on train seeds decides whether a setting exists that lowers silent failure without hurting clean calibration. If none does, that is an early warning for claim 1.
- The simple baseline the goal reviewer will ask for (Bound with confidence lowered after any near-pass) is not yet implemented. It is cheap and belongs in M1.
- Real replays: not run here (check (g) SKIPPED; no data on this machine).

### Open questions for M1
- Tune `d_px` / `w_mix` / `σ_cover` on train seeds. Report clean ECE vs silent failure as a trade-off curve, not a single point.
- Add the "Bound + confidence drop after near-pass" baseline.
- A late cover (the container appears a frame after the cube vanishes) currently yields `{None: 1}`. Check this on real replays.

## M0.1 — fixes after the first review (2026-10-28)
Reviews: `reviews/M0_code.md` (BLOCK), `reviews/M0_goal.md` (ON-TRACK-WITH-CHANGES).

### Code review
- **BLOCK fixed:** `Distribution.from_distances` shifts the exponent by the smallest d², so a valid candidate never underflows (`test_r1_small_sigma_does_not_underflow`, d/σ = 8).
- **E5c made real:**
  - `Params()` is no longer frozen by default.
  - Evaluations load the repo's `params.json` (`scripts/run.py --freeze` wrote it from the M0 defaults; M1's `tune.py` will write it from train seeds).
  - `meta.json` records `params_file_mtime` and `first_data_read`.
- **Per-rule tests:** `tests/test_rules.py` covers R1 (closest wins, no candidate, small σ), R6 (weight formula, no mix at closeness 1, visible cubes ignored), R7 and R2. `tests/test_detectors.py` covers disappear, near-pass needing both containers moving, and pass closing.
- **Retract after mix:** each hypothesis keeps its own evidence, and a key that gets mass for the first time inherits the other key's support. Retracting one container's cover no longer wipes the distribution. This is an approximation after mixing; exact per-origin mass tracking is deferred to M5 (R4 reveal), the first phase that retracts.
- **Minors fixed:**
  - R6 mixes only hidden cubes.
  - R7 uses λ^Δt, includes all distributions and guards against zero containers.
  - `check()` asserts I4 for distributions and enforces the belief source set.
  - R1's `{None: 1}` cites the disappear evidence.
  - Evaluation handles targets that share a t*.
  - Brier counts a lost memory as all mass wrong.
  - The validator rejects `keep` entries that hold no mass and are not cited, and rejects non-list `cites`.
  - `v_move` comes only from `DetectParams`.
  - The debug CLI takes `--condition --system --ep`; traces record `track_q`; baselines get a one-line explanation.
  - The figure filename carries source and p.
  - The rule-mutation test now feeds a disappear and a near-pass.
  - Contract snapshots cover Belief, Distribution, Evidence, Event and MixDist.
  - The notebook has 11 cells.
  - NOTES numbers point to files.
- **Left as is, documented:**
  - Near-pass evidence is emitted when the pass ends, not at the closest approach (e.g. t_min 77, emitted 113). This matters for M4 trigger timing.
  - `EpisodeMeta.E` has length T (no system reads it).
  - Events with conf < 0.5 may lose all support after retraction.
- **Unused, spec-mandated (E1 list):** `ObjectFrame.track_quality/extent/stationary`, `KB.task`. These one-call abstractions are spec stubs: `Detectors`, `Trigger`, `RuleOnlyBackend`, `skills.pick`.

### Goal review: accepted, and it changes the plan
- **The `id_swap` corruption teleports the label.** On train seeds the label jump is 0.116–0.138 m, about 20× the largest true per-frame step. A real tracker cannot swap ids while the containers are 14 cm apart. Any result on this corruption is true by construction (M1 tuning would fit the injected rate; an M2 identity check would catch every jump). It stays only as a sanity check.
- `v_move` was measured on train seeds (0–19, later 0–39), not test seeds.
- The notebook wording is corrected: ArbiterKB was still silently wrong (confidence 0.91 > τ_hi).
- **New plan** (DEVELOPMENT.md §6, v1.1): M1 replaces the corruption with overlap-only swaps (no jump), makes the two trivial baselines mandatory (Bound + haircut, label-jump detector) and does no tuning on the teleport corruption. The November gate is pulled forward as phase G: measure how often and how real trackers swap ids on RoboMME stress variants.

## M0.2 — fixes after the second review (2026-10-28)
Reviews: `reviews/M0.1_code.md` (PASS-WITH-NOTES, one MAJOR), `reviews/M0.1_goal.md` (ON-TRACK-WITH-CHANGES).

- **MAJOR (E5c) fixed.** `evaluate.run` refuses test episodes (detected via `EpisodeMeta.split`, no longer `ep >= 1000`) unless the Params:
  - came from a `params.json` whose stored hash matches its content (hand edits raise);
  - is frozen;
  - has a FREEZE entry in the append-only ledger `results/test_reads.log`.

  A Params built in code has no origin, so it is always refused. Every test read appends a READ line. Tests write to a temporary ledger.
- **params.json note corrected.** It now says the test seeds were already evaluated at 23b53ba before the file existed, and that nothing was tuned. Re-frozen with a new hash; the ledger's first line is that FREEZE.
- **Provenance:** meta.json records `dirty` (git status). `results/latest` and `figures/` are written only with `--keep`, and kept results are regenerated from a clean tree after the commit.
- **Minor fixes:**
  - R7 does not decay in the frame a distribution changed (Δt = 0).
  - `swap_mix` asserts 0 ≤ w ≤ 1.
  - R6 records a swap event only when w > 0.
  - The debug CLI lists the available traces instead of a raw FileNotFoundError.
- **New tests:** a uv-path detector test (both containers moving, never within d_px), the R7 Δt = 0 test, the w-range test, and a deep-copy no-mutation test covering support lists.
- **Known, recorded:**
  - Retracting a near-pass moves half the mass to None; it does not undo the mix. M4's "unlikely" verdict must restore the pre-mix distribution from the trace, not rely on Retract.
  - There is still no test that walks the support chain for every results row (5 episodes are covered).
- **Goal review accepted:** M1 is now plumbing only. OverlapHaircut, the Bayes-optimal rule for `id_swap_overlap`, is added. Tuning, the trade-off curve, R3 and the visibility state machine wait until after G (DEVELOPMENT v1.2).

## M1 — plumbing: overlap swaps, three cheap baselines, R5 (2026-10-28)
Default parameters, nothing tuned. `params.json` re-frozen with the M1 defaults (R6 v1.5 w_max 0.5 / d0 0.02, eps_none 0.01, baseline c 0.5 / d_ov 0.03 / j_max 0.03); FREEZE line in the ledger.

Table: `results/latest/m1_table/summary.csv`. Test seeds 1000–1059, lane and swap duration drawn per episode ("mix"), id_swap_overlap p_swap 0.5, pos_noise σ 15 mm. Command:
`python scripts/run.py --systems all --corrupt none,id_swap_overlap,pos_noise --p 0.5 --lane mix --steps mix --n 60 --keep-as m1_table`

| condition | system | acc | silent | brier | silent on corrupted (n=13) |
|---|---|---|---|---|---|
| none | Bound | 1.00 | 0.00 | 0.000 | – |
| none | OverlapHaircut | 0.85 | 0.00 | 0.100 | – |
| none | ArbiterKB | 1.00 | 0.00 | 0.042 | – |
| id_swap_overlap | Bound | 0.90 | 0.10 | 0.200 | 0.46 |
| id_swap_overlap | BoundHaircut | 0.68 | 0.00 | 0.292 | 0.00 |
| id_swap_overlap | JumpDetector | 0.90 | 0.10 | 0.200 | 0.46 |
| id_swap_overlap | OverlapHaircut | 0.88 | 0.00 | 0.100 | 0.00 |
| id_swap_overlap | ArbiterKB | 0.90 | 0.03 | 0.106 | 0.15 |
| pos_noise | Bound | 0.98 | 0.02 | 0.033 | – |
| pos_noise | JumpDetector | 0.32 | 0.00 | 0.517 | – |
| pos_noise | ArbiterKB | 0.98 | 0.02 | 0.042 | – |

Reading:
- **ArbiterKB − OverlapHaircut ≈ 0** (silent +0.03, Brier +0.006 under id_swap_overlap). Expected: OverlapHaircut is Bayes-optimal for a corruption we wrote. This table says the plumbing works, not that ArbiterKB helps. Only G's measured swaps can say that.
- **ArbiterKB's two silent failures** (1046, 1058) are lane 0.01 passes with min_3d = 0.020 m: w = 0.5·e⁻¹ = 0.18, final p_max 0.808, just above τ_hi 0.8. With the default d0 this is expected. M2 tunes d0 / w_max on id_swap_measured.
- **The haircut baselines lose accuracy on clean data** (0.65 / 0.85). A 50/50 haircut makes the argmax a coin toss. Brier, not accuracy, is the fair score for them.
- **JumpDetector**: it never fires under id_swap_overlap (label jumps median 0.004 m, max 0.023 m < j_max 0.03), so it equals Bound. Under 15 mm position noise it fires constantly (acc 0.32). A label-jump check is the wrong tool for both.
- **R5 (joint exclusivity) emitted 0 ops in this run.** It runs on cover frames only (after mixes the cubes are correlated, and re-normalising marginals was overconfident: 0.8 → 0.94, see test_rules). In VideoUnmaskSwap each cube is covered by a distinct, well-separated container, so the cover marginals are already exclusive. R5 is inert here, against the goal reviewer's expectation. It stays (tested) for layouts where covers are ambiguous.
- **The teleport id_swap is now a pure sanity check.** R6 uses the 3D distance, so lane 0.07 passes (min 3D ≈ 0.12 m) barely mix, and ArbiterKB's silent rate on teleport-corrupted episodes is 1.0, the same as Bound's. The M0 selftest check (c) "ArbiterKB < Bound on teleport" is now INFO. It was a property of the teleport, not of the KB.
- **Fixes during M1:**
  - OverlapSwapSource wrote its log only after the last frame, but scoring at t* reads it while the generator is still open, so n_corrupted was 0. The log is now written before the first frame (with a test).
  - After an earlier swap it swapped the wrong labels. It now swaps the labels currently tracking the two overlapping physical containers.
  - GuardedBound ignores a stale doubt pair once its prediction leaves the pair. Its explain line shows the real confidence and the trigger, not "confidence 1".
- **E2:** explain reaches an observation for all 1080 scored rows of the table run (all six systems), plus a pytest over all three M1 conditions.
- **Figures** (`figures/`, written by the kept run):
  - belief timeline of a corrupted id_swap_overlap episode;
  - reliability diagram (bins with a few rows are noisy at n = 60);
  - label-jump histogram.
- **Not done in M1 (by design, v1.2):** tuning, the trade-off curve, R3, the visibility state machine. These wait for G.

## M1.1 — fixes after the M1 reviews (2026-10-28)
Reviews: `reviews/M1_code.md` (PASS-WITH-NOTES, 3 MAJOR), `reviews/M1_goal.md` (ON-TRACK-WITH-CHANGES). All findings accepted.

**The goal review's point, in one line:** silent failure at a fixed τ_hi can be gamed. Any haircut below τ_hi scores 0 with Bound's accuracy. And ArbiterKB's R6 thresholds the same feature (3D distance) as OverlapHaircut, only softly. So the M1 table could not tell ArbiterKB from a threshold heuristic, and M2 (c) as written was unreachable.

Changes:
- **Metrics:**
  - coverage = P(p_max ≥ τ_hi) and AURC (area under the risk–coverage curve, threshold-free, tie-aware) in every summary;
  - a risk–coverage figure.

  GOALS claim 1 now reads "lower silent rate than Bound **and a haircut on the same feature, at matched clean coverage**", with AURC as the summary. A new falsifier was added: a single feature separates swaps almost perfectly (G's ROC).
- **Spec v1.3:**
  - M2 (c) compares at matched coverage and adds `MarginHaircut`, a threshold on R6 v2's margin feature;
  - M2 evaluates on fresh test seeds (≥ 1060);
  - G's `measure_swaps.py` records every pass, swapped or not (`results/g_passes.csv`), with per-feature ROC/AUC. If the best feature reaches AUC ≥ 0.98, NOTES must say that a threshold rule solves detection, before R6 v2 is built.
- **Code MAJOR 1 (E2 tautology):**
  - Tracker baselines now store the observations each answer rests on, and `debug explain` looks them up; a missing one prints NOT FOUND.
  - The selftest also fails on NOT FOUND.
  - `tests/test_baselines.py` checks that a missing cube gives NOT FOUND.
- **Code MAJOR 2 (untested numbers):**
  - `test_r6_v15_weight` checks w = 0.5 / 0.1839 / 0.0012 at d = 0 / 2 / 12 cm.
  - `test_r6_uses_3d_distance_not_image_closeness` checks that R6 ignores image closeness.
  - The detector min_3d now equals the true closest 3D distance.
  - `tests/test_baselines.py` covers:
    - the OverlapHaircut gate: fires at 2 cm, not at 5 cm;
    - the haircut: fires on any pass;
    - JumpDetector: 0.04 m fires, 0.02 m does not;
    - the stale-doubt guard.
  - Both mutations the reviewer used now fail the tests (checked by hand).
  - The dead M0 closeness branch of R6 and `RuleParams.w_mix` are removed. This changes the hash, so params.json was re-frozen with M1.1 defaults.
- **Code MAJOR 3 / goal risk 3 (R5 scenario):** `tests/test_r5_scenario.py`.
  - Setup:
    - two containers 4 cm apart;
    - the red cube's last sighting is 2.5 cm off (position noise), nearer the wrong container.
  - Bound, and so OverlapHaircut (which sees no pass), puts both cubes in bin_a with confidence 1: a silent failure.
  - ArbiterKB with R5 moves red to bin_b (p 0.79, below τ_hi) and has a lower Brier. Without R5, ArbiterKB is wrong too.

  This is propagation, not a better swap feature. **On stock VideoUnmaskSwap layouts R5 still never fires** (0 ops in the table run): the covers are not ambiguous there. The claim "R5 is the edge over haircuts" is therefore untested on the benchmark. It needs ambiguous covers or reveals (M5's R4).
- **R5 correctness:**
  - R5 now skips cubes whose belief was mixed by R6 (their support contains a near_pass), with a test that fails without the filter.
  - An infeasible joint returns None, and R5 records `exclusive_conflict` evidence for M3's R8 (test).
- **Goal risk 4 / code MINOR (haircut tie):**
  - `Distribution.argmax` breaks ties by key order; it no longer uses the label string.
  - GuardedBound lists Bound's answer first, so a 50/50 haircut keeps it. The haircut baselines' accuracy now equals Bound's (it was 0.68 / 0.88 under id_swap_overlap).
- **Minor fixes:**
  - NoisySource writes its log before the first frame.
  - GuardedBound sets `.t`, and the three GuardedBound systems are in the causality test.
  - The eps_none `None` hypothesis now carries the evidence that triggered R1.
  - The selftest uses train seeds only, so `make check` no longer appends READ lines or dirties the ledger.
- **Correction to the M1 NOTES:** the claim "explain reaches an observation for all 1080 rows, all six systems" was true only for ArbiterKB until this fix. It now holds with a real lookup.

M1.1 table (`results/latest/m1_1_table/summary.csv`; test seeds 1000–1059, same command as M1 plus `--keep-as m1_1_table`; untuned):

| condition | system | acc | silent | coverage | AURC | Brier |
|---|---|---|---|---|---|---|
| none | Bound | 1.00 | 0.00 | 1.00 | 0.000 | 0.000 |
| none | OverlapHaircut | 1.00 | 0.00 | 0.80 | 0.000 | 0.100 |
| none | ArbiterKB | 1.00 | 0.00 | 0.90 | 0.000 | 0.042 |
| id_swap_overlap | Bound | 0.90 | 0.10 | 1.00 | 0.100 | 0.200 |
| id_swap_overlap | OverlapHaircut | 0.90 | 0.00 | 0.80 | 0.020 | 0.100 |
| id_swap_overlap | ArbiterKB | 0.90 | 0.03 | 0.90 | 0.011 | 0.106 |
| pos_noise | Bound | 0.98 | 0.02 | 1.00 | 0.017 | 0.033 |
| pos_noise | ArbiterKB | 0.98 | 0.02 | 0.97 | 0.019 | 0.042 |

Reading:
- ArbiterKB and OverlapHaircut are two operating points on the same feature:
  - ArbiterKB answers more (clean coverage 0.90 vs 0.80) and misses 2 of the 6 wrong rows;
  - OverlapHaircut catches all 6 and answers less.
- ArbiterKB's AURC is lower (0.011 vs 0.020). But that rests on 6 wrong rows out of 60 (n is small), and both systems use the feature we chose for the corruption. **No claim moves.** Only G's measured swaps, and MarginHaircut in M2, can change that.

## M1.2 — closing M1 after the M1.1 reviews (2026-10-28)
Reviews: `reviews/M1.1_code.md` (PASS-WITH-NOTES, 1 MAJOR), `reviews/M1.1_goal.md` (ON-TRACK-WITH-CHANGES). With the user (option A): keep the KB scaffolding, fix the real bug, write the review's points into the spec, and move to G. M1 is not re-reviewed.

- **R5 double counting fixed (code MAJOR).**
  - The bug: a later cover re-combined cubes that an earlier R5 had already combined, which counted the constraint twice. A far, unrelated cover moved red from 0.79 to 0.91.
  - Fix: R5 now joins only the cubes covered in the same frame. Their R1 beliefs are independent, so that joint is exact. Covers on different frames are not linked (§3.5 says so).
  - Regression test: `test_a_later_far_cover_does_not_recombine_earlier_cubes`. The mixed-cube test still holds.
- **Weak tests closed:**
  - a row with p_max exactly τ_hi counts as covered;
  - the `None` hypothesis rests on the trigger;
  - explain prints NOT FOUND when the predicted container is unseen.
- **What ArbiterKB adds today (goal review, accepted).**
  - On the M1.1 test rows ArbiterKB's answer equals Bound's in every row (`results/runs/synthetic_test_2f707774/rows.csv`, per the reviewer).
  - Its AURC edge over OverlapHaircut (0.011 vs 0.020, `results/latest/m1_1_table/summary.csv`) comes from grading confidence by distance (R6's formula) against a binary haircut. A KB-free `SoftHaircut` with the same formula would very likely tie.
  - So far the rule layer is "a calibrated distance score". That is not a verdict on claim 1: the belief layer's own contributions are propagation (R5 on ambiguous covers, retraction, containment as a defeasible default, §7 q5) and handing evidence to the LLM. Neither has been tested on real data.
- **Spec v1.3 (amended):**
  - M2 (c) is against SoftHaircut(margin): AURC with bootstrap CI, and silent rate at matched clean coverage (thresholds swept on train). It uses n ≥ 300 fresh seeds and drops Frozen / BoundHaircut / JumpDetector.
  - **M4 adds arm C, an LLM reading the raw tracker log (no KB).** C vs D (KB + LLM) is the main test of whether the KB helps (user discussion). GOALS gains that falsifier.
  - G's gate uses the best rule with ≤ 2 features (cross-validated logistic), not just single-feature AUC.
  - "Bayes-optimal" is reworded everywhere: OverlapHaircut is hard-label optimal for our corruption, not ranking optimal.
- **Known, not fixed (MINOR, recorded):**
  - R5's mixed filter is on/off (moot now that R5 only joins same-frame covers);
  - the haircut trigger is not traced to the pass's observations;
  - coverage has no Wilson interval yet (M2 needs it);
  - ArbiterKB's argmax tie order is implicit (insertion order);
  - train clean accuracy is 0.983: ep 48 has two exact-contact passes, i.e. two coin flips, which is correct math (`results/runs/synthetic_train_*`);
  - the pos_noise row of OverlapHaircut was missing from the M1.1 NOTES table (AURC 0.017, Brier 0.050, better than ArbiterKB's 0.019 / 0.042 on AURC only);
  - the M1 haircut accuracy numbers "0.68 / 0.88" come from `results/latest/m1_table/summary.csv`.

## G — Mac-side code (2026-10-28)
Written, not yet run on a pod (the user's go is needed for GPU time).
- `stress/stress_env.py`: `VideoUnmaskSwapStress` (registered env id) subclasses RoboMME's VideoUnmaskSwap @ 53673b3, read from the source:
  - knobs `swap_steps`, `lane_offset`, `n_swaps` (≤ 3), `cam_shift`;
  - `step()` is a copy of the stock one with those parameters;
  - it logs actor poses each step (`pose_log`), so ground truth exists for every video frame.
- `stress/render.py`: plays the conditioning video through RoboMME's DemonstrationWrapper, with the same flags as sg_pilot's `make_env`, and saves sg_pilot Replay npz files.
  - Wrist arrays are 1×1 placeholders to halve the size.
  - Seven settings: stock, fast, close, fast_close, fast_close_3swaps, cam_left, cam_right.
- `stress/perceive.py`: sg_pilot's PerceptionWorker over each replay (a copy of `perceived_graph` that also keeps bbox extents), plus `oracle_graph`. It writes small npz files.
- `arbiter_kb/swapstats.py` + `scripts/measure_swaps.py` (CPU):
  - matches slots to oracle nodes (`assign_slots_seeded`, copied verbatim into sg_compat);
  - records **every pass**, swapped or not, with features: min 3D distance, min image distance, max bbox IoU, worst bbox-area ratio, max one-frame jump, min keep-vs-exchange margin;
  - writes `results/g_passes.csv` and `g_summary.json` and prints the gate decision.
  - Per-feature ROC AUC, and the best ≤ 2-feature cross-validated logistic rule.
  - Tests on fake perceived episodes built from synthetic data, with one injected exchange, plus an end-to-end CLI test.
- `stress/POD_RUN.md`: the steps, a smoke test first, about 3–4 h and about $2–3 (rough), and "stop the pod" last.
- **Risk:** the pod-side code has never run against the simulator. The hand-off prompt says to fix only `stress/` and to stop after the smoke test.

## M2 infra — SoftHaircut, CIs, matched coverage (2026-10-28)
Built while G runs on the pod (HANDOFF "Parallel work" 1). No parameter changed (params hash 4e44acb946f6, no re-freeze); no test seed was read.
- **`SoftHaircut(feature)`** (`arbiter_kb/systems/soft_haircut.py`): R6 with no KB.
  - It starts from Bound's answer as a point mass. Each near-pass seen while the cube is hidden applies R6's own `swap_mix` with w = w_max·exp(−feature/d0), using R6's parameters.
  - After one pass on Bound's container, the confidence is 1 − w_max·exp(−feature/d0), as specified.
  - **Deviation from the HANDOFF wording ("Bound's answer"):** a first version kept Bound's answer and multiplied (1 − w) over passes. On train seeds 0–299 (clean + id_swap_overlap, 600 rows; a scratch run, not kept), 36 rows then had a partner with more mass than the answer. Sequential `swap_mix` is what R6 does, so SoftHaircut uses it. Its answer can then differ from Bound's after several passes, as ArbiterKB's does (4 of 300 id_swap_overlap rows below, `same_answers` vs bound in `results/latest/m2_infra_train/compare.csv`).
  - The feature is `min_3d` now; M2 passes the margin.
  - It has no cover kernel, no R5 and no `elsewhere` floor. Whatever ArbiterKB adds beyond SoftHaircut is the KB.
- **Metrics:**
  - `summary.csv` has a Wilson interval on coverage (`cov_lo`, `cov_hi`).
  - `arbiter_kb/compare.py` + `scripts/compare.py`:
    - AURC with a 95% bootstrap CI, resampling episodes (2000 draws, seed 0);
    - the paired AURC difference ref − base, with the same episodes drawn for both;
    - matched coverage: on a **train** calibration run, θ is the baseline confidence whose clean coverage is nearest the reference's clean coverage at τ_hi; then, on the eval run, the baseline's silent rate at θ (Wilson) next to the reference's at τ_hi.
    - A binary haircut can reach only a few coverages, so the achieved calibration coverage is reported next to the target.
- **CLI:** `--first K` skips a split's first K seeds (M2: `--split test --first 60 --n 300` = seeds 1060–1359); it refuses to leave the 1000-seed range. `--systems m2` = bound, overlap_haircut, soft_haircut, arbiterkb (no Frozen / BoundHaircut / JumpDetector). `--systems all` is unchanged (the M1 table).
- **Selftest:** the m2 preset runs, and every SoftHaircut row is explainable (E2).

**ArbiterKB vs SoftHaircut under the M1.1 conditions, train seeds only.** θ is calibrated on seeds 0–149 (`results/runs/synthetic_train_bdb06d98`) and the table is on seeds 150–449 (n = 300 rows per cell; `results/latest/m2_infra_train/summary.csv` and `compare.csv`, commit 73fab80, clean tree). Untuned defaults, p_swap 0.5, lane and steps "mix", pos_noise 15 mm.

| condition | system | acc | silent | coverage [Wilson] | AURC [boot] | Brier |
|---|---|---|---|---|---|---|
| none | Bound | 1.00 | 0.000 | 1.000 [0.987, 1] | 0.000 | 0.000 |
| none | OverlapHaircut | 1.00 | 0.000 | 0.683 [0.629, 0.733] | 0.000 | 0.158 |
| none | SoftHaircut | 1.00 | 0.000 | 0.803 [0.755, 0.844] | 0.000 | 0.081 |
| none | ArbiterKB | 1.00 | 0.000 | 0.803 [0.755, 0.844] | 0.000 | 0.082 |
| id_swap_overlap | Bound | 0.847 | 0.153 | 1.000 | 0.153 [0.113, 0.197] | 0.307 |
| id_swap_overlap | OverlapHaircut | 0.847 | 0.000 | 0.683 | 0.049 [0.032, 0.069] | 0.185 |
| id_swap_overlap | SoftHaircut | 0.857 | 0.047 | 0.800 | 0.0275 [0.017, 0.040] | 0.179 |
| id_swap_overlap | ArbiterKB | 0.857 | 0.047 | 0.800 | 0.0275 [0.017, 0.040] | 0.179 |
| pos_noise | Bound | 0.987 | 0.013 | 1.000 | 0.013 [0.003, 0.027] | 0.027 |
| pos_noise | OverlapHaircut | 0.987 | 0.013 | 0.957 | 0.014 [0.003, 0.028] | 0.048 |
| pos_noise | SoftHaircut | 0.987 | 0.013 | 0.967 | 0.012 [0.000, 0.028] | 0.032 |
| pos_noise | ArbiterKB | 0.980 | 0.020 | 0.967 | 0.044 [0.014, 0.080] | 0.047 |

Paired AURC difference ArbiterKB − baseline [95% bootstrap] and matched coverage (θ from seeds 0–149):

| condition | baseline | ΔAURC [CI] | θ (calib coverage, target 0.80) | ArbiterKB coverage / silent at τ_hi | baseline coverage / silent at θ |
|---|---|---|---|---|---|
| id_swap_overlap | SoftHaircut | 0.000 [0, 0] | 0.816 (0.800) | 0.800 / 0.047 | 0.793 / 0.047 [0.028, 0.077] |
| id_swap_overlap | OverlapHaircut | −0.021 [−0.030, −0.013] | 1.0 (0.727) | 0.800 / 0.047 | 0.683 / 0.000 [0, 0.013] |
| id_swap_overlap | Bound | −0.126 [−0.155, −0.094] | 1.0 (1.000) | 0.800 / 0.047 | 1.000 / 0.153 |
| pos_noise | SoftHaircut | +0.031 [+0.007, +0.059] | 0.816 (0.800) | 0.967 / 0.020 | 0.967 / 0.013 |

Reading:
- **ArbiterKB and SoftHaircut tie exactly under id_swap_overlap and on clean input.** Same answers in 300 of 300 rows, ΔAURC 0 with a zero-width CI, and the same silent rate at matched coverage. As expected, the rule layer is today a scoring formula: R6's distance score, which needs no KB. This moves no claim. M2 (c) must show a gap on the margin and on `id_swap_measured`, or say so in NOTES.
- **Under pos_noise ArbiterKB is worse than SoftHaircut** (ΔAURC +0.031, CI excludes 0). The cause is six rows where ArbiterKB answers `None` ("elsewhere") with p = 1.0, a confident wrong answer. The noisy last sighting of the cube leaves no container within r_cover, so R1 returns {None: 1} (seeds 201, 218, 253, 259, 297, 371 in `results/runs/synthetic_train_9ec184d5/rows.csv`). In two of those rows SoftHaircut is right. **Here the KB adds a failure mode a tracker does not have.** It is not fixed here: it is a rule change, so it belongs in M2 with a re-freeze. A candidate fix is to give the nearest container mass when no container is within r_cover, or to cap {None} below τ_hi.
- OverlapHaircut cannot reach coverage 0.80 (it is binary: 0.727 on calib). The matched comparison against it is therefore only approximate: 0.683 vs 0.800 on eval.
- In the M1.1 test table (seeds 1000–1059) ArbiterKB's answers equalled Bound's in every row. On train seeds 150–449 they differ in 4 of 300 id_swap_overlap rows and 6 of 300 pos_noise rows (`same_answers`). So "equal in every row" was a property of those 60 seeds.

### M2 infra, R1 fix (2026-10-28, with the user's go)
The pos_noise finding above is fixed. The table above is **before the fix** (params 4e44acb946f6). Its calibration run dir `results/runs/synthetic_train_bdb06d98` has since been overwritten by the post-fix run with the same arguments (the run tag hashes only the arguments); to reproduce it, check out e770821.
- **R1 change** (`rules.py`, spec §3.5):
  - If no container is within r_cover, the belief is {nearest visible container: 0.5, None: eps_far = 0.5}. The cover was not seen, so the answer gets even odds, never a confident one. The container is listed first, so it stays the answer.
  - Only when no container is visible at all is the belief still {None: 1}, with the cube `unknown`.
  - A first attempt, all visible containers against `eps_far`, let `None` win the argmax when there were several far containers (commit f371e94). It was replaced in 1b0e7cf.
  - New parameter `RuleParams.eps_far = 0.5`. It is set by reasoning (even odds), not tuned. Re-frozen: params.json hash **f59aaea5e5b2** (FREEZE lines in `results/test_reads.log`; no READ since).
- **Tests:**
  - no container within r_cover → {nearest: 0.5, None: 0.5}, `hidden`;
  - two far containers → only the nearest;
  - no visible container → {None: 1}, `unknown`.
- **After the fix**, same commands, seeds 150–449 (`results/latest/m2_infra_train_r1fix/`), pos_noise 15 mm:
  - ArbiterKB: acc 0.993, silent 0.000, coverage 0.947 [0.915, 0.967], AURC 0.0001, Brier 0.018;
  - SoftHaircut: acc 0.987, silent 0.013, coverage 0.967, AURC 0.012, Brier 0.032;
  - ΔAURC ArbiterKB − SoftHaircut = −0.012 [−0.028, −0.0001].
  - id_swap_overlap and clean are unchanged: an exact tie, 300 of 300 answers the same.
  - Two rows (seeds 201, 297) still answer `None` at p 0.5. A later R6 pass splits the nearest container's 0.5, so 'elsewhere' leads. They are not silent.
- **Out-of-sample check:** the fix was designed on seeds 150–449, so it was re-run on **train seeds 450–749**, which I had not looked at (`results/latest/m2_infra_train_holdout/`, θ from seeds 0–149).
  - pos_noise: ArbiterKB AURC 0.0001 vs SoftHaircut 0.0126, ΔAURC −0.0125 [−0.026, −0.0002]. Silent 0.000 vs 0.013 [0.005, 0.034], at coverage 0.937 vs 0.957 (θ cannot match it exactly).
  - id_swap_overlap and clean: an exact tie again.
- **What this does and does not show:**
  - The pos_noise gain is real on unseen train seeds. But it is a **cover-distance check** ("no container near where the cube vanished → even odds"), i.e. one more feature. A tracker baseline could copy it in two lines; SoftHaircut lacks it only because it copies R6, not R1.
  - It is not propagation and not evidence for the belief layer. It also lives on our own pos_noise (σ = 15 mm, chosen by us).
  - **No claim moves.** For M2 (c), SoftHaircut should get the same cover-distance check, so that the comparison isolates the KB again.

## Direction v2 — the KB as an LLM controller's memory (2026-10-28, user decision)
The user, in discussion: "I definitely want the LLM to output the path. Our goal is to use a more complete KB as the memory module that helps a large model control the robot. We want either a smaller model to reach the same result, or to save a large number of tokens: not looking at many photos, just being given the latest KB, can do the same thing." The reference is arXiv 2610.17350:
- GPT-6-Astra reads the whole image history (a median of 148 images) and outputs one world-frame waypoint per call;
- 92.1% on RoboMME, $0.39 per episode;
- its code is released.

**What changes:**
- `docs/GOALS.md`:
  - New research question and claims:
    1. **smaller model:** a small open model with the KB approaches the frontier result;
    2. **fewer tokens:** the KB + the current frame matches the full image history with the same model.
  - The old claim ("knows when it is wrong") stays as claim 3: calibration and silent failure. G and M2 serve it.
  - New falsifiers: the raw image history does as well as the KB; the raw tracker log does as well; KB + current frame loses needed information.
  - New traps: an oracle KB is an upper bound only; the arms must differ only in memory.
  - "An LLM planner" is removed from the non-goals. Counting and imitation come after permanence, once their KB layers exist.
- `DEVELOPMENT.md` v1.4:
  - §3.8: the LLM controller in the 2610.17350 harness (one waypoint per call, with target and cites). Its validator checks schema, workspace and reach; the distance from the waypoint to the cited target is logged, not enforced.
  - Memory arms: H (full images), R (raw tracker log), K (KB + event log), K−L3, K+A (with the arbiter). The paper's GPT-6 number is a labelled reference row. The rule planner becomes a baseline. The arbiter (§3.7) becomes an optional ablation.
  - §6 phases:
    - **M3:** a complete KB: L3 events (grasp / release / place / press / reveal), L2 Held / On / Near / LeftOf, L0 / L1 extent and stationary intervals, L4 progress, and the event-log view with where-from and where-to. Each is checked against simulator truth.
    - **M4:** the controller harness, ported from the paper's code, with a replay-driven dry run on the Mac, plus a cheap test of whether a small VLM or the rules should fill each of L2–L4.
    - **M5:** the closed-loop comparison on a pod (the user's go needed).
    - **M6:** the final test-split evaluation, the claim 3 table and a failure analysis (KB wrong / KB missing something / controller misused it).
  - Compute: GPU in G, M5 and M6 only. E6: controller actions cite KB items. E8: closed-loop caps are set per experiment by the user.
- **Not changed:** code, parameters (hash f59aaea5e5b2), G (running on the pod), M2.

**What this means for the work so far.** M0–M2 built the KB core, the evidence and belief layer, and the evaluation. All of it stays. It becomes claim 3, and the memory that arm K reads. The M2 infra finding (rules = a scoring formula on swap detection) is about claim 3 only. Claims 1–2 have not been tested at all yet.

**Open points for the user:**
- After G, decide whether M3 / M4 go before M2 (they do not depend on it).
- Decide on the model for claim 2's larger-model arm (budget).
- The closed-loop runs need a pod with the simulator. Rough cost: 9.5 calls × 18 s per call in the paper's harness, so about 3 min of model time per episode plus the simulator. A per-arm estimate comes in M4 (c).
- **Addendum (user, same day):** "We can accept a frontier model as the planner in the end. The structured memory module has its own meaning." GOALS now says that claims 1 and 2 are alternatives, and that claim 2 with a frontier model is a full result. The user chose to start M3 and M4 directly, without the small feasibility probe.

## M3 — a complete KB, first part (2026-10-28)
Built (commit 7b17641; params re-frozen to **b0a2dd3cb30b**, no test read):
- **Gripper evidence** (`detectors.py`): `grip_close` / `grip_open` with hysteresis (`g_close` 0.07, `g_open` 0.075 m of total finger width), carrying the end-effector xyz. `move_start` / `move_stop` now carry the object's xyz.
- **R3 grasp / place** (`rules.py`):
  - On `grip_close`, the nearest non-hidden object within `r_grasp` (5 cm) of the end effector is grasped (event `grasp`).
  - On `grip_open`, every held object is placed (event `place`) onto the object right below it (xy within `r_on`), else onto the table.
  - Held objects are derived from the event log (`held()`), because `observe()` marks a visible object `visible` every frame.
  - A held object out of sight follows the end effector (R2).
  - R1 ignores a held cube that vanishes (fingers, not a cover).
- **L2 geometric relations** (`views.geometric`, computed on demand, never stored): `Near` (< 8 cm), `LeftOf` (robot's left, +y, by > 2 cm), `On` (xy < 4 cm, 0.5–8 cm above), with p = the poses' confidence. Visible or held objects only. `Held` is in `facts()`.
- **L4:** the goal is stored at reset. Progress counts grasps, places, covers and container exchanges.
- **Event log and controller view** (`views.event_log`, `views.controller_view`): goal, progress, objects with 3D centres, relations with probabilities and support ids, then the log in time order with ids. A move reads "moved (x, y) → (x', y') (t=68–112)".
  - A hidden cube's position is **not** printed as a coordinate. Its pose is a probability-weighted mean, which after a swap can lie between two containers. The line says "see Inside" instead.
  - R6's `swap` event is shown as "exchanged places (identity may be confused: see Inside)", because the event only records that two containers exchanged places, not an identity swap.
  - A synthetic episode's view is about 360 tokens; the budget is 1,500.
- **Tests:** `tests/test_m3_kb.py` (pick-place on hand-made frames: grasp, place onto bin, follow when hidden in the hand, not a cover, On / LeftOf, log order and ids, view budget, hidden pose not printed).

**Truth check** (`scripts/measure_kb.py`, train seeds 0–99, synthetic VideoUnmaskSwap; `results/latest/m3_kb_truth/summary.json`; the run's tree was dirty only in NOTES.md):

| | cover precision / recall | near-pass precision / recall | move from→to xy error (median / max) | moves logged |
|---|---|---|---|---|
| clean | 1.00 / 1.00 | 0.94 / 0.86 | 0.0 / 0.0001 m | 380 |
| pos_noise 15 mm | 1.00 / 1.00 | 0.33 / 0.09 | 0.016 / 0.037 m | 38 |

- On clean input the observations are the truth, so these numbers check plumbing only.
- A "true pass" here is any pair moving at the same time, while the detector flags only passes closer than `d_px`. Recall below 1 is partly by definition; precision 0.94 means 6% of flagged passes do not line up in time with a true one. Not investigated further.
- **Finding:** under 15 mm iid noise the motion detector breaks. Every frame exceeds `v_move` (1 mm/frame), so objects never "stop": 38 moves logged instead of 380, and passes are missed (recall 0.09). The event log is therefore only as good as the motion detector. The fix is M2 item 1 (hysteresis / smoothing on speed). It should be tuned on G's real tracker noise, not on our iid noise. Not fixed here.

**Not built yet (M3 list):**
- `press` and `reveal` (R4): no data has them yet. Synthetic never manipulates, and the oracle replays are not on the Mac.
- extent, yaw and `occluded_by_arm`: they need masks / replays;
- stationary intervals as such (the event log's from→to covers the same information);
- task-template parsing for L4 progress against a goal;
- R8 consistency checks.

Grasp / place have no truth check beyond the unit tests, because no source manipulates objects.

## M4 — controller, Mac side (2026-10-28)
- **The 2610.17350 harness was read** (`harness/astra_waypoint_eval_v3.py` at github.com/jackbjed/embodied-astra-robomme, MIT; read via the GitHub API, nothing run).
  - It uses the official RoboMME loop (`BenchmarkEnvBuilder(..., action_space="waypoint", max_steps=1300)`) and the OpenAI Responses API, with all turns chained through `previous_response_id`.
  - Per step: front frames every 3 control steps plus the last, the current wrist frame, TCP pose, finger positions, elapsed steps, camera K / extrinsics. The answer is `{"description", "waypoint", "gripper"}`; the model's own "description" is part of its memory.
  - Endings: `waypoint: null` (declared complete), 3 no-op actions, an unparseable answer, 150 calls.
  - **Differences to record:**
    - they use RoboMME commit 856bc3a, our pod has 53673b3;
    - they use no depth, while our KB's 3D centres come from depth (perception), so arm K has information arm H lacks. To be reported; an "H + depth" arm may be needed for fairness.
- **`arbiter_kb/llm.py`:** OpenRouter via urllib, the key from `.env` or the environment (never printed or cached), DeepInfra pinned with no fallbacks, a disk cache keyed by the full request, the `ARBITER_MAX_USD` cap (a call that would start above it raises), usage counting (calls, hits, tokens, images, $, s). Tested with a fake transport: no network call was made.
- **`arbiter_kb/controller.py`:**
  - Arms H / K / K-L3 / R.
  - The system prompt reuses the paper's robot and interface text, adds a memory section per arm and asks for `target` and `cites`.
  - The paper's `parse_answer` and `to_action`.
  - `check()`: workspace box, a **coarse** reach test (a sphere of 0.855 m around the Panda shoulder; the real IK / mplib check is still to come), and that target and cites are real KB ids.
  - `target_distance()` is logged only.
  - Arm R's raw tracker log.
- **`scripts/dry_run.py`:** fills the KB up to t* on train episodes and asks each arm for one action. Scored: parse kind, check problems, xy error to the TRUE target container (evaluation only), whether `target` is the right container, tokens / images / $. `--print` shows the prompt for free; `--call` spends, up to `--max-usd`.
- **Not run:**
  - No API call was made. There is no `.env` on this Mac, and the user has not approved spending.
  - Arm H needs RGB replays (`data/replays/` is empty). The pod's G tarball has perception outputs only.
- **Not built:**
  - the closed-loop harness on the pod: robomme env, the KB updated online after each motion (from oracle state for the upper bound, from the perception worker for the claims), the paper's endings;
  - the reach IK;
  - freezing the prompt hash into params.json.

## M3a — fixes after the M3a reviews (2026-10-28)
Reviews:
- `reviews/M3a_code.md`: PASS-WITH-NOTES, 3 MAJOR. The agent could not write files; the builder session saved its report verbatim.
- `reviews/M3a_goal.md`: ON-TRACK-WITH-CHANGES.
- No BLOCK and no OFF-TRACK.

The user asked to fix the code review's findings. The goal review's plan-level proposals are listed at the end and left for the user.

**Code MAJORs:**
- **Place target.** `below()` now considers only visible or held objects. Before, a container put back over its hidden cube was logged as `place(bin_a, cube_r)`, because a hidden cube's display pose is its container's. Test: `test_container_put_back_is_not_placed_on_its_hidden_cube` now gives `place(bin_a, table)`.
- **E2 for the new items.**
  - Every trace row now stores `obs["ee"] = [True, ee_xyz, gripper_width]`. Grip evidence cites `o<t>:ee`, and move evidence cites the observation of the frame it starts or ends at.
  - `debug.explain_item(rows, id)` and `debug explain --item <id>` walk any evidence, event or observation id.
  - The selftest checks that **every event-log line resolves to an observation**: 136 lines over 10 train episodes, 0 failing.
  - Test: a grasp event explains down to `o6:ee`.
- **One source of parameters.**
  - `left_margin`, `on_dz_min` and `on_dz_max` moved into RuleParams.
  - `geometric`, `facts` and `controller_view` take the frozen `params.rule`, and dry_run passes it. Test: the view changes with `r_near`.
  - Re-frozen: params.json hash **65d1345cf1be**.

**Code MINORs:**
- Geometric lines in the view now cite the two pose observations and read "from positions". Objects cite their last pose observation.
- A held object that disappears is set to `held`, so it is no longer listed as both Visible and Held (test).
- `move_start` records `t_at = t − 1`, the frame its from-position is taken from. The log's "t=a–b" is now right, and measure_kb uses it.
- `event_log(last=30)` keeps cover / grasp / place events and says "(N earlier lines omitted)" (test).
- New tests:
  - gripper hysteresis on the reviewer's width sequence;
  - no grasp when nothing is within `r_grasp`;
  - move evidence carries xyz and its start time;
  - the hidden-pose test can no longer pass vacuously.
- The Makefile uses `.venv/bin/python` when it exists, so plain `make check` works.
- Stale docs:
  - README status;
  - DEVELOPMENT G / M2 now say claim 3, not claim 1;
  - HANDOFF's "arm C vs D".
- The NOTES M3 figure "about 360 tokens" was one episode and uncited. measure_kb now reports the view size. Over train seeds 0–99: median 430, max 534 (chars / 4) (`results/latest/m3_kb_truth/summary.json`).

**Goal-review points fixed here, because they are wording or measurement, not plan:**
- **Truth for passes.** It no longer reuses the detector's `v_move`. A true pass is a local minimum, below 15 cm, of the true distance between two containers that are both moving. With that truth, see the table at the end of this section; with the old definition it was 0.94 / 0.86 (`results/latest/m3_kb_truth/summary.json`). Under 15 mm noise the detector is still broken.
- **The log no longer states certainty it does not have.** R6's `swap` event reads "passed close (identity may be confused: see Inside)" and Progress says "close passes". A near_pass line says "(detected)". Geometric relations print no probability: theirs was the product of pose confidences, not a relation probability.

**Test-read ledger:** the code reviewer ran one evaluation without `--split train`, which appended 6 READ lines (hash b0a2dd3cb30b, seeds 1000–1019) to `results/test_reads.log`. Nothing was tuned, and those seeds had been read many times before. The lines are kept, since the ledger is append-only.

**Open, for the user (goal review, plan-level):**
- **Arm K0:** the current frame plus a current object table with 3D centres, and no relations, events or history. It separates "memory" from "being given coordinates", since arm K has depth-derived coordinates that H lacks.
- **A one-sentence arm:** the current frame plus "the target cube is under <argmax Inside>".
- **A feasibility probe before the closed-loop harness:** about 20 train episodes, four arms, scored on whether `target` is the true container.
- **Fix the success margin and n before running.** 50 test episodes give about ±6 pp.
- **M3 is not done:** relation precision / recall and L2 calibration are still unmeasured (no manipulation data), and so is an oracle-replay check.

M3a truth check after the fixes (train seeds 0–99, clean tree b20eff4, params 65d1345cf1be; `results/latest/m3_kb_truth/summary.json`):

| | cover precision / recall | near-pass precision / recall (true passes) | moves logged | view tokens median / max |
|---|---|---|---|---|
| clean | 1.00 / 1.00 | 1.00 / 0.83 (272) | 380 | 430 / 534 |
| pos_noise 15 mm | 1.00 / 1.00 | 0.33 / 0.08 | 38 | 292 / 452 |

- The clean misses are true closest approaches between about 6 and 15 cm, which the detector's 30 px threshold does not flag.
- Under noise the view is smaller because fewer events are logged: the detector breaks (M2 item 1).

## Spec v1.5 — the target loop written down (2026-10-28, user discussion)
After a long design discussion with the user, `DEVELOPMENT.md` §2 was rewritten. The user approved the picture and asked for it to be "in the spec, with the whole reasoning and how it works".
- **§2.1:** the loop as a Mermaid diagram (it renders on GitHub), plus a built / not-yet table.
- **§2.2:** one step in order: perception → observation table → observe / detectors / rules → KB → VLM annotator at event moments → views (KB text + event log, annotated frame) → LLM controller → checks → motion planner → new frames.
- **§2.3:** the decisions and their reasons:
  - record task-agnostically, retrieve task-specifically (Clio as the contrast);
  - code every frame, language models only at moments that need them;
  - every item is a belief with support, whoever wrote it;
  - an open relation table, with the relations most useful to a waypoint controller;
  - the observation table as the only door between pixels and memory;
  - real-time notes: an in-memory interface that is logged for replay, per-second thresholds, timestamps;
  - what is and is not ours (ConceptGraphs, DovSG, ReMEmbR, 2610.17350).
- **§3.7:** the arbiter is merged into the VLM annotator, with its input, output schema, checks and how it writes beliefs.
- **M3:** adds the wider observation table (contract v2), the open relation table and scoring every cube.
- **M4:** adds the annotated frame (Set-of-Mark), and item 5 now uses the annotator.

**Findings from the discussion, recorded here:**
- **DINOv2 is wasted work in G.** sg_pilot's PerceptionWorker computes a 768-d DINOv2 feature per object per frame. `stress/perceive.py` does not save it, and nothing reads it. The containers are identical, so appearance cannot tell them apart anyway. Neither the builder nor the reviewers caught this, because the worker sits outside the reviewed code (`scene-graph-robot/notebooks/pilot/perception_worker.py`).
  - Action: `/review-phase G` must explicitly review `stress/` and the sg_pilot worker it calls: what each model computes, saves and uses.
  - Drop DINOv2 in future pod runs, if Cutie does not need it (unchecked).
- **Keep `/workspace/g/replays` on the pod volume.** Object size needs masks + depth. G saves no masks, so sizes can only be recomputed from those replays, without re-running the simulator.
- **Two task-bound parts:** the evaluation scores only the target cube, and perception sees only words in its prompt. Both are now M3 items.

## v1.6 — trim to the main line (2026-10-28, user decision)
The user asked whether the complex rules, the Hungarian check and the many baselines still matter under v1.5. Agreed answer: focus on claims 1–2 (M3–M5) and keep claim 3 small, sized by G.
- **M2 is now conditional on G's branch** (spec §6 M2):
  - **always:** motion smoothing with per-second thresholds, tuned on G's real noise; one claim 3 table (Bound vs SoftHaircut vs ArbiterKB) with the M2-infra CIs and matched coverage;
  - **swaps under 2%:** stop;
  - **AUC ≥ 0.98:** use the feature as it is, with no Hungarian step;
  - **frequent and hard to predict:** only then build `identity.py` (Hungarian margin), R6 v2, MarginHaircut and tuning.
- **Rules:**
  - R7 (decay, never switched on) is deleted: code, `RuleParams.lam`, two tests. `flatten` stays for R8.
  - R5 is frozen: kept, not extended.
  - Re-frozen: params.json hash **40d899eba31d**.
- **Baselines:** `--systems m2` = bound, soft_haircut, arbiterkb, and `compare.py` defaults to soft_haircut, bound. OverlapHaircut, BoundHaircut, JumpDetector and Frozen stay only in the M1 table (`--systems all`), for history.
- **Bound's role now:**
  - claim 3's deterministic baseline;
  - the content of the new arm **S**: one sentence per hidden cube with the KB's most likely container, which on VideoUnmaskSwap is Bound's answer. If S ties K, the KB's structure adds nothing on that task, which is the strongest falsifier for claims 1–2.
- **New memory arms** (`controller.py`, `dry_run.py`, spec §3.8):
  - **K0:** the current frame's visible objects with 3D centres; no relations, events or history. It separates "memory" from "being given coordinates" (M3a goal review).
  - **S:** see above. S gets no coordinates, so like H it must read positions from images; it is meaningful only with RGB replays.
  - Spec §3.8 now says to run K vs S and K vs K0 first, and to fix the success margin and n before any run.

## M4 — first real calls: text-only dry run (2026-10-28)
The user put an OpenRouter key in the git-ignored `.env`, with a $40 limit on the key; `ARBITER_MAX_USD` is 2.00 per run.
- **Smoke test:** `qwen/qwen3.6-35b-a3b` through DeepInfra works, at about 5 s and $0.0003 per short call.
  - It is a **reasoning model**: with `max_tokens` 50, all tokens went to reasoning and the content came back None.
  - `llm.py` now accepts a `reasoning` setting, returns "" for an empty answer (counted as `empty`) and counts reasoning tokens.
  - `dry_run` now uses `max_tokens` 4000.
- **Dry run** (`results/latest/m4_dry_text/`: dry.csv + meta.json; train seeds 0–19, clean synthetic, text only, one decision per episode at t*, total **$0.17**). Each arm gives one waypoint for "pick up the container that hides the <colour> cube".

| arm | reads | target = true container | waypoint within 3 cm of it | input / output tokens per call |
|---|---|---|---|---|
| K | KB view + event log | 19 / 20 | 19 / 20 | 1,488 / 2,099 |
| K-L3 | KB view, no event log | 20 / 20 | 20 / 20 | 1,010 / 2,081 |
| R | raw tracker log (every 10th frame) | 18 / 20 | 13 / 20 | 2,273 / 2,190 |
| K0 | current visible objects only, no memory | 5 / 20 | 5 / 20 | 660 / 2,239 |

Reading (n = 20, synthetic, oracle observations, text only, a single step: **plumbing, no claim**):
- **Memory matters.** K0 is at chance (5 / 20, with 3–4 containers), because the cube is hidden and the current frame cannot tell where it is. So K's 19 / 20 is not just "being given coordinates", at least on this task. This answers the goal review's K0 question in this setting.
- **The structure barely beats the raw log here, on naming the container.** R names the right container in 18 / 20, but its waypoint is within 3 cm in only 13 / 20. The 5 misses are each 14–15 cm off, which is exactly the distance from the container's current position to **where the cube was first seen** (eps 2, 3, 9, 11, 12; checked against the generator). So the model named the right container but aimed at the cube's original spot, not at where the container had moved to. (Not staleness: R's log includes the current frame.) The KB arms print the container's current centre and Inside, so they never made this mistake.
- **The event log adds nothing on this task.** K-L3 does at least as well as K, because Inside already holds the answer. That fits VideoUnmaskSwap, where the KB's answer equals Bound's. Arm S (Bound's answer in one sentence) still needs images to be tested.
- **K's one miss (ep 9)** is where the KB itself was unsure (bin_2 0.62, bin_1 0.37, after a 1 cm close pass): the model picked the less likely container.
- **Most output tokens are reasoning:** about 2,100 per call, against an input of 660–2,300.
- **A bug the run exposed:** almost every "problem" was the model citing an observation id (`o173:bin_1`), which the view prints but `controller.check` rejected. Fixed: `_citable` accepts `o<t>:<object | ee | image>`. Still rejected: cites of a relation's text ("Inside(cube_red): …") or a bare "o32". Relation lines need their own ids.
- **Next:** RGB replays for arms H and S and for the annotator. Clean synthetic text is too easy to separate the arms; the informative tests are perceived input, images and tasks beyond permanence.

**Also built (commit a1fbfd4):**
- **`annotate.py`, the VLM annotator** (spec §3.7):
  - checks each item on its own: ids exist, cites exist, strength is one of three words, the predicate is one short word, no numbers anywhere;
  - writes beliefs into the KB's open fact table (`kb.facts`, `SetFact`, source `vlm`, p from `ArbiterParams.p_clear / p_ambiguous / p_unlikely`);
  - writes descriptions only when an image was shown;
  - its output shows in the controller view.
- **`som.py`, the annotated frame:** ids, plus "red? 0.92" on containers. Labels can overlap when two containers are close; not fixed yet.
- Re-frozen: params hash **2941773a1dd4**.
- Tests: `tests/test_annotate.py`. The annotator has not been called on the real model yet.

**Security:** the key was pasted in the chat. The user should revoke it and put a new one only in `.env`.
- **Why K missed ep 9 and K-L3 did not** (read from the cached reasoning):
  - The event log showed "cube_red disappeared at (0.01, −0.01)", and after three swaps bin_1 happened to sit at that spot again. The model reasoned "the disappearance location is the strongest cue" and overrode `Inside(cube_red): bin_2 0.62 | bin_1 0.37`.
  - This is the same error as R's 5 misses: assuming the cube stays where it was seen, not that it moves with its container.
  - Candidate fixes, **not applied** (the user wants changes batched and tested together, not piecemeal):
    - merge disappear + cover into one line ("covered by bin_2 at …; moves with bin_2");
    - state the containment rule in the prompt;
    - rerun with n ≥ 50.

## Perception without privileged scene information; task items (2026-10-28, user decision)
- **Finding:** G's perception (`stress/perceive.py` → sg_pilot) builds the GroundingDINO word list from the simulator scene: `"container." × n_bins` plus one `"<colour> cube."` per cube in the episode. The detector is therefore told how many objects exist and which colours, which a real robot does not know. It also sees nothing outside that list (table, arm, buttons).
  - That is fine for G's question (does the tracker swap ids, given the right objects?).
  - It is privileged for the claims: arm K's results on such perception are optimistic.
- **Change (user: "change it per v1.5 now; mark task_item where the task gives key information"):**
  - `arbiter_kb/task_items.py`:
    - `parse_task_items(goal)` reads the object phrases from the **instruction text only**: "pick up the container that hides the red cube" → container, red cube. It is a deterministic parser for RoboMME-style instructions; an LLM extractor (as scene-graph-robot's `llm_entities.py`) is the general version.
    - `open_vocabulary(items)` = `container.` ×5, then the task's colour phrases, then `cube.` ×4. The fixed slot counts are upper bounds (unused slots stay empty) and are the same for every episode.
    - `is_task_item()`.
  - The KB stores `task["items"]`. The controller view lists "Task items (named by the goal)" and tags matching objects "(task item)". Everything else is still recorded: marking, not filtering.
  - `stress/perceive.py --vocab open` (pod side) uses that word list. The scene is read only to store the evaluation truth in meta. `--vocab scene` stays the default, so G's run is unchanged and comparable. POD_RUN §6b has the optional re-run on G's replays.
  - **Untested against the simulator:** extra empty slots may cost GroundingDINO calls every frame. Generic `cube.` slots carry no colour, so a cube not named by the task needs its colour from the annotator or the image. Slot ↔ truth matching (`swapstats.assign_slots`) with more slots than objects is unchecked.
- **Claims:** results with `--vocab scene` are marked privileged. Claims 1–3 are made on `--vocab open` perception.
- Tests: `tests/test_task_items.py`.

## G — gate result and `id_swap_measured` (2026-10-28)
- **Full pod run:** 9 settings × 20 episodes = 180 perceived episodes (`--vocab scene`, i.e. privileged word list), 353 passes. Outputs: `results/g_passes.csv`, `results/g_summary.json` (reproduced on the Mac from `data/g_perceived/`; differences vs the pod's copy are float noise at 1e-9).

| setting | identity acc. | passes | swaps | swap rate [Wilson 95%] |
|---|---|---|---|---|
| stock (50, 0.07) | 0.9998 | 44 | 0 | 0.00 [0.00, 0.08] |
| cam_left / cam_right | 0.9998 | 35 / 37 | 0 / 0 | 0.00 |
| fast (20, 0.07) | 0.980 | 33 | 6 | 0.18 [0.09, 0.34] |
| fast_lane045 | 0.946 | 30 | 12 | 0.40 [0.25, 0.58] |
| close (50, 0.02) | 0.871 | 57 | 21 | 0.37 [0.26, 0.50] |
| fast_lane03 | 0.879 | 33 | 22 | 0.67 [0.50, 0.80] |
| fast_close (20, 0.02) | 0.839 | 33 | 27 | 0.82 [0.66, 0.91] |
| fast_close_3swaps | 0.744 | 51 | 42 | 0.82 [0.70, 0.90] |

- **Gate: swaps exist** (≥ 2% in 6 of 9 settings), and **no threshold suffices**: best single feature AUC 0.874 (`max_iou`), best ≤ 2-feature rule 0.926 (`max_iou + area_ratio`, cross-validated) < 0.98. So M2's "swaps are frequent and hard to predict" branch applies.
- **Stock RoboMME never swaps** (0 / 44 passes, 0 / 116 with the camera shifts). Claim 3 on real errors needs the stress settings.
- **What a real swap looks like** (130 swapped vs 223 clean passes):
  - **Distance:** container centres never come closer than 5.1 cm (they are solid; lane 0.02 gives min_3d ≈ 6 cm, not the 4 cm of our synthetic path). Swaps happen at min_3d 5–9 cm (median 6.0 cm); rate 0.70 below 7 cm, 0.29 at 7–10 cm, 0.02 above 10 cm.
  - **Speed matters as well as distance:** at min_3d ≈ 11.6 cm, stock (50 steps) 0 / 44 vs fast (20 steps) 6 / 33.
  - **A mask dropout comes first:** 88% of swaps have a track absent for ≥ 1 frame around the pass (median 6 frames), vs 39% of clean passes (median 0). `area_ratio` is 0 for most swaps: one mask vanishes, then the track re-acquires on the other container.
  - **Timing:** the exchange shows median 3 frames after closest approach (10–90%: −2 to +9).
  - **Label jump at the exchange:** median 7.1 cm (10–90%: 3.1–10 cm). This is far from `id_swap_overlap`'s ≤ 3 cm and close to M0's "teleport" `id_swap` (12 cm). M1's premise that real swaps only happen where tracks overlap (≤ 3 cm) is wrong for this tracker.
  - After a swapped pass a track is lost in 18% of cases (clean: 7%).
- **`id_swap_measured` (built):** `MeasuredSwapSource` in `arbiter_kb/sources/realistic.py`, condition `id_swap_measured` in `scripts/run.py`. Parameters are read from `results/g_signature.json`, which `scripts/measure_swaps.py` now writes (new per-pass columns: `speed`, `swap_dt`, `dropout`):
  - p(swap) = logistic(−78.5·min_3d + 180·speed + 4.26), min_3d clamped to G's range [5.1, 23] cm;
  - a swapped pass exchanges the ids at t_min + a `swap_dt` drawn from G's, after one of the two containers is hidden for a dropout length drawn from G's swapped passes; a clean pass hides one for a length drawn from G's clean passes;
  - all draws are seeded per pass.
  - **Calibration on G's own passes** (fit vs real rate): stock 0.04 vs 0.00, cam 0.03–0.04 vs 0.00, fast 0.09 vs 0.18, fast_lane045 0.34 vs 0.40, close 0.52 vs 0.37, fast_lane03 0.64 vs 0.67, fast_close 0.73 vs 0.82. A two-feature linear fit: it overpredicts slow close passes and underpredicts fast far ones.
  - **Known gap:** synthetic lanes 0 / 0.01 / 0.02 bring centres to 0–4 cm, which real containers cannot. They are clamped to 5.1 cm, so they get G's highest swap rate (about 0.75 at 50 steps), above the real `close` setting's 0.37.
  - On synthetic `mix`, the swap rate is 0.52 of passes, with a median label jump of 5.3 cm.
- **First table (train seeds only, no test read):** `--lane mix --steps mix --systems m2`, calibration 150–449, eval 450–749 (`results/runs/synthetic_train_94f0be4a/compare.csv`):
  - Bound: accuracy 0.573, silent 0.427, AURC 0.427.
  - ArbiterKB: accuracy 0.577, silent 0.25, coverage 0.753, AURC 0.169; ΔAURC vs Bound −0.258 [−0.284, −0.228].
  - **SoftHaircut ties ArbiterKB exactly** again (ΔAURC 0.0 [0.0, 0.0001], 300 / 300 same answers).
  - Accuracy is low for every system: about 0.42 of rows are wrong, and graded confidence only ranks them. R6 still uses d0 = 2 cm, tuned to the 3 cm overlap swaps, while real swaps happen at 5–9 cm.
- **Next (M2 branch "frequent and hard to predict"):** the identity check (`identity.py`, Hungarian margin) and R6 v2, tuned on train seeds against `id_swap_measured`, with SoftHaircut given the same feature. Then a freeze and one test read. The `--vocab open` re-run (POD_RUN §6b) is still needed before claims.

## Which small VLM can list the objects? (2026-10-28, user request)
User idea: let a VLM list the "clue objects" (6–10) in a frame, and run GroundingDINO → SAM → Cutie on those phrases, instead of the simulator's object list. With thinking off, how small can that VLM be?
- **Test** (`results/latest/vlm_list_bench/`: the 4 frames, `bench.py.txt`, `bench.log`):
  - Four real RoboMME VideoUnmaskSwap front frames, cropped from `scene-graph-robot/notebooks/pilot/figures/02_frames.png` (episode 0):
    - f21: red, green and blue cubes;
    - f56: 3 containers (cubes covered);
    - f105: 3 containers mid-swap;
    - f325: 3 containers, one held by the robot, plus a green cube.
  - The prompt asks for every movable object on the table (not the table, not the robot) as JSON `{name, color, count}`. Thinking off; default OpenRouter routing; one call per frame.
- **Scoring:** a frame is right when the total count is right, the coloured cubes are exactly right, and no other kinds of object are named. Nearly every model calls the white containers "white/gray cubes" (they look like small white boxes). That is accepted, since the phrase only has to work as a GroundingDINO prompt.

| model | right | s / call | $ for 4 calls | failures |
|---|---|---|---|---|
| qwen/qwen3-vl-8b-instruct | **4 / 4** | 0.9 | 0.00026 | — |
| qwen/qwen3.5-9b (thinking off) | **4 / 4** | 1.1 | 0.00013 | — |
| qwen/qwen3-vl-30b-a3b-instruct | **4 / 4** | 2.6 | 0.00018 | — |
| qwen/qwen3.6-35b-a3b (thinking off) | 3 / 4 | 1.0 | 0.00021 | f56: 4 objects for 3 |
| google/gemma-4-26b-a4b-it | 3 / 4 | 1.6 | 0.00018 | f325: missed the held container |
| google/gemma-3-4b-it | 2 / 4 | 3.9 | 0.00008 | invents a "robot" and a "button" |
| mistralai/ministral-3b-2512 | 2 / 4 | 1.3 | 0.00010 | cubes called "balls"; f325 one extra |
| mistralai/ministral-8b-2512 | 1 / 4 | 2.9 | 0.00015 | "spheres", an extra "object" |
| google/gemma-3-12b-it | 0 / 4 | 2.3 | 0.00012 | an extra object in every frame |

Reading:
- About 8–9B is the floor here: qwen3-vl-8b and qwen3.5-9b get all four, in about 1 s and for a fraction of a cent. Below that, models invent objects or misname the coloured cubes.
- **This is a tiny test:** 4 frames from 1 episode, frames resampled from a figure, counting only, and no case with many or unusual objects. Before choosing, re-run on G's replays: dozens of frames with simulator truth, frames before and after covers, and while held. Also count misses and inventions separately.
- In this test, thinking off costs about 1 s per call, against 5–40 s with thinking. That settles the keyframe VLM latency.

## Spec v1.7 (2026-10-28)
`DEVELOPMENT.md` now holds the perception and timing discussion:
- **§2.5 perception design:**
  - a small VLM (thinking off) lists the clue objects as the detector's phrases, task items tagged;
  - one multi-object Cutie; no DINOv2; SAM encodes once per keyframe;
  - keyframe triggers and work, including Hungarian matching that never re-matches during a close pass and writes `id_margin` when unsure;
  - keyframes asynchronous.
- **§2.6 time budget:** before, v1.7 unoptimised and v1.7 optimised, against the paper, with the source of each number. The controller's thinking (about 40 s per call by default) decides most of the episode time.
- **§2.1:** the status table is updated.
- **M4 items 7–9:** the thinking-effort test, the clue-object lister and its re-benchmark on G's replays, and the batch of KB / view fixes, to be tested together.
- **M5 item 5:** closed-loop perception.

## Batch A (Mac) — the batch of fixes, and the test fixed before it runs (2026-10-28)
**Fixes, applied together** (spec §6 M4 item 9):
- **Cover lines:** one line per cover: "cube_red covered by bin_2 at (x, y) (conf 0.99) [e9]; from then on it moves with bin_2". The separate "disappeared at" line is gone; in ep 9 it led the model to the cube's old spot.
- **Containment rule in every prompt:** all arms' prompts (shared `ENDING`) say: "an object covered by a container moves with that container until it is seen again; positions recorded earlier are where things were at that time". Every arm gets it, so the comparison stays fair.
- **Relation ids:** relation lines carry an id (`[Inside(cube_red)]`, `[On(a, b)]`). `views.citable_ids` lists every printable id, and dry_run checks cites against it.
- **Set-of-Mark labels:** greedy placement (`som.place`): above, below, right, left, then farther out with a leader line. A test checks four objects within 20 px.
- **Clue-object lister** (`clue_objects.py`, spec §2.5):
  - VLM prompt, per-entry checks, and a cap of 10;
  - phrases without a repeated colour;
  - task items from the instruction text OR the VLM's flag. Its own flags were inconsistent on the real frames.
  - qwen3-vl-8b on the 4 real frames: all right except f325, where it listed 1 container instead of 3. Re-test on G's replays.
- **dry_run:** `--effort default,low,off` (controller thinking), `--first` (seed offset), `--workers` (parallel). `scripts/dry_report.py` reports per (arm, effort) with Wilson CIs and a paired episode bootstrap against a reference.

**The test, fixed before running:**
- **Seeds:** train seeds 100–149 (n = 50), disjoint from the prompt-development seeds 0–19. Clean synthetic, text only, one decision at t*.
- **Grid:** arms K, K-L3, K0, R × effort default, low, off; qwen/qwen3.6-35b-a3b via DeepInfra; cap $3.
- **Primary metric:** the right-container rate. **Secondary:** waypoint within 3 cm, tokens, seconds and $ per call.
- **Decision rule** (`dry_report.py`, reference K at default effort), on the paired difference with a 95% bootstrap CI:
  - **better / worse:** the CI excludes 0;
  - **same:** the CI lies within ±10 percentage points;
  - **undecided:** otherwise.
- **Questions:**
  - Does lower thinking keep K's accuracy (time budget, spec §2.6)?
  - Do K-L3 / R / K0 differ from K once the log and prompt fixes are in?

**Batch A results** (`results/latest/m4_batchA/report.json` + `dry.csv`; run `results/runs/dry_synthetic_20261028_100910`, commit 89aa725, clean tree; 600 calls, **$0.92**; train seeds 100–149, clean synthetic, text only, one decision at t*):

| arm | effort | right container [Wilson 95%] | within 3 cm | input tok | reasoning tok | s / call |
|---|---|---|---|---|---|---|
| K | default | **0.96** [0.87, 0.99] | 0.96 | 1,561 | 1,217 | 20.9 |
| K | low | 0.92 [0.81, 0.97] | 0.92 | 1,561 | 1,191 | 20.4 |
| K | off | 0.68 [0.54, 0.79] | 0.66 | 1,563 | 0 | 3.5 |
| K-L3 | default | 0.92 [0.81, 0.97] | 0.92 | 1,068 | 1,191 | 20.3 |
| K-L3 | low | 0.92 | 0.94 | 1,068 | 1,182 | 20.1 |
| K-L3 | off | 0.70 | 0.68 | 1,070 | 0 | 2.9 |
| R | default | **0.96** [0.87, 0.99] | 0.94 | 2,171 | 1,358 | 23.5 |
| R | low | 0.96 | 0.94 | 2,171 | 1,327 | 23.3 |
| R | off | 0.58 [0.44, 0.71] | 0.48 | 2,173 | 0 | 3.1 |
| K0 | default / low / off | 0.38 / 0.34 / 0.30 | | 692 | | |

Against K at default effort (paired episode bootstrap; rule fixed before the run):
- K-low: −0.04 [−0.10, 0.00], **same**.
- K-off: −0.28 [−0.40, −0.16], **worse**.
- K-L3 default / low: −0.04 [−0.10, 0.00], **same**.
- R default / low: 0.00 [−0.08, 0.08], **same**.
- K0: −0.58 to −0.66, **worse**.
- Without thinking, against K-off: R-off −0.10 [−0.28, 0.08], K-L3-off +0.02 [−0.12, 0.16]. Both **undecided**.

Reading (plumbing on clean synthetic data; no claim moves):
- **Thinking matters, and "low" saves nothing.** With thinking off, K drops from 0.96 to 0.68. "low" kept about the same number of reasoning tokens (1,191 vs 1,217) and the same latency, so this model and provider effectively ignore it. A controller call costs about 20 s with thinking, or 3.5 s and much less accuracy without. Spec §2.6's "40 s" estimate is replaced by the measured 20 s per call (8 calls in parallel, latency per call). **The time budget therefore needs either thinking (about 20 s per call, about 200 s per episode) or a better non-thinking model**; to be tested.
- **With the fixes, the raw tracker log is as good as the KB** (R = K = 0.96). R's earlier misses (aiming at the cube's first-seen spot) disappeared once every prompt states the containment rule. On this task, with clean observations and a thinking model, the KB's structure adds **no accuracy**. It saves input tokens: K 1,561 and K-L3 1,068 against R's 2,171 (−28% / −51%). Synthetic episodes are short; the raw log grows with episode length (RoboMME up to 1,300 steps), and the KB view does not.
- **The event log still adds nothing** here (K-L3 = K), as before.
- **Memory is still necessary** (K0 at 0.30–0.38, near chance).
- **Without thinking, the KB looks slightly ahead** (K 0.68, K-L3 0.70, R 0.58), but n = 50 cannot decide it.
- **Implication for claims 1–2:** on clean synthetic VideoUnmaskSwap with text only, "the KB beats the raw log" is **not supported**, as the M3a goal review predicted for permanence tasks. A difference can only come from:
  - perceived observations with real id swaps and noise (G's data), where the raw log copies the tracker's errors silently;
  - long episodes, where the raw log grows;
  - images (arm H: about 58 images against 2);
  - a non-thinking or smaller controller;
  - tasks beyond permanence, where the answer is not one Inside line.

  These are the next tests, not this one.

## G_M4 reviews: the five MAJORs fixed (2026-10-28)
Reviews: `reviews/G_M4_code.md` (PASS-WITH-NOTES) and `reviews/G_M4_goal.md` (ON-TRACK-WITH-CHANGES). Scope: G plus M4 up to 89aa725; Batch A's results commit (9876a81) came after the reviews. MINORs and the goal review's plan changes are not done yet.
1. **Swap shape (`swapstats._outcome`, `MeasuredSwapSource`).**
   - Every pass now has a `kind`: exchange / merge (both slots on one container) / lost (the other slot on none) / clean.
   - G: 77 / 30 / 23 of 130 swaps. `kind_counts` is in `g_signature.json`; the per-setting table has `swap_exchange` / `swap_merge` / `swap_lost`.
   - Passes no slot sat on are now counted (`unmatched`, 51 of 404) instead of dropped silently.
   - `id_swap_measured` draws the kind at G's rates. A merge leaves two labels reporting one container. A lost swap hides the other label. Both last to the end of the episode, as most of G's do.
   - New tests: one per kind in swapstats, plus merge and lost in the source.
   - **The id_swap_measured table changes** (train seeds; calibration 150–449, eval 450–749):
     - Bound AURC 0.407;
     - ArbiterKB 0.182 against SoftHaircut 0.180: ΔAURC +0.002 [0.0, 0.005], 290 / 300 same answers.
     - Merges and lost tracks make ArbiterKB very slightly *worse* than SoftHaircut, not better.
     - This replaces the "First table" numbers in "G — gate result" (0.169 / 0.427, exact tie).
2. **Cite check.**
   - `controller.check` accepts only ids the arm's prompt printed (`controller.shown_ids`: bracketed ids plus object names in the text).
   - `o99999:bin_0`, a future frame, `o<t>:image` on a text arm and LeftOf ids that are never printed are now `unknown cite`.
   - Batch A was scored with the old check: 192 of 600 rows already had problems, and the new check can only add to them. Its right-container numbers do not use cites.
3. **Cache and budget (`llm.py`, `dry_run.py`).**
   - A cache hit reports the original call's tokens, seconds and cost (`cached_usd`), and `usd` stays 0.
   - dry.csv has `cache_hit`, and its `usd` is the call's cost. A replay writes the same dry.csv apart from `cache_hit` (test).
   - One client and one locked budget serve all threads. A call is refused when spend + calls in flight + itself, each at the dearest price seen, would pass the cap. Test: a $0.25 cap at $0.10 a call stops at $0.20.
   - Rows are written as they arrive, so a budget stop keeps the paid rows.
   - Batch A has no zero-token rows, so its token numbers stand.
4. **Provenance (E4).**
   - dry_run writes `meta.json`: commit, dirty, args, params hash, contract version and LLM usage incl. hits / refused.
   - `g_summary.json` / `g_signature.json` carry `provenance`: commit, args and one sha256 over the 180 inputs, `3f33698888bc…`. Regenerated from the clean tree at 52f7ca4 (only the provenance changed).
   - Run meta records `g_signature_sha256` when id_swap_measured runs.
5. **G Done-when (b).** `figures/g_features.png` (per-feature histograms, swapped vs clean, with AUC) and `figures/g_roc.png` (one-feature ROCs), written by measure_swaps.

`make check`: 159 passed; quality 0 problems; selftest 16 PASS, 1 SKIPPED (no replays).

## Beyond permanence: the RoboMME dataset, robot-state memory (2026-10-28)
Decision with the user: test tasks where memory is about order, counts and "where it went", not one Inside line. Data: RoboMME's LeRobot dataset (`huggingface.co/datasets/lerobot/robomme`, revision 92796ab).
- **Downloaded:** only the tables and metadata, **84.8 MB**, into git-ignored `data/robomme_lerobot/`. The videos (9.5 GB) were not downloaded.
- **What the tables hold:** 1,600 episodes, 10 fps. Per frame: state (7 joint angles + finger), actions, `is_demo`, simple and grounded subgoals (the latter with pixel coordinates). **No depth, segmentation, object poses or camera parameters.**
- `tools/convert_robomme_lerobot.py` turns the tables into one npz per episode. It runs with the system Python's pyarrow: pyarrow is not on the dependency whitelist, so the tool lives in `tools/`, outside `quality.py`, and the project reads only the npz files.
- **`arbiter_kb/fk.py`, Panda forward kinematics** (modified DH, flange 0.107 m + TCP 0.1034 m, base at (−0.615, 0, 0)), checked on the data:
  - at the 1,025 grip open / close moments, the TCP height is median 0.019 m (p10 0.012, p90 0.022), the height of a cube's centre;
  - one homography maps the TCP xy at those moments to the subgoal pixel coordinates with a median residual of 2.3 px (p90 4.7) on 256-px frames.
  - Tests: `tests/test_fk.py`.
- **Can robot-state memory alone answer the event-order tasks? Mostly not.** The answer was rebuilt from the hand's positions when the gripper opened, then compared with the official execution-phase target (the grounded subgoal):
  - VideoPlaceOrder: 58 / 100 within 10 px;
  - VideoRepick: 57 / 100.
  - The reason, checked on VideoRepick episodes: in the demo the robot puts the cube back where it picked it up, but **at execution the "same cube" is somewhere else**. The scene is rearranged, so the task asks which cube (colour / appearance), not where the hand went.
  - These tasks need vision to identify the object: for example, the annotator naming the cube in the gripper at each grasp event. Front-video files: VideoRepick 0.67 GB, VideoPlaceOrder 0.92 GB (files 018–023, partly shared, about 1.4 GB together).
  - **Counting tasks** (PickXtimes, SwingXtimes, BinFill) depend on the robot's own actions, so they can be tested from robot state alone.

## Counting tasks from robot state: the test, fixed before running (2026-10-28)
- **Data:** RoboMME dataset, the first 30 episodes each of PickXtimes, SwingXtimes and BinFill. Decision points are the first frame after each completed place / put-into-bin / left-side move in the execution phase: 234 in all.
- **Truth:** the action class of the next subgoal, from the dataset authors' labels. It is "repeat" (pick again / move right) or "wrap up" (press the button / put the cube on the table). Always answering "repeat" scores 64% / 55% / 65%: the floor.
- **Arms:**
  - N: no memory.
  - P: raw robot log, hand xyz + gripper every 10 frames.
  - K: the KB run on robot state only. It shows grip close / open evidence, the new `hand_pause` evidence (hand still for `n_pause` = 4 frames at < `v_pause` = 2 mm/frame), progress counts and the event log, via `views.controller_view`.
  - KO: oracle, the completed subgoals as text.
- Effort off and default; qwen/qwen3.6-35b-a3b (DeepInfra); cap $4. The answer is one action from a fixed 7-item list shared by all tasks.
- **Rule** (`scripts/count_test.py`, report below): accuracy per (task, arm, effort) with Wilson CIs; the paired difference against K at the same effort, bootstrapped over decision points (2000 draws, seed 0):
  - **better / worse:** the CI excludes 0;
  - **same:** the CI lies within ±10 pp;
  - **undecided:** otherwise.
- **The question:** does a structured record of the robot's own actions help the controller know where it is in the task, against a raw log, and with and without thinking?
- **Code:**
  - `arbiter_kb/sources/lerobot.py` (robot-state observation tables, subgoal segments);
  - the `hand_pause` detector (params re-frozen to **354e33c10d24**);
  - the progress line now counts grip and pause evidence;
  - `scripts/count_test.py`.

## VideoRepick is a shell game with identical cubes (2026-10-28)
- `tools/extract_frames.py` (system Python's PyAV; AV1 MP4) pulls the frames these tests need out of the dataset's front videos, for VideoRepick and VideoPlaceOrder: grip-close frames, the first frame, and the first execution frame. 200 episodes, 279 MB in git-ignored `data/robomme_lerobot/frames/`, in about 1 minute.
- **What the frames show** (episode 1401, checked by eye):
  - all three cubes are the **same red**;
  - in the demo the robot picks the top-left cube and puts it back;
  - then, during the "static" part of the video (frames about 220–340), **the identical cubes slide and exchange places**;
  - at execution the "same block" is therefore at another spot (the official target is the bottom-right one).
- So VideoRepick is a shell game: the memory must track which identical cube went where through the shuffle. Robot state cannot do that (57 / 100 ceiling), and neither can a colour label.
- **That is the identity-through-swaps problem of claim 3, in a stock RoboMME task, with real swaps of identical objects.** It needs visual tracking of the cubes (Cutie on the pod, or a cheap classical tracker: red blobs + assignment over frames). Where cubes pass close, the KB's identity uncertainty (R6-style mixing) has a natural role.
- VideoPlaceOrder's 58% ceiling is not yet explained (to check: does the scene also change there?).

**Counting test results** (`results/latest/count_robot_state/`: report.json, rows.csv, meta.json; run `results/runs/count_20261028_110748`, 1,872 calls, **$1.03**, all answers valid). **Scoring correction made after the run:** SwingXtimes's instruction says "finally press the button" and never mentions putting the cube down, while the dataset puts it down first. So for that task a wrap-up decision accepts either "put the cube on the table" or "press the button", for every arm alike. Strict-scoring numbers are in rows.csv.

Right-decision rate (234 decision points per arm and effort) [Wilson 95%]:

| effort | N (no memory) | P (raw robot log) | K (KB on robot state) | KO (oracle step list) |
|---|---|---|---|---|
| off | 0.48 [0.42, 0.54] | 0.58 [0.51, 0.64] | 0.49 [0.43, 0.56] | **0.98** [0.95, 0.99] |
| default | 0.62 [0.55, 0.68] | 0.80 [0.75, 0.85] | 0.82 [0.76, 0.86] | **0.97** [0.94, 0.99] |

- Per task, at default effort:
  - PickXtimes: P 0.98, K 0.94, KO 1.00;
  - BinFill: P 0.96, K 0.98, KO 1.00;
  - SwingXtimes: P 0.38, K 0.45, KO 0.89. No arm without the oracle does better than N's 0.55.
- Against K (paired):
  - off: P +0.085 [+0.017, +0.154], **better**; N −0.01, **same**; KO +0.49, **better**;
  - default: P −0.01 [−0.06, +0.04], **same**; N −0.20, **worse**; KO +0.15, **better**.
- Cost per call:
  - KO: 244 input tokens; 0.5 s (off), 4.0 s (default);
  - K: 694 tokens; 0.6 s / 8.3 s;
  - P: 1,370 tokens; 0.6 s / 8.6 s.

Reading:
- **A semantic record of completed steps is what helps, and with it thinking is unnecessary.** KO, the list of completed subgoals in words, is near perfect at both efforts: 0.98 without thinking, at 0.5 s and 244 tokens per call. The best non-oracle arm needs thinking to reach 0.80, at about 8.5 s and 690–1,370 tokens.
  - **This is claims 1–2 in miniature.** A memory that says what was done lets a fast, non-thinking model do the job, with fewer tokens than the raw log.
  - It holds **only if the KB can produce that record**. KO comes from the dataset authors' labels.
- **The KB as built from robot state does not produce it.** Grip and hand-pause events are low-level, the same information as the raw log in another form: K = P with thinking, and K is slightly worse without thinking, where P's regular samples are easier to read than a list of pauses. SwingXtimes's moves to the right / left targets are not grip events at all, so only the oracle gets them.
- **Next for the KB:** recognise semantic actions as L3 events, each one checked against the dataset's subgoal labels before any arm uses it:
  - "picked up / put down" (grip close / open with the hand low);
  - "pressed the button" (hand down and up at one spot without a grasp);
  - "held above the robot's left / right side" (a hand pause with the gripper closed at height, labelled by the sign of y);
  - "put into the bin / onto the target" (needs the object positions: vision or the VLM annotator).

## VLM choice for the lister and the annotator (2026-10-28, user decision)
- **Prices are set by providers, not by model size** (OpenRouter endpoints, read 2026-10-28):
  - qwen/qwen3-vl-8b-instruct: **one** provider, Parasail, $0.25 / $0.75 per M tokens;
  - qwen/qwen3.5-9b (newer; image + video input): **six** providers, $0.08–0.17 / $0.13–0.25 (DeepInfra bf16 $0.10 / $0.15).
  - An image costs about the same in tokens on both: about 160 input tokens with the prompt, at 256 px.
- **Default:** `ArbiterParams.vlm_model = qwen/qwen3.5-9b`, `vlm_thinking = False`, `vlm_provider = "DeepInfra,Parasail"`.
  - Only those two, in that order, both bf16, with no fallback to others.
  - Pinning DeepInfra alone failed at once: HTTP 429, "temporarily rate-limited upstream", twice.
  - The provider actually used is in each cached response.
- `llm.vlm_client(params.arbiter)` builds the client. The controller's model stays separate (`qwen/qwen3.6-35b-a3b`).
- Params re-frozen: **84d49928248d**.
- **Check:** the lister through vlm_client on the 4 real frames (`results/latest/vlm_list_bench`).
  - All right: f21 green / blue / red cube; f56 and f105 3 containers; **f325 3 containers + the green cube** (qwen3-vl-8b had listed 1 container there).
  - Task items right (green cube, containers).
  - 0 reasoning tokens, $0.0001 for 4 calls, but **about 4 s per call via Parasail** (1.1 s in the earlier benchmark): latency depends on provider load.
- Still to do: compare the two models on many frames with truth (VideoRepick / VideoPlaceOrder keyframes) before freezing for a claim.
- **Latency check, same 4 frames, one provider each** (2026-10-28):

  | model | provider | s / call | result |
  |---|---|---|---|
  | qwen3.5-9b | DeepInfra | 34–65, then HTTP 429 | unusable right now |
  | qwen3.5-9b | Parasail | 1.1–1.6 | 3, 3, 3, 4 objects: all right |
  | qwen3.5-9b | SiliconFlow / Together | — | HTTP 404 "no endpoints": likely they do not support the reasoning switch |
  | qwen3-vl-8b | Parasail | 1.0–1.8 | f325 again 1 object: wrong |

  The 4 s seen above was the DeepInfra-first order: each call waited for the rate-limited provider before falling back. The order is now **Parasail, then DeepInfra**; params re-frozen. The model stays qwen3.5-9b: as fast on the same provider, more accurate, cheaper.

**Counting test v2 results: the KB with semantic actions** (`results/latest/count_robot_state_v2/`; run `results/runs/count_20261028_113317`; 1,872 calls, **$0.97**, all answers valid; decisions 5 frames after each step for every arm; SwingXtimes wrap-up accepts put-down or press, as in v1):

| effort | N | P (raw log) | **K v2** (semantic actions) | KO (oracle) |
|---|---|---|---|---|
| off | 0.46 [0.40, 0.53] | 0.60 [0.54, 0.66] | **0.85** [0.79, 0.89] | 0.98 [0.95, 0.99] |
| default | 0.61 [0.54, 0.67] | 0.79 [0.73, 0.83] | **0.91** [0.86, 0.94] | 0.98 [0.96, 0.99] |

- Per task, off / default: PickXtimes K 0.90 / 0.96 (P 0.71 / 0.94); **SwingXtimes K 0.65 / 0.80 (P 0.42 / 0.36)**; BinFill K 0.94 / 0.93 (P 0.64 / 0.96).
- Paired against K (rule fixed before v1):
  - off: P −0.24 [−0.33, −0.16], **worse**; N −0.39, **worse**; KO +0.13 [+0.08, +0.18], **better**;
  - default: P −0.12 [−0.18, −0.06], **worse**; N −0.30, **worse**; KO +0.08 [+0.04, +0.12], **better**.
- Input tokens per call: K 505, P 1,385, KO 244, N 191. Seconds per call, off: about 0.8 for every arm. With thinking, K 15 s against P 23 s (the latencies are inflated by the parallel runs).

Reading:
- **This is the first result where the KB beats the raw log, and the gap is largest without thinking**: 0.85 vs 0.60 at about 0.8 s per call, with about a third of the input tokens.
  - A non-thinking model reading the KB (0.85) beats a thinking model reading the raw log (0.79): **claims 1–2 in miniature, on RoboMME data.**
  - The gain over v1 (K 0.49 → 0.85 off) comes only from turning grip / pause signals into semantic actions.
- **Caveats, all to close before any claim:**
  1. The action thresholds were read from these same three tasks' data, and the test ran on episodes from them. **Confirm on unseen episodes (31–100) and on other tasks.**
  2. KO's wording carries ordinals ("for the second time"), so part of its 0.98 comes from the labels themselves. K has no ordinals yet; v3 adds them.
  3. The decisions are offline and single-step; the answers are action choices, not waypoints. There is no closed loop and no frontier-model reference yet.
  4. The same model throughout (qwen3.6-35b-a3b), 30 episodes per task.
- **Next:** v3 view (ordinals, merged transit holds, no empty sections or coordinates; changes batched), confirmed on unseen episodes, with a frontier-model reference arm. The rule is to be written here before the run.

## Spec v1.8 — four roles and a periodic reviewer (2026-10-28, user discussion)
The user asked how the KB is kept correct, and whether one small VLM can, at once, list objects, describe them, give relations, say what happened and check the KB.
- **Decision: split the VLM work into narrow questions** (spec §2.3 "Four roles"):
  - **rules** decide what geometry and robot state decide;
  - the **annotator** adds descriptions and relations from one marked frame, only when the rules cannot decide (at most one call per keyframe, questions batched);
  - the **reviewer** checks the KB against video;
  - the **controller** acts.

  Each answer is then checkable against truth, and a mistake in one does not spoil the others.
- **The reviewer runs on a fixed period (~3 s, overlapping windows) plus once before each action on a target, rather than only on triggers.** Triggers only catch the risks we foresaw; the user proposed the fixed period, and it also catches unforeseen ones.
  - It reads the window's frames (ids drawn) and the KB's events for the window.
  - It returns: confirm / contradict / cannot tell per event; events the KB missed; an identity answer for objects that passed close; a one-line summary anchored to the KB's events. This is the running text summary the user asked for at the start.
  - The answers are written as evidence, never overwriting the rules; a contradiction retracts through the support chain.
  - Estimated cost: about 20–30 calls per episode, about half a cent, run asynchronously.
- The v1.3 arbiter's swap verdict becomes the reviewer's identity check.
- **Spec updates:**
  - §2.1 diagram and status (adds the semantic actions and the counting-test v2 results);
  - §2.2 steps 6 and 6b; §2.3; §2.5 keyframe work; §2.6 cost row; §2.7 code map;
  - §3.7 reviewer protocol and JSON schema;
  - M4 item 10: the reviewer, measured (a) on the downloaded RoboMME videos against the subgoal labels and (b) on G's close passes against per-pass truth (claim 3's real test; needs the pass frames exported from the pod).

## Counting test v3: the view fixes, on unseen episodes, with a frontier reference (fixed before running, 2026-10-28)
- **The v3 view** (`arbiter_kb/action_view.py`, all changes batched):
  - ordinals per action ("pick-up #2", "right hold #1");
  - a hold followed by a release at the same spot (within `merge_xy` = 5 cm) is one line, "held it up on the robot's right side, then placed it down at table level", and the hold still counts;
  - neutral release wording ("placed it down at table level" / "set it down from a small height" / "released it from above (as into a container)");
  - with no objects in the KB, no coordinates and no empty Objects / Relations sections;
  - the progress line counts only actions that happened;
  - no repeated goal line. Params re-frozen.
- **Episodes:** **61–100 of each task** (40 each). The action thresholds were read from episodes 1–60, so these are unseen.
- **Small model:** arms N, P, K (v3 view), KO × effort off / default; qwen/qwen3.6-35b-a3b on DeepInfra; cap $3.
- **Frontier reference:** arms P and K with **anthropic/claude-sonnet-5.5** at its default settings (OpenRouter routing); cap $8. GPT-6-Astra, the paper's model, would cost about $40.
- **Rule** (as in v1 / v2): right-decision rate with Wilson CIs; paired against K at the same model and effort; better / worse = the CI excludes 0, same = the CI within ±10 pp.
- **Questions:**
  1. Does v2's gain over the raw log hold on unseen episodes?
  2. Does the v3 view close part of the gap to KO?
  3. Does the small model + KB without thinking reach the frontier model reading the raw log?

**v3 results on unseen episodes 61–100** (`results/latest/count_robot_state_v3/`: report.json, rows and meta per run; 310 decision points per arm).
- **Incident:** the two runs, started 2 s apart, created the same run directory in the same second and overwrote each other's rows.csv (2,468 of 3,100 rows survived, without a model column).
- **Fixed:** `count_test` now names the directory after the model and the process id, refuses an existing directory, and writes a `model` column.
- Both runs were then replayed **entirely from the cache** (3,100 hits, $0 new). The original spend was **$1.26** (qwen) + **$1.90** (Sonnet).

| model | arm | effort | right [Wilson 95%] | tokens in + out / call | median s | $ / 1000 calls |
|---|---|---|---|---|---|---|
| qwen3.6-35b-a3b | N | off / default | 0.47 / 0.61 | 204 / 1,133 | 0.4 / 6.1 | 0.03 / 0.92 |
| qwen3.6-35b-a3b | P (raw log) | off | 0.61 [0.55, 0.66] | 1,370 | 0.6 | 0.15 |
| qwen3.6-35b-a3b | P (raw log) | default | 0.82 [0.77, 0.86] | 2,736 | 10.4 | 1.45 |
| qwen3.6-35b-a3b | **K (v3)** | off | **0.83** [0.78, 0.86] | 382 | 0.5 | **0.05** |
| qwen3.6-35b-a3b | **K (v3)** | default | **0.94** [0.90, 0.96] | 1,208 | 6.4 | **0.84** |
| qwen3.6-35b-a3b | KO | off / default | 0.96 / 0.99 | 257 / 873 | 0.4 / 4.3 | 0.04 / 0.62 |
| claude-sonnet-5.5 | P (raw log) | default | **0.94** [0.91, 0.96] | 1,417 | 2.9 | **4.37** |
| claude-sonnet-5.5 | K (v3) | default | 0.98 [0.96, 0.99] | 506 | 1.3 | 1.76 |

- Per task, qwen K off / default against qwen P: PickXtimes 0.78 / 0.91 vs 0.72 / 0.99; SwingXtimes 0.71 / 0.88 vs 0.42 / 0.43; BinFill 0.96 / 1.00 vs 0.67 / 0.99. Sonnet P: 1.00 / 0.84 / 0.96.
- Paired (rule fixed before the run):
  - qwen P − K: off −0.22 [−0.28, −0.15], **worse**; default −0.12 [−0.17, −0.07], **worse**;
  - qwen KO − K: off +0.14, default +0.06, **better**;
  - Sonnet P − Sonnet K: −0.04 [−0.07, −0.02], **worse**;
  - **Sonnet P − qwen K (default): +0.003 [−0.03, +0.04], same**;
  - Sonnet P − qwen K (off): +0.11 [+0.07, +0.16], better.

Reading (offline single-step decisions; RoboMME robot state; one small and one frontier model; no closed loop):
1. **v2's gain holds on unseen episodes.** With the KB the small model beats the raw log by 22 pp without thinking and 12 pp with thinking, at a third (off) or less than half (default) of the tokens.
2. **The small model with the KB matches the frontier model with the raw log** (0.94 vs 0.94, paired difference +0.003, CI ±3 pp), at about **one fifth of the cost per call** ($0.84 vs $4.37 per 1,000 calls). It is slower with thinking (6.4 s vs 2.9 s). Without thinking it is about **90× cheaper** and faster (0.5 s, $0.05 / 1,000), but 11 pp below. **This is claim 1 in miniature, on unseen RoboMME data.**
3. **The KB also helps the frontier model** (0.94 → 0.98, with a third of the tokens): structured memory is not only a crutch for small models (claim 2).
4. **The v3 view did not close the gap to the oracle step list** (KO − K: off +0.14, default +0.06; v2 on episodes 1–30: +0.13 / +0.08). The ordinals and merges did not change much. What remains is mostly SwingXtimes, where qwen K reaches 0.71 / 0.88 against KO's 0.92 / 0.97.
5. **Limits:**
   - the decisions are action choices, not waypoints; no closed loop;
   - three counting tasks only;
   - KO's wording carries ordinals;
   - one run each.

## Spec v1.9 — two VLM calls (2026-10-28, user decision)
The user asked why the parts keep growing. Answer: the VLM work is two calls, not three roles.
- **Keyframe call:** the clue-object lister and the annotator ask about the same marked frame, so they are one call. At the first frame it lists all movable objects; at later keyframes, new objects, descriptions, and the relations the rules cannot decide.
- **Review call:** unchanged (every ~3 s over a window of frames, plus once before acting on a target).
- The two differ in what the VLM reads (one frame when something happened vs a window on a timer), so they stay apart.
- Spec updated: header, §2.1 diagram and status, §2.2 step 6, §2.3 roles, §2.5 keyframe work, §2.6, §2.7, §3.7 (the output JSON gains `clue_objects`), M4 items 5, 6, 8, 10, M5 item 1.
- **Code merged (same day):** `arbiter_kb/keyframe.py` replaces `clue_objects.py` and `annotate.py`.
  - One prompt, one call (`keyframe_call`); the answer has `clue_objects`, `relations` and `objects`, each item checked alone.
  - The image is always shown; relations and descriptions cite the keyframe (`o<t>:image`).
  - At the first frame no object has an id, so only the clue list comes back; a new object is described at the next keyframe.
  - Tests: `tests/test_keyframe.py` (the clue tests moved there from test_task_items). The merged prompt has not been called on the real model; the 4-frame benchmark was of the lister prompt alone.

## Keyframe call check: the merged prompt lists worse (2026-10-28)
`scripts/keyframe_check.py` (`results/latest/keyframe_check/summary.json`; qwen3.5-9b via `vlm_client`, thinking off; 8 calls, **$0.0014**, about 4 s per call). The same 4 real frames and scoring as the lister benchmark. Goal text: "pick up the container that hides the green cube". Two memories: none (as at the first frame), and a synthetic KB with 7 objects whose ids do not match the picture. Exploratory: no rule was fixed before the run, and the frames are the ones the lister prompt was written on.

| memory | f21 (3 cubes) | f56 (3 containers) | f105 (3 containers) | f325 (3 containers + green cube) | right |
|---|---|---|---|---|---|
| lister prompt alone (earlier, NOTES "VLM choice") | right | right | right | right | **4 / 4** |
| merged, no memory | + a container | 1 container + 3 cubes (4) | 1 container + 2 cubes (3) | right | 2 / 4 |
| merged, 7-object memory | + a container | + a green cube | 4 containers + a green cube | 4 containers | **0 / 4** |

- **The merged call reads the text, not only the image.**
  - With no memory, f21 gains a container: likely primed by the goal's "container".
  - With the memory, a green cube appears in every frame (the goal and the memory name one), and the container count follows the memory, not the picture.
- **It also annotates what it cannot see:**
  - with no memory it described objects under made-up ids "1"–"4" (all dropped by the checks);
  - with the memory it wrote 10–12 relations and descriptions per frame about ids that are not in the picture. All passed the checks, which test ids, cites and format, not truth.
- **Reading:** on these 4 frames, putting listing into the annotator's prompt costs accuracy. Listing needs the image alone; annotating needs the memory. The memory-mismatch condition is harsher than a real keyframe (where ids are drawn on the picture), but it shows the model trusts the memory text over the image.
- **Options (user decision):**
  1. keep one keyframe moment, but two calls: a listing call that sees only the image (the 4 / 4 prompt), then the annotation call with the memory. The keyframe module stays one file.
  2. one call, with the prompt changed so the list comes from the image only. That is prompt tuning on the same 4 frames, and it needs a larger set with truth to judge.

## Spec v1.10 — listing from the image alone; the review call annotates (2026-10-28, user decision)
After the keyframe check above, the user chose to split the call again and to give the annotator's work to the reviewer: it watches the window, writes the summary, gives relations for the objects, and checks the KB.
- **`keyframe.py`:** listing only, from the frame and the instruction, no memory text (`list_objects`, the earlier 4 / 4 prompt).
  - Re-checked on the same 4 real frames and the same goal: **4 / 4** (`results/latest/keyframe_check_v110/summary.json`; 4 calls, $0.00014, about 1.1 s per call; run from an uncommitted tree, so `dirty` is true).
  - The f21 container that the merged prompt invented is gone, so it came from the memory text, not the goal.
- **`review.py` (new):** one call over a window of frames (ids drawn), with the KB's objects and the window's events.
  - Returns checks, missing events, relations, descriptions, identity and a summary.
  - Every item names the frames it rests on and is checked alone: ids exist, the event is in the window, the frames were shown, no numbers in the text.
  - The prompt says to judge from the frames only, because the merged prompt copied from the text.
  - Written as evidence (`vlm_check`, `vlm_missing`, `vlm_identity`, `vlm_summary`) and beliefs (relations, descriptions, p from the strength table), each supported by the frames named. The event log shows contradictions, missing events, identity changes and summaries (`views._vlm_line`); confirmations are not shown.
  - Tests: `tests/test_review.py` (fake model).
  - **Not yet:** a real call on video windows (needs frames with tracker ids drawn: G's replays on the pod, or the RoboMME videos with a tracker); retraction through the support chain on a contradiction.
- **Risk to measure first:** the review call also has the memory text in its prompt, which is what made the merged keyframe call copy from text. Its truth check (M4 item 10) must count events it confirms that did not happen, and relations it writes that the frames do not show.
- Spec updated to v1.10: header, §2.1, §2.2 steps 6 and 6b, §2.3, §2.5, §2.7, §3.7, M4 items 5, 8, 10.

## Review call on real video: the test, fixed before running (2026-10-28)
- **Question:** does the review call check the KB's events from the frames, or does it confirm whatever the memory text says? This is the risk the v1.9 keyframe check exposed.
- **Data:** the RoboMME front videos already on disk (files 018–023): VideoRepick and VideoPlaceOrder, the first 10 episodes of each that have video.
- **KB:** built from robot state only (`ArbiterKBSystem` on `sources/lerobot`), so its events are the semantic actions: pick-ups and releases. No tracker runs, so no ids are drawn, and **relations and descriptions are not tested here**.
- **Windows:** up to 3 actions per episode (chosen with seed 0). Each window holds 7 frames, at t − 12 … t + 12 in steps of 4 (2.4 s at 10 fps).
- **One planted false event per window**, written into that window's copy of the KB:
  - a release while the true event is a pick-up, or a pick-up while it is a release;
  - at t + 8, so the frames after the true event show whether it happened.
- **Model:** qwen3.5-9b via `vlm_client`, thinking off; one call per window; cap $0.50.
- **Scored with Wilson 95% CIs:**
  - **true events:** confirm / contradict / cannot tell / not answered;
  - **planted events:** the same; a "confirm" here is a **false confirmation**.
- **Decision rule:** usable as a checker if the planted false-confirmation rate is ≤ 0.10 **and** the true-event confirmation rate is ≥ 0.80. Otherwise not usable as it stands.
- **Also reported, not scored:**
  - the number of "missing" items and of dropped items (problems);
  - ten summaries, read by hand against the dataset's subgoal labels.
- **Caveats:**
  - the true events come from the action detectors, which were checked against labels on the counting tasks only (precision 0.99 for pick-up and release), not on these two tasks;
  - one planted kind per window.

**Results** (`results/latest/review_video/summary.json`; log `results/latest/logs/review_video.log`; 60 windows in 20 episodes, 420 images, **$0.012**, about 5.9 s per call).
- **Setup detail fixed before the run** (in `review.window_lines`): the window lists semantic actions only, without grip / pause lines or hand coordinates when no objects are tracked, as in the controller view. Otherwise the planted event could be refuted from the text alone.

| events | confirm [Wilson 95%] | contradict | cannot tell | not answered |
|---|---|---|---|---|
| true (60) | **0.58** [0.46, 0.70] | 0.07 | 0.05 | 0.30 |
| planted false (60) | **0.68** [0.56, 0.79] (false confirmation) | 0.18 | 0.05 | 0.08 |

- **Rule: fails** (false confirmation 0.68, against ≤ 0.10; true confirmation 0.58, against ≥ 0.80).
- **The model does not tell true events from false ones:** it confirms the planted events at least as often as the true ones. It mostly agrees with the memory text, as the merged keyframe prompt did.
- **The summaries echo the text.** Example: "The robot picked up a red cube and then placed it back down" for a window whose release was planted.
- **Dropped items: 221.** Mostly relations and descriptions about invented ids ("blue block", "red_cube_1"), since no objects are tracked. The checks dropped them all, as designed.
- **Caveats:**
  - 256-px frames, with the cube small and often under the gripper. A pick-up and a release can look alike at this size, so the task may be hard for any model; there is no reference model yet.
  - The true events are the detectors', unchecked on these two tasks.
  - It is the prompt as written, with no tuning.
- **Reading:** as it stands, qwen3.5-9b cannot be the KB's checker. A "confirm" from it carries no information, so writing its verdicts into the KB would add noise, not correction.
- **Before changing the design:** run the same 60 windows with a frontier VLM (one call per window). If that model separates true from planted events, the problem is the small model. If it does not, the problem is the frames (size, occlusion), and the review needs the wrist camera or larger crops.

**Reference run, fixed before it runs:** the same 60 windows, prompt, planted events, scoring and rule, with `anthropic/claude-sonnet-5.5` at its default settings (OpenRouter routing; cap $2). If it passes, the small model is the problem; if it fails as well, the frames are.

**Reference results, Sonnet 5.5** (`results/latest/review_video_sonnet/summary.json`; log `results/latest/logs/review_video_sonnet.log`; 60 calls, **$0.27** in all, $0.11 of it in a first attempt that a network timeout stopped, then replayed from the cache):

| events | confirm | contradict | cannot tell | not answered |
|---|---|---|---|---|
| true (60) | 0.05 | 0.02 | **0.67** [0.54, 0.77] | 0.27 |
| planted false (60) | 0.15 [0.08, 0.26] (false confirmation) | 0.05 | **0.80** [0.68, 0.88] | 0.00 |

- **Rule: fails as well** (true confirmation 0.05).
- **Sonnet fails in the honest way:** it mostly answers "cannot tell", and its summaries say why, for example "the frames do not clearly show any cube being lifted or released". qwen3.5-9b confirmed both kinds of event instead.
- **So the frames are the problem, not only the model.** At 256 px from the front camera, the cube is small and under the gripper, so a pick-up and a release look alike.
- **Also learned:** when the frames cannot decide, the small model agrees with the text, while the frontier model abstains. Any checker needs a "cannot tell" that is measured, not assumed.
- **Next, to decide:** the dataset also has a **wrist camera** video (`videos/wrist_image`, not downloaded), where a held cube fills the view. The same test on wrist frames would show whether the review can work at all. Alternatives: larger crops around the gripper from the front frame; or limit the review to what the front view can show (objects moving, appearing, disappearing).

## What can a small VLM see in a window, with no memory text? The test, fixed before running (2026-10-28)
User question: the review call failed because the memory text was in the prompt and the frames were hard to read. Can a small VLM, given only frames, say what happened and give relations?
- **Windows:** the same 60 (VideoRepick / VideoPlaceOrder episodes 1–10 of each, 7 frames from t − 12 to t + 12).
- **No memory text.** The prompt holds only the frames, each titled with its frame number, and the question.
- **Overlay:** no tracker runs on these videos, so object ids cannot be drawn. The only overlay is the **hand**, a mark at the gripper.
  - Its pixel comes from the robot state (forward kinematics) through a camera homography.
  - The homography is fitted on other episodes (11–30 of each task): hand xy at gripper closings against the grounded subgoal pixels.
  - It is camera calibration, a constant of the setup, not information about the episode.
  - It is exact only near table height.
- **Answer (JSON):**
  - `happened`: one of `pick_up`, `put_down`, `both`, `moving_only`, `nothing`, `cannot_tell`;
  - a summary of up to 25 words;
  - relations between objects named in words, since there are no ids (for example "red cube" `on` "table").
- **Truth for `happened`:** the actions the robot-state detectors found in the window. Pick-up only → `pick_up`; release only → `put_down`; both → `both`.
- **Conditions:**
  - **A:** the 7 full frames with the hand mark;
  - **B:** the 7 frames cropped to 96 × 96 px around the hand and enlarged to 256 px.
  - Each with qwen3.5-9b (thinking off) and anthropic/claude-sonnet-5.5 (reference). Cap $2.
- **Scored:** accuracy of `happened` with Wilson 95% CIs, the `cannot_tell` rate, and the confusions.
  - **Rule:** the small model can report what happened if its accuracy is ≥ 0.80 in A or B.
  - Relations and summaries are not scored automatically (no object truth here); 10 are read by hand.

**Results** (`results/latest/describe_{qwen,sonnet}_{A,B}/summary.json`; logs `results/latest/logs/describe_*.log`; 240 calls, **$0.47** in all, qwen $0.012). The homography fit 194 points with a 0.6 px median residual. The dataset's labels turned out to be `<row, column>`; this was seen on the first overlay image and fixed before any call.

Truth: 27 windows pick_up, 33 put_down (always answering pick_up scores 0.45).

| model | condition | `happened` right [Wilson 95%] | most common answer |
|---|---|---|---|
| qwen3.5-9b | A full frames | 0.08 [0.04, 0.18] | "nothing" (45 / 60) |
| qwen3.5-9b | B crops at the hand | 0.40 [0.29, 0.53] | "pick_up" (50 / 60) |
| Sonnet 5.5 | A full frames | 0.02 [0.00, 0.09] | "moving_only" (52 / 60) |
| Sonnet 5.5 | B crops at the hand | 0.12 [0.06, 0.22] | "moving_only" (35 / 60) |

- **Rule: fails** in every condition, the reference included. The crops help (qwen 0.08 → 0.40, Sonnet 0.02 → 0.12), but qwen B is no better than always answering pick_up.
- **Why, from the hand heights:** the windows end 1.2 s after the gripper closes or opens. At that moment the hand is still 1–4 cm above the table; it rises to about 15 cm only by about t + 30.
  - So the outcome that tells the two apart (the cube rising with the gripper, or staying behind) is not in the frames.
  - A put-down looks like a pick-up in reverse: the hand comes down holding the cube, which reads as "comes down and grasps".
  - **The window was badly chosen (my design), not only the models.** Sonnet's summaries describe what the frames show: "moves down toward a blue ball … not clearly lifted".
- **Summaries and relations:**
  - Sonnet B's are accurate about what is visible: "blue object held_by gripper", "red cube on wooden table", "gripper near blue ball".
  - qwen B's are short, and the same in every window ("gripper held_by blue object", with the relation's direction reversed; "grey object on the floor").
- **Not yet shown:** whether a window that covers the outcome (the lift, until about t + 30) lets either model decide. That is a new test, with its rule to be fixed before it runs.

**Longer windows (v2): the test, fixed before running.** Everything is as above except:
- **Window:** 9 frames from t − 10 to t + 30 in steps of 5 (4 s), so the frames include the lift after a pick-up or a put-down.
- **Windows:** the same events; any whose t + 30 runs past the episode is left out, and n is reported.
- **Frames** go to `data/robomme_lerobot/review_frames_long/`; the v1 frames stay as they were.
- **Truth:** the robot-state actions inside the longer window, so `both` can now occur.
- **Prompt:** the same, except "about 4 seconds".
- **Conditions:** A and B, each with qwen3.5-9b and Sonnet 5.5. Cap $2.
- **Rule:** the same; accuracy ≥ 0.80.

**v2 results** (`results/latest/describe_long_{qwen,sonnet}_{A,B}/summary.json`; logs `results/latest/logs/describe_long_*.log`; 56 windows; truth 27 pick_up, 29 put_down, no `both`; 224 calls, **$0.54**, qwen $0.012):

| model | condition | `happened` right [Wilson 95%] | v1 (2.4 s) | pattern |
|---|---|---|---|---|
| qwen3.5-9b | A full frames | 0.43 [0.31, 0.56] | 0.08 | pick_up on 34 / 56 |
| qwen3.5-9b | B crops | 0.46 [0.34, 0.59] | 0.40 | pick_up on 46 / 56; put-downs called pick_up 20 / 29 |
| Sonnet 5.5 | A full frames | 0.46 [0.34, 0.59] | 0.02 | put-downs called moving_only 13 / 29 |
| Sonnet 5.5 | B crops | 0.52 [0.39, 0.64] | 0.12 | put-downs called moving_only 11 / 29 |

- **Rule: fails** in every condition. Always answering put_down would score 0.52.
- **qwen still answers "pick_up" almost regardless.** Its summary is the same sentence each time ("closes around a blue object and lifts it"), and its relation is often reversed ("gripper held_by red object").
- **Sonnet now sees what happens, but does not name it as the event.** For put-downs it writes "the white gripper … pulls back and up. The ball stays on the wooden table and is not lifted", which is a correct description of a release, yet it answers "moving_only". Its last-frame relations are mostly right: "blue cube held_by gripper" after pick-ups, "blue ball on wooden table" after put-downs.
- **Reading:**
  1. Naming the robot's own action from the front video is hard even for the frontier model (≤ 0.52), while robot state gives it at 0.99. **The VLM should not judge robot actions**: they come from robot state.
  2. What a VLM can add is **state at a frame**: what is held, what is on what. Sonnet's relations are mostly right here; qwen3.5-9b's are not (one stereotyped, often reversed, relation). This is judged by eye, with no object truth here.
  3. For relations, the small VLM chosen for listing is not good enough on this evidence. A relation test with truth (G's replays with ids drawn, on the pod) should compare a larger VLM, as well as qwen3.5-9b, with Sonnet.

## Does a richer KB text make the facts easier to extract? The test, fixed before running (2026-10-28)
User question: can the KB give the LLM richer, more semantic context, so that it extracts what an instruction needs more easily?
- **Arms:** the same KB and the same episodes, different text:
  - **R:** the raw tracker log (every 10th frame);
  - **K:** the current controller view;
  - **KR (new, `narrative.py`):** the same KB written in words. Per object: where it is in words (the robot's left / right, near / far, order left to right and near to far), what each container probably holds, how often it moved and with what, and a timeline with names.
- **Data:** synthetic VideoUnmaskSwap train seeds 100–129 (30 episodes), lane and steps "mix" as in Batch A; clean, oracle observations. The robot is at (−0.615, 0); its left is +y.
- **Questions:** 8 per episode, all in one call, answered as JSON. Truth comes from the generator (evaluation only).
  1. Q1 (×3): which container hides the red / green / blue cube now?
  2. Q2: how many times did container X move? (X chosen at random, seed 0)
  3. Q3: which container is now farthest to the robot's left?
  4. Q4: which two containers took part in the most recent exchange of places?
  5. Q5: which cube is hidden in the container nearest the robot now? ("none" if it holds none)
  6. Q6: has the container that hides the <target> cube moved since it covered the cube? (yes / no)
- **Bias risk:** I wrote the KR text knowing these question types. Q5 is not stated directly in KR; it needs two facts combined.
- **Model:** qwen/qwen3.6-35b-a3b (DeepInfra), effort off and default; cap $2.
- **Score:** accuracy per question type and overall, Wilson CIs. Paired bootstrap over episodes of the overall per-episode accuracy (KR − R, KR − K). Rule as before: better / worse = the CI excludes 0; same = the CI within ±10 pp.
- **Question:** is KR better than R and K, at each effort?

**Results** (`results/latest/extract_kb/summary.json`; log `results/latest/logs/extract_kb.log`; 180 calls, **$0.24**, no empty answers). Accuracy over 240 answers per arm and effort (30 episodes × 8 questions):

| arm | effort | all | Q1 where is each cube (×3) | Q2 move count | Q3 farthest left | Q4 last exchange | Q5 cube in nearest container | Q6 moved since covered | prompt tokens (chars / 4) |
|---|---|---|---|---|---|---|---|---|---|
| R raw log | off | 0.40 | 0.33–0.40 | 0.10 | 0.57 | 0.77 | 0.07 | 0.63 | ~790 |
| R raw log | default | 0.53 | 0.43–0.50 | 0.13 | 0.80 | 0.90 | 0.17 | 0.80 | |
| K view | off | 0.66 | 0.57–0.73 | 0.67 | 0.83 | 1.00 | 0.07 | 0.77 | ~640 |
| K view | default | 0.86 | 0.97–1.00 | 0.77 | 0.90 | 0.97 | 0.57 | 0.73 | |
| **KR words** | off | **0.91** | 0.93–0.97 | 0.80 | 0.87 | 1.00 | **0.93** | 0.80 | **~530** |
| **KR words** | default | **0.93** | 0.97 | 0.80 | 1.00 | 1.00 | 0.93 | 0.80 | |

- **Paired over episodes** (rule fixed before the run):
  - KR − K: off **+0.25** [+0.18, +0.32], default **+0.07** [+0.03, +0.11]; both **better**.
  - KR − R: off +0.51 [+0.45, +0.57], default +0.40 [+0.33, +0.49]; both better.
  - K − R: off +0.26, default +0.34; both better.
- **Reading:**
  1. **Writing the same KB in words helps, most without thinking.** KR without thinking (0.91) beats K with thinking (0.86), with fewer input tokens (~530 against ~640) and no reasoning tokens. This mirrors counting v3: phrasing, not new information.
  2. **The combined question shows it most.** Q5 (the cube in the container nearest the robot) is not stated in any text and needs two facts joined. K gets 0.07 without thinking and 0.57 with it; KR gets 0.93 at both.
  3. **The ceilings are the KB's, not the text's:**
     - Q2 tops out at 0.80 because the motion detector merges back-to-back exchanges of the same container into one move, so the KB undercounts.
     - Q6 tops out at 0.80 because the truth is "ended more than 3 cm from where it covered the cube", while a container that went away and came back did move.
     - Both are limits of what the KB knows, which no wording fixes.
- **Caveats:**
  - Clean synthetic data with oracle observations, one task family, one model.
  - I wrote KR knowing the question types; only Q5 was held out by design.
  - The next check should use questions written by someone else, or taken from RoboMME instructions, on perceived data.

## Counting with the KB in words: the test, fixed before running (2026-10-28)
- **Arm KR:** the counting KB (robot state only) written in words (`narrative.robot_narrative`). It holds:
  - what the gripper holds now;
  - the last action;
  - the counts so far per kind (pick-ups, put-downs by how, presses, holds per side);
  - the numbers the goal mentions, as written ("two times" → 2), with no task-specific reading of them;
  - the same action timeline as K (ordinals, merged holds).
- **Against K:** the v3 view, the same prompt skeleton, the same decision points.
- **Data:** RoboMME PickXtimes / SwingXtimes / BinFill, episodes 61–100 (as v3; 310 decision points). The v3 K rows come from the cache, so only KR costs anything.
- **Model:** qwen/qwen3.6-35b-a3b on DeepInfra, effort off and default; cap $2.
- **Rule** (as in v1–v3): right-decision rate with Wilson CIs; KR − K paired over decision points. Better / worse = the CI excludes 0; same = the CI within ±10 pp.
- **Caveat:** episodes 61–100 were read in v3 (results only, not used for tuning), and the wording is mine. KO (0.96 off / 0.99 default) remains the ceiling.

**Results** (`results/latest/count_robot_state_words/`: report.json, rows.csv, meta.json; log `results/latest/logs/count_words.log`; 1,240 calls, 620 of them K from the v3 cache, which reproduced v3's K exactly; **$0.25** new). Scored as in v1–v3 (SwingXtimes wrap-up accepts put-down or press).

| arm | effort | right [Wilson 95%] | PickXtimes | SwingXtimes | BinFill | input tokens | median s |
|---|---|---|---|---|---|---|---|
| K (v3 view) | off | **0.83** [0.78, 0.86] | 0.78 | 0.71 | 0.96 | 354 | 0.5 |
| KR (words) | off | 0.74 [0.68, 0.78] | 0.70 | 0.57 | 0.91 | 449 | 0.7 |
| K (v3 view) | default | **0.94** [0.90, 0.96] | 0.91 | 0.88 | 1.00 | 352 | 6.4 |
| KR (words) | default | 0.91 [0.87, 0.94] | 0.86 | 0.86 | 1.00 | 447 | 6.6 |

- **Paired:** KR − K: off **−0.09** [−0.13, −0.06], **worse**; default −0.03 [−0.05, −0.003], **worse** (the CI excludes 0 but lies within ±10 pp).
- **Where KR loses** (KR wrong, K right, no thinking): 31 decisions, all at a **wrap-up**.
  - SwingXtimes: "move right" instead of "put down", 13;
  - PickXtimes: "pick up" instead of "press", 9;
  - BinFill: "pick up" instead of "press", 9.
  - The model continues when it should stop, even when the counts line says, for example, "picked up an object 4 time(s)" and the goal says four.
- **A likely cause, from reading the prompts:** the "Numbers the goal mentions" line.
  - In BinFill "two red cubes and three green cubes" becomes "two (2), three (3)", which can anchor on 3 rather than 5.
  - The long "So far" sentence with zeros and transit holds ("held an object up on the left 4 times" in PickXtimes) adds counts that do not matter. This was not tested separately.
- **Reading:**
  1. **Words are not better by themselves.** In the synthetic questions, KR won by stating joined facts (places, contents, exchanges) that K left to the reader. Here, K already gave what the decision needs (ordinal action lines and a progress line), and KR added counts and numbers that are irrelevant or misleading. A non-thinking model followed them.
  2. **What to keep:** facts the reader would otherwise have to join, written once and only when they bear on a decision. Not every count in words.
  3. **Not tuned after the result:** the wording stays as run. Any change (drop the numbers line, drop zero counts) is a new test, with its rule fixed first.

**Trimmed words (arm KT): the test, fixed before running.**
- **KT** = KR without the "Numbers the goal mentions" line, with the "So far" sentence listing only the kinds that happened (no zeros, no put-down breakdown by how); everything else as KR.
- **Same** episodes (61–100), decision points, model, efforts and scoring; K from the cache. Cap $1.
- **Rule:** KT − K paired, as before. The question is whether the two removed parts caused KR's loss: if KT ≈ K or better, they did.

**Trimmed results** (`results/latest/count_robot_state_words_trim/`: report.json, rows.csv, meta.json; log `results/latest/logs/count_words_trim.log`; 620 new calls, **$0.25**; K again from the cache, identical to v3):

| arm | effort | right [Wilson 95%] | PickXtimes | SwingXtimes | BinFill | input tokens |
|---|---|---|---|---|---|---|
| K (v3 view) | off | **0.83** [0.78, 0.86] | 0.78 | 0.71 | 0.96 | 354 |
| KR (words) | off | 0.74 | 0.70 | 0.57 | 0.91 | 449 |
| KT (trimmed) | off | 0.70 [0.64, 0.75] | 0.67 | 0.57 | 0.83 | 398 |
| K (v3 view) | default | **0.94** [0.90, 0.96] | 0.91 | 0.88 | 1.00 | 352 |
| KR (words) | default | 0.91 | 0.86 | 0.86 | 1.00 | 447 |
| KT (trimmed) | default | 0.91 [0.88, 0.94] | 0.87 | 0.86 | 1.00 | 396 |

- **Paired:** KT − K: off **−0.13** [−0.17, −0.09], **worse**; default −0.02 [−0.04, −0.01], **worse** (within ±10 pp).
- **The losses are again all wrap-ups:** without thinking, 43 decisions lost against 3 won.
  - BinFill "pick up" instead of "press", 18;
  - SwingXtimes "move right" instead of "put down", 13;
  - PickXtimes "pick up" instead of "press", 12.
- **So the goal-numbers line and the zero counts were not the cause:** removing them did not help, and without thinking it got slightly worse.
- **What is left in KR / KT and not in K:**
  - the sentences "Right now the gripper is empty" and "Last thing the robot did: placed it down …";
  - the absence of K's "Task items" line.
  - **Hypothesis, not tested:** the two "now" sentences prime the locally natural next step (empty gripper after a put-down → pick up again), and a non-thinking model follows that instead of checking the total against the goal.
- **Reading (combining the three wording tests):**
  1. **Writing facts out in words helps when it joins facts the reader would otherwise have to combine** (synthetic questions: KR 0.91 vs K 0.66 without thinking, and the held-out joined question 0.93 vs 0.07).
  2. **It hurts when it adds "current situation" sentences to a counting decision:** the model acts on the most recent state rather than the count. For counting, K's ordinal action lines plus one progress line remain the best format we have.
  3. **Rule of thumb for the view:** state the joined facts a decision needs (contents, places, counts against the goal when the goal is parsed); do not narrate the present moment.
- I stop varying the wording here. Any further variant (for example K plus the joined facts only) needs its rule fixed first.

## The merged view (K + joined facts): the test, fixed before running (2026-10-28)
- **Arm KH:** the v3 controller view (K) unchanged, plus a section "Joined facts (computed from the memory above)". It holds, from `narrative.py`:
  - the containers from the robot's left to its right, and from nearest to farthest;
  - per container: what it may hold, and how often it moved, with which container;
  - per hidden cube: its container(s) and how often each moved since the cover.
  - **No "now" sentences**, no goal numbers. With no objects in the KB (robot state only), the section is empty, so KH is K word for word.
- **Tests:**
  1. the synthetic questions (seeds 100–129, 8 questions; R, K, KR from the cache, KH new);
  2. counting on episodes 61–100. KH must equal K there; this is checked by comparing the prompts, and the run is then a cache replay.
- **Rule:** KH − K and KH − KR paired, as before.
- **Expectation, written before the run:** KH ≈ KR on the questions, and KH = K on counting.

**Results** (synthetic questions: `results/latest/extract_kb_merged/summary.json`, log `results/latest/logs/extract_kb_merged.log`; 240 calls, 180 of them R / K / KR from the cache, reproducing the earlier table; **$0.08** new):

| arm | effort | all | Q1 (×3) | Q2 | Q3 | Q4 | Q5 joined | Q6 | prompt tokens (chars / 4) |
|---|---|---|---|---|---|---|---|---|---|
| K | off | 0.66 | 0.57–0.73 | 0.67 | 0.83 | 1.00 | 0.07 | 0.77 | ~640 |
| KR | off | 0.91 | 0.93–0.97 | 0.80 | 0.87 | 1.00 | 0.93 | 0.80 | ~530 |
| **KH** | off | **0.91** | 0.90–0.97 | 0.80 | 0.93 | 1.00 | **0.97** | 0.80 | ~920 |
| K | default | 0.86 | 0.97–1.00 | 0.77 | 0.90 | 0.97 | 0.57 | 0.73 | |
| KR | default | **0.93** | 0.97 | 0.80 | 1.00 | 1.00 | 0.93 | 0.80 | |
| KH | default | 0.87 | 0.90–0.93 | 0.77 | 0.93 | 0.90 | 0.90 | 0.73 | |

- **Paired:**
  - KH − K: off **+0.25** [+0.19, +0.32], **better**; default +0.01 [−0.08, +0.08], **same**.
  - KH − KR: off 0.00 [−0.03, +0.04], **same**; default −0.06 [−0.14, +0.00], **undecided**.
- **Counting:** KH's prompts are identical to K's on all 310 decision points (checked), so KH = K = 0.83 / 0.94 there.
- **Reading:**
  1. **Without thinking, the merged view keeps both gains:** the same as KR on the questions (0.91, the joined question 0.97) and the same as K on counting (0.83).
  2. **With thinking, it gains nothing over K on the questions** (0.87 vs 0.86), and may trail KR (0.93). Q1 drops from 0.97–1.00 to 0.90–0.93.
     - A guess: the longer prompt with two descriptions of the same facts lets the thinking model second-guess one with the other.
     - KH is also the longest text (~920 tokens against ~640 / ~530).
  3. **No single format wins everywhere.**
     - For a fast non-thinking controller, the case claims 1–2 care about, KH is the best choice we have.
     - With thinking, the plain KR words are best on object scenes, and K on counting.
- Next options, each a new test: KH with the event log trimmed (to cut the duplicate facts and the length); or choose the format by whether the KB holds objects (KR words for object scenes, K for robot-only).

## A goal counter kept by the KB: the test, fixed before running (2026-10-28)
User idea: keep the count against the goal up to date in the KB, so the controller reads a ready number instead of working it out.
- **Goal parse (`goal_counter.py`):** once per episode, the controller's model reads only the instruction and returns
  `{"count": "pick_up|release|press|hold_left|hold_right", "target": <int>, "then": "<what comes after>"}`.
  - The spec is checked; a bad spec means no counter line.
  - qwen/qwen3.6-35b-a3b on DeepInfra, thinking default (one call per episode).
- **Counter:** every frame, the KB counts its own robot-state actions of that kind (`actions.py` evidence; holds by side).
- **Arm KC:** the v3 view (K) plus one line, `Goal progress: <kind> <n> of <target>`, with "— the repetitions are done" once n ≥ target. No "now" sentences.
- **Data and scoring:** RoboMME counting episodes 61–100 (no demo phase in them), as in v3; K from the cache. Effort off and default; cap $1.
- **Parse accuracy:** against the dataset's execution subgoals. The truth kind is release for PickXtimes / BinFill and hold_left for SwingXtimes; the truth target is the number of such subgoals. Reported per task.
- **Rule:** KC − K paired, as before. KO (0.96 / 0.99) is the ceiling.
- **Caveat:** if the parse is right, the counter does most of the decision's work. The score then measures parse + counter + reading, and the parse errors are reported apart.

**Results** (`results/latest/count_robot_state_counter/`: report.json with the parse breakdown, rows.csv, meta.json with every goal spec; log `results/latest/logs/count_counter.log`; K from the cache, identical to v3; **$0.17** new, goal parses included):

| arm | effort | right [Wilson 95%] | PickXtimes | SwingXtimes | BinFill |
|---|---|---|---|---|---|
| K (v3 view) | off | 0.83 [0.78, 0.86] | 0.78 | 0.71 | 0.96 |
| **KC (K + goal counter)** | off | **0.94** [0.91, 0.96] | 0.98 | 0.82 | 1.00 |
| K (v3 view) | default | 0.94 [0.90, 0.96] | 0.91 | 0.88 | 1.00 |
| KC (K + goal counter) | default | 0.95 [0.92, 0.97] | 0.97 | 0.87 | 0.99 |
| KO (oracle, v3) | off / default | 0.96 / 0.99 | | | |

- **Paired:**
  - KC − K: off **+0.11** [+0.08, +0.15], **better** (39 decisions won, 4 lost);
  - default +0.01 [−0.01, +0.04], **same** (10 won, 6 lost).
- **The counter gives the non-thinking controller what thinking gave it.** KC without thinking (0.94, about 0.5 s and no reasoning tokens) equals K with thinking (0.94, about 6 s), and is 2 pp under the oracle step list.
- **Goal parse accuracy:**
  - **By the pre-registered rule** (kind and target both exact): PickXtimes 5 / 40, SwingXtimes 8 / 40, BinFill 5 / 40. Low, mostly because the parser counted pick-ups where the rule expected releases.
  - **In practice,** at these decision points (just after a put-down) pick-ups and releases are equal, so the number is what matters:
    - PickXtimes: target right 39 / 40 (1 parse failed);
    - BinFill: target right 35 / 40 (5 failed);
    - **SwingXtimes: 21 / 40, with 16 failed parses** and kinds spread over hold_left / hold_right / pick_up / release / press.
  - The back-and-forth instruction is the hard one to parse. SwingXtimes is also where KC still loses most: "move right" instead of "put down", 13 times without thinking.
- **Reading:**
  1. **Pre-computing the count against the goal works.** It is the user's idea, and it is L4 of the spec: the largest gain of any view change without thinking (+11 pp), at no cost per decision beyond one parse per episode.
  2. What remains is **the parse**, not the counter or the reader. A wrong or missing spec falls back to K.
  3. The parser prompt should name the back-and-forth case, or count pairs, before the next run. That is a new test with its rule fixed first, and it should use other episodes than 61–100, which have now been read three times.
- **Caveat:** the counter only counts the robot's own actions (no colours, no objects), so it cannot count "two red, then three green" separately; it counts five put-downs.

## Goal parse v2: the test, fixed before running (2026-10-28)
- **Why the SwingXtimes parses failed:** the model wrote its reasoning, with a draft JSON, into the answer. `parse_answer` takes everything from the first `{` to the last `}`, which then spans two objects and fails. The parser itself was mostly right ("hold_right, 3").
- **Changes, all together:**
  1. the goal parse reads the **last** JSON object in the answer;
  2. the prompt adds two general rules: "if the instruction gives no number, the target is 1", and "a motion repeated back and forth N times means N holds on each side".
- **Parse accuracy, old against new parser:** all 100 episodes of each counting task. Scored on the target number (the kind may be any action whose count equals the target's at the decision points: pick_up or release in PickXtimes / BinFill, hold_left or hold_right in SwingXtimes); a missing spec is wrong.
- **Decisions:** episodes **31–60** of each task. They were not used in any model test, but were among those whose subgoals set the action thresholds. K and KC, effort off and default, qwen on DeepInfra; cap $1.5.
- **Rule:** KC − K paired, as before; and new-parser target accuracy ≥ 0.95 on each task.

**Results.**

**Goal parse accuracy** (`results/latest/goal_parse/summary.json`; log `results/latest/logs/goal_parse.log`; 600 parses, v1 for episodes 61–100 from the cache; **$0.25** new):

| task | v1 (first run) | v2 (last JSON + two rules) |
|---|---|---|
| PickXtimes | 97 / 100 | **100 / 100** |
| BinFill | 83 / 100 | **100 / 100** |
| SwingXtimes | 35 / 100 | **77 / 100** |

- **Rule (≥ 0.95 on each task):** met for PickXtimes and BinFill, **not met for SwingXtimes**.
- **All 23 SwingXtimes errors are on goals with no number** ("… repeating this back and forth motion, finally press the button": once).
  - The target 1 is right, but the kind is wrong: pick_up 15, press 8.
  - With pick_up 1, the counter says "done" right after the first pick-up, which is misleading. With press 1, it never helps.

**Decisions with the counter** (`results/latest/count_robot_state_counter_v2/`: report.json, rows.csv, meta.json with the 90 specs, none missing; log `results/latest/logs/count_counter_v2.log`; episodes 31–60, 224 decision points per arm; 896 calls, **$0.29**):

| arm | effort | right [Wilson 95%] | PickXtimes | SwingXtimes | BinFill |
|---|---|---|---|---|---|
| K | off | 0.85 [0.80, 0.89] | 0.81 | 0.73 | 0.99 |
| **KC** | off | **0.95** [0.91, 0.97] | 0.99 | 0.84 | 1.00 |
| K | default | 0.91 [0.86, 0.94] | 0.87 | 0.83 | 1.00 |
| **KC** | default | **0.94** [0.90, 0.97] | 0.94 | 0.87 | 1.00 |

- **Paired:** KC − K: off **+0.10** [+0.06, +0.14], **better** (22 decisions won, 0 lost); default **+0.04** [+0.01, +0.07], **better** (10 won, 2 lost).
- **The counter result holds on episodes no model test had used,** now at both efforts. Without thinking, KC (0.95) beats K with thinking (0.91).
- **What is left is SwingXtimes:** without thinking, 10 decisions answer "move right" where the truth is "put the cube down" after the last cycle. These are mostly the no-number goals, where the spec's kind is wrong.
- **Next fix (a new test):** the parse rule "a back-and-forth with no number means one cycle: count the holds on the side it ends on". Or let the parser return the cycle as two kinds. It must be checked on goals with and without numbers.

## Goal parse v3: the test, fixed before running (2026-10-28)
- **One more general rule in the prompt:** "When the instruction moves an object back and forth between two places, count the holds, whether or not it gives a number." Nothing else changes.
- **Parse accuracy:** all 100 episodes of each task, v2 (from the cache) against v3, scored as in v2. **Rule:** v3 ≥ 0.95 on every task, with no task below v2.
- **Decisions:** episodes 31–60 again, arm KC with the v3 specs against K (both K and the KC prompts whose spec did not change come from the cache). Same scoring and paired rule.
- **Caveat:** episodes 31–60 have now been read once for the counter; the change targets the error seen there. Only the parse accuracy over all 300 goals is a fresh check.

**Results.**

**Goal parse v3** (`results/latest/goal_parse_v3/summary.json`; there the column "v2" is the current parser, i.e. v3; log `results/latest/logs/goal_parse_v3.log`; 94 new parses):

| task | v1 | v2 | v3 |
|---|---|---|---|
| PickXtimes | 97 | 100 | 99 |
| BinFill | 83 | 100 | 100 |
| SwingXtimes | 35 | 77 | **86** |

- **Rule: fails.** SwingXtimes 86 < 95, and PickXtimes is 1 below v2. That one episode's spec is now "press 1".
- **The remaining SwingXtimes errors (14) are all no-number goals:**
  - "press 1", 10: the model reads "no repetition" and counts the final press;
  - "pick_up 1", 4.

**Decisions with the v3 specs** (`results/latest/count_robot_state_counter_v3/`; log `results/latest/logs/count_counter_v3.log`; episodes 31–60; K and unchanged KC prompts from the cache; 154 new calls):

| arm | effort | right [Wilson 95%] | PickXtimes | SwingXtimes | BinFill |
|---|---|---|---|---|---|
| K | off | 0.85 [0.80, 0.89] | 0.81 | 0.73 | 0.99 |
| KC (v2 specs) | off | 0.95 | 0.99 | 0.84 | 1.00 |
| **KC (v3 specs)** | off | **0.97** [0.94, 0.99] | 0.99 | 0.92 | 1.00 |
| K | default | 0.91 [0.86, 0.94] | 0.87 | 0.83 | 1.00 |
| **KC (v3 specs)** | default | **0.96** [0.93, 0.98] | 0.96 | 0.92 | 1.00 |

- **Paired:** KC − K: off **+0.12** [+0.08, +0.17], **better** (28 won, 1 lost); default **+0.06** [+0.02, +0.09], **better** (15 won, 2 lost).
- **Reading:**
  - The fast non-thinking controller with the counter is at **0.97**, against the oracle step list's 0.96 / 0.98 on episodes 61–100 (KO was not run on 31–60). The counter closes most of the remaining gap.
  - The decisions improved even though the parse rule failed: the remaining parse errors sit on no-number SwingXtimes goals, where one cycle is short and K already does well.
  - **Caveat:** the v3 rule was written after seeing the v2 errors on these same episodes. The parse accuracy over all 300 goals (86–100) is the fresh check.
- **I stop tuning the parser here.** A further rule written for the no-number back-and-forth case would be fitting one phrasing of one task.

## Before the full rerun: code changes 1–5 (2026-10-29, user decision)
1. **`scripts/count_report.py`:** the counting reports come from a script now, not from code typed into the shell.
   - Inputs: run directories (rows*.csv + meta*.json).
   - Scoring: the SwingXtimes wrap-up rule.
   - Output: Wilson CIs per (model, arm, effort) and per task; pairs against `--ref` and any `--pair` across models (bootstrap, decisions won / lost, common errors); goal-parse accuracy when specs are in meta.
   - Checked: it reproduces the v3 table, the v3 Sonnet pairs (+0.003 [−0.03, +0.04]; +0.11 [+0.07, +0.16]) and the counter-v3 table exactly.
2. **The latest version is the default.**
   - `views.controller_view` is v4: the v3 view, plus the goal counter's line when `kb.task["counter"]` holds a parsed goal, plus the joined facts when the KB holds containers. `version="v3"` gives the old view.
   - `ArbiterKBSystem.reset(meta, counter=...)` stores the parsed goal.
   - **Arm names now:** counting K = latest (the counter runs called it KC), K3 = the v3 view (called K there). Questions K = latest (called KH), K3 = v3 (called K).
   - Checked: the new K prompts equal the old KC prompts on all 224 decision points of episodes 31–60, and the question test replays all 240 calls from the cache.
3. **Frozen prompts:** `params.json` now also holds `prompts_hash`, a sha256 over controller / views / action_view / narrative / goal_counter / keyframe / review. `load_params` refuses to run when it differs, so any edit to what a model reads needs `run.py --freeze --force`.
   - count_test's meta records its system prompt's sha and the prompts hash.
   - Re-frozen: params **4565b592715c**, prompts **ab198bbca95d**.
4. **`controller.parse_answer`** returns the last top-level JSON object (nested ones are skipped), fenced or not; the goal parse uses it too. Before, 5 of 2,756 recent controller answers were unreadable.
5. **Which data each test has read.** The final run must use data that the latest version has not been tuned on.
   - **Counting episodes, by range:**
     - 1–60: their subgoals set the action thresholds;
     - 1–30: v1 and v2 with the older views;
     - 31–60: the goal counter, whose parse rules were written after its errors there;
     - 61–100: v3, the two worded views and the first counter run;
     - all 300 goals: the parse accuracy checks, so every parse error has been seen.
   - **Proposed final set on the Mac: counting episodes 1–30,** where the latest view (v4 with the counter) has never run. Caveats: the thresholds came from them, and their goals were in the parse checks.
   - **Synthetic data:** the LLM tests used train seeds only (0–19, 100–149). The final permanence runs should read **test seeds** (≥ 1000) once, recorded in the E5c ledger.
   - **A clean set without these caveats** needs new RoboMME episodes generated on the pod.

## Motion detector v2 (item 6): the test, fixed before running (2026-10-29)
- **Real tracker jitter** (G's 180 perceived episodes, container slots, frame-to-frame xy, from t = 40):
  - still frames: median 0.0 mm, p90 7.5 mm, p99 29.7 mm; **26% exceed the current 1 mm threshold**;
  - frames inside a swap: median 4.7 mm; **35% are below 1 mm**.
  - So the one-frame speed test makes false starts and stops on real data. On synthetic data with 15 mm iid noise it already broke (M3 truth check).
- **Detector v2** (`detectors._motion`, causal):
  - the xy position is the median of the last `m_win` frames;
  - the speed is the smoothed displacement over `k_lag` frames, divided by `k_lag`;
  - a move starts after `n_on` frames above `v_start` and stops after `n_off` frames below `v_stop`;
  - while moving, a velocity turned by more than 120° from the move so far (cos < −0.5, once the move is over 2 cm) **splits** the move: back-to-back swaps.
- **Grid, chosen on training data only:** m_win ∈ {1, 3, 5, 7}, k_lag ∈ {1, 3}, v_start ∈ {0.001, 0.002, 0.003, 0.005} m/frame, v_stop = v_start / 2, n_on = 2, n_off ∈ {2, 4}.
  - m_win 1, k_lag 1, v_start 0.001, n_on 1 is the old detector; it is reported as the reference.
- **Score:**
  - G episodes: training = episodes 0–9 of each setting, test = 10–19.
  - Truth: the swap intervals in each episode's meta (two containers move in each). Identity is not needed.
  - A detected move segment is **matched** when it overlaps exactly one swap interval for at least half of its own length. It is **merged** when it overlaps two or more swaps, and **spurious** when it overlaps none.
  - recall = mean over swaps of min(matched segments in it, 2) / 2; precision = matched / all segments.
  - **Selection:** the highest F1 on G training. Synthetic clean (train seeds 0–99) must keep recall ≥ 0.95. Ties go to the smaller m_win.
  - The chosen setting is read **once** on G test, and on synthetic clean and 15 mm noise (train seeds).
- **Also reported, not used for selection:** near_pass recall against true passes (`swapstats.oracle_passes` on the oracle tracks, matched in time), and the old measure_kb numbers.
- **Counting is unaffected,** since it uses robot state only. The synthetic question and permanence views change, which the full rerun will cover.
- **Change after the first grid** (training data only; G test not read):
  - **v1 itself fails the synthetic floor:** clean synthetic recall 0.79, with 46 merged segments, so "recall ≥ 0.95" could not be met. The floor is now min(0.95, v1's recall).
  - **The reversal split catches few back-to-back swaps,** because a container often goes on to a third place. Every scripted swap slows almost to zero at its end, so a **speed-valley split** was added: below `valley` × the move's peak, then back above twice that.
  - The grid adds valley ∈ {0, 0.2, 0.35, 0.5}. The selection rule is otherwise unchanged.

**Results** (`results/latest/motion_v2/`: grid.json with all 256 training settings, read.json with the one test read):
- **Chosen on training:** m_win 1 (no median), k_lag 1, v_start 0.003 m/frame (v_stop 0.0015), n_on 2, n_off 2, valley 0.35, rev_cos −0.5.
  - The settings with a median filter scored higher on G training (F1 up to 0.86), but merged more back-to-back swaps on clean synthetic data (recall 0.45–0.66, below the floor of 0.79), because smoothing fills the speed valleys.
  - Re-frozen: params **49a8577dc73f**.

| data | detector | recall | precision | F1 | segments | merged | spurious | pass recall |
|---|---|---|---|---|---|---|---|---|
| G test (episodes 10–19), real tracker | v1 | 0.955 | 0.525 | 0.678 | 2,564 | 141 | 1,029 | **0.932** |
| G test | **v2** | **0.989** | **0.687** | **0.811** | 2,691 | 194 | **590** | 0.889 |
| synthetic clean (train 0–99) | v1 | 0.788 | 0.879 | 0.831 | 380 | 46 | 0 | 0.849 |
| synthetic clean | v2 | 0.821 | 0.821 | 0.821 | 424 | 76 | 0 | 0.915 |
| synthetic, 15 mm iid noise | v1 | 0.068 | 0.108 | 0.084 | 399 | 265 | 19 | 0.113 |
| synthetic, 15 mm iid noise | v2 | 1.000 | 0.570 | 0.726 | 6,101 | 472 | 1,726 | **0.000** |

- **On the real tracker (G test), v2 is clearly better:** spurious moves 1,029 → 590 (−43%), F1 0.68 → 0.81. The close-pass recall drops slightly, 0.93 → 0.89.
- **On clean synthetic data it is about the same:** more back-to-back swaps are split, but the extra splits also cut some single swaps.
- **Under 15 mm iid noise both fail, in opposite ways:**
  - v1 never stops a move;
  - v2 fragments every move into many short ones (6,101 segments), and with containers rarely "both moving" for long enough, **no close pass is detected**.
  - iid noise of 15 mm on every frame is much harsher than G's tracker (median 0, p90 7.5 mm). It remains a known weakness; R6's near-pass evidence would need its own smoothing for such data.
- **Still open:** 590 spurious segments on G test (about 3 per episode). They come from the tracker's position jumps (p99 30 mm), which a one-frame-lag detector cannot tell from real starts.

## Spec v1.11 and item 7 (2026-10-29, user decision)
- **`review.py` removed, `ask.py` added:** one closed question about one object in one marked frame, no memory text. Two questions: occluder (robot arm / container / out of view / cannot tell) and appearance (≤ 12 words + colour).
  - Answers are checked; occluder → `vlm_occluder` evidence (shown in the event log), appearance → the description fact.
  - Tests: `tests/test_ask.py`. Not yet measured with truth (pod, M4 item 10).
- **Removed with it:** `tests/test_review.py`, `scripts/review_test.py`; its windows helpers moved to `scripts/video_windows.py`, which `describe_test.py` uses. Reproducing the review-call results needs commit 8d0787f.
- **The window summary**, if one is wanted, comes from the KB's event log by code, not from a VLM.
- **DEVELOPMENT.md is now v1.11** (header, §2.1 diagram and status, §2.2 step 6b, §2.3, §2.5, §2.6, §2.7, §3.7, M3 item 4, M4 items 5, 6, 10, 11). HANDOFF.md is rewritten for the current state.
- `prompts_hash` now covers `ask.py` instead of `review.py`; re-frozen: params **49a8577dc73f**, prompts **ba6ba13b8dbd**.

## Is the counting win ArbiterKB's, or the counter's? Two controls, fixed before running (2026-10-29, goal review)
The goal review asked: the counting win comes from `actions.py` (robot-state events) plus one counter line, while the KB's own parts (beliefs, relations, support, cites) are not used. Two controls never run before:
- **PC = raw log + the same counter line.** The raw robot log (arm P) with the "Goal progress: … n of N" line on top, computed from the same robot-state actions and the same goal spec.
- **RULE = no LLM per decision.** With the parsed spec:
  - if the KB's count of the spec's action is ≥ its target, answer the wrap-up ("press the button"; for SwingXtimes either wrap-up is scored right, as for every arm);
  - otherwise repeat: "move to the top of the right-side target" when the spec counts holds, else "pick up a cube";
  - with no spec, repeat.
  - The goal is still parsed once per episode by the LLM; only the per-decision call is gone.
- **Against:** K (v4 view + counter) and K3 (v3 view), the same decisions and specs (parse v3).
  - **Episodes 31–60:** K and K3 come from the cache.
  - **Episodes 61–100:** K3 and P come from the v3 cache; K and PC are new.
  - qwen3.6-35b-a3b, effort off and default; cap $1.5.
- **Readings fixed in advance:**
  - **PC − K "same" (CI within ±10 pp) or better:** the gain is the counter on preprocessed events, not the KB view.
  - **RULE ≥ 0.95:** on these tasks the per-decision LLM adds nothing once the goal is parsed, and claim 1 has no content here.
  - **Both readings are also made per task.** The overall number is reported but does not decide: a parse error in SwingXtimes can pull RULE's total below 0.95 while PickXtimes and BinFill are at ceiling. RULE ≥ 0.95 on a task means the per-decision LLM adds nothing on that task.
  - **If RULE ≥ 0.95 on PickXtimes and BinFill:** these tasks test counting, not reasoning over memory. Claims 1–2 move to tasks that need it (VideoPlaceOrder order, permanence after swaps), which need perception, so the pod closed-loop pilot moves up.
  - Either finding is reported as it is.
- **Also, free:** accuracy and input tokens split by the goal's target count (from the dataset's execution subgoals), for P, K3, K, PC, RULE and KO, from existing and new rows.

**Results** (`results/latest/count_controls/report_31_60.json`, `report_61_100.json`; runs `count_20261029_021320_…_70083` and `…_021321_…_70084`; qwen3.6-35b-a3b, parse v3 specs from the cache; $0.95 new, all decisions answered):

| arm / effort | 31–60 | 61–100 | Pick (61–100) | Swing (61–100) | BinFill (61–100) | median input tokens |
|---|---|---|---|---|---|---|
| RULE (no call) | **0.996** | **0.981** | 1.00 | 0.94 | 1.00 | 0 |
| K off / default | 0.973 / 0.964 | 0.965 / 0.952 | 1.00 / 0.98 | 0.88 / 0.88 | 1.00 / 0.98 | ~370 |
| PC off / default | 0.902 / 0.942 | 0.855 / 0.903 | 0.99 / 1.00 | 0.57 / 0.70 | 0.96 / 0.98 | ~1,290 |
| K3 off / default | 0.853 / 0.906 | 0.826 / 0.935 | 0.78 / 0.91 | 0.71 / 0.88 | 0.96 / 1.00 | ~350 |
| P off / default | — | 0.610 / 0.819 | 0.72 / 0.99 | 0.42 / 0.43 | 0.67 / 0.99 | ~1,280 |
| KO off / default | 0.951 / 0.973 | 0.965 / 0.990 | 1.00 / 1.00 | 0.92 / 0.97 | 0.97 / 1.00 | ~240 |

Paired against K (bootstrap 95%, decisions won / lost):
- RULE − K: 31–60 +0.022 [+0.004, +0.045] off, +0.031 [+0.009, +0.054] default; 61–100 +0.016 [+0.003, +0.029] off, +0.029 [+0.013, +0.048] default. **RULE never loses a decision to K** (0 lost in all four).
- PC − K: 31–60 −0.071 [−0.112, −0.031] off, −0.022 [−0.054, +0.009] default; 61–100 −0.110 [−0.145, −0.074] off, −0.048 [−0.081, −0.019] default.

**Readings, as fixed:**
- **RULE ≥ 0.95 on PickXtimes and BinFill (1.00 on both ranges), and on SwingXtimes too (0.98, 0.94 — just under on 61–100).** Once the goal is parsed, the per-decision LLM adds nothing on these tasks; it costs 2–3 pp. Claim 1 has no content on counting. By the rule fixed in advance, claims 1–2 move to tasks that need reasoning over memory, and the pod closed-loop pilot moves up.
- **PC − K:** "same" (CI within ±10 pp) with thinking on both ranges, although K is ahead on 61–100. Without thinking, K is better (lower bounds −11 and −14.5 pp). The gap is SwingXtimes (PC 0.57–0.79 vs K 0.88–0.92); on PickXtimes and BinFill PC = K. So most of the gain is the counter on robot-state events; the KB view adds on top only for the hold task without thinking.
- **Where K loses to RULE:** at target 1 (K 0.71–0.80, RULE 1.00): the model reads "1 of 1 — the repetitions are done" and still repeats.

**By target count** (free split): K's input stays at 315–425 tokens for targets 1–5; P / PC grow from ~880 to ~1,900. P without thinking is 0.51–0.78 at every target. With targets of only 1–5, the curve is short; a longer one needs new episodes.

## Does the KB-in-words gain survive real perception? G perceived episodes (2026-10-29; approved by the user before any call)
Goal: test whether the synthetic result (KR 0.91 vs raw log 0.40 without thinking, NOTES "Does a richer KB text …") holds on the real tracker, and whether the KB's beliefs help when the tracker swaps ids (claim 3).

**Check first, $0 (G train, episodes 0–9 of each setting, 270 cubes): the perceived cube tracks do not disappear when covered.**
- The oracle shows every cube hidden from t ≈ 31. Of the perceived cube slots, only 26 / 270 disappear after the cover; the rest stay "visible" to the end. The tracker's cube mask moves onto the container that covers it (perceived z ≈ 0.05 m, the container's top, against the cube's 0.017 m).
- Where the ghost track ends, it sits on the true container in stock 25 / 27, cam_left 24 / 24, cam_right 20 / 20, but only in 8–19 of 24–26 in the fast / close settings: the ghost swaps with the container tracks.
- **So on G's perception, R1 (cover on disappearance) almost never fires, and the KB thinks the cubes are on the table.** Run as designed, this test would measure the perception hand-over, not the KB.

**Proposed design (to approve before anything is built):**
1. **Occlusion step, chosen on G train only:** a cube slot counts as not visible when its centroid lies over a container (xy within r of a container track) and above the cube's height (z > z_cube + dz). r and dz from a small grid; the score is cover recall against the oracle's hide frames (±5 frames). **Gate: recall ≥ 0.90 on G train**, otherwise stop and report. This is a FrameObs-side fix (§3.11: the input side), not a KB change.
2. **Reference row (privileged):** the same episodes with the cubes' visibility from the oracle and everything else perceived. It isolates the container tracker's errors.
3. **Arms:** R (raw tracker log), K (v4 view), KR (words); qwen3.6-35b-a3b, effort off and default; G train only (90 episodes; G test stays unread for this question).
4. **Questions:** Q1–Q6 as in the synthetic test. **Q1 (which container hides each cube) is the main metric**, because it is the VideoUnmaskSwap instruction itself; Q2–Q6 are mine and secondary. Q1 also asks for a confidence 0–1 per cube.
5. **Truth:** from the oracle tracks, mapped to the perceived ids at the last frame (`swapstats.identities`: the slot sitting on the true container). A question whose true container has no slot on it at the end is dropped and counted.
6. **Readings fixed in advance** (paired bootstrap over episodes; better / worse = CI excludes 0; same = CI within ±10 pp):
   - KR − R and K − R on Q1, overall and split stock / stress: does the gain transfer to real perception?
   - **Claim 3 through the LLM:** on the cubes whose container took part in a tracker swap (`swapstats.episode_passes`), the AUROC of the stated Q1 confidence for correctness, per arm. K / KR better than R by CI → the KB's probabilities reach the controller usefully.
7. **Cost:** 90 × 3 × 2 = 540 calls plus the reference row, about $1; cap $2.
- **Approved by the user (2026-10-29); occlusion step chosen, before any call:**
  - grid r ∈ {0.03, 0.05, 0.08} m × dz ∈ {0.01, 0.02, 0.03} m on G train (270 cubes). A cube counts as covered right when its first hidden frame is within ±5 of the oracle's, it was visible 8 frames before, and it stays hidden in ≥ 90% of the frames from 5 after.
  - recall: dz 0.03 fails everywhere (0.32–0.33); dz 0.01 / 0.02 give 0.87 (r 0.03), 0.90 (r 0.05), 0.90–0.91 (r 0.08).
  - **Chosen: r 0.08, dz 0.01, recall 0.907 ≥ 0.90: the gate passes (narrowly).** `arbiter_kb/sources/perceived.py` (PerceivedSource, tests in `tests/test_perceived.py`), `scripts/perceived_qa.py`.
  - Truth coverage: 8.3 of 9 question keys per episode kept; Q1 kept for 241 / 270 cubes. Tracker-swapped Q1 cubes (the slot on the true container changed since the cover): stock 1 / 30, cam_left 2, cam_right 9, fast 7, fast_lane045 11, close 13, fast_close_3swaps 14, fast_lane03 16, fast_close 21 (of 30 each).

**Results** (`results/latest/perceived_qa/summary.json`; 1,080 calls, **$1.05** new, no empty answers, 12 / 1,446 Q1 answers without a confidence). Q1 = which container hides each cube (Wilson 95%):

| row / arm | Q1 off | Q1 default | Q1 stock | Q1 stress | Q1, tracker-swapped cubes | AUROC of Q1 confidence, swapped (off / default) | Q5 joined |
|---|---|---|---|---|---|---|---|
| perceived / R | 0.49 | 0.61 | 0.53–0.67 | 0.48–0.60 | 0.13–0.24 | 0.48 / 0.29 | 0.21 |
| perceived / **K** | **0.78** | **0.77** | 0.97 | 0.74–0.75 | 0.27 | 0.51 / 0.48 | 0.63–0.64 |
| perceived / KR | 0.76 | 0.77 | 0.97 | 0.74 | 0.27 | 0.49 / 0.49 | 0.63–0.65 |
| ref (oracle cube visibility) / R | 0.48 | 0.64 | 0.53–0.80 | 0.47–0.61 | 0.15–0.21 | 0.50 / 0.54 | 0.19–0.25 |
| ref / K | 0.71 | 0.71 | 0.97 | 0.67 | 0.015 | (0.98 / 0.41, 1–2 right) | 0.49–0.51 |
| ref / KR | 0.70 | 0.69 | 0.97 | 0.65–0.66 | 0.00–0.015 | — / 0.33 | 0.48–0.52 |

Paired over episodes (Q1 per episode, bootstrap 95%):
- perceived: K − R off **+0.29** [+0.22, +0.37], default **+0.16** [+0.09, +0.22]; KR − R off +0.28, default +0.15; **KR − K off −0.01 [−0.03, 0.00], default −0.004 [−0.01, 0.00]: same.**
- stock only: K − R +0.43 [+0.17, +0.67] off, +0.30 [+0.10, +0.50] default. Stress only: +0.27 [+0.19, +0.35] off, +0.14 [+0.07, +0.21] default.
- ref: K − R off +0.22 [+0.14, +0.30], default +0.07 [+0.00, +0.14].

**Readings, as fixed:**
1. **The KB's gain over the raw log transfers to real perception:** better at both efforts, in stock and in stress. Q1 0.77 vs 0.49–0.61; the joined question Q5 0.63 vs 0.21.
2. **The words add nothing over the v4 view here** (KR = K). The synthetic KR − K gain was against the v3 view; v4 already carries the joined facts. Q4 (last exchange) is worse in words (0.47–0.51 vs 0.80–0.81).
3. **Claim 3 through the LLM is not supported.** On cubes whose container's id the tracker swapped, every arm is mostly wrong (K 0.27; R 0.13–0.24), and no arm's stated confidence separates right from wrong (AUROC 0.29–0.54, at or below chance). The K view printed "0.98" for containers that had been swapped: R6 moved almost no mass, so the KB was confidently wrong and the model repeated it.
4. **Stock is near ceiling (K 0.97), stress is not (0.74).** The loss is the tracker's swaps that the KB does not catch, not the view.

**Also seen:**
- **ref is not an upper bound.** With the cover at the exact frame, K is lower (0.71) and the swapped cubes drop to 0.015. In the perceived row, some cube tracks reappear at table height and are covered again later (the occlusion step's release), which re-runs R1 on the container there and sometimes repairs a tracker swap.
- **Q2 (move count) 0.08–0.19 and Q6 (moved since covered) 0.03–0.18 for every arm:** the stock containers' tracks "moved" 11–14 times in one episode. The motion detector still makes spurious moves from tracker jumps (NOTES "Motion detector v2": about 3 per episode on G test), so these questions measure the detector.
- **Caveats:** G train only (the occlusion step and motion v2 were chosen on it); one task family; Q2–Q6 are mine; the swapped subset is small (67 cubes).

**What it means:** a structured memory doubles a cheap model's chance of naming the right container on real perception, as long as the tracker keeps the ids. When the tracker swaps them, the KB neither catches the swap nor lowers its confidence. So the next bottleneck for claim 3 is swap evidence (R6's pass detection and weight on real passes), not the text.

## Fixing the tracker's swaps: re-linking container tracks by motion (2026-10-29, user decision)
User: if the tracker can be fixed, fix it, and look for other places that need the KB more.
- **`arbiter_kb/relink.py`:** each container identity keeps a position and a smoothed velocity (alpha 0.3; steps over 5 cm/frame are tracker jumps and do not update it). Every frame the visible container slots are assigned to the predicted positions (Hungarian, xy). The containers are identical, so the direction of motion is the only cue through a crossing.
- **Chosen on G train only**, by id consistency: a container slot counts as consistent when it sits on the same true container at the end as just after the cover (`swapstats.identities`).
  - raw tracker 0.636; re-linked 0.69–0.73 over alpha {0.3, 0.6, 1.0} × v_max {0.02, 0.05}; coasting unseen tracks and gating did not help (0.44–0.73). **Chosen: alpha 0.3, v_max 0.05, 0.728.**
  - Per setting (of 36): fast_close 11 → 23, fast_lane03 16 → 22, fast_lane045 22 → 29, close 22 → 24, fast_close_3swaps 16 → 18; stock, cam_left, cam_right unchanged (34, 32, 25).
  - **Why it stops near 0.73:** in 28% of frames after t = 40 some container has no slot on it; the tracker drops container tracks (26% of frames have an unseen slot). Re-linking cannot restore observations that are missing; that needs the tracker itself (one multi-object Cutie, spec §2.5, pod).
- **Then:** the perceived-QA test again on the re-linked tracks, arms R and K, same questions, occlusion step and truth mapping (truth now in the re-linked ids). Compared with the earlier perceived row on Q1, stress, and the swapped subset. Cap $2.

**Results** (`results/latest/perceived_qa_relinked/summary.json`; 360 calls, **$0.21**):

| arm | Q1 off / default | Q1 stress | Q1, cubes still tracker-swapped | AUROC, swapped (off / default) | Q5 joined |
|---|---|---|---|---|---|
| R, raw tracker → re-linked | 0.49 → 0.54 / 0.61 → 0.68 | 0.48–0.60 → 0.55–0.68 | 0.21–0.33 | 0.47 / 0.29 | 0.27 |
| **K, raw tracker → re-linked** | **0.78 → 0.85 / 0.77 → 0.84** | **0.74–0.75 → 0.83** | 0.35 | 0.53 / 0.48 | 0.72–0.74 |

- **Re-linked − raw, paired over episodes:** K +0.07 [+0.01, +0.14] at both efforts (stress +0.08 [+0.01, +0.16]); R +0.07 to +0.08. Both better.
- K − R on the re-linked tracks: +0.29 [+0.21, +0.38] off, +0.15 [+0.08, +0.22] default, the same margins as before.
- Tracker-swapped Q1 cubes: 94 → 75 of 270. On those that remain, every arm is still mostly wrong and no confidence separates right from wrong (AUROC 0.29–0.53).
- Caveats: the truth uses the re-linked ids in this row (the container physically there, named by the ids the memory saw); G train only, where relink was chosen.
- **Reading:** the fix helps every arm by about the same, and the KB's margin over the raw log stays. The remaining loss is lost container tracks, which needs the tracker on the pod. The KB still does not know when it is wrong.

## Overnight plan (2026-10-29, user decisions before leaving)
User: zoom out from the tests; as a memory module, how can the KB organise more information that is useful to the LLM controller? Do A, B and C from the brainstorm, step by step, and analyse. Decisions: **budget $5 in total, qwen3.6-35b-a3b only**; commit each step to main, no push; B goes into a harness skeleton with offline checks; the code-reviewer and goal-reviewer agents run at the end and their findings are fixed.
- **A:** say how each fact is known, and give explicit alternative hypotheses instead of one number.
- **B:** memory of the controller's own actions and their outcomes, its past decisions, and "what changed since your last decision" first.
- **C:** memory as a tool: a short working memory plus queries the controller can make.

## A — explicit alternatives and how each fact is known: the test, fixed before running (2026-10-29)
**Free check first** (G train, re-linked tracks, 241 Q1 cubes, no model call):
- The KB's top container is right for 0.83 of the cubes. **Its own probability already separates right from wrong: AUROC 0.731** (median p 0.986). The model's stated confidence in the perceived QA was lower (0.66–0.71). The model loses information the KB has.
- **Alternative set** per hidden cube: start from its top container; going back in time from the end, every near pass after the cover with min_3d < d joins the partner.
  - d = 0.10 m: the truth is in the set for 0.944 of the cubes, mean size 1.57, size 1 for 0.51; the top is right for 0.90 when the set has size 1.
  - d = 0.06 m: 0.906, size 1.18. d = ∞: 0.949, size 1.78.
  - **Chosen before any call: d = 0.10 m** (it keeps 0.94 coverage at about one and a half containers per cube).

**Arms** (G train, re-linked, 90 episodes, effort off and default; cap $1):
- **K:** the v4 view, as before.
- **KA:** v4 plus a section "How the memory knows (per hidden cube)": the most likely container, how that is known (seen covered at t, followed with the container's track since; the tracker's ids can be wrong), and the alternatives with their reason ("bin_1 and bin_2 passed 4 cm apart at t = 110; the tracker may have mixed them up"), or "no close pass since the cover".
- Same KB, same numbers; KA only adds this section. Code: `arbiter_kb/epistemic.py` (frozen in the prompts hash).

**Questions:** Q1–Q6 and the Q1 confidence as before, plus **q1_alt**: "if not sure, which other container could it be? (container id or none)". Both arms get the same questions, so K is asked again.

**Readings fixed in advance:**
1. **Calibration (main):** the AUROC of the stated Q1 confidence over all Q1 cubes, KA − K, bootstrap over cubes. Better = the CI excludes 0. Reference: the KB's own p (0.731).
2. **Top-2:** the share of cubes where the Q1 answer or q1_alt is right, KA − K (paired bootstrap over episodes).
3. **No cost to the main answer:** Q1 accuracy KA − K within ±5 pp, or better.

## C — memory as a tool: the test, fixed before running (2026-10-29)
- **Arm KT** (`arbiter_kb/memtool.py`): a working memory whose size does not grow with the episode (the goal, the two container orders, each cube's most likely container), plus four queries the model can make, one per turn, at most 4:
  - `history(object)`: its moves and partners;
  - `passes(container)`: its close passes with distances;
  - `how_known(colour)`: A's line for that cube;
  - `events()`: the whole event log.
- **Against K** (the full v4 view, one call), on the same data, questions and scoring as the perceived QA (G train, re-linked, Q1–Q6 + Q1 confidence; no q1_alt), effort off and default. K comes from the cache.
- **Tokens:** the sum of input tokens over all turns of an episode, so the queries are paid for.
- **Readings fixed in advance:**
  1. Q1 KT − K within ±5 pp: "same"; the CI below −5 pp: worse.
  2. Input tokens per episode, KT / K (median).
  3. **Scaling (free):** K's view tokens against the number of KB events per episode, and KT's working memory against the same. The claim-2 argument is that KT stays flat while K grows; G's episodes are short, so this shows the slope, not a long-episode result.
  4. Which queries the model makes, and whether Q2 / Q4 / Q6 (which need the queries) fall.
- Cap $1.

## B — the controller's actions and their outcomes: built, checked offline (2026-10-29)
- **`arbiter_kb/self_memory.py`:** each controller decision is recorded in `kb.task["decisions"]` with its intent and a snapshot of the beliefs it was made on.
  - The intent is the controller's own "intent" field (an explanation, like "target" and "cites", never a constraint), or is read off the waypoint (gripper closes with nothing held → pick_up; opens while holding → release; already closed, nothing held, low → press; else move).
  - `outcome()`: "done (pick_up at t=… [e…])", "not done yet", or "not done; instead …", from the robot-state actions since the decision. No vision.
  - `changes()`: events and belief changes since the last decision (the most likely container changed; p fell by ≥ 0.1).
  - `self_view()`: the section that goes **first** in the prompt: the changes since the last decision, then the last three decisions with their outcomes. It suggests no action.
- **`arbiter_kb/harness.py`:** the loop skeleton (the KB reads the frames → prompt = self view + controller view → decide → record → env.step → the KB reads the new frames), an `Env` protocol for the pod's RoboMME wrapper, and `ReplayEnv` (a recorded dataset episode; a decision at each execution subgoal's start; the actions are ignored).
  - **Not yet the 2610.17350 port:** its prompt skeleton, feedback text and completion rules have to be copied from its released code before M5 (downloading it needs the user's go).
- **Offline check** (`scripts/check_self_memory.py`, counting episodes 31–100, no model call): a scripted controller declares the current subgoal's action.
  - Right intent → the outcome should say "done" (recall). Wrong intent (a different action declared) → it should not (control).
  - This mostly re-measures `actions.py` (recall 0.97–1.0 against the labels, NOTES "Actions vs subgoal labels") through the new code path. Its value is a working loop and outcome text, not a new result.
  - A smoke run on 5 episodes per task was looked at while building (all right intents "done" except SwingXtimes holds 13 / 16; wrong intents 0 "done").
- **Result** (`results/latest/self_memory_check/summary.json`, 210 episodes):
  - right intent says "done" **0.984** (1,415 / 1,438): every pick_up, release and press 1.00; SwingXtimes holds 0.927 (291 / 314), the known miss of `hold_above` on repeated holds;
  - wrong intent says "done" **0.000** (0 / 1,618): it never reports an action that did not happen;
  - size: the self view is a median of 229 characters against 912 for the controller view.
- **Reading:** the outcome text is as reliable as `actions.py`, at about a quarter of the view's length. Whether it helps a controller (failure categories such as premature completion or a late press) can only be measured in the closed loop (M5).

**A — results** (`results/latest/perceived_qa_A/summary.json`; 360 calls, **$0.55**, no empty answers):

| arm / effort | Q1 | Q1 swapped | AUROC of Q1 confidence, all cubes | top-2 (Q1 or q1_alt right) |
|---|---|---|---|---|
| K off / default | 0.842 / 0.842 | 0.354 / 0.354 | 0.675 / 0.682 | 0.921 / 0.913 |
| KA off / default | 0.842 / 0.826 | 0.354 / 0.333 | 0.675 / 0.687 | 0.921 / 0.905 |

- KA − K: AUROC 0.000 [0.000, 0.000] off, 0.000 [−0.053, +0.049] default; top-2 0.000 / −0.007 [−0.037, +0.015]; Q1 0.000 / −0.015 [−0.041, 0.000].
- **Readings, as fixed:** calibration **same** (no gain); top-2 same; no cost to Q1 (within ±5 pp). **A, as built, adds nothing.**
- **Why** (looked at after the readings): 57 / 90 KA answers without thinking are identical to K's, word for word. The stated confidences are the KB's own numbers copied: 0.99 (242 of 723), 0.98, 0.97. The K arm already names the same alternatives in q1_alt, read from the "passed close" lines of the event log, so top-2 is 0.92 in both arms.
  - So the model's calibration is the KB's calibration: it copies p. The KB's p has AUROC 0.731 here, and the model's 0.68.
  - **Hypothesis:** the numbers anchor the answer, so the reasons and alternatives are not used to set the confidence.

## A2 — the same alternatives without the numbers: the test, fixed before running (2026-10-29)
- **Arm KN:** KA with every probability removed from the text (the numbers 0.00–1.00 in the view and the joined facts), so only the reasons and alternatives remain. Code: `epistemic.strip_probs`.
- Against K and KA from the A run (cache), same questions (with q1_alt), G train, re-linked, effort off and default. Cap $0.6.
- **Readings fixed in advance:**
  - AUROC KN − K over all cubes: better if the CI excludes 0. Better = the reasons can carry calibration when the numbers do not anchor it.
  - Q1 KN − K within ±5 pp, or worse (the numbers were doing work).
  - Also reported: the spread of stated confidences (share at ≥ 0.97).

**C — results** (`results/latest/perceived_qa_C/summary.json`; 494 calls, the first run about $0.6; the report was replayed from the cache after a report bug, the `queries` field being read as a question):

| arm / effort | Q1 | Q1 swapped | AUROC all | Q4 last exchange | Q6 | input tokens per episode (median, all turns) | turns (mean) |
|---|---|---|---|---|---|---|---|
| K off / default | 0.846 / 0.842 | 0.354 / 0.354 | 0.69 / 0.66 | 0.80 / 0.81 | 0.12 / 0.13 | 2,486 / 2,484 | 1 / 1 |
| KT off / default | 0.813 / 0.672 | 0.312 / 0.25 | 0.63 / 0.64 | 0.06 / 0.13 | 0.90 / 0.28 | **523** / 1,151 | 1 / 2.49 |

- KT − K on Q1: off −0.043 [−0.074, −0.015]; default **−0.183 [−0.256, −0.115]**.
- **Readings, as fixed:** without thinking **undecided** (the CI is neither within ±5 pp nor below it), at **0.21× the input tokens**; with thinking **worse**.
- **Without thinking the model never queried** (90 / 90 answered at once). So the off row measures the working memory alone: it keeps 0.81 of K's 0.85 on Q1 for a fifth of the tokens, and loses the questions that need history (Q4 0.06). Q6 jumps to 0.90, likely because "yes" is the majority answer and KT answers yes without history (not checked).
- **With thinking it queried** (history 73, how_known 38, events 20, passes 3), but **18 / 90 episodes never answered**. 15 ran out of the 4 queries, and in others the model wrote a bare {"tool": …} without the "query" wrapper, which the loop took as the final answer. On the 72 that answered, Q1 is 0.83 against K's 0.87 on the same episodes. So most of the loss is the protocol, not the memory.

## C2 — the same, with the query protocol fixed: the test, fixed before running (2026-10-29)
- **Fixes:** a bare {"tool": …, "arg": …} counts as a query; on the last allowed turn the tool result ends with "No more queries: answer the questions now." Nothing else changes (working memory, tools, questions, limit of 4).
- **Arm KT2** against K (cache), effort off and default, G train, re-linked. Cap $0.8.
- **Readings fixed in advance:** the same as C (Q1 KT2 − K within ±5 pp = same; CI below −5 pp = worse; tokens KT2 / K), plus the share of episodes without an answer, which should be 0.
- **C scaling (free, reading 3; `results/latest/perceived_qa_C/scaling.json`):** over the 90 episodes (14–172 KB evidence items, median 72), K's view grows by **4.4 tokens per evidence item** (523–1,308 tokens, chars / 4). KT's working memory stays at 205–225 tokens (0.05 per item). The gap widens with episode length. G's episodes are short, so this is the slope, not a long-episode result.

**C2 — results** (`results/latest/perceived_qa_C2/summary.json`; **$0.06** new, since the calls up to the first query are cached):
- **Without thinking:** unchanged (it never queries): Q1 0.813, −0.043 [−0.074, −0.015] against K, at 523 vs 2,486 input tokens.
- **With thinking:** episodes without an answer fall from 18 to 5 (all five used up the 4 queries and asked again). Q1 0.772, **−0.081 [−0.137, −0.031]: undecided** (the CI neither within ±5 pp nor wholly below). Tokens 1,151 vs 2,484.
- On the 85 that answered: Q1 0.81 vs K's 0.85 on the same episodes. Episodes with queries: 0.81 vs 0.83; without: 0.81 vs 0.87.
- **Reading:** a working memory of about 220 tokens keeps most of Q1 (−4 to −8 pp) for a fifth to a half of the tokens. **Querying does not recover the rest:** with thinking, the model asks for history (75), how_known (38) and the event log (20), yet ends no better than when it does not ask. It loses the history questions outright (Q4 0.06–0.21 vs 0.80).
- So on G's short episodes the full view is better. The token case for KT needs long episodes, where the view grows by 4.4 tokens per evidence item. The query policy would need work (the model asks the wrong things, or too much).

**A2 — results** (`results/latest/perceived_qa_A2/summary.json`; 360 calls, **$0.27** new):

| arm / effort | Q1 | AUROC all | AUROC swapped (not a reading) | top-2 | stated confidence ≥ 0.97 |
|---|---|---|---|---|---|
| K off / default | 0.842 / 0.842 | 0.675 / 0.682 | 0.50 / 0.51 | 0.921 / 0.913 | 0.75 / 0.75 |
| KN off / default | 0.842 / 0.830 | 0.620 / 0.580 | 0.52 / 0.62 | 0.929 / 0.905 | 0.44 / 0.34 |

- KN − K: AUROC −0.055 [−0.130, +0.015] off, −0.103 [−0.218, +0.012] default; Q1 0.000 / −0.011 [−0.026, 0.000].
- **Readings, as fixed:** calibration **not better** (both point estimates lower; the CIs include 0); Q1 within ±5 pp.
- Without the numbers the model spreads its confidence (0.9 becomes the mode), but the spread does not follow right and wrong as well as the KB's numbers did.
- **What A and A2 say together:** the model's calibration comes from the KB's numbers. Alternatives and reasons in words neither add to it (A) nor replace it (A2). The KB's own p is the best signal on this data (AUROC 0.731). **To make the controller better calibrated, the KB's probabilities must be better** (R6's swap evidence on real passes); presentation will not do it.

## Overnight goal review: what it changed (2026-10-29, `reviews/overnight_2026-10-29_goal.md`)
Verdict: on track with changes. Checked and acted on:
1. **The LLM adds nothing on Q1 over the KB's own top answer.** Checked (free, `results/latest/perceived_qa_A/baselines.json`): the KB's most likely container, with no model, is right for **0.842** of the 241 Q1 cubes (stress 0.825), the same as K with the LLM (0.842–0.846). **So "K beats the raw log on real perception" means: the KB's cover-and-follow rules beat an LLM reading the raw tracker log.** It is the counting result again (RULE = K), not evidence that the KB helps an LLM decide. Claims 1–2 need the closed loop, where the controller has to act on it.
2. **Calibration baselines.** The KB's own p (best container): AUROC **0.701** overall and 0.704 within the stress settings. A one-bit "stock vs stress" feature: 0.558 (the review computed 0.641 from other rows; mine is on the KB's 241 cubes). "Any near pass since the cover → low": 0.52.
   - **Correction:** A's free check reported 0.731. It scored the 7 cubes whose top hypothesis is "elsewhere" at p = 0, while the model is asked for a container. With the best container's p, it is 0.701. The conclusion is the same: the model's stated confidence (0.68) is a little below the KB's p.
   - **Claim 3 is about the KB's p, which a controller can read directly.** The LLM's stated confidence is the wrong metric for it and is dropped as a claim-3 measure.
3. **C is true by construction for Q1:** the working memory states each cube's most likely container, so KT's 0.81 restates the KB's 0.84. "A fifth of the tokens" compares against K, not against claim 2's reference (the image history, 16.5k tokens and 148 images in 2610.17350). **C is stopped**; `memtool.py` stays for a closed-loop arm only if that is needed.
4. **A / A2:** null results, so the leak of d = 0.10 into A does no harm. "Calibration comes from the KB's numbers" is a hypothesis consistent with A2, not a measurement (the KN CIs include 0, and ties dominate: 75% of confidences ≥ 0.97). **No further view arms.**
5. **B's control was too easy.** Added a **premature** control (the next subgoal's action declared early):
   - at first it read "done" for 0.83 of the SwingXtimes holds, because the outcome matched the action type, not its side;
   - **fixed:** a decision may carry a `detail` (side or how), and the outcome must match it. Premature now says "done" 0.019 overall (holds 0.073); right intent is still 0.984, wrong intent 0.000 (`results/latest/self_memory_check`).
   - B is frozen until M5.
6. **In-sample:** G train chose the occlusion step, the relink setting and A's d, and all the QA rows are scored on it. Relink had no reading fixed in advance, so its +0.07 is weak evidence. One frozen G test read is the remedy (next step).
7. **Rules and results in the same commit:** today's counting controls (9001dd4) committed the rule together with its results. From now on, each rule gets its own commit before its run.
8. **Next (replaces HANDOFF Next 1):**
   - turn the relink cost margin into `id_margin` evidence for R6;
   - score the KB's p with no LLM (AUROC / AURC on swapped vs clean cubes, within stress) against SoftHaircut with the same margin feature, the setting bit and the near-pass heuristic;
   - tune on G train, commit the rule, then one frozen G test read, adding R / K / the KB's top answer on Q1 (about $0.4);
   - if the KB does not beat SoftHaircut, claim 3 is negative;
   - after that, only the closed loop (H / R / K) can move claims 1–2.

## Overnight code review: fixes, before rescoring (2026-10-29)
The reviewer recomputed every stored perceived-QA report from its rows and got the same numbers. Found and fixed:
1. **BLOCKER, Q6 truth was "no" in all 90 episodes:** the container's position was taken at the cube's last visible frame, where the oracle container is still NaN (containers are absent before t ≈ 32). It now uses the container's first finite position at or after the cover, with a test. **Every Q6 number above is void**, and so are the readings built on it: "Q6 measures the detector", and C's "KT answers yes" (KT-off answered "no" 81 / 90 and was scored right only because the truth was always "no").
2. **MAJOR, "tracker-swapped" counted missing tracks as swaps** (17 of 67 raw, 17 of 48 re-linked). Now None (unknown) when no slot sat on the true container just after the cover or at the end; unknowns are left out of the swapped subset. Every "Q1 swapped" and "AUROC swapped" figure above changes (the reviewer's recomputation: K's Q1 on swapped cubes 0.27 → 0.08 raw, 0.35 → 0.10 re-linked). The reading "mostly wrong on swapped cubes" gets stronger.
3. **MAJOR, A2's KN arm still showed the cover confidences** ("(conf 0.99)" in the event log, 303 times over 90 views). So KN was not the registered arm. `strip_probs` now removes them too (test added); **KN is run again as registered** (new calls), and the first KN numbers are void.
4. **MINOR, two slots on one true container:** Q1 now accepts any slot sitting on the true container at the end (about 0.4 pp).
5. **MINOR, the AUROC-difference bootstrap** now resamples episodes, not cubes.
6. **Not changed (MINOR):** A's alternatives propagate back from the end, not forward in time (sets differ for 11 / 225 cubes; truth-in-set 0.960 vs 0.956); the harness decides 8 frames into a subgoal, which may explain some of B's SwingXtimes hold misses.
- **Then:** every perceived-QA run is re-scored from the cache (no new calls), and KN is re-run. The corrected tables follow.

## Corrected tables after the code review (2026-10-29; these supersede the perceived-QA numbers above)
All runs were re-scored from the cache ($0). KN was re-run as registered (**$0.27**). Q1 truth now accepts any slot on the true container, Q6 truth is fixed, and "swapped" leaves out unknown tracks (re-linked: 31 swapped, 17 unknown of 241 Q1 cubes). Files: `results/latest/perceived_qa*/summary.json`, `perceived_qa_A/baselines.json`.

**Q1 (which container hides each cube), off / default:**

| row | KB top answer, no LLM | K | KR | R | KA | KN | KT2 |
|---|---|---|---|---|---|---|---|
| raw tracker | — | 0.784 / 0.780 | 0.772 / 0.776 | 0.490 / 0.614 | | | |
| re-linked | **0.846** | 0.851 / 0.846 | | 0.548 / 0.685 | 0.846 / 0.830 | 0.846 / 0.838 | 0.817 / 0.776 |
| ref (oracle cube visibility) | — | 0.710 / 0.714 | 0.705 / 0.697 | 0.477 / 0.639 | | | |

- **Swapped cubes (re-linked):** the KB's top 0.129; K 0.129 / 0.129; R 0.129 / 0.065. On the raw tracker: K 0.10, R 0.04. Every arm is mostly wrong there.
- **Q6 (moved since covered), now valid:** K 0.73 / 0.72, R 0.86 / 0.76, KA 0.68, KT 0.19 / 0.26. The raw log is as good as or better than K here; K's view has many spurious moves.
- **Calibration (re-linked, AUROC of the stated Q1 confidence over all cubes):** K 0.67–0.68, KA 0.67–0.68, KN 0.60 / 0.57; **the KB's own p 0.694** (0.698 within stress); the stock / stress bit 0.558.
- **Paired, unchanged in sign and decision:**
  - K − R: +0.29 / +0.15 (re-linked), +0.30 / +0.16 (raw);
  - KA − K: AUROC 0.000 / +0.001 [−0.059, +0.054];
  - **KN − K (as registered): AUROC −0.073 [−0.169, +0.017] off, −0.105 [−0.233, +0.010] default**;
  - KT2 − K: Q1 −0.043 / −0.081.

**Readings after the corrections:**
1. **On Q1 the LLM adds nothing over the KB's own answer** (0.846 = 0.846). "K beats the raw log" is the KB's rules beating an LLM that reads the raw log. The controller is not yet doing anything the KB could not.
2. **A and A2 unchanged:** alternatives in words add nothing, and removing the numbers (now all of them) lowers the stated confidence's AUROC (not significant). The stated confidence stays at or below the KB's own p (0.69).
3. **C unchanged** for Q1. KT's Q6 drops to 0.19–0.37 (it has no history), so the earlier "KT answers yes" note was wrong.
4. **Claim 3's measure is the KB's p** (AUROC 0.69 overall, 0.70 within stress), against the trivial baselines: the setting bit 0.56, near pass since the cover 0.52. The next step compares it with SoftHaircut using the same margin feature.

## Claim 3 on G: the KB's own probabilities, with the re-linking margin as swap evidence. The test, fixed before running (2026-10-29, user go)
**Question:** does the KB's belief layer give better-calibrated answers than the same swap score without a KB (SoftHaircut), on real perception? No LLM is involved: claim 3 is about the KB's p, which a controller can read directly (overnight goal review).

**New evidence:**
- `relink.relink_margin`: per identity and frame, how much worse (m) the assignment would be if it exchanged slots with its closest competitor. Near 0 = the motion cannot tell the two apart.
- `PerceivedSource(relinked=True)` puts it in each container's `track_q` = min(1, margin / 0.10 m). The near-pass detector records the pass's lowest track_q as `min_q`.
- R6 and SoftHaircut take the feature `min_3d` (scale d0, as before) or `min_q` (scale q0): w = w_max·exp(−feature/scale), w_max 0.5 for both.

**Systems:** Bound (a point mass); SoftHaircut(feature); ArbiterKB (R6 inside the KB, plus R1's cover kernel, R5, the "elsewhere" floor), each with min_3d or min_q. Same perceived frames: G, re-linked tracks, the occlusion step.

**Per Q1 cube:** the system's answer (its most likely container) and that answer's p; truth from `perceived_qa.truth` (any slot on the true container at the end).

**Metrics:** accuracy; AUROC of p for right / wrong; AURC (the mean risk over coverage, answers sorted by p; lower is better). Overall and within the stress settings.

**Tuning on G train only** (`scripts/claim3_g.py --grid`): d0 ∈ {0.02, 0.05, 0.10, 0.20} m; q0 ∈ {0.05, 0.10, 0.20, 0.50}. The choice per (system, feature) is the lowest train AURC, ties to the smaller scale. It is saved to `results/latest/claim3_g/choice.json` and frozen.

**One read on G test** (episodes 10–19 of each setting; never read for this question; motion v2 read it once for motion segments only), with a ledger READ line.

**Readings fixed in advance** (paired bootstrap over test episodes, 1,000 draws):
1. **Main: ArbiterKB(min_q) − SoftHaircut(min_q), AURC.** KB better if the CI is below 0. Otherwise claim 3 is **not supported on real perception**: the gain is the feature, not the belief layer.
2. The same for min_3d.
3. ArbiterKB(min_q) − ArbiterKB(min_3d), AURC: does the margin carry swap information that distance does not?
4. Reported with each: accuracy (R6 can change the answer) and AUROC; Bound as the floor.

**Then the LLM rows on G test** (Q1, as in the perceived QA, re-linked, with the chosen KB setting): R, K and the KB's top answer, effort off and default, about $0.4. They show whether the train results (K = the KB's top answer, K − R +0.15 to +0.29) hold out of sample.
- **Grid on G train** (`results/latest/claim3_g/grid.json`, 241 Q1 cubes), AURC (lower is better) / AUROC / accuracy:
  - Bound 0.1295 / 0.50 / 0.834;
  - SoftHaircut: min_3d best at d0 0.02, **0.1127** / 0.66; min_q best at q0 0.05, **0.0950** / 0.67; accuracy 0.834 (0.809 at d0 0.2);
  - ArbiterKB: min_3d best at d0 0.02, **0.0908** / 0.69; min_q best at q0 0.05, **0.0874** / 0.71; accuracy 0.846.
  - **Chosen (committed before the test read):** soft/min_3d 0.02, soft/min_q 0.05, kb/min_3d 0.02, kb/min_q 0.05. q0 = 0.05 is the grid's edge for both systems; the grid stays as registered.
  - On train, the KB is ahead of SoftHaircut with either feature, and min_q helps both.

**Claim 3 — the G test read** (`results/latest/claim3_g/read.json`; 235 Q1 cubes; ledger READ a810942859b9):

| system | accuracy | AUROC | AURC | AUROC stress | AURC stress |
|---|---|---|---|---|---|
| Bound | 0.821 | 0.50 | 0.1432 | 0.50 | 0.1372 |
| SoftHaircut min_3d (d0 0.02) | 0.821 | 0.640 | 0.1269 | 0.644 | 0.1365 |
| SoftHaircut min_q (q0 0.05) | 0.817 | 0.608 | 0.1218 | 0.603 | 0.1238 |
| **ArbiterKB min_3d (d0 0.02)** | 0.817 | **0.742** | **0.0962** | 0.732 | 0.1109 |
| ArbiterKB min_q (q0 0.05) | 0.809 | 0.723 | 0.1056 | 0.714 | 0.1206 |

Paired over test episodes:
- **KB − SoftHaircut, min_q (main): AURC −0.016 [−0.088, +0.019]; AUROC +0.115 [+0.033, +0.201].**
- KB − SoftHaircut, min_3d: AURC **−0.031 [−0.074, −0.004]**; AUROC +0.102 [+0.032, +0.192].
- KB min_q − KB min_3d: AURC +0.009 [+0.002, +0.020] (min_q worse).

**Readings, as fixed:**
1. **Main (min_q): not supported.** The AURC CI includes 0. By the rule, the KB's advantage with the margin feature is not shown on test.
2. **With min_3d, the KB is better than SoftHaircut** (AURC CI below 0), and its AUROC is higher with either feature (CIs exclude 0). The belief layer adds calibration beyond R6's swap score: Bound's errors get lower p in the KB, through R1's cover kernel, R5 and the "elsewhere" floor (which of these does the work is not measured).
3. **The margin does not carry swap information out of sample:** min_q was best on train and is worse than min_3d on test, for the KB. The train choice (q0 at the grid's edge) overfitted.
- **Summary (rewritten after the goal review, `reviews/claim3_g_goal.md`):** **the registered main test is not supported**, and the new margin evidence overfitted (min_q is worse than min_3d on test). Secondary: with min_3d, the KB's p ranks its own errors better than a swap-score haircut (AURC −0.031 [−0.074, −0.004], upper end near 0; one of two tests; d0 at the grid's edge), at Bound's accuracy. **Where that comes from is not measured**, and it is probably not swap detection: at d0 0.02, R6's weight at the 5–9 cm passes where G's real swaps happen is about 0.5·e^(−2.5) ≈ 0.04, and the KB is right on only 0.09–0.12 of the tracker-swapped cubes. The likely source is cover-time ambiguity (R1's kernel, the eps_far floor, R5), which SoftHaircut's point-mass start never has. G test is now spent for claim 3; a confirmatory claim needs fresh episodes.

## Claim 3 follow-up: the LLM rows on G test (2026-10-29; registered in "Claim 3 on G")
- R and K on Q1, G test, re-linked, effort off and default; K's KB uses the train choice min_q, q0 0.05 (`--r6 min_q:0.05`), as registered before the read. The KB's top answer with the same setting: 0.809 (above).
- Ledger READ line with the base params hash. Cap $1.
- **Results** (`results/latest/perceived_qa_test/summary.json`; 360 calls, **$0.51**, 2 empty answers; commit e91ee66, clean tree):

| arm | Q1 off / default | stock | stress | Q1 on tracker-swapped cubes (33) | AUROC of stated confidence | Q5 joined |
|---|---|---|---|---|---|---|
| KB top answer, no LLM | 0.809 | | | | (the KB's p: 0.723) | |
| K | 0.813 / 0.809 | 1.00 | 0.79 / 0.78 | 0.12 / 0.09 | 0.62 / 0.63 | 0.80 / 0.77 |
| R | 0.489 / 0.736 | 0.55 / 0.83 | 0.48 / 0.72 | **0.33 / 0.36** | 0.52 / 0.57 | 0.31 / 0.30 |

- K − R: off **+0.311 [+0.219, +0.404]**; default **+0.067 [−0.006, +0.143]**.
- **Out of sample:**
  1. **K = the KB's top answer again** (0.81 = 0.809): the LLM adds nothing on Q1.
  2. **K beats the raw log without thinking; with thinking the gap shrinks to +0.07, and its CI includes 0** (train: +0.15). With thinking, a 35B-A3B model reading the raw tracker log gets 0.74.
  3. **On the cubes the tracker swapped, R is better than K (0.33–0.36 vs 0.09–0.12):** reading the raw positions, the model sometimes follows the true container where the KB follows the wrong id. The KB is confidently wrong exactly where it matters, and its p does not drop enough there (the KB's AUROC lead over SoftHaircut comes from elsewhere).
  4. The model's stated confidence (0.62–0.63) is below the KB's own p (0.72), as on train.
- **Process note (from the goal review):** the rule commit (e5b30fe) and the grid commit are 68 s apart. The grid was run after the rule commit (its grid.json records commit e5b30fe, clean tree); no train AURC was seen before the rule was written.
- **Falsifier 3 (the raw log as good as the KB) is nearly triggered with thinking** on G test: K − R +0.067 [−0.006, +0.143]. R reads fewer prompt tokens (1.65k vs 2.5k) but spends reasoning tokens.

## Claim 3 attribution: where does the KB's calibration lead come from? Fixed before running (2026-10-29, goal review)
Post-hoc attribution of the read above, **$0, no LLM**: run on G train first; then on G test, labelled as post-hoc, with a ledger READ line. G test is spent for confirmatory claims, so nothing here is tuned on it.
- **Systems:** the KB as read (min_3d, d0 0.02) and four ablations:
  - (a) **R6 off** (w_max = 0);
  - (b) **no cover kernel or floor** (sigma_cover → 0.001 m, eps_far = eps_none);
  - (c) **R5 off**;
  - (d) **cover only** (R6 and R5 off: R1's distribution at the cover, carried forward with the container; the equivalent of a "CoverHaircut").
  - Plus the earlier systems: SoftHaircut min_3d, Bound, the stock / stress bit, and "any near pass since the cover → low".
- **Scored on the same rows:** accuracy, AUROC, AURC; also **AUROC on swapped vs clean cubes** (does p drop on the cubes the tracker swapped?), which the overnight review asked for and the read did not score.
- **Readings fixed in advance:**
  - if (a) ≈ the KB (AURC within ±0.01) and (d) ≈ the KB, the calibration lead is the cover feature, not swap handling or the belief layer as a whole;
  - if (b) loses most of the lead, the cover kernel is the source.
  - No change to the claim-3 verdict above is made from this.

## Claim 3 code review: the AURC was wrong (2026-10-29, `reviews/claim3_g_code.md`)
- **BLOCKER:** `claim3_g.aurc` broke ties by row order (episode, then cube) instead of letting them enter together, as `metrics.aurc` and the spec do (DEVELOPMENT §3.10). The ties are large: Bound is all p = 1, and SoftHaircut min_q has 206 / 241 train rows at p = 1 (`round(w, 6)` makes w = 0 when min_q ≈ 1). Shuffling within ties moves SoftHaircut min_q's train AURC from 0.070 to 0.167. **Every AURC above and all paired AURC CIs are void**; accuracy and AUROC are unaffected.
- **Fixed:** `claim3_g.py` now calls `metrics.aurc`. The reviewer checked that the train choice is the same under the spec AURC for all 16 settings. So the **test read is redone with the committed `choice.json`, as a correction** (ledger READ). The grid is re-run only to record the corrected train table; the choice must come out the same.
- **Also fixed:** `auroc(...) or 0.5` turned a real 0.0 into 0.5. Tests added for `relink_margin` (small at a crossing, inf alone) and for R6's feature / scale. `ArbiterKBSystem(skip=...)` was added for the ablations.
- **Not changed (MINOR):** the margin counts only competitors matched in that frame, and its minimum may come from a third container; `--split test --print` builds test prompts without a READ line; 8 of the 360 G-test LLM calls were cache hits (source not found); run meta records the replaced params hash (40e166310e73), the ledger the frozen base (a810942859b9).
- The attribution ablation registered above now also has its code (`claim3_g.py --ablate train|test`).

## Claim 3, corrected read and attribution (2026-10-29; these supersede every claim-3 AURC above)
**Grid re-run with the spec AURC** (`results/latest/claim3_g/grid.json`): the same choice (checked against the saved file). Train AURC: Bound 0.166; SoftHaircut min_3d 0.121 / min_q 0.117; ArbiterKB min_3d 0.099 / min_q 0.095.

**Corrected G test read** (`read.json`; ledger READ marked CORRECTION; 235 cubes):

| system | accuracy | AUROC | AURC | AUROC clean vs swapped (p lower on swapped) |
|---|---|---|---|---|
| Bound | 0.821 | 0.50 | 0.179 | 0.50 |
| SoftHaircut min_3d | 0.821 | 0.640 | 0.139 | 0.812 |
| SoftHaircut min_q | 0.817 | 0.608 | 0.147 | 0.703 |
| **ArbiterKB min_3d** | 0.817 | **0.742** | **0.104** | 0.812 |
| ArbiterKB min_q | 0.809 | 0.723 | 0.115 | 0.763 |

- KB − SoftHaircut: **min_q (main) AURC −0.033 [−0.069, +0.004]**; min_3d **−0.034 [−0.069, −0.004]**; AUROC +0.10 to +0.11 (CIs exclude 0). KB min_q − KB min_3d AURC +0.011 [+0.001, +0.022].
- **Readings, unchanged after the correction:** main **not supported** (the CI includes 0, narrowly); min_3d: the KB is better; the margin feature is worse than distance out of sample.
- Trivial baselines on the same rows: the stock / stress bit 0.56–0.58, "no near pass since the cover" 0.50.

**Attribution** (`ablate_train.json`, `ablate_test.json`; the test run is post-hoc, with a ledger READ). AURC Δ against the KB [95% CI], train / test:
- (a) **R6 off: +0.040 [+0.002, +0.078] / +0.053 [+0.003, +0.100]**; AUROC −0.14 / −0.16; clean-vs-swapped AUROC 0.72 / 0.81 falls to 0.50.
- (b) **no cover kernel or floor: +0.007 [−0.010, +0.028] / +0.037 [+0.010, +0.069]**; AUROC −0.01 / −0.09.
- (c) R5 off: −0.000 / −0.001 (nothing).
- (d) cover only (R6 and R5 off): +0.040 / +0.052, the same as (a).

**Readings, as fixed:** neither "(a) ≈ KB" nor "(d) ≈ KB" holds, so **the lead is not the cover feature alone**. R6 does most of the calibration work. The cover kernel adds the rest on test: without it, the KB's test AURC (0.141) is SoftHaircut's (0.139). R5 adds nothing on G.

**What it means:**
1. **The goal review's guess is refuted in part:** R6 does act on real swaps. With R6, p is lower on swapped cubes (clean-vs-swapped AUROC 0.81 on test), even though w ≈ 0.04 at a 5–9 cm pass. The drop is small: the median p is 0.92 on swapped cubes against 0.98 on clean ones, and 65% of the KB's wrong answers still have p ≥ 0.9. **The ranking is right but the magnitude is overconfident**, which is why a model that copies the numbers stays overconfident.
2. **The KB's lead over SoftHaircut is R6 on a cover-time distribution instead of a point mass.** R1 starts each hidden cube with mass over the nearby containers, so R6's mixing has something to spread. A KB-free haircut that also kept the cover kernel would likely match it. That is a feature, not the belief layer's propagation (R5 is inert here; retraction is untested). By falsifier 6's logic, **claim 3's "belief layer" is not shown on G**, though the KB's p is the best-calibrated signal measured.
3. **Next for claim 3, if pursued:** fresh episodes (G test is spent), d0 / w_max tuned for magnitude (ECE / Brier as well as AURC), and a SoftHaircut with the cover kernel as the fair baseline. Lower priority than the closed loop.
- An extra look at p's magnitude on the test rows (the median p values above) was computed post-hoc from the same systems.

## Closed loop: the 2610.17350 harness ported, pod runbook (2026-10-29, user go)
- **Their code, downloaded with the user's go:** `github.com/jackbjed/embodied-astra-robomme` at commit 79dcc7f (MIT), shallow clone in the session scratchpad, read only, never run. Their loop is `harness/astra_waypoint_eval_v3.py`:
  - memory = the whole conversation (every sampled front frame stays in context, chained with previous_response_id);
  - endings: declared complete, unparseable / invalid answer, env error, 3 consecutive no-op actions, 150 calls;
  - one front frame every 3 control steps plus the last; reasoning effort medium; their 92.1% run used the left/right note.
- **`arbiter_kb/closed_loop.py`** ports that loop:
  - arm H re-sends the whole conversation (OpenRouter has no previous_response_id; the content is the same);
  - the memory arms (R, K, …) get a fresh request each step: memory text + the current front and wrist frames + their state text (TCP pose, fingers, elapsed steps, camera calibration);
  - K's memory text starts with `self_memory.self_view` (B), and every decision is recorded in the KB;
  - `check()` (workspace, reach, and for memory arms the cited ids) gives one retry, for every arm alike.
  - Tests: `tests/test_closed_loop.py` (fake env: success, the three endings, H's chained conversation, the memory arm's fresh request with its self view, the retry).
- **`controller.ENDING`** now carries their MODEL_NOTES and left/right note verbatim, and the calibration sentence. This changes the system prompt for every controller arm (the counting and QA tests use their own prompts and are unaffected). Re-frozen: prompts c8d7e42cdcaa.
- **`arbiter_kb/sources/online.py`** (`OnlineObs`): perception slot rows → FrameObs one frame at a time, with the incremental `relink.Relinker` (now behind `relink_margin`, bit-identical on all 180 G files) and a causal occlusion step. **On all 180 G episodes the online FrameObs equal the batch source's: 158,427 object-frames, 0 differences.**
- **Pod side:**
  - `stress/perceive.slot_row` (one frame through the worker, shared with the offline run);
  - `stress/closed_loop_pod.py` (RoboMME env with depth for perception only, the open word list, arms H / R / K, resumable results, ledger READ for `--dataset test`);
  - `stress/POD_CLOSED_LOOP.md` (setup, smoke test, val pilot on VideoUnmaskSwap and VideoPlaceOrder, 10 episodes per arm; about $3–4 GPU and under $1 LLM).
  - **Not run:** the pod script cannot run on the Mac (RoboMME, the worker), so its first real run is the smoke test. The runbook lists what to verify there (observation keys, double reset, the K prompt).
- **Model:** `qwen/qwen3.6-35b-a3b` takes images on OpenRouter (262k context), so one model serves all arms.

**Smoke test on the pod (A40; 2026-10-29), first findings, merged back:**
- **Double reset:** only the first `reset()` of a fresh RoboMME env gives the episode's own scene (169 vs 151 frames on a second reset). `closed_loop_pod.episode` now probes the goal and K on a throwaway env and runs the arm on a fresh one (the pod session's fix, merged).
- **Image limit:** DeepInfra takes at most 30 images per request; arm H sends 58 on its first turn and about 150 by the end. Probes (under $0.10) found **Parasail** takes 160; **every arm now uses Parasail** (`--provider`, default Parasail).
- **Token cap:** at 4,000 completion tokens, H's and R's first calls ended in reasoning (finish_reason length, empty answer → "unparseable"). **`--max-tokens` 16,000 for every arm.** K's calls did not hit the cap.
- **Perception:** about 1.05 s per perceived frame including model load (169-frame video, 57 frames perceived).
- **Our bug, fixed on the Mac:** with the open word list, the four generic "cube." slots were all named `cube_cube.` in `OnlineObs` and overwrote each other. They are now `cube_s<slot>` (a coloured phrase named once keeps `cube_<colour>`; test added; G unchanged). Tracker slots that never found an object are no longer listed in the controller view. Re-frozen.
- The K prompt was checked on the pod: the memory lists the containers and the green cube, and after the first action it starts with "Since your last decision".
- **Smoke run with the fixed upload (Parasail, 16,000 cap):** H success in 3 calls ($0.007, 123 s; 58 / 78 / 86 images accepted). **R and K ended "unparseable": their reasoning looped until the cap** (K: 450 lines, 69 distinct, one sentence repeated 25+ times). Perception about 0.9 s per frame on the A40.
- **Sampling, decided 2026-10-29:** every call so far used temperature 0. Qwen warns that greedy decoding in thinking mode leads to endless repetition, and 2610.17350 used its API defaults (temperature 1, top-p 0.98). **The closed loop now uses Qwen's recommended thinking settings for every arm: temperature 0.6, top_p 0.95, top_k 20** (`closed_loop_pod.py --temperature / --top-p / --top-k`; `LLM.call(sampling=...)` adds them, so earlier cache keys are unchanged). Results are no longer exactly replayable; the pilot and M5 rely on episode counts, not on one sample.
- **Three bugs of ours the smoke run exposed, fixed (tests added):**
  1. the one retry did nothing for the fresh-request arms (R, K): the problem note was only added to H's conversation, so the identical request went out again, the cache returned the same answer, and it was executed. Now the note and the model's last answer ride on the next fresh request;
  2. a never-seen generic cube slot was listed as "visible" in the joined facts while the object list left it out; never-seen slots are now left out of both (`narrative.seen`);
  3. `check()` rejected cites written as one string ("e32,e37,o49:bin_2") or in brackets ("[o74:bin_2]"); they are now split as `shown_ids` splits them.
- **Noted, not fixed:** a generic cube slot sat at x = −0.59 (off the table: likely a detection on the robot or the edge). A workspace filter on perceived objects is a candidate after the pilot.
- **Smoke run on CoreWeave with retries (2026-10-29; $0.11):** VideoUnmaskSwap H and R, and VideoPlaceOrder R and K, ended at the token cap (loops); **VideoUnmaskSwap K ran 8 calls to a real outcome (`env:fail`, wrong pick)**. VideoPlaceOrder H was rejected before any call: its opening video is 1,078 frames (360 sampled front images + wrist), and OpenRouter's pre-flight counts 1,536 tokens per image (554k > 262k), while the model uses about 66 tokens per 256-px frame (5,334 prompt tokens for 58 images). Perception about 1.0 s per frame on the opening video, 0.6–0.7 s later. No 429 on CoreWeave.
- **Loop-rate probe** (first-turn requests of H, R and K from VideoUnmaskSwap, 5 seeds each, CoreWeave, $0.33; cap hits or looping lines per 15):

| setting | looped / 15 | parseable / 15 | median reasoning tokens (H, R, K) | median s (H, R, K) |
|---|---|---|---|---|
| S0: T 0.6, top_p 0.95, top_k 20 | 3 | 12 | 4.4k, 5.1k, 4.2k | 43, 79, 37 |
| **S1: S0 + presence_penalty 1.5** | **0** | **15** | 9.0k, 4.6k, 1.9k | 94, 64, 19 |
| S2: S0 + effort low | 5 | 11 | 16k, 4.4k, 2.9k | 168, 45, 25 |
| S3: S0, thinking off | 1 | 14 | 0 | 11, 5, 7 |

- **Decided (2026-10-29), every arm alike:** presence_penalty 1.5 (S1); provider CoreWeave; the pod session's 429 / 5xx retry (6 tries, 5·2^n s) merged into `llm.py`; **arm H's sampled frames go 3 × 3 into one image** in reading order (`closed_loop.GRID`, `grid`), followed by the current front frame alone. Every frame is kept; VideoPlaceOrder's first turn becomes 40 grids + 2 images. The grids are ours (2610.17350 sent single frames) and are stated in H's prompt. Caveats: n = 5 first turns; S1 is untested on whole episodes, so the next smoke run checks it.
- **Smoke run with S1, CoreWeave and H's grids (2026-10-29; $0.35):** no loops (2 of 48 calls hit the 16k cap, both R's last call, 96–97% of reasoning lines distinct: long reasoning, not repetition). H's grids fit (largest request 53 images; VideoPlaceOrder's first turn 42). Outcomes: VideoUnmaskSwap H `declared_complete` (call 3), R `unparseable` (call 32 at the cap), K `env:fail` (3 calls); VideoPlaceOrder H `declared_complete` (call 4), R `unparseable` (call 3 at the cap), K `harness_error`. Perception 1.1–1.25 s per frame on the opening video (VideoPlaceOrder: 360 frames, 387–452 s), about 4 s per later batch. Reasoning on VideoPlaceOrder: 10k–16k tokens per call for R and K.
- **Two bugs of ours, fixed (tests added; re-frozen, prompts 05918c9eb8a7):**
  1. R's every target was "unknown" (`_problems` took its ids from the KB, which R does not have; R's log prints bare ids), so every R decision cost a retry: VideoUnmaskSwap R's 32 calls were 16 decisions. R's ids now come from the frames (`Memory.ids`), and its log leaves out tracker slots that never found their object, as K's view does (`raw_tracker_log(keep=)`; the offline R arms are unchanged).
  2. K crashed on VideoPlaceOrder (`KeyError('table')`): a put-down away from every container is the event `place(cube, "table")` (rules.py), and `narrative.name` looked "table" up as an object. Non-objects are now named by their id.
- **Token cap 32,000 for every arm** (`--max-tokens`): 16,000 cut R's genuine reasoning on VideoPlaceOrder. Under OpenRouter's pre-flight that leaves room for about 149 images per H request (largest so far 53).
- **Open:** both H episodes declared completion before the env registered success; the next smoke run reports H's last answers to tell a misread scene from a premature "done".
- **Smoke run with the fixes and the 32k cap (2026-10-29; $0.46):** no cap hits, no harness errors; K ran on VideoPlaceOrder. **A third bug of ours, fixed:** R still retried about half its calls (VideoUnmaskSwap 15 of 31, VideoPlaceOrder 7 of 15), now on "unknown cite": R's log lines have no ids, so R cited whole lines ("t=20: bin_0 0.008, ..."). The answer format asks every arm for cited ids, but only K's memory has them, so R's cites are no longer checked (its target still is; test extended; re-frozen, prompts f9dafdf84d96).
- **Same run, the rest (pod report):** K finished VideoPlaceOrder (`declared_complete`, 3 calls); no call hit the 32k cap (largest 21.6k reasoning tokens). H's two `declared_complete` endings: on VideoUnmaskSwap H read the scene correctly (right container lifted, green cube visible) but declared done at z = 0.15, while the env needs a higher lift (run 3 reached `env:success` after lifting to z = 0.25): a premature "done", the same prompt for every arm (2610.17350's ending, verbatim), so left as is. On VideoPlaceOrder H misjudged depth (closed the gripper at z = 0.30, in the air; cubes at z ≈ 0.025) and then reported a placement that never happened: the failure H's lack of 3D positions predicts, and where K's memory should help.

**Val pilot, interim read (2026-10-30 01:46; 24 of 60 episodes; run untouched):**
- **One success in 24** (VideoUnmaskSwap ep 2 K); none on VideoPlaceOrder. Endings: declared_complete 10, unparseable 4 (all the 32k cap, empty content, VideoPlaceOrder R / K), env_error 4 (3 the mani_skill zero-step `follow_path` bug, 1 a genuine planning failure), invalid_action 2, env:fail 2, env:timeout 1, success 1.
- **Floor effect:** with qwen3.6-35b-a3b as the controller every arm is near 0, so the pilot cannot separate H, R and K. 2610.17350's 92.1% used a frontier model; this model's failures are mostly motor and planning (H closes the gripper in the air, repeats one waypoint for 15+ calls; R flips between plans; K repeats grasp-and-release), not memory.
- **One memory-side problem:** on VideoPlaceOrder, K carried a bin onto another bin ("the blue cube is believed to be inside bin_0"): the open word list's containers and the KB's Inside relation are VideoUnmaskSwap's ontology, and on a placing task they mislead.
- The zero-step bug is RoboMME's / mani_skill's (the prompt's 2 mm note is 2610.17350's warning about it); it hits every arm alike.
- **Decided (user, 2026-10-30):** the pilot was stopped at 24 / 60 episodes (results kept on the pod under `/workspace/cl/pilot`). Next: a stronger controller, starting with 2610.17350's own model (`openai/gpt-6-astra`, on OpenRouter at $10 / $50 per M tokens), first one H episode per task to measure cost and check that our port reproduces their behaviour.
- **Fixes for that (tests added; code review in the session, its findings fixed; re-frozen, prompts 88111cd4ccc1):**
  - **Targets are not containers:** when the goal names targets and no container (`task_items.place_word`; VideoPlaceOrder, PickXtimes, SwingXtimes), the generic slots look for "target." and are named `target_<s>` (cls "target"): they hide nothing, and a put-down on one is `place(cube, target_k)` with `On(cube, target_k)`. `rules.below` keeps a target even when the cube on it hides it. The first slots keep sg_pilot's container area floor (without it spare slots seed on cubes; the pilot's "bins" were the targets, so targets pass it); to verify on the pod at frame 0.
  - **Pod runner:** `--model`, `--provider auto` (CoreWeave for Qwen, OpenAI for openai/; others must be named), non-Qwen models on their API defaults (no temperature, seed or sampling sent; recorded as None), `--max-usd-episode` (default 3): an episode over it ends `limit:episode_usd` and is **unscored** (H costs most, so a scored cap would favour the memory arms), and the run stops at the global budget instead of failing every later episode.
  - `prompts_hash` now also covers `task_items` and `sources/online` (the goal parse and the object ids K reads).
- **GPT-6-Astra probe (2026-10-30, val ep 0, arm H, OpenAI):** **2 / 2 success** (VideoUnmaskSwap 3 calls $0.17; VideoPlaceOrder 5 calls $0.65, 171k prompt tokens: H re-sends the image history every call). Qwen solved neither in the pilot. Our port reproduces 2610.17350's behaviour on these episodes; the controller model was the pilot's bottleneck.
- **Perception check, VideoPlaceOrder val ep 0–2 (no LLM):** the "target." phrase does **not** find the targets (flat discs at z = 0.005). target_0 sits on the robot base at (-0.577, -0.002, 0.054) in all three episodes (the same false detection as VideoUnmaskSwap's `cube_s8` in the smoke run); target_2–4 are never seen; target_1 visits each target's position mid-video, most likely riding the cube as it is placed. So the place word fixes the KB's ontology but not detection. Open: which generic phrase finds flat targets (to be measured against the simulator's target positions before any pilot), and a workspace filter for the robot-base detection.
- **Decided (user, 2026-10-30): both at once.** (1) The VideoUnmaskSwap val pilot with `openai/gpt-6-astra` for every arm (H / R / K, episodes 0–9; its perception was validated on G). (2) VideoPlaceOrder's target phrase, chosen on val before any VideoPlaceOrder pilot.
- **Robot-base filter (tests added; re-frozen, prompts 7aade10cec0b):** `online.not_objects` marks hidden any detection within 10 cm (xy) of the robot base or more than 2 cm below the table, before re-linking. The detector's phrase per place word is now `task_items.PLACE_PHRASE`, apart from the slot names.
- **Rule for the target phrase, fixed before the measurement:** candidates "target.", "container.", "circle.", "disc.", "ring." (generic shape words; no colour, which the goal does not give). On VideoPlaceOrder val episodes 0–4, opening video, with the robot-base filter, against the simulator's target positions: **found** = targets with a place slot within 3 cm in xy (z < 0.03) at frame 0; **coverage** = mean over perceived frames of the share of targets so matched; **hijack** = place slots within 3 cm of a cube at frame 0; and the goal's named cube must still be found at frame 0. Choose the highest mean found, then coverage; a phrase with any hijack or a lost named cube is out. If no phrase finds at least 3 of 4 targets on average, VideoPlaceOrder's K arm is not run, and NOTES says so.
- **GPT-6-Astra VideoUnmaskSwap val pilot, ep 0–9, H / R / K (finished 2026-10-30 05:24; pod report, files under
  `/workspace/cl/astra_pilot`; OpenRouter shows $13.78 for the run):**

  | arm | success | endings | median calls | total $ | median $ / ep | median prompt tok / ep | median min / ep |
  |---|---|---|---|---|---|---|---|
  | H | **9 / 10** | env:fail 1 (ep 1) | 5 | 2.76 | 0.278 | 51,673 | 1.0 |
  | R | 6 / 10 | env:fail 2, env_error 2 (ep 7, 9: RRT* failed three times; not the zero-step bug) | 16 | 6.18 | 0.603 | 40,957 | 7.3 |
  | K | **8 / 10** | env:fail 2 (ep 0, 8) | 5.5 | 3.52 | 0.279 | 19,558 | 4.2 |

  - Qwen 3.6-35B had 1 success in 15 VideoUnmaskSwap pilot episodes; with Astra all arms work, so the floor effect was
    the model's.
  - **K against H, n = 10, a direction only:** 8 vs 9 successes (not separable at this n), **−62% prompt tokens**
    (19.6k vs 51.7k), the same median cost per episode ($0.279 vs $0.278: K's text prompt is cheaper but it makes
    slightly more calls; with output at $50 / M, output tokens probably dominate, not yet split per arm). Claim 2
    shows on tokens, not yet on dollars, with this model. K's wall time is perception (4.2 vs 1.0 min).
  - **R is the weakest and costliest** (16 calls median, 2 RRT* failures): the raw log does not give the model a
    stable plan; the KB view does.
  - **Caveat:** ep 1 R's first attempt ended in harness_error, OpenRouter HTTP 402 (credits low, given in-flight
    requests); its rerun took the first 25 of 29 calls from the LLM cache (the same answers) and 4 new ones, and
    succeeded. Its first attempt's spend ($1.32) is in OpenRouter's total, not in any row.
  - K's two failures and H's one are to be read (memory wrong vs read wrong vs execution) before any claim; not done.
  - **Why K's fewer tokens did not cost less: OpenAI's prompt caching** (pod, from the LLM cache's responses, free;
    2026-10-30 06:14). Per arm over the pilot (cache files, so H includes the probe's ep 0 and R ep 1's aborted try):

    | arm | prompt tok | cached tok | cached share | completion tok | reasoning tok | cost $ | input $ | output $ | $ / M prompt tok |
    |---|---|---|---|---|---|---|---|---|---|
    | H | 650,160 | 500,685 | **0.77** | 11,258 | 7,054 | 2.93 | 2.37 | 0.56 | **4.51** |
    | R | 461,342 | 0 | 0 | 34,713 | 17,651 | 7.50 | 5.77 | 1.74 | 16.26 |
    | K | 244,858 | 0 | 0 | 9,152 | 2,190 | 3.52 | 3.06 | 0.46 | **14.37** |

    - Input dominates every arm's cost (output 13–23%), so the earlier guess "output sets the cost" was wrong.
    - Solving the input costs: about $12.50 / M uncached (a cache write) and about $1.00 / M cached. H's request is
      one growing conversation, so 77% of its prompt is a cached prefix; R and K send a fresh request each call with
      the changing memory text and new frames near the start, so they pay the write price and never read it back.
    - **K sends 38% of H's prompt tokens but pays 3.2× more per token, so the dollars come out even.** Claim 2 holds
      on tokens; on dollars it depends on the provider's caching and on K's prompt layout.
    - Counterfactual, not measured: with H's cached share, K's input would be about $0.89 instead of $3.06 (total
      about $1.35 vs H's $2.93). A cache-friendly K (fixed text first, changing view and frames last, or one growing
      conversation) is a prompt change: a new version, re-frozen, measured, never patched into these results.
    - `llm.py` does not record `cached_tokens` per call; to add before the next paid run.
- **Target-phrase test (VideoPlaceOrder val ep 0–4; pod report 2026-10-30 05:57; frame rows in
  `/workspace/cl/phrase_check/<phrase>_ep<e>.npz`, `summary.json`):**

  | phrase | found at frame 0 (of 4), mean | coverage | hijack | named cube found at frame 0 |
  |---|---|---|---|---|
  | target. | 0.00 | 0.002 | 0 | 5 / 5 |
  | container. | 0.00 | 0.017 | 0 | 5 / 5 |
  | circle. | **3.20** | **0.678** | 0 | **4 / 5** (lost in ep 2) |
  | disc. | 2.00 | 0.596 | 0 | 4 / 5 (lost in ep 2) |
  | ring. | 0.00 | 0.060 | 0 | 5 / 5 |

  - **By the rule fixed before the measurement, no phrase passes:** "circle." and "disc." lose the named cube in
    ep 2, which puts them out; the other three find no target. **So VideoPlaceOrder's K arm is not run.** "circle."
    is the only phrase that finds the targets (3.2 of 4 at frame 0, two-thirds over the video); running it would
    need the user to change the rule, recorded as a deviation.
  - **Deviation in the measurement:** hijack was measured against the perceived cube slots, not the simulator's
    cubes (their poses at frame 0 of the opening video are not available after reset). The pod saw no hijack with
    any phrase.
  - Not yet known: whether "circle." loses the cube in ep 2 by taking its slot or the cube is missed anyway at
    that frame (the npz rows can tell, free).

## Layer 1 on real scenes: how accurate is the model's first call-out? The test, fixed before running (2026-10-29, user request)
Question: on the real RoboMME scenes the closed loop already ran, does the model's first waypoint point at the right
object, and does its first close happen at the object's height? Layer 1 of docs/HANDOFF_EXECUTION.md; the evidence so
far (M4, Batch A) is synthetic text with oracle positions only. No LLM call: it scores answers already paid for.
- **Data:** every `<task>_ep<e>_<arm>/result.json` under `/workspace/cl/pilot` (qwen/qwen3.6-35b-a3b) and
  `/workspace/cl/astra_pilot` (openai/gpt-6-astra, the probe included). A row's own `model` field wins; the run
  directory gives it otherwise.
- **Truth:** per (task, episode, dataset) a fresh env (`closed_loop_pod.make_env`), **one reset** (only the first reset
  gives the episode's scene; RoboMME's reset returns after the opening video, so this is the scene of the first
  decision), poses read from `env.unwrapped` (`pod_checks.truth`). Objects: actors within the table box
  (|x|, |y| ≤ 0.6 m, −0.01 ≤ z ≤ 0.30 m), not within 10 cm (xy) of the robot base, not named table / ground / floor /
  camera / wall / goal_site / robot; actors at the same position (≤ 1 mm) are one object. **Covered objects** (within
  3 cm in xy of an object at least 1 cm higher: VideoUnmaskSwap's cube under its container) decide the right object
  but are left out of the nearest / second-nearest distances (added while writing the code, before any run: otherwise
  every pick of the hiding container is "uncertain").
- **The right object, from the goal and the simulator:** the goal's first coloured cube (`task_items.parse_task_items`);
  if the goal also names a container, the right object is the container nearest that cube in xy (the one hiding it;
  VideoUnmaskSwap), else the cube itself (VideoPlaceOrder). The cube is the object whose actor name contains the colour;
  if none or several do, the right object is **unknown** and the episode is reported, not scored. A `--list` pass
  prints every episode's objects and right object before scoring, to be read first.
- **Fallback when the goal names no coloured cube** (e.g. VideoPlaceOrder's goal may only say "in the order of the
  video"; decided 2026-10-29, before any listing): **the right object stays unknown and the task is not scored for
  "right"**. Its rows still report what needs no right object: d1 (does the waypoint point at any object at all),
  uncertain, and dz at the first close. Not chosen: "the first cube picked in the video". RoboMME's reset returns after
  the video, so the simulator's poses cannot show it; it would need an env attribute whose name and meaning we have
  not read, and whether that cube is the right *first* pick depends on the task's rule, which we would be guessing
  after seeing the data. If the listing shows such an attribute, a rule for it is written here before any scoring.
- **First waypoint:** the first step with an executed-or-attempted action (retries without an action skipped). d1, d2 =
  xy distance to the nearest and second-nearest object.
  - **right:** nearest = the right object and d1 < 3 cm;
  - **uncertain:** d2 − d1 < 2 cm (checked first; never counted right; reported apart);
  - **wrong:** otherwise; **no action:** the episode ended before any action; **unknown:** no right object.
  - Also logged: the xy distance to the right object.
- **First close:** the first action with gripper −1: dz = waypoint z − the centre z of the object nearest it in xy, and
  that object's name. A waypoint is the TCP pose, so dz ≈ 0 is a centre grasp.
- **Report:** one CSV row per episode; per task × arm × model: n, right / uncertain / wrong / no action / unknown,
  median and max d1, median and max distance to the right object, median and max |dz| at the first close.
- **Limits, fixed now:** only the first waypoint and the first close (later steps need replays whose determinism is
  unverified); about 30 episodes, so a direction, not a conclusion; the first waypoint can be a pre-grasp above the
  object, so only its xy is scored. Run on the pod after the target-phrase test, with the user's go.
- **Run procedure:** the code goes to its own directory (`/workspace/score`), never over the pilot's copy; the pod is
  not touched until the target-phrase test has finished; stage 1 is `--list` only, its output comes back and the
  right objects are confirmed here before stage 2 (scoring) runs.
- **Stage 1 listing (pod, 2026-10-30 06:21; `/workspace/cl/callout_list.txt`): unusable, right object null in all 14
  episodes** (VideoPlaceOrder 0–3, VideoUnmaskSwap 0–9). Our bug: `pod_checks.truth` keyed objects by env attribute
  and index (`spawned_bins[0]`, `all_cubes[0]`, `targets[2]`), so no name held the goal's colour; the colour is in the
  actor's own name (`cube_red_0`, `target_0`, per the earlier perception check). No scoring was run.
  - The listing also showed: VideoPlaceOrder's goals name a coloured cube ("place the red cube on the first target it
    was previously placed on"), so the fallback rule is not triggered there; VideoPlaceOrder lists `button` and
    `cap_link[0]` 7 mm apart (one button); every VideoUnmaskSwap cube sits under a bin at the same xy, as expected
    after the video; **VideoUnmaskSwap val ep 9 lists two cubes at the same position** (-0.021, -0.144, 0.017),
    unexplained so far.
- **Rule changes after the listing, before any scoring** (the reason in each):
  - objects are keyed by the actor's own name; one actor reached through two attributes is kept once (`pod_checks`);
  - the ≤ 1 mm "same position" merge is gone from `objects`: with names it is not needed, and in ep 9 it would drop
    one of two distinct cubes, perhaps the goal's;
  - for the nearest / second-nearest distances only, actors within 1 cm (3D) are one object's parts (the button and
    its cap_link), kept under the first name; covered objects are still left out there;
  - the goal's cube must be an actor whose name contains both the colour and "cube" (a coloured bin cannot match).
  - Stage 1 is rerun with these before stage 2.

## Astra VideoUnmaskSwap val pilot and the target-phrase test: pod report (2026-10-30)
Read from the pod session's report (pasted by the user); the files stay on the pod (`/workspace/cl/astra_pilot`,
`/workspace/cl/phrase_check/`).
- **Astra pilot, VideoUnmaskSwap val ep 0–9, every arm `openai/gpt-6-astra`:**

| arm | success | endings | median calls | $ total | median $ / ep | median prompt tok / ep | median min / ep |
|---|---|---|---|---|---|---|---|
| H | 9 / 10 | success 9, fail 1 (ep 1) | 5 | 2.76 | 0.278 | 51,673 | 1.0 |
| R | 6 / 10 | success 6, fail 2, env_error 2 (ep 7, 9) | 16 | 6.18 | 0.603 | 40,957 | 7.3 |
| K | 8 / 10 | success 8, fail 2 (ep 0, 8) | 5.5 | 3.52 | 0.279 | 19,558 | 4.2 |

  - R's two env_error endings are genuine planning failures (RRT* returned −1 three times each), not the zero-step
    bug. Qwen solved 1 of 15 VideoUnmaskSwap episodes in its pilot.
  - **ep 1 R's first attempt ended in OpenRouter HTTP 402 (credits too low for the in-flight requests)**; it left no
    result, was rerun once and succeeded, but 25 of its 29 calls were cache replays of the aborted attempt. The run
    spent $13.78 on OpenRouter against $12.46 in the results (the difference is that attempt).
  - **Prompt caching (from the LLM cache's usage fields, no call made):** about 77% of H's prompt tokens were cache
    reads (OpenAI caches H's re-sent history); K and R never hit (their fresh request per step changes the text at
    the start). Uncached prompt tokens cost about $12.50 per million, cached about $1.00. So H's per-token cost is
    about 3× lower than K's; **any cost comparison for claim 2 must count cached tokens and say so**.
  - n = 10 per arm: H 9 vs K 8 vs R 6 is a direction, not a difference.
- **Target-phrase test (rule fixed above, VideoPlaceOrder val ep 0–4; found at frame 0 / coverage per episode):**

| ep | target. | container. | circle. | disc. | ring. |
|---|---|---|---|---|---|
| 0 | 0 / .011 | 0 / .000 | 3 / .463 | 3 / .474 | 0 / .000 |
| 1 | 0 / .001 | 0 / .047 | 3 / .604 | 1 / .615 | 0 / .000 |
| 2 | 0 / .000 | 0 / .013 | 3 / .741 | 1 / .402 | 0 / .218 |
| 3 | 0 / .000 | 0 / .027 | 3 / .730 | 1 / .637 | 0 / .084 |
| 4 | 0 / .000 | 0 / .000 | 4 / .851 | 4 / .851 | 0 / .000 |

  - circle.: mean found 3.20, coverage 0.678, hijack 0, named cube found at frame 0 in 4 / 5 (missed in ep 2, as with
    disc.). No phrase had a hijack (measured against the visible cube slots: the simulator's cube positions at frame 0
    are not available after reset).
  - **Under the rule as fixed, circle. is out**: "a phrase with any hijack or a lost named cube is out", and circle.
    lost the named cube in ep 2; so is disc. The remaining phrases find 0 targets, so the rule's last line applies:
    **VideoPlaceOrder's K arm is not run.** The pod session called circle. "clearly best", which is true of found and
    coverage but not the rule's verdict. Changing the rule now (e.g. allowing one lost cube in five) would be a change
    after seeing the data; it is the user's decision and would be stated as such.
- **Decided (user, 2026-10-30): the rule stands.** circle. and disc. are out, no phrase qualifies, and VideoPlaceOrder's
  K arm is not run; VideoPlaceOrder compares H and R only. The rule is not changed after seeing the data.
- **Stage 1 rerun (pod, 2026-10-30 06:32; commit b5240fa in `/workspace/score`): right object filled in all 14
  episodes, checked here one by one against the listed poses.** VideoPlaceOrder 0–3: the goal's cube (`cube_red_0`,
  `cube_red_0`, `cube_blue_0`, `cube_blue_0`). VideoUnmaskSwap 0–9: the bin at the xy of `target_cube_<goal colour>`
  (bin_2, bin_1, bin_1, bin_2, bin_1, bin_0, bin_2, bin_0, bin_1, bin_1); every one matches to ≤ 1 mm. Stage 2 may run.
  - **ep 9's "two cubes at one position" was ours, not the scene's:** the env reaches the same actors through many
    aliases (`target_cube`, `target_cube_0/1/2`, `target_cube_for_bin_0/1/2`, `target_cube_<colour>`, `other_cubes`),
    and the old attribute-keyed walk listed one actor under several keys. The three cubes are at three positions.
  - **Do not use the env's `target_cube` as the right object:** in ep 9 it is the red cube (`target_cube_color` 'red')
    while the goal asks for green, then blue. The goal's colour is the reference, as the rule says.
- **Stage 2 results (pod, 2026-10-30 06:39; `/workspace/cl/callout/callout_rows.csv`, `callout_summary.csv`,
  raw lines `/workspace/cl/callout_score.txt`, not copied). No unknown or no_objects rows.**

| task | arm | model | n | right | uncertain | wrong | no action | d1 median / max (m) | d_right median / max | \|close dz\| median / max |
|---|---|---|---|---|---|---|---|---|---|---|
| VideoUnmaskSwap | H | astra | 10 | 10 | 0 | 0 | 0 | 0.0049 / 0.0173 | 0.0049 / 0.0173 | 0.0253 / 0.0284 |
| VideoUnmaskSwap | R | astra | 10 | 10 | 0 | 0 | 0 | 0.0137 / 0.0294 | 0.0137 / 0.0294 | 0.0138 / 0.0244 |
| VideoUnmaskSwap | K | astra | 10 | 9 | 0 | 1 | 0 | 0.0135 / 0.0294 | 0.0142 / 0.1962 | 0.0083 / 0.0217 |
| VideoUnmaskSwap | H | qwen | 5 | 1 | 2 | 2 | 0 | 0.0701 / 0.1743 | 0.096 / 0.1743 | 0.006 / 0.4467 |
| VideoUnmaskSwap | R | qwen | 5 | 4 | 0 | 1 | 0 | 0.0124 / 0.0149 | 0.0138 / 0.1647 | 0.0067 / 0.0133 |
| VideoUnmaskSwap | K | qwen | 4 | 2 | 0 | 2 | 0 | 0.0118 / 0.0134 | 0.088 / 0.1962 | 0.005 / 0.0083 |
| VideoPlaceOrder | H | qwen | 4 | 0 | 0 | 4 | 0 | 0.0354 / 0.0619 | 0.0656 / 0.1808 | 0.03 / 0.28 |
| VideoPlaceOrder | R | qwen | 4 | 3 | 0 | 0 | 1 | 0.0079 / 0.0094 | 0.0079 / 0.0094 | 0.02 / 0.08 |
| VideoPlaceOrder | K | qwen | 3 | 3 | 0 | 0 | 0 | 0.0072 / 0.0079 | 0.0072 / 0.0079 | 0.03 / 0.03 |

  (Qwen's n are the pilot's episodes before it was stopped; there are no Astra VideoPlaceOrder rows: the probe's
  VideoPlaceOrder episode is not under `/workspace/cl/astra_pilot`.)
- **Reading (about 70 episodes over 9 cells; a direction, not a conclusion):**
  - **With memory (R, K) the first call-out is right in most episodes for both models:** Astra 19 / 20, Qwen 12 / 15
    of the episodes that acted (corrected from 12 / 16 when the CSVs were checked). The memory arms' xy errors repeat to 0.1 mm across arms and across models (VideoUnmaskSwap
    ep 1: 0.0098 for Qwen K, Qwen R, Astra K, Astra R; ep 2: 0.0113; ep 4: 0.0149), so **their waypoint copies the
    perceived position, and their 0.3–2.9 cm error is perception's, not the model's.** Two of those rows sit near the
    3 cm line (Astra K / R ep 9: 2.94 cm).
  - **The memory arms' wrong picks are of two kinds.** VideoUnmaskSwap ep 0: Qwen K and Astra K both chose bin_1 at the
    same distance, while R chose right with both models: K's memory pointed at the wrong bin there (a KB error, to
    read in that episode's view). ep 3: Qwen K and Qwen R both chose bin_1, Astra K and R chose right from the same
    kind of memory: the model's choice, not the memory.
  - **From images (H) it depends on the model:** Astra H is right in 10 / 10 and the most precise of all rows (median
    4.9 mm, better than perception's ~1.3 cm); Qwen H is right in 1 / 9, with two "uncertain" first moves 9–17 cm from
    every object and VideoPlaceOrder first moves 3.4–3.7 cm from the right cube (ep 0, 2) or at another object
    (ep 1, 3).
  - **So "a small model can find the place" holds only when the memory gives the positions;** Qwen cannot localise
    from the images. Qwen K / R's low pilot success comes mostly after a first move that is right.
  - **The first close already shows layer-2 failures for Qwen:** H closed 28 cm (VideoPlaceOrder ep 0) and 45 cm
    (VideoUnmaskSwap ep 2) above the object's centre (in the air). On VideoPlaceOrder the memory arms closed 1–8 cm
    above the 4 cm cube's centre (K: 1, 3, 3 cm; R up to 8 cm), i.e. at or above its top face for most, where a grasp
    likely fails. Astra's closes on bins are consistently 1–3 cm below the bin's centre.
- **The CSVs on the Mac (2026-10-30):** `data/cl_pod/callout/callout_rows.csv` (55 rows: Qwen 25, Astra 30; sha256
  394fc61a…) and `callout_summary.csv` (9 rows; c0fea8e2…), downloaded from the pod's Jupyter; `data/` is git-ignored.
  Rerunning `callout.summarize` on the rows reproduces the summary exactly (0 differences). Columns: `first_xyz` /
  `nearest` / `d1` / `d2` / `d_right` / `category` for the first waypoint; `close_xyz` / `close_object` / `close_dz` for
  the first close (dz = waypoint z − that object's centre z); `status` / `end_reason` the episode's ending; `how` the
  right-object derivation. `callout_score.txt` (the raw lines) stays on the pod.
- **Read from the rows, beyond the summary:**
  - **First call-out against the episode's outcome:** Astra's 23 successes all began with a right call-out, and 6 of its
    7 failures did too (the 7th is K ep 0, the wrong bin); Qwen began right in 13 episodes and succeeded in 1. So for
    both models most failures happen after a right first move, in layers 2–5.
  - **The target can change between the first move and the first close:** Qwen K and R on VideoUnmaskSwap ep 3 moved
    first above bin_1 (wrong) and closed at bin_2 (right); Qwen R ep 1 moved above bin_1 (right) and closed at bin_0
    (wrong). The first waypoint alone does not settle which object the model meant.
  - **Qwen H on VideoPlaceOrder ep 0 closed on its first action**, at z = 0.30 above the cube (the "closed in the air"
    case of the smoke run), then declared completion.
  - **Astra's failures with a right call-out:** H ep 1 and R ep 3 / ep 7 close within the successes' range (−0.6 to
    −2.8 cm); K ep 8 closed 2.2 cm above bin_1's centre, and R ep 8 / ep 9 at +0.1 cm, the highest of R's closes
    (R's successes: −0.6 to −2.4 cm). A weak hint that closing high goes with failing; n is far too small, and the
    cause may be later. To be read in those episodes' later steps.

## Astra's 7 VideoUnmaskSwap failures after the first move: the reading, fixed before reading (2026-10-30, user request)
Question: in the 7 failed Astra episodes (K ep 0, 8; H ep 1; R ep 3, 7, 8, 9), where did it go wrong, and do the
failures point to HANDOFF_EXECUTION option 3 (skills) or option 4 (KB as completion verifier)? Free: `result.json` only.
- **Data:** `/workspace/cl/astra_pilot/*/result.json` (all 30, so the 23 successes give the reference), copied to
  `data/cl_pod/astra_pilot/`; scene objects and goals of VideoUnmaskSwap val 0–9 from the stage-1 listing
  (`data/cl_pod/callout/vus_objects.json`, transcribed from the pod's output).
- **What a result.json has:** per executed step the 7-vector action, frames recorded, env status; the ending. **Not**
  finger widths, object poses after reset, or the model's answers (except rejected retries). So "the grasp missed"
  cannot be observed, only inferred.
- **Trace per episode:** each close (gripper −1) is mapped to the nearest object at its **reset** position (after a bin
  is lifted and put down it moves, so a later close can be mis-mapped: flagged when the close is > 3 cm from every
  object); "held" from a close until the next open (+1); the lift of a hold = the highest waypoint z while held.
  The goal's sequence of right objects = the container hiding each coloured cube the goal names, in order.
- **One label per failed episode, first rule that applies:**
  1. **planning failure:** ended `env_error`;
  2. **wrong object at close:** a hold on an object that is neither the next in the goal's sequence nor the one just
     picked (re-lifting it after a slip is not wrong), followed by a lift ≥ 5 cm; (re-grasp clause added while writing
     the code, before reading any data)
  3. **premature "done":** ended `declared_complete` (waypoint null) without the env's success;
  4. **lift too low:** a hold on the right object whose lift is below the lowest lift of that object-pick in any
     successful episode (all arms), then the episode ended without success;
  5. **grasp missed (inferred):** a hold on the right object lifted at least as high as the successes', and still no
     success: unconfirmed, as result.json has no finger widths;
  6. **other:** anything else, described.
- **Pointing (fixed now):** labels 2–5 that a deterministic pick (grasp height from the KB centre, fixed lift) would
  remove point to option 3; label 3 (and a "done" after a missed grasp) points to option 4; label 1 is the planner's
  and points to neither; label 2 is a decision / memory error and points to neither. n = 7: a direction only.
- **Read (2026-10-30; `scripts/read_failures.py` on `data/cl_pod/astra_pilot/`, 30 result.json from the pod's Jupyter;
  no LLM call).** Reference from the 23 successes: lowest lift of the first right pick 0.13 m, of the second 0.16 m.
  The rule's label is reported as computed; the raw steps of every failure were then read by hand, because most later
  closes are flagged (> 3 cm from every reset position: the bins had been moved).

| episode | goal picks | rule label | what the steps show | layer | points to |
|---|---|---|---|---|---|
| K ep 0 | 1 | wrong object at close | first move to bin_1 (KB's wrong bin, as the call-out found), lifted → fail | memory | neither (a KB error) |
| H ep 1 | 2 | grasp missed (inferred) | pick 1 right (bin_1 to 0.19 m), then **opened at 0.16 m directly above bin_2**, dropping bin_1 onto it; closed there and lifted → fail (most likely bin_1 again) | 3 (put-down) | option 3 |
| R ep 3 | 2 | grasp missed (inferred) | pick 1 right (bin_2, four closes, to 0.18 m), put down at (−0.23, −0.24); **then re-grasped and lifted the same bin_2** → fail; bin_0 never tried | 3 / progress | option 4 |
| R ep 7 | 2 | planning failure | pick 1: closed at bin_0 but lifted only to 0.085 m and opened; repeated closes pushed bin_0 out to x ≈ 0.20; 5 answers rejected by our reach check; RRT* exhausted | 2 (grasp / lift) | option 3 |
| K ep 8 | 2 | wrong object at close | pick 1 right (bin_1 to 0.23 m, put down); pick 2 bin_0 **lifted only to 0.14 m (< 0.16) and released**; then went back and lifted bin_1 again → fail | 3 (lift) + progress | options 3 and 4 |
| R ep 8 | 2 | grasp missed (inferred) | pick 1 right (bin_1 to 0.23 m, put down); **then hovered over and re-grasped bin_1 three times and lifted it** → fail; bin_0 never tried | progress | option 4 |
| R ep 9 | 2 | planning failure | pick 1 (bin_1) lifted to 0.17 m and **released in the air**; bin_1 landed near x ≈ 0.19; the model kept returning to bin_1, not bin_2 (2 answers rejected by reach); RRT* exhausted | 3 (put-down) + progress | options 3 and 4 |

- **The rule mislabelled three episodes** (H ep 1, R ep 3, R ep 8 as "grasp missed"): its re-grasp clause (added to
  spare a re-lift after a slip) also spared re-lifting the first bin **after it had been put down**. A better clause:
  re-lifting the just-picked object counts as wrong once an open has put it down. Not applied retroactively; the
  table keeps the rule's labels next to the hand reading.
- **What the 7 show (n = 7, one model, one task: a direction):**
  - **Two-pick goals are where Astra fails:** single-pick episodes (0, 2, 4, 5, 6) 14 / 15 success, the one failure
    the KB's wrong bin; two-pick episodes (1, 3, 7, 8, 9) 9 / 15. Six of the seven failures are in two-pick goals.
  - **The most common failure is losing track of progress:** in 4 episodes (R ep 3, K ep 8, R ep 8, R ep 9) the model,
    after a successful first pick, went back to the first bin instead of the second. No failure is a premature
    "done": none ended `declared_complete`.
  - **Execution failures are put-downs and low lifts, not grasps at the wrong height:** dropping a bin from 0.16–0.17 m
    (H ep 1, R ep 9), lifting to 0.085 m and 0.14 m (R ep 7, K ep 8 pick 2). A bin dropped or pushed toward x ≈ 0.2
    ends in reach rejections and an RRT* failure (R ep 7, 9): both `env_error`s are consequences, not the planner's
    fault alone.
- **Which option:**
  - **Option 4, reshaped:** the evidence is not for a "done?" verifier (no premature done) but for **progress facts
    against the goal**: "pick 1 of 2 done: bin_1 lifted to 0.23 m and put down at (x, y); next: the container hiding
    green = bin_0 at (x, y)". 4 of 7 failures. This is a memory contribution (fits claims 1–2). K already shows its
    own decisions (self_memory), yet K ep 8 still went back to bin_1: decisions alone do not say which subgoal is done,
    so the section must be phrased against the goal's sequence.
  - **Option 3** would remove the put-down and lift failures (4 of 7: H ep 1, R ep 7, K ep 8, R ep 9): a fixed lift
    (≥ the successes' 0.16 m, or the env's threshold once read) and a put-down that lowers to the table at a free spot
    away from the other bins and the reach limit. A separate system claim, as HANDOFF_EXECUTION says.
  - **Neither** for K ep 0 (the KB's own wrong bin: a perception / memory error to read in that episode's view).
  - Overlap: K ep 8 and R ep 9 count for both. **Order suggested by this reading:** option 4's progress section first
    (cheap, every memory arm, fits the claims), then option 3 as the separate system.

## Does K lose track on two-pick goals? The test, fixed before running (2026-10-30, user decision)
Why: of the 4 "went back to the first bin" failures above, 3 are R's (a baseline we do not change) and 1 is K's
(ep 8). So the evidence for adding progress facts to K is one episode, and K at 8 / 10 cannot show a gain on those 10.
First measure whether K itself loses track, on new episodes.
- **Detector, written and checked on the pilot before this run** (`callout.went_back`): on a two-pick goal, P = where
  the first right pick was put down (the first open after its close; the gripper command is taken to act at the end of
  the motion: assumed, not read in RoboMME's executor); "yes" when a later hold lifted ≥ 5 cm is nearer P than the
  second right object's reset position; "ambiguous" when P is within 3 cm of that object. On the pilot it says yes for
  R ep 1 (a success: it moved the dropped first bin aside), R ep 3, 7, 8, 9 and K ep 8, ambiguous for H ep 1. R ep 7's
  texts say "I will grasp bin_0 first": it never finished pick 1, so a "yes" is not by itself losing track.
- **Logging added (not read by the model):** each executed step in result.json now carries the model's `description`
  and `target`. `closed_loop.py` is in the prompts hash, so params were re-frozen (params a810942859b9, prompts
  82d83ed1f52d); the prompts and views the model reads are unchanged. Also `llm.py` now counts the provider's prompt
  cache (`cached_tokens`, `cache_write_tokens` from `prompt_tokens_details`; not our disk cache), and the episode
  totals in result.json include `cached_tokens`.
- **Episodes, chosen from goals only, before any call:** VideoUnmaskSwap **val**, episodes 10 onward in order
  (`stress/list_goals.py`, free: one reset each), those whose goal names exactly two coloured cubes; the first 20
  (fewer if the split ends; reported).
- **Run:** arm K only, `openai/gpt-6-astra` (OpenAI, API defaults), the pilot's harness otherwise; `--max-usd 8`
  (estimate ~$6: K's pilot median $0.28 per episode, two-pick episodes run longer); the default per-episode cap ($3,
  unscored). Results in `/workspace/cl/astra_k2pick`, copied to `data/cl_pod/astra_k2pick/`.
- **Per episode:** success; `went_back`; and, for every failure, a label by the rule above with one correction
  written now: re-lifting the first bin **after it was put down** counts as "wrong object at close" (the old re-grasp
  clause stays for re-closes before any open).
- **"K loses track" in an episode** = it failed, `went_back` is "yes", and its logged descriptions after the put-down
  name the first container (or the first cube's colour) as the one to pick now: read by hand and quoted here.
- **Decision rule:**
  - **≥ 2 loses-track episodes** → design option 4's progress section (below);
  - **0 or 1** → the KB already carries progress well enough on this task; the finding is the R-vs-K gap on going
    back (pilot: R 5 of 5 two-pick episodes "yes", K 1 of 5), and we look for K's failures on harder tasks instead.
  - Also reported, not decided on: K's success on two-pick goals with a Wilson interval, the other failure labels.
- **If option 4 is built (constraints fixed now):** facts only (what was lifted, to what height, put down where; which
  goal items remain), with the goal's items parsed generically by `task_items`; no thresholds learned from these
  episodes (no "lifted enough" at 16 cm or any other number read off them); a new prompt version, re-frozen, whose
  results are never mixed with this run's or the pilot's.
- n ≈ 20, one model, one task: a direction. Ask the user before the run (it is paid; OpenRouter answered 402 for low
  credit in the pilot).
- **Goal listing (pod, 2026-10-30 07:33, commit dec1e03, free; `/workspace/cl/goals.jsonl`).** VideoUnmaskSwap val
  has **50 episodes (0–49)**: for 50–59 RoboMME silently built one identical default scene (same cube positions,
  115-frame video, goal) instead of raising, so `list_goals.py` never stopped. Those lines are ignored. Fixed after the
  listing: `pod_checks.in_split` raises for an episode outside the split (`get_episode_num()`), used by both
  `pod_checks.make_env` and `closed_loop_pod.make_env` (stress/ only, not in the prompts hash).
  - Episodes 10–49: 18 one-cube goals (10, 12, 13, 14, 18, 20, 21, 22, 26, 28, 30, 33, 34, 36, 38, 42, 46, 48) and
    **22 two-cube goals** (11, 15, 16, 17, 19, 23, 24, 25, 27, 29, 31, 32, 35, 37, 39, 40, 41, 43, 44, 45, 47, 49); the
    pod session's summary said 16 / 24, its own lists say 18 / 22.
  - **Chosen by the rule (the first 20 two-cube episodes from 10): 11, 15, 16, 17, 19, 23, 24, 25, 27, 29, 31, 32,
    35, 37, 39, 40, 41, 43, 44, 45.** Not yet run.
- **The dec1e03 run, pod report (2026-10-30 08:57; with the gripper-0 bug below; read by the rule: not yet):**
  K **9 / 20** success on two-pick goals (env:success 9, env:fail 9, declared_complete 2: ep 23, 31). Median 9 calls,
  median $0.46, total about $12.5 (ep 16: 33 calls $2.02, ep 17: 31 calls $1.94). `cached_tokens` 0 in every episode.
  Results on the pod in `/workspace/cl/astra_k2pick/`, packed as `/workspace/cl/astra_k2pick_results.tar.gz`
  (md5 728f8e71a08ce33a8b75fe22fd4d9901); not yet on the Mac. No H on these episodes, so 9 / 20 does not say whether
  two-pick goals are hard for every arm or for K.
- **User verdict (2026-10-30):** this run was sent while K's gripper record was wrong, so it cannot answer the question
  and the spend was wasted. From now on every paid run gets a proofread of the exact package and a free preflight that
  compares the model-facing text with simulator truth before the user is asked for go (HANDOFF.md, "Rules").

## K's request layout and the provider's prompt cache (2026-10-30, user request; free, no LLM call)
Question: make K's requests cheap under the provider's prompt cache without lowering accuracy. Background: in the
Astra pilot K sent 38% of H's prompt tokens and paid as much ("Why K's fewer tokens did not cost less" above).
- **K's request now (`closed_loop._fresh`, `controller.user_content`, `Memory.text`):** one system message
  (`system_prompt("K")`: robot text, K's memory section, the notes; the same at every call), then **one** user message:
  `Task goal: …` (left out at the first call, where the view starts with `Goal:`), the memory text
  (`self_view`: changes since the last decision and the last three decisions; then `controller_view`: goal, progress,
  task items, objects with positions and `last seen`, relations, event log of the last 30 lines, joined facts), the
  current front and wrist frames, and `Robot state:` JSON (TCP pose, fingers, elapsed steps, wrist extrinsic, and the
  camera calibration, which is the same at every call). Tokens (o200k_base, a proxy for the model's tokenizer):
  system 865, goal 21, calibration 187, the rest of the state ~130, memory text median 2,089 (of which the event
  log ~1,270) on the rebuilt episodes below; the pilot's K averaged 3,710 prompt tokens per call (244,858 / 66).
- **The provider's rules** (OpenAI GPT-5.6 and later: developers.openai.com/api/docs/guides/prompt-caching;
  openrouter.ai/docs/features/prompt-caching; both read 2026-10-30):
  - shortest cacheable prefix 1,024 tokens; reuse needs the whole rendered prefix up to a breakpoint to match;
    images are part of the prefix (text, images, documents are cached);
  - breakpoints: implicit mode puts them at message ends (the last developer message of the first block, the end of
    each user message); explicit mode (`prompt_cache_options: {"mode": "explicit"}`, top level) only at text blocks
    carrying `prompt_cache_breakpoint: {"mode": "explicit"}`; explicit with no marked block disables caching;
  - price: a cached read 0.1× the input rate, a write 1.25× (30-minute TTL; 2.5× for 12 h), anything else 1×;
    OpenRouter: sticky routing is skipped with a manual `provider.order` (ours names one provider, OpenAI, so all
    calls go there anyway).
- **Why K hit 0%, explained by these rules:** its system prompt (865) is under 1,024, and the next breakpoint is the
  end of the one user message, whose content changes at every call. So every call writes its whole prompt at 1.25×
  and never reads. Two checks against the pilot: 0 cached tokens for K and R (observed), and ~$12.50 / M = 1.25 ×
  Astra's $10 / M input (observed). H's growing conversation ends each turn on a user message, so its history is
  re-read at 0.1×.
- **Free simulation (`tools/cache_sim.py`; tiktoken is outside the whitelist, so tools/):** requests rebuilt on the
  Mac: G's perceived opening videos (train episodes 0–9 of the listing, real perception) through `ArbiterKBSystem`,
  then a scripted two-pick continuation (8 calls: above bin A, close, lift, put down, above bin B, close, lift, done;
  3 mm noise on visible objects per frame; the cube under a lifted bin becomes visible); the memory text built as
  `Memory.text()` builds it and each decision recorded as `Episode.execute` does. Rebuilt mean per call 3,370 or
  3,750 tokens (66 or 255 per image; pilot 3,710). The cache model is the rules above plus two assumptions (a
  request reads the longest earlier breakpoint prefix it starts with; an
  episode's calls reach a machine holding its earlier writes). Image tokens per 256-px frame are not known for
  Astra: 66 and 255 both run.

  | layout | mode | read share | cost vs now (66 / 255 image tok) | text the model reads |
  |---|---|---|---|---|
  | now (`v1`) | implicit | 0 | 1.00 / 1.00 | — |
  | now (`v1`) | explicit, no breakpoint | 0 | **0.80 / 0.80** | byte-identical |
  | (a) reorder only: goal first | either | 0 | 1.00 or 0.80 | same text, goal moved |
  | `fixed_first`: goal + calibration first, breakpoint after them | explicit | 0.28 / 0.25 | **0.61 / 0.63** | same keys and values; see below |
  | (b) append-only log + rewritten current state | explicit | 0.63 / 0.57 | **0.45 / 0.49** | changed (tokens +22%) |
  | (c) one growing conversation, full K content each turn | implicit | 0.78 / 0.78 | **1.03 / 1.06** | changed (tokens 3.7×) |

  - **(a) as asked, a pure reorder, saves nothing:** the only parts that are the same at every call are the system
    prompt and the goal, 886 tokens, under the 1,024 minimum. Only explicit mode's billing (no write premium on the
    changing part) helps: −20%, with the text byte-identical.
  - **`fixed_first` needs the calibration moved too:** 865 + 21 + 187 = 1,073 tokens by o200k, **49 over the minimum**;
    the model's own tokenizer may count differently, so this is to be checked on the first paid call
    (`cached_tokens` ≈ 1,070 from the second call of an episode; if 0, the fixed block is under 1,024 and the result
    is explicit mode's −20% only).
  - **(c) is out:** 78% of its tokens are cache reads, but it sends 3.7× the tokens (each turn's full view stays in the
    history), so it costs more than now. A negative result; not built.
  - **(b) saves the most but changes what the model reads** (and its log grows with the episode, so its share
    depends on episode length); it needs the offline test below before any closed-loop use.
  - In dollars on the pilot's K input ($3.06): explicit ~$2.45, `fixed_first` ~$1.9, (b) ~$1.4–1.5 (scaled by the
    ratios; not measured). Output ($0.46) is unchanged by any layout.
- **Built (defaults unchanged, so a run without the new flags sends what it sent before):**
  `controller.user_content(..., layout="fixed_first", breakpoint=)`: `Task goal: …` at every call, then
  `Robot state, fixed part (camera calibration): {wrist_camera_K, front_camera_K, front_camera_extrinsic…}` (marked
  when `breakpoint`), the memory text unchanged, the frames, `Robot state: {the other keys}`. Differences from `v1`
  in what the model reads: the goal line also at the first call (where the view already has `Goal:`); the state's
  keys split into two JSON objects with that heading. Tests: the same keys, values, memory text and images as `v1`;
  the fixed block byte-identical across calls. Pod runner: `--layout v1|fixed_first`, `--prompt-cache
  implicit|explicit` (OpenAI only; fresh arms only, H keeps its implicit cache), both recorded in result.json.
  `closed_loop.py` and `controller.py` are in the prompts hash, so params are re-frozen (below); **any run with
  `--layout fixed_first` is a new prompt version, never mixed with the pilot's or the two-pick test's results.**
- **(b), designed, not built; its offline test, fixed before running (needs the user's go: Qwen, paid):**
  - Layout (explicit mode): the fixed block as in `fixed_first` (breakpoint); then one appended message per call,
    never edited afterwards (breakpoint at its end): `Memory log, t=a–b (append-only):` the new event-log lines
    (the whole log, never truncated), the decision just taken, and each belief change since the last decision written
    as `Correction at t=…: Inside(cube_x) now bin_k p; supersedes the belief stated at t=…`; last, rewritten every
    call: `Current state (authoritative; it overrides anything above):` the view without its event log
    (`self_view`, goal, progress, task items, objects, relations, joined facts), the frames, the rest of the state.
    The current state is always stated in full, because R (a log the model must reduce itself) did worst in the
    pilot (6 / 10, median 16 calls).
  - Data: synthetic VideoUnmaskSwap train seeds 100–129 (as Batch A: clean, text only) with two-cube goals and
    `tools/cache_sim.py`'s scripted two-pick continuation; the robot follows the script whatever the model answers (open
    loop), and the model is asked at each of the 7 calls before "done".
  - Arms: K `v1` (now) and K (b), Qwen 3.6-35B-A3B, the closed loop's Qwen settings; the same frames (none: text only).
  - Score per call: `target` names the container the script acts on at that call (bin A for calls 1–4, bin B for
    5–7), and the waypoint's xy is within 3 cm of that container's true centre (as M4). Primary: the paired
    difference (b − v1) in the share of calls with both right, 95% paired bootstrap CI over episodes.
  - Rule: (b) may replace `v1` for a closed-loop run only if the CI's lower bound is above −0.05 (a margin chosen
    now as a design choice, not read off any data); otherwise (b) is dropped and the result is recorded as it is.
    Also reported, not decided on: calls 5–7 alone (after the put-down, where losing track would show).
- **Re-frozen:** params a810942859b9 (unchanged), prompts 7148880935b2 (`python scripts/run.py --freeze --force`).
  The two-pick test runs from its own package (dec1e03, prompts 82d83ed1f52d) with the `v1` layout.
- **Not touched:** the two-pick test's section above and the pod upload package. (This section's first draft was
  swept into commit 8eee38f by the other session's `git add`; this commit is its final text.)
- **Approved by the user (2026-10-30), one change before the run: `--max-usd 15`, not 8.** All 20 are two-pick
  episodes, which take more steps than the pilot's mix (K averaged ~$0.35 per episode over one- and two-pick goals);
  hitting the cap only stops the run and costs a resume, and the cap is not spend. OpenRouter balance ≥ $20 before
  starting. Run from `/workspace/score` at dec1e03 (no re-upload; the episode guard of e75ff1f does not affect these
  20). The pod upload archive is not rebuilt until this test's result is in NOTES (the cache-optimisation session
  works in parallel).

## Bug: "keep" (gripper 0) recorded as open in K's self-memory (found 2026-10-30 by the cache session; checked here)
- `closed_loop.py` passed `action[6] < 0` to `self_memory.record` as `gripper_closed`, so gripper 0 ("keep") while
  holding was recorded as **open**; `self_memory.intent` then reads open-while-held as a "release", which never
  happens, and K's view shows "release -> not done yet". Astra's K had 21 such steps in the pilot. It corrupts exactly
  the progress record the two-pick test is about. The offline harness (`harness.py`) passes real booleans and is not
  affected; neither are H and R (no self-memory).
- **Consequences, fixed now:** the pilot's K rows and the dec1e03 two-pick K run (started 2026-10-30 07:51 on the pod)
  measure K **with this bug**. That run is still read and reported by its rule, labelled "dec1e03, with the gripper-0
  bug", but **the decision on option 4 is taken on the fixed version**, run K-only on the same 20 episodes (with the
  user's go), and go-backs are compared per episode, bug vs fixed. The two versions' results are never pooled.
- **Fixed (2026-10-30):** `closed_loop.gripper_closed(command, before)`: −1 closes, +1 opens, **0 keeps the state
  before it**; the state starts from the robot's finger positions (`fingers_closed`: total opening < 1 cm) and then
  follows the commands. K's self-memory now records a keep while holding as holding. Test: a close, two keeps, an open
  and a keep record closed = [T, T, T, F, F] (the bug gave [T, F, F, F, F]). **This changes K's text: new prompt
  version, re-frozen (params a810942859b9, prompts 4750fdef3d00; includes the cache session's eff65c8, whose layout
  defaults are unchanged).** Results under 4750fdef3d00 are never pooled with dec1e03's (82d83ed1f52d) or the pilot's.
- **Fixed-version run, not yet approved:** K only, `openai/gpt-6-astra`, the same 20 episodes, same cap ($15), same
  rule and decision rule as the dec1e03 run; extracted into its own directory (`/workspace/score_fix`), started only
  after the dec1e03 run has finished (extracting over `/workspace/score` would change the running job's code).
  Compared per episode with the dec1e03 run: success, `went_back`, and the confirmed "loses track" episodes.

## K2: initial state, change log, current state (2026-10-30, user design; replaces the (b) layout above)
Why: K's view mixes what changed, what is true now and the history, in an order no reader would choose, and it starts
with text that changes every call, so the provider can never cache it ("K's request layout and the provider's prompt
cache" above). The user's design, a new arm K2 (`arbiter_kb/view_k2.py`), with K's KB and K's perception unchanged:
- **A. Initial (fixed for the episode):** `Task goal: …`, the camera calibration, and every object as observed in the
  first frame (`seen at (x, y, z)` or `not seen yet`), from the KB's trace (`system.rows[0]`).
- **B. Change log (one message per call, appended, never edited):** the evidence and events since the previous call:
  moves as displacement and end point (`bin_1 moved by (dx, dy, dz) to (x, y, z) (t=a-b)`), objects seen (with
  where), no longer seen, covers, close passes, the robot's pick-ups and put-downs, and each Inside belief when it first
  appears and when it changes (`correction: … supersedes … stated at t=…`; a change = another top container or the top
  probability down by ≥ 0.1, self_memory's existing rule). The first frame's sightings are in A, not repeated in B.
- **C. Current state (rewritten every call, last, "authoritative"):** the hand and what it holds; every object
  (visible with position and seen time, held, or hidden with its Inside probabilities); relative positions (left to
  right, near to far, On / Near); the goal's coloured items in order, each with where the memory puts it and what the
  robot did to the container believed to hide it (`bin_1: picked up at t=174, put down at t=234` / `not picked up
  yet`); counts of the robot's actions; a summary of what happened (narrative's timeline plus pick-ups and
  put-downs). Then the current frames and the changing state.
- K2 has no `self_memory` section (the controller's own decisions): C's task lines and the log say what was done.
- **What is new against K, so the test cannot attribute a difference to one part:** the order (A → B → C), the
  log's wording (displacements), the per-item task lines (option 4's progress facts, facts only, no threshold) and
  the dropped decisions section. If K2 differs from K, a follow-up would separate them.
- **Requests:** system (K2's memory section in `controller.MEMORY["K2"]`), then user messages A, B1…Bn, C; in
  explicit cache mode (`--prompt-cache explicit`) A and every B carry a breakpoint, C none. Tests: every earlier message
  is byte-identical in the next request; C comes last and has no breakpoint; a move is logged once with its
  displacement.
- **Free simulation (`tools/cache_sim.py`, the rebuilt G episodes above, 10 × 8 calls):** K2 sends 1.56× K's tokens
  (the log is never truncated: the opening video's first message is ~2,400 tokens, cached after the first call), 0.67
  of them cache reads; cost vs K now **0.53** (66 image tokens) / **0.55** (255), against `fixed_first` 0.61 / 0.63.
  (Corrected below: these are against K with its implicit cache; against K in explicit mode, 0.66 / 0.69.)
  C is median 778 tokens (max 1,132). The opening log is mostly moves of a few millimetres (perception noise in the
  rebuilt episodes; the pilot's real logs are on the pod and not yet read).
- The scripted continuation moved to `scripts/two_pick.py` (shared by `tools/cache_sim.py` and the test below); it now
  records "keep" (gripper 0) as the state before it (`closed_loop.gripper_closed`, the fix in 038ddf1), so the
  simulation's K text changed slightly (cost ratios moved by ≤ 0.01).

### K vs K2 offline: the test, fixed before running (needs the user's go: Qwen, paid)
- **Data:** synthetic VideoUnmaskSwap train episodes 100–129 (`SyntheticSource(swap_steps="mix", lane_offset="mix")`,
  as Batch A: clean, text only, no frames), goal rewritten to two cubes (`gt.targets[0]`, then the next cube in the
  ground truth's order), then `scripts/two_pick.py`'s seven scripted actions (open loop; 3 mm noise on observations
  after the video). Both arms read the same KB at every call; K's request is `user_content(..., layout "v1")` (the
  closed loop's K now, with the gripper fix), K2's is `view_k2.request` (no breakpoints offline).
- **Model:** Qwen 3.6-35B-A3B on CoreWeave with the closed loop's Qwen settings (temperature 0.6, top-p 0.95, top-k 20,
  presence penalty 1.5, 32k cap, default thinking). One call per (episode, call, arm): 30 × 7 × 2 = 420 calls; cap
  `--max-usd 1.5` (estimate ~$1 from Batch A's $0.0015 per call and longer prompts).
- **Score per call:** the right container = the goal's first container for calls 1–3 (before its pick is done), the
  second for calls 5–7 (after the first was put down); `both` = `target` names it and the waypoint's xy is within
  3 cm of its true centre. Call 4 (first container just lifted: put it down or go on are both reasonable) is reported
  only (which container `target` names).
- **Primary:** per episode, the mean of `both` over calls 1, 2, 3, 5, 6, 7; K2 − K with a 95% paired bootstrap CI over
  episodes (10,000 resamples, seed 0). **Rule: K2 may replace K for a closed-loop run only if the CI's lower bound
  is above −0.05** (the margin from the (b) test above, a design choice not read off any data). Otherwise K2 is not run
  in the closed loop, and the result is recorded as it is.
- **Also reported, not decided on:** calls 5–7 alone (where losing track would show), call 4's targets, mean prompt
  tokens per arm, spend.
- Script: `scripts/k2_offline.py` (`--print` shows episode 100's requests for free; `--call` runs). Results in
  `results/runs/k2_offline_<time>/` (rows.csv, summary.json, meta.json with commit and params hash).
- **Re-frozen:** params a810942859b9, prompts 946e17f79534 (K2's memory text, `view_k2.py`, now in the prompts hash, and the runner flags); K's
  own text is unchanged by this commit.
## The dec1e03 two-pick K run, read by the rule (2026-10-30; with the gripper-0 bug; belongs to "Does K lose track")
- **Reading the dec1e03 run by the rule: one change, fixed before looking at its data (2026-10-30).** The pod is
  stopped and object positions after reset exist on the Mac only for episodes 0–9 (`vus_objects.json`); `went_back`
  needs the second right object's reset position. A positions-free variant (`callout.went_back_free`):
  - first pick = the first hold lifted ≥ 5 cm after which the env stays "ongoing" (lifting a wrong bin ends the
    episode `env:fail` on that step in this task: pilot K ep 0, K ep 8);
  - P = the first open after it (the put-down);
  - "yes" = a later hold lifted ≥ 5 cm whose close is within **5 cm** of P (container centres never come closer than
    5.1 cm in RoboMME VideoUnmaskSwap, G measurement above), else "no"; "n/a" without a first pick or a put-down.
  - **Checked first on the pilot's 30 episodes against `went_back`** (both computable there); the agreement is
    reported, and if it disagrees on any two-pick K episode the variant is not used.
  - Then, as fixed: every failure with "yes" is confirmed or not from K's logged descriptions after the put-down,
    quoted here. The result is reported as "dec1e03, with the gripper-0 bug"; it does not decide option 4 (the user's
    verdict above), it describes what K did under the bug.
- **dec1e03 run, read by the rule (2026-10-30; `data/cl_pod/astra_k2pick/`, free). WITH THE GRIPPER-0 BUG.**
  - **Variant check on the pilot:** `went_back_free` agrees with `went_back` on all 5 two-pick K episodes and on 13 of
    15 two-pick episodes overall; the 2 disagreements are not K's (H ep 1: "ambiguous" is invisible without poses;
    R ep 1: the dropped bin landed 6.3 cm from the put-down). So the variant is used, as fixed.
  - K 9 / 20 success. `went_back_free` "yes" in 5 episodes: 16, 31, 32, 45 (failed) and 41 (succeeded; a missed grasp
    re-tried before any put-down was read as one, a behaviour only).
  - **Confirmed "loses track" from K's descriptions after the put-down: 2 (ep 16, ep 32).**
    - ep 16 (red, then green): lifted bin_0, lowered it with "keep" and never opened, then for 25 calls: "I will lift
      bin_0 to complete the first pickup before approaching bin_2", "bin_0 has not yet been confirmed picked up",
      finally re-grasped bin_0 → fail (33 calls, $2.02).
    - ep 32 (green, then red): the same pattern: lowered bin_0 with "keep", "I will lift it higher to complete the first
      pickup", released at 7.5 cm, then "I will lower onto bin_0 and close the gripper to grasp it first" → fail.
    - **Both follow exactly the bug's footprint:** a held bin lowered with gripper 0, which the self-memory recorded as
      an unexecuted "release", and K then said the first pickup was not complete. So these 2 are most likely the bug's.
  - **Not losing track (descriptions name the second container):** ep 31: released bin_0 at 0.21 m directly above
    bin_3 (the second), grasped there, declared complete (a put-down error and a premature "done"); ep 45: K's memory
    said "bin_1 ... has moved" to where K had just put bin_2 down, K lifted there → fail (a tracking error caused by
    the robot's own put-down).
  - **All 11 failures, one label each:** wrong container from K's memory **7** (first pick: 15, 29, 39; second pick:
    11, 19, 37, 45); loses track **2** (16, 32; the bug's footprint); premature "done" **2** (23, 31). (Wrong-container
    reads use the env: a wrong lift ends `env:fail` on that step; poses after reset are not on the Mac for 11–45.)
  - **Reading:** by the rule's count (≥ 2) this would say "design option 4", but both cases carry the bug's footprint,
    and the user's verdict stands: this run does not decide option 4. What it adds: **on two-pick val episodes 11–45
    K's main failure is the memory's wrong container (7 / 11), not progress**; and one new failure kind, the KB
    re-identifying a bin after the robot moved one (ep 45). The fixed-version package (K fixed + H, maybe K2) answers
    the progress question; H on the same 20 episodes says whether 9 / 20 is K's or the task's.

## K2 fixes after the review (2026-10-30, user go; before the K vs K2 offline test, which is unchanged)
A review of the K2 sketch, checked against K2's real output (synthetic episode 100, call 5) and OpenAI's docs:
- **Fixed in the KB (all arms, K included): rule R4 reveal** (`rules.r4_reveal`, the "R4 reveal" the spec had
  deferred). Before: after bin_1 was lifted and cube_blue was seen at its spot, cube_green still read
  `bin_2 0.92 | bin_1 0.07`. Now: a covered cube seen again while a container is held, within `r_cover` (existing) of
  where that container was grasped, gives an event `reveal(container, cube)`, and every other hidden cube loses its
  mass on that container (`Distribution.without`, renormalised; R5's at-most-one-cube-per-container assumption; no new
  parameter). A lifted container that shows nothing changes nothing (an unseen spot is not an empty one). K's event
  log gets one sentence for it. The opening videos have no grasps, so no earlier offline result changes.
- **Fixed in K2's view (`view_k2.py`):**
  - grasp and release lines give numbers, not labels: where the hand closed, where it opened, and the highest hand z
    while holding (the "lifted how high" of option 4's constraint); "put down at table level" (actions.py's label from
    z_low) is gone; a release onto an object names it;
  - every decision of the controller is logged (`[d<n>] you: waypoint …, gripper … after it, target …`), not one;
  - each belief line has its own id (`[b<n>]`), stamped with the time the belief changed; a later line says which it
    replaces; a line is written when the top container changes, the top falls by ≥ 0.1, or the set of candidates
    changes (the last clause is new, so R4's removals are logged);
  - "nearest to the robot" now says "the robot's base (-0.615, 0)", as `narrative.py` computes it;
  - the summary in Now lists only semantic events (covers, close passes, grasps, releases, reveals). The old one came
    from `narrative._timeline`, which joined millimetre noise moves into a false "bin_0 and bin_1 and bin_2 and bin_3
    exchanged places (t=133-247)";
  - Now starts with time, hand, finger opening and what is held, then the goal's items, then an object table.
- **Already true in the real output, sketch errors only:** move lines carry ids and displacements; distributions
  print `elsewhere`; "moved back" was the sketch's wording.
- **Breakpoints (OpenAI docs, read 2026-10-30):** at most 4 cache writes per request; in explicit mode lookups at the
  first 2 and the latest 50 breakpoints. K2 writes one new log message per call and the previous call's last
  breakpoint is always among the latest 50, so the limit does not bind.
- **Not fixed:** millimetre moves are still logged (the motion detector's own output); dropping them needs a noise
  threshold, to be set from train-data perception noise if at all, not from any pilot.
- Simulation after the fixes (`tools/cache_sim.py`, 10 × 8 calls; against K implicit, see the correction below): K2 costs **0.51 / 0.54** of K now (66 / 255 image
  tokens), Now median 650 tokens (max 882).
- Re-frozen: params a810942859b9, prompts 08c6cb1474ce. `make check`: tests pass; quality fails only on
  `arbiter_kb/callout.py` (253 lines > 250), changed in ada07a5 by the other session, not touched here.

## Move threshold from real perception noise, and a noisy condition for K vs K2: rules fixed before measuring (2026-10-30, user decision)
Why: the motion detector (motion.py v2, speed-based) reports moves of a few millimetres while a container stands
still; K's event log and K2's log carry them as "moved". The user's rule: below a threshold measured on real
perception noise, no move is reported (K and K2 alike).
- **Data:** G's perceived train episodes (`PerceivedSource().episodes("train")`, episodes 0–9 of each setting),
  container slots only (moves are reported for containers only). Never the synthetic 3 mm, never any pilot.
- **Still frames, from the simulator (G's oracle rows; ground truth, train only, used to set a parameter):** a
  container is still at frame t when its oracle xy differs from frame t−1 by less than 0.1 mm; a still segment is a
  maximal run of still frames of one container in which the perceived track is seen in every frame.
- **Measure:** within each still segment, the perceived xy displacement |p(t) − p(s)| for every pair of frames
  s < t (what a "move" between those frames would report while nothing moved). Reported: segments, pairs, the 50th,
  95th, 99th percentiles and the maximum.
- **Threshold (the rule): T = the 99th percentile of those pair displacements.** A move whose start-to-stop xy
  displacement is below T is not reported in K's event log, in narrative's move counts and exchanges (K's joined
  facts), or in K2's log. The evidence stays in the KB (traceability); the motion detector itself is unchanged.
- **Noise model for the offline test, from the same frames:** residual = perceived xyz − the segment's median
  perceived xyz, pooled over all still frames. The noisy condition adds to every visible object in every frame (the
  synthetic video and the scripted continuation) one residual drawn at random from that pool (seeded per episode).
  An empirical draw, not a Gaussian, because real tracks jump (motion.py's docstring). The clean condition adds none
  (the continuation's synthetic 3 mm is removed).
- **K vs K2 offline test, amended (supersedes its rule above):** the same 30 episodes run in both conditions, clean
  and noisy (30 × 7 × 2 arms × 2 conditions = 840 Qwen calls, ~$2, cap $3). **K2 passes only if, in each condition
  separately, the 95% paired bootstrap CI of K2 − K has its lower bound above −0.05.** Otherwise K2 is not run in the
  closed loop. Reported per condition as before.
- Both changes alter what K and K2 read: new prompt version, re-frozen, never mixed with earlier results.

### K vs K2 offline, redesigned so that every scored call depends on memory (2026-10-30, goal review; supersedes the scoring above)
The goal review found that of the six scored calls only two (where to go first, where to go after the put-down)
needed memory; the other four had the hand already above the right container. Redesign, fixed before running:
- **Script (`scripts/two_pick.py`, open loop):** after the opening video the hand is at home (0, 0, 0.45). Asked at
  five states, each with the hand **not** above the container the goal needs:
  1. home, nothing done — right: A (the container hiding the goal's first cube);
  2. hand above D1 (the container nearest A in xy, other than A and B), open — right: A;
  then, unasked: above A, down and close, lift to 0.22, put A down to one side, back home;
  3. home, A put down — right: B (the container hiding the second cube);
  4. hand above where A was put down — right: B (going back to A here is the "lost progress" failure);
  5. hand above D2 (the container nearest B, other than A and B) — right: B.
  Moves to home and to D1 / D2 / A's spot are recorded as decisions with no target (they are the script's, not
  the model's); the pick of A is recorded with target A.
- **Score per asked call:** `both` = `target` names the right container (brackets and spaces stripped, so `[bin_1]`
  = `bin_1`) and the waypoint's xy is within 3 cm of its true centre. An answer that does not parse into a valid
  action scores 0 (it is recorded, never crashes the run; a replay from the cache gives the same rows).
- **Pairing:** the comparison uses only (condition, episode, call) triples where both arms have an answer; triples
  lost to an API error or the budget are counted and reported, never silently dropped.
- **Primary, per condition (clean, noisy):** per episode the mean of `both` over its paired calls; K2 − K with a 95%
  paired bootstrap CI over episodes (10,000 resamples, seed 0). **K2 passes only if the lower bound is above −0.05 in
  both conditions.** Also reported: calls 3–5 alone (after the first pick), the share of call-4 answers naming A,
  mean prompt tokens per arm, spend.
- Calls: 30 episodes × 5 asks × 2 arms × 2 conditions = 600 (cap $3). Needs the user's go.

### Move threshold: measured, applied; K2 and test fixes (2026-10-30)
- **Measurement (`scripts/still_noise.py`, rule above):** the first run paired perceived and oracle containers by
  slot index; the slots are numbered differently (perceived slot 0 = oracle slot 1 in episode 0), so "still" was
  wrong and its numbers (p99 0.178 m) are **discarded**. The track's object is now the oracle container nearest it in
  xy at that frame, a segment keeps one object (the rule's meaning: the displacement a still container shows).
  Result: 90 train episodes, 723 still segments, 417,957 pairs; xy displacement p50 0.0020 m, p95 0.0279, **p99
  0.0416 = T**, max 0.112. Residual std (x, y, z) 0.0065, 0.0056, 0.0057 m over 17,428 still frames; the pool is
  `data/g_perceived_still_residuals.npy` (git-ignored, rebuilt by the script).
- **Applied:** `DetectParams.move_min = 0.0416`; `motion.reported(xyz0, xyz1)`; moves below it are left out of K's
  event log, `narrative.moves` (so K's joined facts' move counts and exchanges) and K2's log. The detector and the
  evidence are unchanged.
- **"Fixed K" is not only the gripper fix.** Against dec1e03's K, the fixed K differs by: (1) the gripper-0 fix
  (038ddf1), (2) R4 reveal, which changes K's beliefs and adds a log sentence (f6fad9a), (3) the move threshold, which
  removes move lines and changes move counts in K's text (this commit). A difference between the fixed-version run and
  dec1e03 cannot be put on the gripper fix alone.
- **K2:** belief ids are numbered in time order; a held container's moves between its grasp and its release are not
  logged (the grasp and release lines give them); the initial state lists only the first frame's objects (spare
  tracker slots that never find an object are never listed; objects first seen later enter through the log).
- **Offline test script (eval review):** a malformed answer scores 0 and never raises (also on a cache replay);
  `[bin_1]` = `bin_1`; API errors and budget stops are recorded as rows with an error, and the comparison pairs only
  calls both arms answered (counts reported); the redesigned probe script and the two conditions as fixed above.
  Tests: `tests/test_k2_offline.py`.
- **Cost, corrected (user):** the earlier "K2 = 0.53–0.55 of K" compared K2 in explicit mode with K in its implicit
  cache mode (K now). Against K in explicit mode (−20% for K with no content change) the same simulation gave
  **0.66–0.69**. After the review fixes: 0.51–0.54 vs K implicit = 0.64–0.68 vs K explicit. **Now, with the move
  threshold (in both K and K2) and real-noise residuals in the scripted continuation instead of the synthetic 3 mm:**
  K's prompt fell to 235k tokens (from 269k), K2 sends 1.21× K's tokens; **K2 explicit = 0.45 / 0.49 of K implicit
  and 0.56 / 0.61 of K explicit** (66 / 255 image tokens).
- **Re-frozen:** params b4caad02074a (move_min added), prompts 9eefe9411062.

## K's view after a cube is seen again: one fact line instead of an open distribution (2026-10-30, user; rule before code)
Found in episode 100 call 5 (a229e35): cube_blue is visible after bin_1 was lifted (R4 reveal at t=232), yet K's
Relations still print `Inside(cube_blue) bin_1 0.92 | bin_2 0.07 | elsewhere 0.01` and its joined facts say "bin_1 may
hold the blue cube (0.92)" (the open question 1a kind of stale belief). K2 already shows only hidden cubes'
distributions and is not changed.
- **Rule (K's view and `narrative`'s joined facts only):** a cube whose Inside belief exists but which is not hidden
  now is no longer listed as an open distribution. Its Inside line becomes one fact:
  - after an R4 reveal: `[Inside(cube)] was under <container>, revealed t=<t> [<event id>]`;
  - seen again without a reveal: `[Inside(cube)] seen again at t=<t>, now at x, y, z; the earlier belief
    (<container> <p>) no longer applies` (t: its last appear; the position is its current one).
  Joined facts: a container no longer "may hold" a cube that is not hidden; after a reveal its line says "hid <the
  cube> (revealed t=<t>)", and the cube's line says "visible; it was under <container> (revealed t=<t>)".
- The KB's distributions are unchanged (only the text); `self_memory` is unchanged.
- Two notes on the offline test (user): (1) the open-loop script's moves (home, D1, D2, A's new spot) are recorded as
  decisions with target None, so both arms show them as "your" decisions (K: "move -> moved"; K2: `[d<n>] you: …`);
  a side effect of the open loop, the same for both arms. (2) K keeps the "placed it down at table level" label
  (actions.py's z_low label): K is the existing arm and is not changed for this; K2 gives the numbers instead.
- **Done:** `narrative.revealed` / `narrative.seen_again`; K's Relations and joined facts follow the rule; tests
  (revealed: `Inside(cube_red) was under bin_0, revealed t=…`, "hid the red cube", no "may hold"; seen again without a
  reveal: `seen again at t=8, now at …`). `scripts/k2_offline.py --print [--noisy]` writes
  `results/runs/k2_print/ep<first>_<clean|noisy>.txt`. Re-frozen: params b4caad02074a, prompts 643e351eb3f1.

## Model screen: the small model changes, rules fixed before any call (2026-10-30, user decision)
**Decided (user, 2026-10-30):**
1. Every model goes through OpenRouter except Gemini Robotics-ER 2, which is the last resort (it needs a Google key),
   considered only if no OpenRouter model passes.
2. **Claim 1 stays; its small model changes.** `qwen/qwen3.6-35b-a3b` (3B active) is too small: val pilot 1 / 24, H
   localises 1 / 9 (NOTES "Layer 1 on real scenes"). The small model becomes a mid-size model chosen by this screen.
   Claim 1 itself ("a smaller model with the KB reaches the frontier model's success") is unchanged. GOALS.md still
   names the old primary; that line is to be updated when the screen has chosen (the user's edit or go).
3. The frontier model is not fixed either: the screen also sets the ceiling.

**Question.** With memory that is right, can a model write the right waypoint at each stage of a two-pick
VideoUnmaskSwap episode? A coarse screen (5 episodes × 7 stages × 2 passes per model), not a claim.
`scripts/model_screen.py` (states, calls, output), `scripts/screen_rules.py` (rules, mock model), tests in
`tests/test_model_screen.py`.

**States.**
- Synthetic VideoUnmaskSwap train episodes built as `scripts/k2_offline.py` builds them (`SyntheticSource(swap_steps=
  "mix", lane_offset="mix")`, two-cube goal "pick up the container that hides the X cube, then … the Y cube"), clean
  (frames are simulator truth), containers moved to RoboMME's height (bin origin z = 0.052; the synthetic source puts
  them at 0.03).
- **Episode selection (fixed now):** from seed 100 upward, the first 5 episodes in which the KB's top Inside belief for
  both goal cubes is the true container with p ≥ 0.8 (the screen tests execution, not doubt; with `lane_offset="mix"`
  some memories are coin flips: 101, 103, 109 have 0.49–0.50). This gives **100, 102, 105, 107, 108** (one with 4
  containers, four with 3).
- Request: the closed loop's K request (`controller.user_content`, layout v1, system prompt K), **text only**: the
  synthetic source has no images, so the K arm's two current frames are missing (a stated deviation; every candidate
  takes images, for the closed loop later).
- Seven stages, each asked once: (1) hand at home → above A; (2) above A → close on A; (3) closed on A → lift;
  (4) lifted → put A down; (5) after the put-down → above B; (6) above B → close on B; (7) after the scripted lift
  of B → done (`waypoint: null`). Between stages the robot follows `two_pick.plan` (open loop: the script acts whatever
  the model answers; the KB records each scripted action as "your" decision, as in k2_offline) at RoboMME's heights:
  above / lift 0.2 (`solve_pickup_bin`), release 0.07 (`solve_putdown_whenhold`), **grasp 0.045, not RoboMME's 0.062**
  (next point). After the release the container settles at z 0.052.
- **Found while proofreading the print (a KB issue, not fixed here; `actions.py` belongs to another window):** the KB
  records a `pick_up` only when the gripper closes with the hand below `z_low` = 0.05 m (`actions.py`, read off the
  dataset's cube picks: pick 0.020). RoboMME grasps a container at TCP 0.062. With the scripted grasp at 0.062, every
  later view said "pick_up bin_1 -> not done yet" while the container was held at 0.2 m. **The closed-loop K arm has
  the same rule**: a container closed on above 5 cm is never "picked up" in K's self-memory (Astra's first closes on
  containers were |dz| 0.8–2.5 cm (medians per arm) from the centre z 0.052, sign not recorded in NOTES, so some may
  have been above 5 cm). This may matter for K's two-pick
  progress; to check on the dec1e03 / pilot K logs before the next K run. The screen scripts the grasp at 0.045
  (inside the grasp band below) so its memory is right.

**Rules (per answer; every number from RoboMME's code, github.com/RoboMME/robomme_benchmark @ 69a2fe2,
`src/robomme/robomme_env`, or marked "ours").** Geometry: `build_bin` makes an open box turned upside down, inner side
and wall height 2.5 × cube_half_size = 0.05 (cube_half_size 0.02, ManiSkill `PICK_CUBE_CONFIGS["panda"]`), walls 5 mm,
floor plate 4 mm (now on top, its top at 0.054), a 4 cm handle cube at the origin (0.032–0.072); origin at rest z =
0.052. Success: `is_bin_pickup` = container origin z > 0.15; two-pick task list = pick A, `is_bin_putdown` (origin
≤ 0.07, not grasped, TCP z > 0.05), pick B; picking any other container fails.
- Every action must pass `controller.check`'s workspace box and reach (as the loop does); target and cites not scored.
- An answer that does not parse into a valid action or `null` is wrong (counted in the JSON-valid rate); a `null` at
  stages 1–6 is wrong ("premature done"); at stage 7 only `null` is right.
- **Above (stages 1, 5):** gripper 0 or 1; xy ≤ 0.03 m from the container's centre (walls' outer half width
  0.025 + 0.005); 0.10 ≤ z ≤ 0.40 (0.10 = handle top 0.072 + `FINGER_LENGTH` 0.025 from `solve_pickup_bin`, rounded
  up; 0.40 ours, below the script's home 0.45). Also right: a direct grasp (gripper −1 under the grasp rule).
- **Grasp (stages 2, 6):** gripper −1, or 0 / 1 ("descend first, close next": the prompt lets the next call close);
  0.02 ≤ z ≤ 0.077 (ours on both ends: fingertips off the table, at most 5 mm above the handle top 0.072); xy ≤ 0.02 m
  if z ≥ 0.054 (the handle's half width) else ≤ 0.01 m (fingers 0.08 m open, prompt "0.04 m each", must straddle the
  0.06 m walls). Not scored: yaw (the 4 cm handle fits the open gripper at any yaw; the 6 cm body does not at 45°,
  diagonal 8.5 cm; the K view gives no container yaw).
- **Lift (stage 3):** gripper −1 or 0; 0.143 ≤ z ≤ 0.40: 0.143 = 0.15 + (scripted grasp 0.045 − origin 0.052), the
  TCP height at which the origin passes `is_bin_pickup`'s 0.15. xy free (a sideways lift still lifts).
- **Put down (stage 4):** release: gripper 1, 0.06 ≤ z ≤ 0.12 (the container's bottom reaches the table at TCP 0.062
  minus 2 mm, ours; a drop ≤ 5.8 cm, ours; RoboMME releases at 0.07), xy ≥ 0.075 m from every other container's centre
  (`spawn_random_bin`: 2 × half 0.0275 + min_gap 0.02). Also right: carry first (gripper −1 / 0, 0.143 ≤ z ≤ 0.40,
  the hand moved ≥ 0.03 m in xy (ours), ending ≥ 0.075 m from every other container).
- Scored against the truth at that state (A held at the hand for stages 3–4; B and the others where they are).
- Checked on the mock: RoboMME's own script (`screen_rules.canonical`) passes every stage of every selected episode
  (test).

**Models and settings (OpenRouter models API, 2026-10-30; every one lists image input).** OpenRouter's defaults: no
temperature, seed, reasoning or provider is sent (OpenRouter routes; the provider used is logged per call); sent:
`max_tokens` = min(32000, the model's max output on OpenRouter) and `usage.include`. Pass 2 has its own disk cache
(`results/llm_cache/model_screen/pass{1,2}`), so it is a second sample, and a rerun resumes from the cache.

| model | group | total / active params (B) | $ / M in / out | max_tokens |
|---|---|---|---|---|
| meta-llama/llama-4-scout | small | 109 / 17 | 0.10 / 0.30 | 16384 |
| nvidia/nemotron-3.5-super-vl | small | 120 / 12 | 0.15 / 0.50 | 32000 |
| google/gemma-4-31b-it | small | 30.7 dense | 0.20 / 0.40 | 16384 |
| qwen/qwen3.5-122b-a10b | small | 122 / 10 | 0.26 / 2.08 | 32000 |
| z-ai/glm-4.6v | small | 106 / 12 (**unverified**: not on the OpenRouter page) | 0.30 / 0.90 | 32000 |
| qwen/qwen3.6-27b | small | 27 dense | 0.32 / 3.20 | 32000 |
| google/gemini-4-argon | strong | not published | 2.00 / 10.00 (until 2026-11-30) | 32000 |
| openai/gpt-6-sol | strong | not published | 2.00 / 10.00 | 32000 |
| anthropic/claude-opus-5.5 | strong (added 2026-10-30, user) | not published | 4.00 / 20.00 | 32000 |
| openai/gpt-6-astra | strong (reference; pilot) | not published | 10.00 / 50.00 | 32000 |

Parameter counts from each model's OpenRouter description / Hugging Face id. Not chosen: Kimi K2.6 (1T total, not
mid-size), Mistral Small 4 (119B, active not published), Gemini 3.5 Flash-Lite / 3.8 Flash (closed, size unknown;
the backup if the open group fails). Claude Opus 5.5 was first left out for the $10 cap; added with the cap at $15
(user, 2026-10-30). OpenRouter models API, 2026-10-30: `anthropic/claude-opus-5.5`, input text / image / file, $4 /
$20 per M (cache read $0.20, write $5), max output 128,000 (so max_tokens 32000), context 1,000,000.

**Run.** Models run cheapest first under one total cap (`--max-usd`, default and agreed cap **$15**, user
2026-10-30; was $10); a call the cap
cannot cover is recorded as `budget` and not scored. Estimate (request ≈ 1.85–2.0k tokens by characters / 4; output
unknown: 0.3k per call as Astra's pilot K, up to 3k for thinking models): the six small models $0.1–0.7 each
(≈ $2 together, ≈ $5 if they all think long); Sol ≈ $0.5–2.4; Argon ≈ $1–2.5; Opus 5.5 ≈ $2–10 (2k in + 0.3–3k out per call, 70 calls); Astra ≈ $3.5 (pilot: $0.05 per K
call). The total may reach the cap at Astra, which runs last; if so, Astra is run with `--passes 1` (its pilot exists).
**Not run: the paid run waits for the user's and the orchestrator's go.**

**Output per model** (`results/runs/model_screen/<run>/<model>/`): rows.csv (every call: stage, valid, pass, variant,
reason, z, gripper, xy distance, $, seconds, tokens, provider, the answer), summary.json and report.md (pass rate per
stage, overall, JSON-valid rate, agreement between the two passes per state, $ and seconds per call; every failure with
the model's own text).

**Choice rule (decided, user 2026-10-30; was "proposed"):** computed over stages 1–6 only, 60 calls per model (5
episodes × 6 stages × 2 passes); **stage 7 (done) is reported on its own and never enters the choice.** A small model
is a candidate for claim 1 if, over those 60 calls, its pass rate is ≥ 0.8, its JSON-valid rate ≥ 0.95, and each of
stages 1–6 has a pass rate ≥ 0.6. Among candidates: the fewest active parameters first, then the cheapest (mean $ per
call in this screen). Code: `screen_rules.selection` / `screen_rules.choose`; each model's summary.json carries
`pass_1_6`, `valid_json_1_6`, `stage_min_1_6`, `stage7_done`, `meets`; the run writes `choice.json`. Calls the cap
or the API stopped are not answered and not counted (reported as errors). The strong group's best overall pass rate is the ceiling the small
models are read against. n = 5 episodes: a screen, then the closed loop decides.

**Limits, stated now:** no images; open loop (an answer is never executed, so recovery is not tested); clean memory
(no perception noise); no physics (a pass means the waypoint satisfies RoboMME's geometry and success rule, not that
the grasp holds); yaw not scored; one task.

- **Done (free):** `--print` → `results/runs/model_screen_print/ep100.txt` (episode 100's seven requests and the truth
  each is scored against); `--mock` (canonical answers, plus prose at the first episode's stage 2 and a premature null
  at the second's stage 3): pass 0.943 = 33 / 35 per pass, JSON-valid 0.971, agreement 1.0, as built in. `make check`
  passes.
- **Decided before the run (user, 2026-10-30):** the choice rule above (stages 1–6, stage 7 apart); Opus 5.5 added
  to the strong group; total cap $15, Astra still last. `docs/GOALS.md` is not changed now: after the screen's
  results, a draft of its change goes to the user for confirmation. The `pick_up` z_low question (RoboMME grasps
  containers at 0.062 > 0.05) belongs to the two-pick window; `actions.py` is not touched here. Mock run after the
  change: pass_1_6 0.933, valid_json_1_6 0.967, stage_min_1_6 0.8, stage7_done 1.0, meets (as built in). **Not run.**

## Grasps above z_low, and K's decision record: the count, then the fixes (2026-10-30, user request; rules before code)
Background (model screen, bdd8afe): `actions.py` records a `pick_up` only when the gripper closes with the hand below
`z_low` = 0.05 m (read off the dataset's cube picks at 0.020); RoboMME's own container grasp is at TCP 0.062.
- **Free count (`data/cl_pod/astra_pilot`, `astra_k2pick`, K episodes; the commanded waypoint z of each close that
  starts a grasp, as the hand's z; dec1e03 = "with the gripper-0 bug"):** pilot 5 of 17 closes at z ≥ 0.05 (ep 3,
  7, 8 ×2, 9); dec1e03 5 of 42 (ep 16 at 0.052, 17 at 0.054, 37 at 0.050 and 0.051, 41 at 0.065). Astra closes on
  containers lower than RoboMME's script (0.03–0.054 mostly). Descriptions after those closes (dec1e03 only; the pilot
  logged none): **one** doubts the pickup: ep 17, after the 0.054 close, "the first container pickup has not been
  confirmed", re-grasped, and the episode **succeeded**; ep 16, 37 (×2) and 41 say the grasp happened. So in these
  runs the z_low rule cost little that shows; it would bite whenever a controller grasps at RoboMME's 0.062.
- **Geometry check on the dataset (free; `data/robomme_lerobot`, 30 episodes each of BinFill, PickXtimes,
  SwingXtimes, VideoPlaceOrder, VideoRepick):** the finger opening the gripper settles at after a close is 0.0366 m
  holding a cube (3.66 cm) and **exactly 0.000 on every empty close**: all 150 button presses, and 29 of 469 labelled
  picks, which are demo recoveries that closed above the cube (e.g. BinFill_1200 t=243 at z 0.059: opening 0, lift,
  reopen, retry). Something between the fingers, not the hand's height, tells a grasp.
- **Rules for the fixes (fixed now, before code):**
  1. **Grasp from the fingers (`actions.py`):** a `grip_close` starts a pending grasp; when the opening has settled
     (changes < 1 mm between observed frames while closed) it is a `pick_up` if the opening is above **w_empty =
     0.005 m** (a margin over the fully closed 0.000; any held object keeps the fingers apart by its thickness),
     else an empty close (no event). An open before settling is an empty close. The hand's height no longer decides
     a pick_up (z_low stays for presses and for how a release is named). Not from the pilot: 0.000 is the gripper's
     closed reading; **to check in the preflight:** the opening when RoboMME's gripper holds a container (its wall
     must be thicker than 5 mm, or w_empty is wrong).
  2. **Record what the step did (`closed_loop.py`, review push_2026-10-30_code):** the decision is recorded after
     `env.step` and before its frames reach the KB; the gripper state changes only when the step executed (n > 0);
     a step that executed nothing (n = 0) or errored is recorded as **not executed**, and K's view says so ("not
     executed: the robot did not move") instead of "pick_up -> not done yet".
  3. **A pick_up intent only on a change from open to closed (`self_memory.intent`):** a keep (or a close) while
     already closed with nothing held is a "press" when the hand is low (as before) and a "move" otherwise, not a
     "pick_up" (the review: every such keep read as a pick_up; button tasks).
  - Not fixed here: a held object slipping out (the opening falls to 0 while closed) is not detected; open.
- **Fixed K (updates "Fixed K is not only the gripper fix" above):** = the gripper-0 fix (038ddf1) + R4 reveal
  (f6fad9a) + the move threshold (a229e35) + these three (grasp from the fingers, record what executed, pick_up only
  on open→closed), re-frozen together. Differences between the fixed-K run and dec1e03 cannot be put on any one part.
- **Implemented (2026-10-30; re-frozen: params fe4e94112a74, prompts 4612ab9e1390).**
  - `actions.py`: a `grip_close` starts a pending grasp; it becomes a `pick_up` when the opening has settled (< 1 mm
    change between observed frames, `DetectParams.w_settle`) above `w_empty` = 0.005 m; an open before settling or a
    settled opening ≤ w_empty is an empty close. The pick_up's position is the hand's at the close.
  - `closed_loop.execute`: steps first, then records the decision (before the step's frames reach the KB) with the
    gripper state after what executed; `self.closed` changes only when n > 0; `record(..., executed=n > 0)`.
  - `self_memory`: `outcome` of a non-executed decision = "not executed: the robot did not move"; `intent` gives
    "pick_up" only on a change from open to closed (a keep while closed and empty: "press" low, "move" otherwise);
    `gripper_closed` moved here from closed_loop (closed_loop is at the 250-line limit; still importable from it).
  - **`scripts/two_pick.py` (shared with K2 / the model screen):** the scripted gripper now acts over 6 control steps
    (2 observed frames, hand still) after its command instead of in one frame, as RoboMME's does (~8 dataset frames
    from open to 0.000). With an instant close no frame after it showed the opening settled, so the screen's KB saw
    the grasp only during the lift (test_model_screen caught it). No paid run used two_pick before this change; the
    K vs K2 offline test and the model screen run under this freeze.
  - **Dataset check, all 100 episodes each of PickXtimes, SwingXtimes, BinFill, VideoPlaceOrder, VideoRepick
    (`check_actions` with the code defaults; HEAD before the change in a separate worktree):** recall unchanged for
    every kind (pick_up 1446 / 1446, press 500 / 500, releases 745 / 745, hold_above 439 / 446); pick_up precision
    1.000 both, with 1448 pick_ups instead of 1459 (the 11 dropped were empty closes the height rule counted); **press
    precision 0.996 → 0.973** (514 presses instead of 502): after an empty close the closed, empty gripper's lift now
    reads as a press, as it physically resembles one. Accepted as is; to look at if a button task is run.
  - Tests: `tests/test_gripper_record.py` (a container grasped at 0.062 is a pick_up; an empty close at 0.02 or
    0.059 is not; a cube settling over frames; open→closed intent; a non-executed step recorded and keeping the
    gripper). `make check`: 253 passed, quality clean, selftest 16 pass / 1 skip.
