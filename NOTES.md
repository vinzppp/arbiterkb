# NOTES — engineering log

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
