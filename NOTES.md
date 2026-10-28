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
