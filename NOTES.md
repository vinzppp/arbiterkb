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
- With the untuned defaults the belief layer barely helps (0.90 vs 1.00 silent). Passes stay 21–36 px apart, so with `d_px` = 30, w = 0.5·(1 − d/30) ≤ 0.15, and confidence in the wrong label stays around 0.9 > τ_hi.
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
