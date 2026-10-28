# Goal review — M0 — 2026-10-28
Verdict: ON-TRACK-WITH-CHANGES

M0 does what a skeleton should: the pipeline runs end to end, and correctness is judged by physical position, not by id. NOTES is honest that M0 is not evidence for claim 1. The problem is the plan for M1–M2: as specified, it would build on a corruption model whose signature makes any result true by construction, in one direction or the other.

## Claim movement
- **Claim 1 (silent failure lower than deterministic memory): none to weak.**
  - On the 20 corrupted episodes in `results/m0_summary_idswap_p1.0.csv`, ArbiterKB is silent on 18/20 (0.90, Wilson CI about 0.70–0.97) and Bound on 20/20 (1.00, about 0.84–1.00). The intervals overlap.
  - ArbiterKB's mean confidence when wrong is 0.91. At p = 0.5 only 5 episodes are corrupted (4/5 vs 5/5), which is too few to read anything into.
  - Done-when (c) is met only in the letter.
- **Clean cost:** confidence when right is 0.94 (ECE 0.059) against Bound's 1.00 and 0. Accuracy is equal (1.00).
- **Falsifier "deterministic memory is rarely confidently wrong":** not touched.
  - Frozen's 0.47 clean silent failure is an artefact of Frozen ignoring all motion. It says nothing about realistic errors.
- **Claim 2 (cost) and falsifier "arbiter adds nothing":** not touched (out of scope for M0).

## Risks of self-deception
1. **The corruption teleports the label, and that one fact decides both M1 and M2.** I measured it on test seeds 1000–1029:
   - At the corruption frame, the swapped label's position jumps by 0.117–0.13 m (median 0.127 m).
   - The largest per-frame step anywhere else is 0.0064 m, about 20 times smaller.
   - The synthetic lanes are 0.14 m apart and the corruption fires at the closest approach (21–36 px), so the label hops across the gap. A real tracker cannot swap ids there because the objects never overlap.

   Two consequences:
   - **M1:** R6 fires identically on clean and corrupted passes, so the belief layer is a fixed confidence cut of (1 − w) after a pass. Tuning `w_mix` by Brier on a "clean + id_swap p = 0.5" mix (M1 build step 4) will set w to roughly the injected corruption rate. Calibration will then look good by construction, because the tuning mix sets the base rate.
   - **M2:** the Hungarian / constant-velocity identity margin will catch a 12 cm one-frame jump nearly 100% of the time. "Silent failure drops in M2" will be true because of how the corruption was generated, not because the method works. `id_swap_prob` has the same problem: its swap probability is a sigmoid of `min_px`, the same feature R6 uses for w.

   **Cheapest control:** report a "label-jump detector" baseline (flag if any container moves more than 3 cm in one frame), and regenerate the corruption with `lane_offset` small enough that the tracks actually overlap. Then swap ids only where positions coincide, so there is no jump.
2. **Spec drift on held-out data:** `v_move` was changed after measuring synthetic data (NOTES deviation 1). NOTES does not say which seeds were used. Confirm they were train seeds (0–999); otherwise that is mild test leakage.
3. **Reproducibility:**
   - The p = 0.5 summary has no `results/runs/` directory behind it. Only the CSV exists, so it cannot be traced to a run.
   - NOTES reports a median step of 0.40 ms, but `meta.json` says 0.10 ms.

## Missing baseline(s)
- **Bound + fixed haircut** (confidence c < 1 after any near-pass involving the target's container). Under oracle data this equals ArbiterKB-RuleOnly by construction, so it must be in M1's table. NOTES already plans it; make it mandatory.
- **Label-jump detector** (above). On the current `id_swap` it would score 20/20 detections with zero clean false positives. If it matches the M2 system, the M2 result says nothing about the claim.

## Cut or defer
- `arbiter.py` `validate()` and its tests: no claim needs them before M4.
- `planner.py` / `skills.py` waypoints with yaw 0, and the monitor stub: these serve no metric before M3/M5.
- R7 wired with λ = 1 is a no-op.
- None of these are costly, but stop extending them until a claim needs them.

## Over-interpretation
- The notebook says ArbiterKB "keeps the alternative". It does, but at about 0.09 probability, while it trusts the wrong label at 0.91 and stays above τ_hi = 0.8. That is still a silent failure. Reword it.

## Recommendation for the next phase
**Modify M1.**
1. **Do not tune `w_mix` against the current `id_swap`.** With no signal that tells corrupted passes from clean ones, any calibration gain is just fitting the base rate the tuning mix injected.
2. **Fix the corruption so swaps only happen where tracks overlap**, and add both trivial baselines (Bound+haircut, label-jump) to the M1 table.
3. **Pull forward the cheapest decisive experiment: the November gate.** Run the real tracker over the RoboMME VideoUnmaskSwap train replays. Count how often ids actually swap against ground truth, and record what the swaps look like (overlap, `track_q` drop, jump size). If real swaps are rare, the falsifier "deterministic memory is rarely confidently wrong" fires and M1–M3 tuning is wasted. If they are common, their measured signature should define `id_swap_prob`, rather than a sigmoid we chose ourselves.

Files used: `results/m0_summary_idswap_p1.0.csv`, `results/runs/synthetic_test_9fc955f2/meta.json`, `arbiter_kb/sources/corrupt.py`, `arbiter_kb/rules.py`, `NOTES.md`.
