# ArbiterKB — Goals (the anchor for the goal reviewer)

Distilled from the plan report "ArbiterKB 项目规划报告" (2026-10-28). Updated after the M0 goal review (label-teleport trap; gate pulled forward). Change this file only when the research direction changes, and record why in NOTES.md.

## Research question
When a robot's memory is wrong, does the system know? Existing robot memories store perception guesses as facts and fail silently.

## Claims to establish
1. **Knows when it is wrong.** Under realistic memory corruption, ArbiterKB's silent-failure rate (wrong with p_max ≥ τ_hi, scored against **ground-truth identity**) is clearly lower than that of deterministic memory (HODOR-frozen, bound rule), with no loss of accuracy on clean input. Calibration (Brier, ECE) is reported.
2. **Cheap.** It approaches the frontier VLM + waypoint result (arXiv 2610.17350, RoboMME 92.1%) using a local-sized open model, at far lower tokens, latency and cost per episode. We compare tokens, seconds and dollars, not call counts (Ledger already uses about 2.7 calls).

## What would falsify it
- Deterministic memory is rarely confidently wrong (a low silent-failure rate) under realistic errors.
- RuleOnly is as good as RuleOnly + LLM arbiter (the arbiter adds nothing).
- Confidence is miscalibrated (says 90%, right 60%).

## What is ours (and what is not)
Ours: an explicit distribution over alternatives with a mutual-exclusion constraint; justification with cascade retraction (TMS); the LLM as arbiter that must cite evidence and never writes coordinates; 3D geometry + explicit relations feeding a waypoint executor; the first ground-truth calibration / silent-failure measurement of robot memory.
Not ours: confidence with time decay (ECoMEM has it); cover-following permanence (Ledger, bound rule); a small planner with structured state (SimpleARM).

## Non-goals (until a claim needs them)
Training any policy; beating 92% on the leaderboard; other RoboMME suites before permanence works; real robots; multi-view; an LLM planner.

## Known traps
- **Circularity:** corruption injected exactly where a rule mixes beliefs makes the result true by construction. M0's `id_swap` at the near-pass has this flaw by design (a skeleton). Claims need `id_swap_prob` (M2) and real tracker errors (M6).
- **Label teleport:** a corruption that moves a label further in one frame than any object can move is trivially detectable (M0 `id_swap`: about 12 cm jump vs at most 0.6 cm true motion). Corruptions must swap ids only where tracks really overlap, or follow the signature measured on the real tracker (phase G). Always compare against BoundHaircut and JumpDetector.
- **Synthetic ≠ RoboMME:** synthetic results guide development; claims are made on RoboMME replays.
- **Stock VideoUnmaskSwap rarely fools the tracker** (the overlap is image-only); the November gate decides whether the main claim survives.
- **Test leakage:** tuning or prompt iteration on test episodes invalidates claims.
