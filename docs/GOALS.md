# ArbiterKB — Goals (the anchor for the goal reviewer)

Distilled from the plan report "ArbiterKB 项目规划报告" (2026-10-28). Updated after the M0 goal review (label-teleport trap; gate pulled forward). Change this file only when the research direction changes, and record why in NOTES.md.

## Research question
When a robot's memory is wrong, does the system know? Existing robot memories store perception guesses as facts and fail silently.

## Claims to establish
1. **Knows when it is wrong.** Under realistic memory corruption, ArbiterKB's silent-failure rate (wrong with p_max ≥ τ_hi, scored **physically** against ground truth) is clearly lower than that of deterministic memory (HODOR-frozen, bound rule) **and of a threshold haircut on the same evidence feature, at matched clean coverage** (share of clean rows answered with p_max ≥ τ_hi), with no loss of accuracy on clean input. The threshold-free summary is AURC (area under the risk–coverage curve). Calibration (Brier, ECE) is reported. Silent failure alone is not enough: any system that lowers its confidence below τ_hi scores 0 (M1 goal review).
2. **Cheap.** It approaches the frontier VLM + waypoint result (arXiv 2610.17350, RoboMME 92.1%) using a local-sized open model, at far lower tokens, latency and cost per episode. We compare tokens, seconds and dollars, not call counts (Ledger already uses about 2.7 calls).

## What would falsify it
- Deterministic memory is rarely confidently wrong (a low silent-failure rate) under realistic errors.
- RuleOnly is as good as RuleOnly + LLM arbiter (the arbiter adds nothing).
- An LLM reading the raw tracker log does as well as the LLM reading the KB, at equal or lower cost (the KB adds nothing to the LLM; M4 arm C vs D).
- Confidence is miscalibrated (says 90%, right 60%).
- A one-feature threshold rule separates real swaps from clean passes almost perfectly (phase G's ROC). Then a haircut on that feature suffices for claim 1, and the belief layer can only add value through propagation (R5, retraction, the arbiter).

## What is ours (and what is not)
Ours: an explicit distribution over alternatives with a mutual-exclusion constraint; justification with cascade retraction (TMS); the LLM as arbiter that must cite evidence and never writes coordinates; 3D geometry + explicit relations feeding a waypoint executor; the first ground-truth calibration / silent-failure measurement of robot memory.
Not ours: confidence with time decay (ECoMEM has it); cover-following permanence (Ledger, bound rule); a small planner with structured state (SimpleARM).

## Non-goals (until a claim needs them)
Training any policy; beating 92% on the leaderboard; other RoboMME suites before permanence works; real robots; multi-view; an LLM planner.

## Known traps
- **Circularity:** corruption injected exactly where a rule mixes beliefs makes the result true by construction. M0's `id_swap` at the near-pass has this flaw by design (a skeleton). Claims need `id_swap_prob` (M2) and real tracker errors (M6).
- **Label teleport:** a corruption that moves a label further in one frame than any object can move is trivially detectable (M0 `id_swap`: about 12 cm jump vs at most 0.6 cm true motion). Corruptions must swap ids only where tracks really overlap, or follow the signature measured on the real tracker (phase G). Always compare against BoundHaircut, OverlapHaircut (hard-label optimal for a corruption we design; not optimal for ranking), SoftHaircut (R6's formula without a KB) and JumpDetector.
- **Our own choices leak into synthetic results:** on any synthetic corruption we design, the best possible system copies our parameters (overlap threshold, swap rate). Synthetic tables show plumbing; only G's measured signature or real RoboMME data can support a claim.
- **Synthetic ≠ RoboMME:** synthetic results guide development; claims are made on RoboMME replays.
- **Stock VideoUnmaskSwap rarely fools the tracker** (the overlap is image-only); the November gate decides whether the main claim survives.
- **Test leakage:** tuning or prompt iteration on test episodes invalidates claims.
