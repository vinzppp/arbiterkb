# ArbiterKB — Goals (the anchor for the goal reviewer)

**Direction changed on 2026-11-01 (user decision; docs/HANDOFF_ORCHESTRATOR.md "Paper direction", 2fd5bda; NOTES
"Memory or policy? Preregistration of the main study").** The paper asks whether memory structure or task policy explains
a learned-memory system's gain on RoboMME's `multi_choice` executor. ArbiterKB is one memory arm, not the protagonist.
The earlier questions (waypoint control with a smaller model or fewer tokens; calibrated beliefs) are kept under
"History" below, with why they stopped. Change this file only when the research direction changes, and record why in
NOTES.md.

## Research question
MEMI's learned text memory lifts GPT-6 Luna on RoboMME-Plan from 62.9% (full text history) to 85.4%, with RoboMME's
privileged `multi_choice` executor. MEMI learns two things at once: what to remember (its fields) and how to act on it
(its subgoal rules: when to pick, put down, press, stop; how to count progress). **How much of the gain is the policy,
and how much is the memory?** Executor, observations (MEMI-Plan's privileged capture), model and episodes are fixed;
only memory and policy text vary.

## Claims to establish
1. **Policy matters.** Giving every memory MEMI's task policy (its subgoal text) raises success by at least 10 points on
   average over the four memory designs (full history, a task-agnostic deterministic ledger, MEMI's fields, the KB).
2. **Memory matters little once the policy is matched.** Under that policy, each memory's success is within ±5 points
   of the deterministic ledger's. Claimed as equivalence only if the sample reaches the TOST size; otherwise reported as
   differences with intervals.
3. **MEMI's gain is mostly policy.** Full history with MEMI's policy recovers at least half of stock MEMI's gain over
   full history with a generic policy.

Every comparison uses one harness: same model and effort, menu, executor, observations, episodes, step and time limits;
policy text identical across memory arms, memory text identical across policy arms; no memory text recommends an action.

## What would falsify it
- Claim 1: the 95% CI of the policy main effect lies entirely below 10 points.
- Claim 2: any memory differs from the ledger by ≥ 5 points with Holm-corrected significance under the matched policy
  (then memory structure does matter, and the paper reports where).
- Claim 3: the 95% CI of (FH-matched − FH-generic) − 0.5 × (MEMI − FH-generic) lies entirely below 0.
- Split fidelity: stock MEMI and "MEMI's fields + MEMI's policy" in our framing differ by > 10 points. Then our
  framing, not the split, carries the effect, and claim 3 is reported with that caveat.

## What is ours (and what is not)
**Ours:** the memory × policy factorial on a fixed executor; the mechanical split of a learned-memory prompt into
memory and policy; a task-agnostic deterministic ledger arm; the KB as one memory arm; the preregistered analysis.
**Not ours:** RoboMME and its `multi_choice` executor; MEMI, its prompts, capture and FullHist; the memory-condition
ablation idea (MemOut); typed task state (SimpleARM, 2609.36595); an external object ledger (Ledger, 2609.34554);
explicit LLM world state (Statler, 2306.17840); probabilistic anchoring (Persson et al., 1902.09937; 2002.10373).

## Non-goals (this paper)
Perception tiers as a main result (direction C; the free snapping check G2 is reported as a finding only); a
costly-verify action or calibration-to-decision study (future work, its free "Gate A"); training; our own executor or
waypoint control; real robots; beating MEMI's score; re-optimizing any prompt.

## Known traps
- **Policy hidden in memory:** a memory that writes "next: put it down" or a demo-inclusive progress count is policy.
  Memory texts hold no recommendation; the KB uses the phase-safe view (demo-inclusive progress removed).
- **Policy text naming memory fields:** MEMI's subgoal text names its fields; the names are substituted the same way
  for every memory arm, and split fidelity is measured.
- **The policy side is not only choice rules:** MEMI's subgoal text also holds perception and update heuristics; the
  claim says "MEMI's subgoal prompt", and the share of such sentences is reported.
- **MEMI's training episodes:** RoboMME train 0–99 is in MEMI's training data and is never used here.
- **Task selection used published test numbers** from the same test episodes we confirm on; all arms are re-run, and
  the re-measured MEMI gap may shrink. Stated in every table.
- **Asymmetric tuning:** MEMI's prompts were optimized for Luna on privileged observations; no other arm is tuned.
  Stated, not corrected.
- **Absence of evidence:** a non-significant memory contrast at ~250 pairs is "not detected", not "no difference".
- **Test leakage:** RoboMME test is read once from a frozen manifest, ledgered in `results/test_reads.log`.

## History (moved, not deleted)
- **2026-10-28 → 2026-10-31 (waypoint line), claims 1–3 below.** Stopped by direction C (2026-10-31): no model passed
  the waypoint screen (NOTES "Model screen", screen1), the strong–weak gap was grasp height, not memory; MEMI reached
  the same goals with learned memory and RoboMME's executor (docs/HANDOFF_LANDSCAPE.md).
- **2026-10-31 → 2026-11-01 (direction C, never adopted in this file):** memory under perception errors
  (docs/DRAFT_DIRECTION_C.md, superseded). Dropped for this paper: stock RoboMME makes our tracker swap 0 / 116 times
  (NOTES "G — gate result"); on G train the KB and a strong ledger tie at 83.40% with 4 / 241 action-relevant
  disagreements (external branch identifiability report); `multi_choice` has no verify or abstain action.

### Waypoint line goals (2026-10-28, superseded 2026-10-31)
The file as it stood before 2026-11-01, headings demoted one level, text unchanged.

Distilled from the plan report "ArbiterKB 项目规划报告" (2026-10-28). Updated after the M0 goal review (label-teleport trap; gate pulled forward). **Direction changed on 2026-10-28 (user decision, NOTES "Direction v2"):** the KB is now the memory module of an LLM controller that outputs the robot's waypoints itself, as in arXiv 2610.17350. The old question (does memory know when it is wrong?) is kept as claim 3. Change this file only when the research direction changes, and record why in NOTES.md.

### Research question
Can a complete, structured memory (the KB) let an LLM control a robot as well as a frontier VLM that reads the whole image history, either with a **much smaller model** or with **far fewer tokens and images**?

The reference point is arXiv 2610.17350:
- A frozen GPT-6-Astra sees every front frame (one per 0.15 s), the wrist frame and the task video. That is a median of 148 images and 16.5k input tokens in the final request.
- It outputs one world-frame waypoint (position, orientation, gripper) per call, which RoboMME's motion planner executes.
- Results: 92.1% on RoboMME (permanence 97.0), about 9.5 calls and $0.39 per episode.
- Its authors state that they did not isolate which part of the context acts as memory.

### Claims to establish
Claims 1 and 2 are alternatives: either one makes the memory module worth having. A frontier model as the controller is acceptable (user, 2026-10-28). The structured memory has value of its own: it is inspectable, it cites evidence and it carries calibrated beliefs. So claim 2 with a frontier model, at a large token saving, is a full result.

1. **Smaller model.** A small open model (primary: `qwen/qwen3.6-35b-a3b`, 3B active) reading the KB approaches the frontier VLM's success on RoboMME. The same small model reading the raw image history does clearly worse.
2. **Fewer tokens.** With the latest KB text plus the current frame(s), the same model matches its own success with the full image history. It must use far fewer input tokens, images and dollars per episode. This is tested for the small model, and for a larger model if the budget allows.
3. **Knows when it is wrong.** The KB's beliefs are calibrated against ground truth: silent failure (wrong with p_max ≥ τ_hi, scored **physically**) is lower than for deterministic memory (HODOR-frozen, bound rule) **and than for a threshold haircut on the same evidence feature, at matched clean coverage**; AURC is the threshold-free summary; Brier and ECE are reported. The controller sees these probabilities, so an uncertain memory can lead to "look first" instead of a confident wrong grasp. Silent failure alone is not enough: any system that lowers every confidence scores 0 (M1 goal review).

Every success comparison uses the same harness:
- the 2610.17350 action format, feedback and executor (its code is released);
- the same tasks, step limit and failure rules;
- only the memory content the controller reads changes.

That controlled comparison is the gap no published work fills (`scene-graph-robot/docs/memory-landscape-2026-10-27.md`).

### What would falsify it
- The small model with the KB stays far below the frontier result, even on permanence tasks, where the KB is most complete.
- The small model with the raw image history does as well as with the KB: the KB adds nothing.
- An LLM reading the raw tracker log (positions per frame, no beliefs, no events) does as well as with the KB at equal or lower cost: the structure adds nothing beyond perception.
- KB + current frame does clearly worse than the full history with the same model: the KB loses information the controller needs. The failure analysis must say which.
- Confidence is miscalibrated (says 90%, right 60%), or errors in the KB turn into confident wrong actions as often as without beliefs.
- A one-feature threshold rule separates real id swaps from clean passes almost perfectly (phase G's ROC). Then claim 3's detection part needs no belief layer.

### What is ours (and what is not)
**Ours:**
- one memory module with all five layers, each item a belief with support (L0 geometry with 3D centroids; L1 objects; L2 relations as distributions with mutual exclusion; L3 an event log with from-where-to-where and a text summary; L4 task progress);
- justification with cascade retraction (TMS);
- a controller that cites the KB items behind each action;
- the controlled comparison of memory content at a fixed executor;
- the first ground-truth calibration / silent-failure measurement of robot memory.

**Not ours:**
- an LLM emitting waypoints for a motion planner (2610.17350);
- a VLM-written action transcript read by a planner (Ledger, 2609.34554);
- confidence with time decay (ECoMEM);
- cover-following permanence (Ledger, bound rule);
- a small planner with structured state (SimpleARM).

### Non-goals (until a claim needs them)
- training any policy;
- beating 92% on the leaderboard (claims 1 and 2 are about matching it cheaply);
- real robots;
- multi-view.

RoboMME suites are added in order: permanence first, then counting and imitation, once the KB layers they need (L3 events, L4 progress) exist.

### Known traps
- **Circularity:** corruption injected exactly where a rule mixes beliefs makes the result true by construction. M0's `id_swap` at the near-pass has this flaw by design (a skeleton). Claim 3 needs `id_swap_prob` (M2) and real tracker errors (M6).
- **Label teleport:** a corruption that moves a label further in one frame than any object can move is trivially detectable (M0 `id_swap`: about 12 cm jump vs at most 0.6 cm true motion). Corruptions must swap ids only where tracks really overlap, or follow the signature measured on the real tracker (phase G). Always compare against BoundHaircut, OverlapHaircut (hard-label optimal for a corruption we design; not optimal for ranking), SoftHaircut (R6's formula without a KB) and JumpDetector.
- **Our own choices leak into synthetic results:** on any synthetic corruption we design, the best possible system copies our parameters (overlap threshold, swap rate). Synthetic tables show plumbing; only G's measured signature or real RoboMME data can support a claim.
- **Synthetic ≠ RoboMME:** synthetic results guide development; claims are made on RoboMME closed-loop episodes.
- **Oracle ≠ perceived:** a KB filled from simulator ground truth makes claims 1 and 2 easy. Report the oracle KB as an upper bound and make the claims on perceived input.
- **Unequal harness:** the arms must differ only in memory content. Same model, prompt skeleton, action format, step limit and failure rules. The paper's GPT-6 number is a reference row from a different run, labelled as such.
- **Stock VideoUnmaskSwap rarely fools the tracker** (the overlap is image-only); the November gate decides whether claim 3 survives.
- **Test leakage:** tuning or prompt iteration on test episodes invalidates claims. 2610.17350 itself says its configuration was chosen with test-split evaluations, so the reference row is optimistic.
