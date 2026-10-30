# Handoff: the landscape after MemOut and MEMI — 2026-10-30

For a conversation that decides what ArbiterKB still contributes. Written by the research-advisor window after reading
the closest work; no code, NOTES or GOALS were changed. Answer the user in Chinese; docs in English. Read with
`docs/GOALS.md`, `docs/HANDOFF_ORCHESTRATOR.md`, `docs/HANDOFF_EXECUTION.md`, NOTES sections named below.

**Status of this document:** findings and inferences only. Nothing here is a direction change; GOALS / NOTES change
only by the user's decision. Inferences are marked *(inference)*.

## 1. Why this matters
Three pieces of work, all public by late October 2026, cover most of claims 1 and 2 on RoboMME:
- **MemOut** (memory-condition ablation with frontier models, skill interface);
- **RoboMME's own `multi_choice` executor** (a ready-made, privileged execution layer);
- **MEMI** (learned structured text memory; a cheap model reaches the frontier's full-history success).

The user's instruction going forward: first get clear on what we have done and what is still ours; the paper's
framing comes later. Execution is a solved problem elsewhere and should not be built by us.

## 2. MemOut — "How Hard Are Current Robot Memory Benchmarks for Agentic Harnesses?"
Source: https://memout-robot.github.io/ (read 2026-10-30). Wu, Ye, H. Chen, H. Chen, Zhou, Ji, Y. Liu, Y. Du.
Paper and code "coming soon": **numbers unverified**, from the project page only.
- **Harness:** every decision a stateless API call; the model picks one of RoboMME's high-level skills + a pixel
  (`{"choice": "b", "point": [97, 87]}`); "agent-built skills" (code written by a coding agent, frozen) execute.
  Whether these re-implement or call RoboMME's own `solve` functions is not stated.
- **Memory conditions on RoboMME:** N = action log (skill, pixel, result, frame count; **no observations**);
  H = N + a 24-frame grid over the episode. Both get the goal, skills, current RGB, per-task notes, and the
  demonstration frames on video tasks.
- **Protocol:** developed on indices 0–9, frozen, evaluated once on test 10–19: 16 tasks × 10 × 2 conditions ×
  2 models = 640 episodes (+300 on RMBench).
- **Results:** Astra 76.3% → 91.9%, Opus 5.5 69.4% → 83.8% (McNemar p 1.6e-6, 3.4e-5). 24 of Astra's 25 net gains
  come from ButtonUnmask (1 → 10), ButtonUnmaskSwap (1 → 10), PickHighlight (4 → 10): cues visible only during
  execution. **VideoUnmaskSwap 10/10 and VideoPlaceOrder 8/8 in both conditions, both models.**
- **Their own open points:** "This does not show that images are necessary if the same facts were stored in text";
  the full log fits in every prompt, so selection / compression are untested; "compact task-state memory,
  world-state graphs, … open-weight planners are left for future comparisons". No cost reported.
- **Relation to earlier work:** essentially RoboMME's `multi_choice` setting (RoboMME paper: Gemini-2.5-Pro + oracle
  planner 47.9%; 2610.17350's oracle-planner diagnostics: Astra 77.8%, gpt-4o 30.8%,
  `docs/TEST_PLAN_COMPONENTS.md` §2) plus a clean memory ablation. Methodologically incremental.

## 3. RoboMME's `multi_choice` executor (code read)
`github.com/RoboMME/robomme_benchmark` @ 69a2fe2: `src/robomme/env_record_wrapper/OraclePlannerDemonstrationWrapper.py`,
`src/robomme/robomme_env/utils/vqa_options.py`, `choice_action_mapping.py`.
- Fixed options per task (VideoUnmaskSwap: a "pick up the container", b "put down the container"; ButtonUnmask:
  press / pick up / put down; etc.).
- The pixel is **snapped to the nearest simulator-registered candidate** (e.g. `env.spawned_bins`); the option's
  `solve` (e.g. `solve_pickup_bin`) runs from **simulator poses**: screw motion, 3 retries, then RRT*, 3 retries.
- So it is privileged execution: grasp height, lift, put-down spot are not the model's. MEMI-Astra-Plan's 99.4%
  shows it almost never fails when the decision is right.
- *(inference)* For a memory study this is acceptable: what is tested is what was remembered and chosen; the KB can
  still come from real perception (KB 3D position → projected pixel → snapped).

## 4. MEMI — "Reverse-Engineering What Robots Should Remember"
Sources: https://memi-robot.github.io/ (blog, read 2026-10-30); code + all logs https://github.com/memi-robot/memi
@ bb6c37c (2026-10-25); leaderboard entries `doc/submission/memi-astra-{plan,act}.md` in RoboMME's repo.
Ryu, **Dai (RoboMME author)**, Du, Finn, Liu. Not peer reviewed.
- **Memory:** field–value pairs the policy LLM rewrites at every step (in Act, fields may name keyframes that are
  shown again). Example, VideoUnmaskSwap Act episode 0: `target_locations: [[87,106],[122,82]]`,
  `completed_lifts: 1`, `final_layout_keyframe: V113`.
- **Learning:** per task, two prompts (annotator: what to remember; policy: how to update and use it), optimised by
  textual SGD over 5 iterations × 5 training demos, under sufficiency / predictability / necessity. No weights.
- **Plan:** observations are **text built from the simulator's semantic segmentation** (each visible object's name
  and pixel position, every 16 steps; containers unlabelled, e.g. `container at [75,120], [88,167], …`); no images;
  RoboMME's oracle executor. **Act:** Embodied-Astra's waypoint harness, real images every 16 steps.
- **Results (RoboMME test, 800 episodes; recounted by us from the released logs, matching the README):**

| policy | FullHist | Init | MEMI |
|---|---|---|---|
| Astra-Plan | 91.3 | 92.4 | 99.4 |
| Fable-Plan | 85.9 | 88.6 | 97.1 |
| Luna-Plan | 62.9 | 66.6 | 85.4 |
| Haiku-Plan | 58.0 | 73.9 | 90.8 |
| Astra-Act | 85.3 | 88.3 | 97.0 |

  FullHist = the full text observation history (Plan) or the full image conversation (Act).
  VideoUnmaskSwap: Haiku-Plan 11 → 48 / 50, Luna-Plan 17 → 41, Astra-Act 45 → 48.
  **Cost, Act (median $ per episode, from `result.json`): FullHist 0.16, Init 0.24, MEMI 0.43.** Memory raised
  success and raised cost.
- **Caveats:** prompts tuned per task on training demos; intermediate prompts (iterations 1–4) were rolled out on
  25 test episodes (the final prompt is the fixed 5th iteration, so probably not selected on test, *inference*).

## 5. Other RoboMME / RoboMME-Plus entries seen (RoboMME repo `doc/submission/`, read 2026-10-30)
- **RoboMME-Plus** (released 10/2026, harder): Embodied-Astra 23.0% (waypoint, $0.95 / ep), one-shot
  Embodied-Astra 36.75% ($2.57 / ep), 3-tier Astra (Astra planner + Qwen3-VL-4B monitor + π0.5) 9.5%.
  VideoUnmaskSwap 1 / 50 for both Embodied-Astra variants. No MEMI entry on Plus found.
- **SimpleARM** (arXiv 2609.36595): typed state (relation / reference / progress / route) for frozen VLAs, 67.17% on
  RoboMME vs 44.51% best non-oracle baseline; abstract only read.
- **RoboMME-Interference** (arXiv 2606.22338): perceptual memory decays as unrelated sessions accumulate; subgoal
  memory degrades less; abstract only read.

## 6. Our claims against this landscape
Our numbers: NOTES "Astra VideoUnmaskSwap val pilot", "Layer 1 on real scenes", "Astra's 7 VideoUnmaskSwap failures",
"The dec1e03 two-pick K run, read by the rule", "Claim 3, corrected read and attribution", "Corrected tables after
the code review", "Is the counting win ArbiterKB's".

| | ours, measured | covered elsewhere | not covered by anyone (found) |
|---|---|---|---|
| Claim 1 (smaller model) | Qwen 35B-A3B 1/24 closed loop; with memory, first call-out right 12/15 vs H 1/9 | **MEMI: Haiku-Plan 90.8 ≈ Astra FullHist 91.3** (privileged perception, oracle execution) | open-weight models; non-privileged perception |
| Claim 2 (fewer tokens) | Astra VUS val 0–9: H 9, R 6, K 8 / 10; K 38% of H's tokens, same $ (H 77% cached) | MEMI-Act 97.0 > FullHist 85.3 (success); **costlier** | a memory that saves cost at matched success |
| Claim 3 (calibration) | pre-registered main not supported; lead from R6 + cover kernel; overconfident on swapped cubes | none | all of it |
| Memory from real perception | KB from GroundingDINO → SAM → Cutie; tracking identity is the main error (7 / 11 dec1e03 failures, swapped cubes ≤ 0.27 for every arm) | none (MEMI-Plan uses sim segmentation; MEMI-Act / MemOut let the LLM read images) | **the gap between privileged and perceived observations for a structured memory** |
| Task-agnostic memory | KB is generic, rule-built, inspectable, cites evidence | MEMI learns a schema per task from demos | generic memory vs learned per-task memory |
| Execution | waypoint interface; failures in layers 2/3/5 | RoboMME `multi_choice` (privileged), MEMI-Plan uses it | — (stop building our own) |

*(inference)* Claims 1 and 2 as written are largely pre-empted on RoboMME. What still looks open:
1. **Perceived vs privileged observations** for a structured text memory: replace MEMI-Plan's simulator-segmentation
   text with our perception / KB, same executor, and measure the drop; then whether KB rules (re-linking, beliefs)
   recover it. MEMI's Plan code is released, so this may need no new harness (to verify by reading `memi/plan`).
2. **Generic vs learned memory:** the KB with no per-task tuning vs MEMI's per-task prompts, at equal perception.
3. **Uncertainty:** no one gives the planner calibrated beliefs; ours exist but are not yet shown to help.
4. **RoboMME-Plus**, where every system is far from the ceiling.
5. **Cue-during-execution tasks** (ButtonUnmask, ButtonUnmaskSwap, PickHighlight), where MemOut's action log fails:
   does a perceived observation memory in text recover the image gain?

## 7. Consequences for current work *(inference; for the user and orchestrator to decide)*
- Our own execution work (option 3 skills layer, waypoint-level fixes for weak models) loses its point; RoboMME's
  `multi_choice` or MEMI's Plan harness replaces it.
- The planned "fixed K + H two-pick" waypoint package mainly measures execution, not memory (MemOut: VUS 10/10 with
  an action log; MEMI-Act FullHist VUS 45 / 50).
- The model screen (waypoint stages) answers an execution question; under a Plan interface it would need redesign.
- Unchanged in value: perception and tracking-identity work, the KB's re-linking and beliefs, claim 3.

## 8. Not yet done
- Read MEMI's `memi/plan` code: how observations are built, whether our KB text can replace them, how FullHist is
  formed; and `memi/act` memory handling.
- Check whether MEMI has RoboMME-Plus results or code paths.
- MemOut: wait for paper / code; confirm whether its executor is RoboMME's `solve`.
- Search again for "perceived vs privileged observation" studies on RoboMME (none found so far).

Working copies (temporary, scratchpad; re-clone if gone): RoboMME @ 69a2fe2, MEMI @ bb6c37c.
