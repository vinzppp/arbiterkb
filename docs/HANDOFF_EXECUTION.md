# Handoff: the execution gap (IK, controller design, models, landscape) — 2026-10-30

For a new conversation about how to raise closed-loop execution success. Read with `README.md`, `HANDOFF.md`
(project state) and `NOTES.md` "Closed loop" (every number below and its file). Answer the user in Chinese; docs in
English.

## The setup in one paragraph
RoboMME (ManiSkill sim, Franka, 16 memory tasks; arXiv 2603.04639) with 2610.17350's harness (arXiv 2610.17350,
GPT-6-Astra 92.1% on 800 test episodes). Each step the LLM returns one JSON: `waypoint` = TCP pose
`[x, y, z, roll, pitch, yaw]` in the world frame, `gripper` −1 / 0 / 1, plus our `target` / `cites`; `waypoint: null`
declares completion. RoboMME's planner does IK + screw motion, RRT* fallback, executes at 20 steps / s and records
frames. The arms differ only in the memory the model reads: H (all past frames, 3 × 3 grids), R (raw tracker log),
K (our KB view); code `arbiter_kb/closed_loop.py`, runner `stress/closed_loop_pod.py`.

## What IK does and does not do (the user's question, settled)
IK + the motion planner turn **a full 6-D hand pose** into joint angles and a collision-free path. They do not
choose that pose or the sequence. Between "the grasp point" and IK sit layers the LLM currently does by itself:

| layer | question | who does it now |
|---|---|---|
| 1 what / where | which object, its 3D position | perception / KB / the LLM |
| 2 grasp pose | approach direction, hand yaw aligned with the cube's faces, closing height, opening | **the LLM** |
| 3 sequence | above → down → close → lift (how high) → move → release | **the LLM** |
| 4 joints and path | IK, collision-free motion | RoboMME planner (works) |
| 5 verification | grasped? lifted enough? done? | **the LLM** |

The hand's orientation (roll / pitch / yaw) is an IK *input*; joint angles are its output. Position-only IK picks an
orientation by joint criteria, never by the object's yaw.

## Evidence so far
- **M4 offline dry run** (NOTES "M4"): with oracle positions, Qwen put the first waypoint within 3 cm of the right
  container in 19 / 20 (K). One step only: layer 1 works; layers 2, 3, 5 were never tested.
- **Qwen 3.6-35B-A3B val pilot** (stopped, 25 / 60, 1 success): failures are layers 2, 3, 5 (gripper closed in the
  air, lift to 0.15 m and "done" where 0.25 m was needed, one waypoint repeated 15+ calls, plans flipping, reasoning
  to the 32k cap), plus 3 of 4 `env_error` from ManiSkill's zero-step `follow_path` bug (docs/UPSTREAM.md).
- **GPT-6-Astra probe:** H 2 / 2 (VideoUnmaskSwap $0.17, VideoPlaceOrder $0.65 per episode). Same harness and
  planner, so the gap is the controller model's. Astra's VideoUnmaskSwap H / R / K pilot is running on the pod
  (`/workspace/cl/astra_pilot`); first four: H 1 / 2, R 1 / 1, K 0 / 1.
- 2610.17350's own 63 failures are mostly premature completion and mistimed button presses, not grasps.

## Literature found (2026-10-30 search; read the papers before relying on numbers)
- **No benchmark compares models in the waypoint-to-planner setting.** Only 2610.17350 (one model). A model
  ladder on this interface is itself a gap we could fill.
- **Gemini Robotics-ER 1.5 / 2** (arXiv 2510.03342; API docs ai.google.dev/gemini-api/docs/robotics-overview):
  outputs text: 2D points `[y, x]` 0–1000, boxes, trajectories; no 3D poses. Evaluated on 15 reasoning / pointing
  benchmarks, and on real robots only as an orchestrator of their own trained VLA (progress ~80% vs 44% VLA alone;
  planning failures 25.5% → 9%). Not on OpenRouter; needs a Google API key (the user's call).
- **Embodied-R1.5, 8B, open weights** (arXiv 2606.11324, Qwen3-VL-8B base): 2D points and traces; claims +17 pts
  over Robotics-ER 1.5 on embodied benchmarks (benchmarks, not task success); a separate VLA variant (trained action
  head) reports 92.4% on SimplerEnv; zero-shot RoboTwin via "pointing + unified motion logic" with no success rate.
  Could be self-hosted on the A40.
- **General VLMs** are already strong on spatial QA; specialists lead mostly on pointing, and on size / cost /
  latency. On OpenRouter with images: gemini-3.5-flash ($1.5 / $9 per M), gpt-6-sol ($2 / $10),
  claude-sonnet-5.5 ($2 / $10), qwen3.8-27b ($0.15 / $1.88), gpt-6-astra ($10 / $50).
- **Closing the gap, four families:**
  1. semantic decision by the model, motion by deterministic code: arXiv 2610.14106 (the LLM never outputs
     coordinates), SynManDex 2606.09798 (primitive library with feasibility checks), KeyMPs;
  2. keypoints / constraints: MOKA 2403.03174 (2D affordance points → 3D), ReKep 2409.01652 (keypoint
     constraints + solver); our KB already supplies the 3D positions they work to get;
  3. verification: FailBench 2609.03611 (VLM success detectors are biased toward "success", even with more
     reasoning: our premature "done"); Manipulate-Anything 2406.18915 (a verifier after each subtask);
  4. embodied-specialist models (above).

## Options on the table (none decided)
1. **Model ladder on the current interface** (fair for claims 1–2): K, plus an oracle-memory diagnostic arm
   (simulator truth written into K's view) to separate execution ability from perception, over qwen3.8-27b,
   gemini-3.5-flash, gpt-6-sol, maybe Embodied-R1.5-8B, with Astra as the reference.
2. **Fair harness fixes for every arm** (each a stated deviation from 2610.17350): patch the zero-step bug, a
   "you repeated this waypoint N times" note, a reasoning-length limit.
3. **Skills layer** (`pick(obj)`, `place(obj, where)`; grasp pose from the KB's 3D centre and a yaw from the mask's
   minimum-area rectangle; lift to a fixed height; grasp checked by finger width). Biggest effect, but only R / K
   have 3D positions, so it becomes a *system* claim (KB + skills + small model vs big model + images), separate
   from claims 1–2.
4. **KB as the completion verifier:** before a `null` waypoint, the K arm's view states the facts (grasped? lifted
   how high? placed where?) from `self_memory`. Targets the premature "done" that FailBench says VLMs cannot fix
   themselves; it is a memory contribution, so it fits claims 1–2.
5. **RoboMME's `multi_choice` action space** may be the "oracle skill interface" the human 90.5% used. To check by
   reading RoboMME's code (free): if so, it measures pure decision ability with perfect execution (a diagnostic
   upper bound, simulator poses, not a claim).

## Open questions for the new conversation
- Which of options 1–5, in what order, and which are claims vs diagnostics.
- Landscape / market: who builds "LLM or VLM as the high-level controller" products (Google's Robotics-ER API,
  open embodied models, VLA companies), where a KB memory module fits, and what a small-model + memory system would
  be worth against frontier-model APIs (cost per episode: Astra H $0.17–0.65 on two tasks).
- Whether to add Gemini Robotics-ER 2 (needs a Google API key).

## Rules that still hold
Rules fixed in NOTES before paid runs; honest negatives; no-LLM baselines; code + goal reviewers after each phase;
commit to main, no push without the user; never print the OpenRouter key; stop the pod when idle; test-split reads
need a freeze (`python scripts/run.py --freeze --force`) and go in `results/test_reads.log`.
