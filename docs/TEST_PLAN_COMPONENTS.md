# Test plan: testing the closed loop's parts separately — 2026-10-29

Status: **plan, nothing run.** Every threshold below is a proposal; each test's rule is copied into NOTES and fixed
before its first paid call (the project rule). Read with `docs/HANDOFF_EXECUTION.md` (the five layers, the evidence)
and NOTES "Closed loop".

## 1. Goal
The Qwen val pilot scored 1 / 24 and we cannot say which part failed: finding the object, turning a position into a
grasp, sequencing, or judging completion. End-to-end episodes are the most expensive way to find out (dozens of
image calls each) and they mix every cause. This plan measures each layer on its own, against simulator truth, mostly
without the LLM, so that:
1. we know whether execution is solvable by dumb code at all (if a scripted pick from true poses does not reach
   ~100%, the problem is the planner / env, not any model);
2. we know how accurate the "where" has to be (a tolerance curve), and whether each arm (H / R / K) meets it;
3. we know how much a model loses when it is the controller (same start, same true position, model vs script);
4. we know how often a model declares "done" falsely, and whether K's facts fix it;
5. every later end-to-end run is spent only on what the parts cannot answer.

Diagnostics, not claims: A–C use simulator truth and say where the losses are. Claims 1–2 still come from
end-to-end runs on the unchanged interface; a skills layer (option 3) would be a separate system claim.

## 2. What we already know (inputs to the design)
- **Layer 1 works with positions in memory:** M4 offline, K arm, first waypoint within 3 cm of the right container
  19 / 20. Untested: H (images only), later steps, R.
- **Qwen's failures are layers 2, 3, 5:** gripper closed in the air (z = 0.30 with cubes at z ≈ 0.025), lift to
  0.15 m where 0.25 m was needed, one waypoint repeated 15+ calls, "done" before the env's success.
- **Astra on the same harness:** H 2 / 2 (VideoUnmaskSwap, VideoPlaceOrder).
- **RoboMME's `multi_choice` is 2610.17350's "oracle-planner" setting** (their README §"Oracle-planner diagnostics";
  per-episode outcomes in their `analysis/oracle_planner/oracle_planner_outcomes.csv`, read 2026-10-29). Base
  setting, 800 test episodes: **gpt-6-astra 622 / 800 (77.8%), gpt-4o 246 / 800 (30.8%)**; VideoUnmaskSwap 39 / 50
  and 10 / 50, VideoPlaceOrder 45 / 50 and 9 / 50. So the skill interface is **not** a perfect-execution upper bound:
  Astra scores lower there than with waypoints (92.1%), and their base setting shows one frame in 30. This corrects
  HANDOFF_EXECUTION option 5: `multi_choice` is a different setting to compare against, not a ceiling.
- **Truth is reachable:** 2610.17350's `analysis/env_facts.py` reads task parameters from `env.unwrapped`
  (`u.target.pose.p`, `u.button`, ...), so object poses come from the unwrapped ManiSkill env. RoboMME also exposes
  `maniskill_obs` and `*_subgoal_online` info fields (their README; never shown to any model here).
- `arbiter_kb/skills.py` already has a four-waypoint top-down `pick` (SAFE_Z 0.20, yaw a TODO, no reach check).

## 3. Strategy
1. **Truth first, LLM last.** Every test scores against the simulator (object poses, grasp state, lift height), never
   against another model's judgement.
2. **Cheapest test that answers the question.** Order: no-LLM (B0, B1) → one call per sample, no execution
   (A, B2', C) → a few calls with execution from a fixed start (B2) → end-to-end. Each stage gates the next.
3. **Same starts for model and script.** B2 starts from the exact states B0 uses, so the difference is the model.
4. **Record once, score many.** The pod writes a state bank (frames, state text, each arm's memory text, truth) once;
   A, B2' and C run on the Mac from it, for any number of models, with no simulator.
5. **Two tasks first:** VideoUnmaskSwap (lift a container, pick a cube) and VideoPlaceOrder (pick and place in order),
   the pilot's tasks. Others only after these are read.
6. **Models:** qwen3.6-35b-a3b (the pilot's), qwen3.8-27b, gemini-3.5-flash; gpt-6-astra as the reference on a small
   sample. One model list, fixed before the first paid call.

## 4. The tests

### B0 — scripted ceiling (layers 2–4, no LLM)
- **Question:** given true poses, does a dumb top-down pick / place succeed in this env with this planner?
- **Procedure:** for each val episode 0–9 of both tasks, at the first moment the robot may act, take every object the
  task can require picking and that is not covered (VideoUnmaskSwap: the containers; VideoPlaceOrder: the cubes); run `skills.pick(true_xyz, true_yaw)`; separately, for VideoPlaceOrder, pick a cube and
  `skills.place` it on a true target. Fresh env per trial (only the first reset gives the episode's scene).
- **Success (pick):** the simulator says the gripper holds the object and its z rose ≥ 0.10 m, held for 20 control
  steps. **(place):** the object rests within 2 cm (xy) of the target after release.
- **Rule (proposed):** B0 ≥ 95% per task. Below that, fix the skill (or report the env / planner failure, e.g. the
  zero-step `follow_path` bug) before any test with a model. Also record what lift height the env's own success needs
  (VideoUnmaskSwap: 0.25 m in the smoke run, not 0.20).
- **Cost:** pod GPU only.

### B1 — tolerance curve (how accurate must "where" be?)
- Same trials as B0, with the commanded position offset by 1, 2, 3, 5 cm in a random xy direction (seeded), and
  separately yaw offset 10°, 20°, 45°; also z offset ±1, ±2 cm for the closing height.
- **Output:** success vs error for each axis. **Tolerance** = the largest xy error with success ≥ 90%; this becomes A's
  pass threshold (so A's threshold is measured, not guessed).
- **Cost:** pod GPU only (~5 × B0).

### A — does the model find the right object and place? (layer 1)
- **State bank:** for val episodes 0–9 of both tasks, the states at the first action and after each subtask, recorded
  from B0 runs (scripted, so the states are clean and known). For each: H's content (grids, current frames, state
  text), R's log, K's view, and truth.
- **Call:** the arm's normal first-turn request (unchanged prompts), one call per state; read `target` and the first
  waypoint.
- **Scores:** right object (by truth, and for H by the waypoint's nearest true object); xy error of the waypoint to the
  true object; within B1's tolerance.
- **No-LLM rows:** the perception's own error (KB / tracker position vs truth, free) — the floor for R and K; and
  "nearest object to the goal word" heuristics where they apply.
- **Rule (proposed):** report per arm and model; an arm "finds" if right object ≥ 90% and within tolerance ≥ 80%.
- **Cost:** ~40 states × 3 arms × models, one call each.

### B2' — does the model write a correct pick sequence? (layers 2–3, no execution)
- **Call:** from A's states, with the true object position given in the state text, ask for the whole sequence
  (waypoints + gripper) in one answer.
- **Rule-based score (proposed, all must hold):** (1) a pre-grasp waypoint above the object (xy within tolerance,
  z ≥ object top + 0.08 m) before any descent; (2) gripper open before the descent; (3) close at z within ±1.5 cm of
  the grasp height measured in B0; (4) a lift to at least the height B0 found the env needs; (5) top-down
  orientation with yaw within B1's yaw tolerance of the object's faces.
- **Output:** the share passing all rules, and which rule fails most (expect (3) and (4) from the pilot).
- **Cost:** one call per state; text-light (no image history needed).

### B2 — the model as controller from a fixed start (layers 2–4 executed)
- **Procedure:** restore B0's start state; give the model the normal harness prompt plus the true position of the
  object to pick; let it run the pick subtask only (≤ 8 calls); score as B0.
- **Comparison:** B2 vs B0 on the same starts = the loss from making the model the controller. Variants (each a
  stated harness change): the procedure written in the prompt ("pre-grasp at +0.10 m, close at the object's centre
  height, lift to 0.25 m"), to test "knowledge summary" against "code skill".
- **Cost:** ≤ 8 calls per trial; run after B2' says it is worth it.

### C — false "done" (layer 5)
- **State bank:** from B0 / B1 runs, labelled by truth: done (lifted enough; placed), and near misses (lifted 0.15 m;
  gripper closed but empty; placed on the wrong target; not yet released).
- **Call:** the arm's normal request at that state; the model's answer is "done" if `waypoint` is null.
- **Scores:** false-done rate on near misses (FailBench's failure mode), miss rate on done states.
- **Variant:** K's view with a completion-facts section from `self_memory` (grasped? lifted to z? placed where?;
  option 4). Same states, so the difference is the facts.
- **Cost:** one call per state.

### D — decision with skills (`multi_choice`), later
- Run RoboMME's `multi_choice` on the two tasks for our models, to compare with 2610.17350's Astra / gpt-4o rows. Only
  after reading RoboMME's `multi_choice` code on the pod (what the options are, which frames the model sees). Not an
  upper bound (§2).

### Order and gates
B0 → B1 → (A, B2', C in parallel; same state bank) → B2 → end-to-end. If B0 < 95%: stop and fix execution code. If A
fails for H but passes for K: the memory gap is real at layer 1. If B2' fails on a model: B2 for that model only
with the prompt variant. End-to-end episodes are next spent only on models / arms that pass A–C.

## 5. Code changes

| file | change | runs on |
|---|---|---|
| `stress/truth.py` (new) | `truth(env, task) -> dict`: object poses (position, yaw about z) from `env.unwrapped` actors, per task a name map (cubes, containers / bins, targets, buttons); `grasped(env, actor)` from the agent's `is_grasping` if RoboMME's agent has it, else finger width + object lifted with the TCP; the env's own success / status. Never shown to a model. | pod |
| `arbiter_kb/skills.py` | finish `pick` (yaw from the true yaw folded to the gripper's ±45°; closing height and lift as parameters, lift default from B0; the reach check from `controller.check`); add `place(xyz)`; add `noise=(dxy, dyaw, dz, seed)` for B1. Unit tests for the geometry. | Mac + pod |
| `stress/scripted_pod.py` (new) | B0 / B1 runner: fresh env per trial, act from `truth`, score with `grasped` and poses; resumable `result.json` like `closed_loop_pod.py`; writes the state bank (below) at the first action and after each subtask. | pod |
| `stress/state_bank.py` (new) | save / load one state: front and wrist frames needed by each arm (H's sampled frames, current frames), `closed_loop.state()`'s text, R's log and K's view (built through the same `Memory`), the truth dict, the label (C). `.npz` + `.json`, copied to the Mac. | pod writes, Mac reads |
| restore a start state | B2 needs it. First choice: ManiSkill's `get_state_dict` / `set_state_dict` on `env.unwrapped` if RoboMME's wrappers do not break it (to check: the recorder's frame lists, `elapsed_steps`). Fallback: rebuild by a fresh env + replaying the recorded action prefix (the sim is deterministic per seed; checked by comparing truth after replay). | pod |
| `arbiter_kb/layer_tests.py` (new) | scoring only, no I/O: A's right-object and error, B2''s five rules, C's false-done / miss; plus the request builders that reuse `controller.user_content`, `closed_loop.h_content` / `_fresh` unchanged, and B2''s "whole sequence" request. Tests in `tests/test_layer_tests.py`. | Mac |
| `scripts/probe_layers.py` (new) | A / B2' / C runner over the state bank: `--test A|B2p|C --arms H,R,K --models ...`, cached calls via `llm.LLM`, budget cap, results per (state, arm, model). | Mac |
| `stress/closed_loop_pod.py` | `--start-from <bank state>` and `--subtask pick:<obj>` for B2 (stop when `grasped` and lifted, ≤ 8 calls); `--prompt-variant procedure` for B2's knowledge variant. Default behaviour unchanged. | pod |
| `arbiter_kb/self_memory.py` | (C's variant only) a completion-facts section in K's view: grasped / lift height / placed where, from the recorded actions and the KB. A stated K-arm change, re-frozen. | Mac |

Unchanged: the controller prompts, H's grids, the endings, `closed_loop.run` for end-to-end episodes. Any change that
reaches an end-to-end arm (the prompt variant, the completion facts) is a stated deviation, applied to every arm it can
apply to, and re-frozen.

## 6. To check on the pod before writing B0 (free, no LLM)
1. The actor names per task on `env.unwrapped` (cubes, bins, targets), and their yaw.
2. Whether the agent has `is_grasping(actor)` under RoboMME's wrappers.
3. Whether `get_state_dict` / `set_state_dict` round-trips (truth and the next step's frames identical), or the replay
   fallback is needed.
4. What `skills.pick` does today on one VideoUnmaskSwap episode (smoke, 1 trial), including the zero-step bug.
5. RoboMME's own scripted solvers (`solve_pickup_bin` etc., the dataset's generator): if they run from truth, B0 can
   reuse them instead of our skill (and say so).

## 7. Cost (rough)
- B0 + B1: about 10 episodes × ~6 objects × (1 + ~12 offsets) trials, ~20–40 s each on the A40 → 4–8 GPU hours.
- A + B2' + C: ~40–80 states × 3 arms × 3 models, one call each → a few hundred calls; H's requests carry the image
  grids. Under $5 without Astra; Astra on 20 states per test adds roughly $5–10.
- B2: ~20 trials × ≤ 8 calls per model.
