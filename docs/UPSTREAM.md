# Upstream contributions: candidates

Bugs and pitfalls we hit in other people's code while porting the 2610.17350 harness to RoboMME, with what we
know, what still needs checking, and what we would send. Nothing here has been filed. Filing is the user's to do
from their own GitHub account; we prepare the text and a minimal reproduction.

Order of work for each item: reproduce on the latest upstream version → search existing issues → open an issue with
a minimal repro → offer a PR once a maintainer agrees.

## 1. ManiSkill: `follow_path` crashes on a zero-step path (high priority)
- **Where:** `mani_skill/examples/motionplanning/base_motionplanner/motionplanner.py`, `follow_path`. Still
  present on `haosulab/ManiSkill` main (checked 2026-10-30; latest release v3.0.1, the pod runs 3.0.0b21).
- **Bug:** `obs, reward, terminated, truncated, info` are assigned only inside `for i in range(n_step + refine_steps)`.
  When the planner returns a path with 0 steps (the target pose is the current pose, or numerically within the
  planner's tolerance) and `refine_steps == 0`, the loop body never runs and the `return` raises
  `UnboundLocalError: cannot access local variable 'obs'`.
- **Effect seen:** RoboMME's waypoint wrapper catches it as a planning failure, so the episode ends `env_error`.
  3 of the 4 `env_error` endings in our Qwen val pilot were this, not a real planning failure. 2610.17350's
  prompt works around it ("a waypoint identical to the current TCP pose also fails: move by at least 2 mm").
- **Proposed fix:** with no steps to follow, return the env's current observation without stepping (or step once
  with the current qpos), so "already there" is a no-op rather than an exception.
- **To do first:** a 10-line repro (reset a pick task, plan to the current TCP pose, call `follow_path`); search
  ManiSkill's issues (our `gh search` was refused, so not yet checked).

## 2. RoboMME: zero-displacement waypoints (medium, depends on 1)
- **Where:** RoboMME's waypoint action path (`robomme/robomme_env/utils/planner_denseStep.py`,
  `env_record_wrapper/MultiStepDemonstrationWrapper.py`), which calls ManiSkill's planner.
- **Proposal:** independent of a ManiSkill fix, treat a waypoint within a small tolerance of the current TCP pose
  (with only a gripper change) as "apply the gripper, no motion", instead of planning. Any LLM or policy that
  issues "close here" hits this today.
- **To do first:** confirm in RoboMME's current code that no such guard exists; ask whether they prefer the guard
  in RoboMME or the fix in ManiSkill.

## 3. RoboMME: a second `reset()` on the same env gives another scene (low; docs)
- **Seen:** `make_env_for_episode(e)` then `reset()` twice: the second reset gives a different episode (169 vs 151
  frames of opening video on VideoUnmaskSwap val ep 0). We had probed the goal with one reset and played on the
  second, so arms ran on different scenes until we used a throwaway env for the probe.
- **Likely by design** (the env iterates episodes). Proposal: one sentence in the README / docstring that each
  `make_env_for_episode` env is good for exactly one `reset()`, or a warning on the second.
- **To do first:** read the wrapper to confirm the intent.

## 4. 2610.17350's harness (jackbjed/embodied-astra-robomme): practical notes (low; docs or issue)
Things anyone re-running it on other models or providers will hit; useful as a "running with other models" note:
- OpenRouter's pre-flight estimate counts ~1,536 tokens per image, so the image-history arm on long-video tasks
  (VideoPlaceOrder: 360 sampled frames on the first turn) is rejected before any call, although the model uses
  ~66 tokens per 256-px frame. Grids (3 × 3 frames per image, every frame kept) fix it.
- Some providers cap images per request (DeepInfra: 30).
- Reasoning models served greedily (temperature 0) loop until the token cap; Qwen's thinking settings plus a
  presence penalty stopped it (our loop probe: 0 / 15 vs 3 / 15 first turns).
- Once item 1 is fixed upstream, the "move at least 2 mm" note can point to it.

## 5. Later, once we have results: a memory-module interface for that harness
Our port (`arbiter_kb/closed_loop.py`) keeps their loop and makes the memory the only switch between arms (image
history, raw tracker log, KB view). Offering that interface back, with the KB as one plug-in, is a contribution in
itself, but only after the closed-loop results are in and the paper says what the KB is worth.

## Not upstream bugs (ours, already fixed; listed so we do not file them)
R's target / cite checks, the `"table"` KeyError, generic cube slots sharing a name, never-seen slots in the
views, the retry not reaching fresh-request arms. "target." not detecting flat discs is a limit of the detector's
vocabulary, not a bug.
