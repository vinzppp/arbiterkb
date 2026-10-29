> **Superseded (2026-10-28) by `DEVELOPMENT.md`.** Kept for history; build from DEVELOPMENT.md.

# Notebook instructions: ArbiterKB, a knowledge base where classical algorithms propose and an LLM arbitrates (offline, RoboMME permanence)

## Goal
Build one Jupyter notebook that walks, step by step and runnable, from cached RoboMME replays to an offline evaluation of **ArbiterKB**:

> A robot knowledge base (KB) is the central carrier of memory. Classical algorithms (tracking, back-projection, Hungarian matching, change detectors, forward rules) write **hypotheses** into it, each with a belief: confidence, time and evidence. A small LLM, the **arbiter**, periodically reviews those hypotheses as a whole, reconciles them and writes validated updates. From the KB, a planner outputs the **key pose (position + orientation)** and **which skill** to run (pick / place / press / verify), which an IK + motion-planning executor would carry out.

Research question for this notebook:
> When the memory is wrong, does the system know? Compared with deterministic memory (HODOR-frozen, bound rule), does ArbiterKB (rules only, then rules + LLM arbiter) keep accuracy on clean input, stay calibrated, and cut the **silent-failure rate** (wrong with high confidence) under memory corruption, at a small cost per episode?

The notebook teaches and evaluates. It must:
- reuse the existing code in `~/Code/active/scene-graph-robot` (read-only);
- split the work into small steps;
- after every step, show the intermediate result: a printed shape and sample with its meaning, and an image or plot wherever the result is visual;
- end with a printed one-line verdict.

All reusable code lives in the package `arbiter_kb/`, which the command-line scripts also import, so the full evaluation later runs exactly the same code outside the notebook. CLI scripts are thin wrappers and can be completed after the notebook (see Output).

## Run modes
Define `RUN_MODE` in the first code cell. Every heavy cell reads it.
- `"smoke"`: VideoUnmaskSwap, 2 training episodes + 2 test episodes, oracle and Q2 inputs only, arbiter backend `RuleOnly` + `Cached` (one hand-written cached response). CPU only. Must finish in under 10 minutes. **This is the mode the notebook is executed in when it is built.**
- `"full"`: VideoUnmaskSwap, train episodes for tuning (≤ 20), all 50 test episodes, every input condition, arbiter backend `LocalHF` if a GPU is present (else `RuleOnly`). Run by the user later.

Flags (default off), checked by every GPU cell: `USE_GPU_PERCEPTION`, `USE_LOCAL_LLM`, `RUN_CLOSED_LOOP`. **Never turn one on during the build.** If a flagged cell is skipped, print `SKIPPED (flag off)` and continue.

## Environment
- Primary: the RunPod pod described in `scene-graph-robot/notebooks/pilot/POD_RUNBOOK.md` (network volume at `/workspace`, venvs `/workspace/envs/sg-main` and `sg-percep`). The cached replays and the 23 GB VideoUnmaskSwap H5 live there. The volume has only about 12 GB free: write nothing large.
- Steps 1–12 need **no GPU and no simulator** if the caches hold what Step 2 needs. They may run on a CPU pod or on the Mac with a copied subset (Step 2.5 decides).
- GPU only for the flagged cells: the perception worker (Grounding DINO → SAM → Cutie, `sg-percep` venv, a separate process because transformers 4.44 conflicts with the main env) and a local LLM.
- Don't spend GPU money without the user's explicit go. Stop the pod when idle.
- The user is new to pods, tmux and git: when a step needs an action from them, print the exact command and where it runs (Mac vs. pod).

## Materials (read all before writing code)
1. **Design**: `scene-graph-robot/docs/system-design-kb-2026-10-27.md` (KB layers L0–L4 + belief overlay, rules R1–R8, model vs. algorithm split). Where this file and the design doc differ, this file wins: the KB stores data by type (object table, relation distributions, event log) and generates predicate tuples only as a view.
2. **Motivation**: `scene-graph-robot/docs/close-reading-simplearm-ledger-2026-10-27.md` and `docs/memory-landscape-2026-10-27.md` (25/27 SimpleARM failures had internally accepted but wrong bindings; no system measures calibration or silent failure).
3. **Existing code**: `scene-graph-robot/notebooks/pilot/sg_pilot.py`, `BUILD_NOTES.md`, `POD_RUNBOOK.md`. Reuse (print the file and line of each when first used):
   - `Replay` (front/wrist RGB, depth in metres, segmentation, `K_front`, `E_front[T,3,4]` world→camera OpenCV, joint/gripper state, `ee_xyz`, `is_video`, `scene`, `obj_xyz` ground-truth positions), `load_replay`, `load_training_episode`.
   - `backproject_mask`, `project_points`, `cam_to_world`.
   - `oracle_graph`, `GraphTracker`, `RuleParams`, `apply_rule`, `target_cube_nodes`, `decision_frame`, `ActiveTarget`, `active_targets`.
   - `corrupt`, `draw_corruption`, `q2_fires` (Q1/Q2).
   - `PerceptionWorker`, `make_robomme_processor`, `PerceivedOnline`, `CONTAINER_MIN_AREA_512`.
   - `assign_slots`, `identity_accuracy`, `draw_markers`, `serialize`, `wilson`, `load_motion`, `pass_fail`.
4. **Course knowledge-base code**: `Robotic Intelligence/Module 7/module07_r_deliberative.pdf` (`unify`, `satisfy`, `instantiate`, `TELL`, `ASK`, forward chaining) and `Module 8/module08_r_knowledge.pdf` (knowledge representation forms, planner–executor–verifier–replanner). The predicate view and the notebook text use this vocabulary.
5. **Benchmark**: RoboMME paper https://arxiv.org/abs/2603.04639; env repo https://github.com/RoboMME/robomme_benchmark (`doc/env_format.md`: action types `joint_angle`, `ee_pose`, `waypoint`, `multi_choice`; world frame +x forward, +y left, +z up, table at z = 0, Panda base at (−0.615, 0, 0), extrinsic XYZ Euler, gripper −1 closed / 1 open).
6. **Comparison point**: arXiv 2610.17350 (frozen frontier VLM + `waypoint` executor: 92.1% overall, VideoUnmaskSwap 96%, about 9.5 calls per episode, 18.3 s per call, about $0.39 per episode).

**Facts to verify, not assume.** Print the evidence for each:
- Which replay caches exist on `/workspace` for VideoUnmaskSwap train/test, and whether each holds front RGB, depth, segmentation, `E_front`, gripper state and `obj_xyz`. The handoff says the slim caches store no DINOv2 features; it does not say whether they hold segmentation and depth. If a field is missing, write exactly what is missing and either (a) rebuild it with `replay_episode` (needs a GPU + Vulkan pod: ask first) or (b) continue with what exists.
- `cache/VideoUnmaskSwap_test/motion.json` gives 31 moved and 19 back-at-start test episodes.
- `oracle_graph` on training episode 0 reproduces the frozen vs. bound gap (frozen error at t* of tens of px; bound under 5 px).
- How RoboMME executes `action_space="waypoint"` (planner library, steps per waypoint, failure signalling), from the source under `src/robomme/`. Code reading only; record it in `NOTES.md`.

## Code layout (create these files under `~/Code/active/arbiter-kb/`)
```
arbiter-kb/
  INSTRUCTIONS.md              this file
  README.md                    what ArbiterKB is, how to run (Mac and pod), status
  NOTES.md                     engineering log: decisions, gotchas, measured numbers
  requirements.txt
  arbiter_kb/
    __init__.py
    beliefs.py                 Belief, Distribution (normalised hypotheses), decay
    kb.py                      ObjectFrame, Evidence, Event, KB (objects / relations / evidence / events / task),
                               tell / ask / retract with truth-maintenance cascade, count()
    views.py                   predicate view (course unify/satisfy), text view, Set-of-Mark image view
    sources.py                 OracleSource, PerceivedSource (cached worker output), CorruptedSource (Q1/Q2)
    geometry.py                centroid back-projection (wraps sg_pilot), optional constant-velocity Kalman smoothing
    visibility.py              visibility state machine
    identity.py                Hungarian identity check with margin → Evidence
    detectors.py               task-agnostic change detectors → Evidence
    rules.py                   R1–R8 + belief updates; baselines Frozen and BoundDeterministic
    monitor.py                 consistency checks, risk flags, decision thresholds
    arbiter.py                 triggers, keyframe selection, prompt builder, JSON schema, validator, apply_patch
    llm_backends.py            RuleOnly, Cached, LocalHF
    planner.py                 rule planner: KB → (skill, key poses); skill library pick/place/press/verify
    fast_loop.py               runs one episode frame by frame through sources → KB → rules → monitor (→ arbiter)
    evaluate.py                metrics and tables
    plots.py
  scripts/
    run_offline_eval.py        CLI over evaluate.run(...)        (thin; may be completed after the notebook)
    build_obs_cache.py         CLI to cache perceived masks/centroids from the worker (GPU; written, not run)
  tests/
    test_beliefs.py test_kb.py test_views.py test_rules.py test_identity.py test_detectors.py test_arbiter_validator.py test_planner.py
  arbiter_kb.ipynb             the notebook; imports arbiter_kb, explains each function where first used
  figures/                     every plot the notebook saves
  results/                     small JSON/CSV results; results/llm_cache/ for every LLM call
```
The notebook calls the same functions as the scripts. Show each key function's source once, at the step where it is first explained (`inspect.getsource`).

## Core data structures (implement as specified; record any change and its reason in NOTES.md)
```python
@dataclass
class Belief:
    value: Any
    conf: float            # 0–1
    t: int                 # frame of last confirmation
    source: str            # "observed" | "inferred" | "prior" | "verified" | "arbiter"
    support: list[str]     # evidence / rule / event ids (truth maintenance)

@dataclass
class Distribution:        # mutually exclusive hypotheses, always normalised
    probs: dict[str | None, float]   # {"cup_B": .55, "cup_C": .45}; None = unknown/elsewhere
    t: int
    support: list[str]
    def argmax(self) -> tuple[str | None, float]: ...
    def entropy(self) -> float: ...
    def mix(self, other: "Distribution", w: float) -> "Distribution": ...

@dataclass
class ObjectFrame:         # L1 with L0 geometry
    id: str; cls: str; description: str; attributes: dict
    pose: Belief           # value np.array([x, y, z]) world metres (+ optional yaw)
    extent: np.ndarray | None
    visibility: Belief     # visible | occluded_by_arm | hidden | held | out_of_view | unknown
    track_quality: float
    t_last_seen: int
    stationary: list[tuple[int, int, np.ndarray]]

@dataclass
class Evidence:            # algorithm output, append-only, ids "e1", "e2", ...
    id: str; t: int
    kind: str              # move_start | move_stop | overlap_start | overlap_end | appear | disappear
                           # | gripper_close | gripper_open | track_drop | id_ambiguous | risk
    args: tuple; data: dict

@dataclass
class Event:               # L3 semantic event from rules or the arbiter; always cites evidence
    id: str; t: int
    type: str              # cover | swap | pick | place | reveal | press
    args: tuple; conf: float; support: list[str]; source: str   # "rule" | "arbiter"

class KB:
    objects: dict[str, ObjectFrame]                 # L0 + L1
    relations: dict[tuple[str, str], Distribution]  # L2, key (predicate, subject), e.g. ("Inside", "cube_green")
    evidence: list[Evidence]
    events: list[Event]                             # L3
    task: dict                                      # L4: goal, plan, progress, status
    def tell(...); def ask(pattern) -> list[tuple[dict, float]]; def retract(id: str)
    def count(event_type: str, obj: str) -> int     # derived from events, never stored
    def snapshot(t: int) -> dict
```
Invariants (tested): positions are attribute values, never part of a key; exclusive hypotheses live in one `Distribution` that sums to 1; counts come only from `count()`; every inferred or arbiter-written item has a non-empty `support`; the arbiter can never write a coordinate.

## Steps
Each step:
- opens with one Markdown line on what goes in and what comes out (shapes and meaning), plus one sentence placing it in the architecture (perception / KB / arbiter / planner / executor);
- runs;
- shows the listed intermediate output;
- ends with a one-line check that prints PASS or FAIL with the measured value.

### Step 1. Setup
1.1. Print Python, numpy, scipy, matplotlib and (if present) torch / transformers versions; whether a GPU is visible; `RUN_MODE` and the flags. Check: imports succeed.
1.2. Add `scene-graph-robot/notebooks/pilot` to `sys.path`; import `sg_pilot`; print the file and line of every reused function listed in Materials 3. Check: all found.
1.3. Set seeds (Python, numpy) to 0. Print the episode lists and input conditions this run mode implies.
1.4. Draw the architecture diagram (matplotlib boxes and arrows): perception → fast loop → KB (L0–L4 + beliefs) ↔ monitor ↔ arbiter (LLM) → planner → skills → IK executor. Save to `figures/01_architecture.png`. Check: file exists.

### Step 2. Data audit
2.1. List the replay caches found for VideoUnmaskSwap train and test with sizes. Check: at least the smoke episodes exist.
2.2. Load one training replay. Print every field with shape, dtype, units and meaning. Check: depth in metres (max < 5) and `E_front` shape `[T,3,4]`.
2.3. Show front RGB, depth and segmentation at 4 key frames (covered, mid-swap, swap done, t*). Save to `figures/02_frames.png`.
2.4. Reproduce the frozen and bound errors at t* on that episode with `oracle_graph` + `apply_rule`. Check: bound < frozen and bound < 5 px.
2.5. Estimate the size of a CPU-only subset (front RGB, depth, segmentation, `E_front`, gripper, `obj_xyz`, metadata) for the full-mode episodes. Print it and the exact command to copy it to the Mac. Do not copy during the build.
2.6. Record in `NOTES.md` how the `waypoint` action type is executed (from source reading). Print a 3-line summary.

### Step 3. The knowledge base (no data needed)
3.1. Implement `beliefs.py`. Show `Distribution` normalisation, `argmax`, `entropy`, `mix`, and `Belief` decay on a toy example. Check: probabilities sum to 1 after every operation.
3.2. Implement `kb.py`. Build by hand a three-cup scene at one frame (3 containers, 3 cubes, one hidden cube with `Inside` = {cup_B: .55, cup_C: .45}). Print the object table and the relation table.
3.3. Implement the predicate view with the course's `unify` / `satisfy`. Run `ask(("Inside", "cube_green", "?c"))` and `ask(("Hidden", "?x"))`. Check: returns `[({"?c": "cup_B"}, .55), ({"?c": "cup_C"}, .45)]`.
3.4. Truth maintenance: add evidence `e20` supporting the `Inside` distribution and a derived `At(cube_green)` belief; `retract("e20")`. Print before and after. Check: every item whose support included `e20` is downgraded or removed.
3.5. Text view (what the arbiter reads), one line per object, e.g. `[cube_green] cube, hidden since t=56; inside cup_B (.55) | cup_C (.45); last swap t=105 had overlap`. Print it and its token count (whitespace tokens). Check: under 200 tokens for this scene.
3.6. Set-of-Mark view: draw ids and top-2 hypotheses with probabilities on a front image (extend `draw_markers`). Save to `figures/03_som.png`.
3.7. Run `pytest tests/test_beliefs.py tests/test_kb.py tests/test_views.py` from the notebook. Check: all pass.

### Step 4. Observation sources
4.1. `OracleSource`: per frame, per object, a mask from simulator segmentation (as in `oracle_graph`). Print the per-frame record format.
4.2. `CorruptedSource`: wraps another source and applies Q1 (σ = 20 / 50 px) or Q2 (wrong container in 13/50 or 25/50 test episodes, nested) through the existing `corrupt` helpers, only for the active target while hidden. Show one Q2 episode's corrupted vs. true target position at t*.
4.3. `PerceivedSource`: reads cached worker output (masks or centroids per frame). If no cache exists, print `SKIPPED (no cache; run scripts/build_obs_cache.py on a GPU pod)` and the command. Keep the known fixes in the worker config: phrases written as `"green cube."`; container boxes at least `CONTAINER_MIN_AREA_512` = 1200 px² at 512 px and not touching the top image edge.
4.4. Check: each available source yields records for every frame of the smoke episodes.

### Step 5. Fast loop I: geometry and visibility
5.1. Centroids from masks + depth via `backproject_mask`; optional constant-velocity Kalman smoothing (flag `SMOOTH`). Plot x, y over time for the three containers against `obj_xyz`. Save to `figures/05_centroids.png`. Check: median error to `obj_xyz` < 3 cm while visible.
5.2. Visibility state machine: visible / occluded_by_arm (EE within `d_occ` of the object's last position when the mask vanishes) / hidden / held (gripper closed within `d_grip`) / out_of_view. Plot a state timeline strip per object. Save to `figures/05_visibility.png`.
5.3. Check: on the smoke episodes, the target cube is `hidden` at t* and its covering container is `visible`.

### Step 6. Fast loop II: identity check (Hungarian matching with a margin)
6.1. Explain the assignment problem in 3 sentences and show a 3×3 cost example where greedy matching fails and `scipy.optimize.linear_sum_assignment` succeeds.
6.2. Build the cost matrix between KB containers and the current detections (3D distance to predicted pose + optional mask IoU term). Run every frame (oracle) or every `N_ID` frames (perceived).
6.3. Margin = (second-best total − best total) / best total. Emit `id_ambiguous` evidence when margin < `margin_thr`. Plot the margin over time with swap intervals shaded. Save to `figures/06_margin.png`.
6.4. Check: the minimum margin falls inside a swap interval on the smoke episodes.

### Step 7. Fast loop III: change detectors → evidence log
7.1. Implement task-agnostic detectors: move start/stop (centroid speed with hysteresis), overlap start/end (mask IoU or 3D distance under one container width), appear/disappear (mask area), gripper close/open, track-quality drop. Cap evidence per frame.
7.2. Print the evidence log of one training episode (id, t, kind, args, data).
7.3. Plot the evidence timeline against the ground-truth container motion from `obj_xyz`. Save to `figures/07_evidence.png`.
7.4. Check: every ground-truth swap produces move_start for both containers and an overlap or near-pass event; no frame emits more than the cap.

### Step 8. Rules and beliefs (the "RuleOnly" ArbiterKB)
8.1. Implement R1–R8 from the design doc:
- R1 cover: object disappears while a container overlaps and descends → `Inside` distribution.
- R2 follow: a hidden object's displayed pose is the probability-weighted container pose; decisions use the distribution.
- R3 held: gripper closed near the object → held; pose follows the EE.
- R4 reveal: container lifted (z > 0.15 m) and the object seen at its old spot → `verified`.
- R5 exclusivity: at most one cube per container; normalise the joint assignment over hidden cubes (enumerate permutations, N ≤ 3).
- R6 swap: a clean swap moves the probability mass; overlap or `id_ambiguous` mixes with weight `w_mix`.
- R7 decay: `conf ← conf · λ^Δt` for unconfirmed beliefs.
- R8 consistency: two objects at one position, a "hidden" object seen, an empty position where a container should be → downgrade + `risk` evidence.
8.2. Implement the baselines in the same interface: `Frozen` (HODOR) and `BoundDeterministic` (the existing bound rule, conf = 1).
8.3. Run the three systems on one Q2-corrupted training episode. Show the target's `Inside` distribution over time (stacked area) for RuleOnly, and the argmax for the baselines, with t* marked. Save to `figures/08_beliefs.png`.
8.4. Print KB snapshots (text view) just before the swap, just after the swap, and at t*.
8.5. Check: on clean oracle input RuleOnly's argmax equals BoundDeterministic's at t* on the smoke episodes; on the Q2 episode RuleOnly's confidence in the wrong container is below 1.

### Step 9. Monitor and decisions
9.1. Decision rule from the KB at a decision point: conf ≥ τ_hi → `act`; τ_lo ≤ conf < τ_hi → `act_on_argmax` (flag the risk); conf < τ_lo → `verify`. Print the decision trace for the smoke episodes.
9.2. Show how a `risk` evidence lowers confidence and changes the decision on the Q2 episode.
9.3. Check: thresholds are read from `results/params.json` (written in Step 12.1 or set to defaults in smoke mode).

### Step 10. The arbiter (LLM consolidation)
10.1. Trigger policy: move_stop after a multi-object motion, appear/disappear, gripper_open, and every `K` frames while anything is hidden. Print the trigger frames for one episode.
10.2. Keyframe selection (1–3 frames): the maximum-overlap frame, the motion start and stop frames, the lowest-margin frame. Show them with Set-of-Mark annotation. Save to `figures/10_keyframes.png`.
10.3. Prompt builder: system instructions (role: reconcile hypotheses; cite evidence ids; never output coordinates; output JSON only), KB text view, new evidence lines since the last call, keyframe references. Print one full prompt and its token count.
10.4. JSON schema and `validate(patch, kb)`: schema check; cited evidence ids exist; object ids exist; R5 holds; no numeric coordinates anywhere. On failure, one retry with the error message, then fall back to RuleOnly. Show one valid and two invalid patches (invented evidence id; a coordinate) and the validator's verdicts.
10.5. Probabilities: the arbiter proposes hypotheses and qualitative strength; numbers come from (a) fast-loop signals through R6, or (b) self-consistency over k = 5 samples (agreement rate). Implement both behind a flag; in smoke mode show (a).
10.6. Backends: `RuleOnly` (no call), `Cached` (responses keyed by prompt hash in `results/llm_cache/`), `LocalHF` (a small open model such as Qwen3-8B-Instruct, or Qwen3-VL-8B for keyframes, 4-bit if needed; flag `USE_LOCAL_LLM`). No paid APIs without the user's approval. Log calls, tokens and seconds per episode.
10.7. End to end on one trigger with the `Cached` backend: prompt → patch → validation → KB diff (before/after text view).
10.8. Check: every patch applied in smoke mode validates; the KB contains no arbiter-written coordinate; replaying from the cache gives an identical KB.

### Step 11. Planner output: skill + key poses
11.1. Rule planner: from the task goal (`target_cube_nodes`, `ActiveTarget`) and the KB, choose a skill (`pick`, `place`, `press`, `verify`) and the target object.
11.2. Skill library: `pick(c)` → 4 waypoints (pre-grasp 10 cm above, descend to grasp height, close, lift to z = 0.20) in RoboMME's waypoint format (xyz, roll-pitch-yaw, gripper ∈ {−1, 0, 1}); positions come only from the KB's L0. Print the waypoint list at t* for one episode.
11.3. Offline check against ground truth: the distance between the planned grasp position and the true target container position (`obj_xyz`) at t*. Plot it on the front image (projected waypoints). Save to `figures/11_waypoints.png`.
11.4. Check: on clean oracle input, the planned grasp is within 3 cm of the correct container in the smoke episodes.
11.5. (Flag `RUN_CLOSED_LOOP`, default off.) Closed-loop smoke: `make_env(..., action_space="waypoint")` with these waypoints on 1 episode; print success and per-step timing. Needs a GPU + Vulkan pod and the user's go.

### Step 12. Offline evaluation
12.1. Tune λ, `w_mix`, `margin_thr`, τ_hi and τ_lo on training episodes only; write them to `results/params.json` before any test episode is read. Print the chosen values.
12.2. Systems: Frozen, BoundDeterministic, ArbiterKB-RuleOnly, ArbiterKB+LLM (each probability method in full mode). Inputs: oracle, Q1 σ20, Q1 σ50, Q2 13/50, Q2 25/50, perceived (if cached).
12.3. Metrics at t* per active target: argmax accuracy; Brier score; ECE (10 bins); **silent-failure rate** (wrong with conf ≥ τ_hi); detection rate (wrong with conf < τ_hi or a risk flag); decision outcome (act-correct / act-wrong / verify); for the arbiter, calls, tokens and seconds per episode. Wilson intervals for rates. Split moved vs. back-at-start.
12.4. One summary table (CSV + printed), one reliability diagram per input condition (`figures/12_reliability_*.png`), and a silent-failure bar chart (`figures/12_silent.png`).
12.5. Cost comparison row: ArbiterKB calls / seconds / tokens per episode vs. the 2610.17350 numbers (Materials 6), clearly labelled as numbers from a different system and setting.
12.6. Check: the table regenerates from `results/` in one cell; Frozen and BoundDeterministic on oracle input reproduce Step 2.4.

### Step 13. Verdict and limits
13.1. Print one line: `ArbiterKB vs BoundDeterministic under Q2: silent-failure X% → Y%, accuracy clean A% → B%, arbiter cost C calls/episode` (in smoke mode, print `= ? (run in full mode)` for values not computed).
13.2. Markdown: what this shows and what it does not. State plainly that on stock VideoUnmaskSwap real tracking errors are rare (perceived ≈ oracle: identity 0.983, 0.3 px at t*), so Q2 corruption stands in for real failures until stress variants exist. Next steps: stress variants (fast or partly out-of-view swaps, camera shift), peek-allowed verify tasks, counting/order events (L3) for the other RoboMME suites, multi-view.

## Constraints
- Don't modify `scene-graph-robot`. Import from it, or copy a function with a comment naming its source file and line.
- Never fake an output. If something can't run, print what is missing and continue.
- No GPU, simulator or LLM use during the build (flags off). No paid APIs.
- Test episodes are read only in Step 12, after `params.json` is written.
- Units: metres, frames, world frame. `E_front` is world→camera (OpenCV `extrinsic_cv`). Depth in metres (ManiSkill raw int16 is mm: convert once).
- Seeds fixed; every result file records the git commit, `params.json` hash, input condition and run mode.
- Keep notebook cells short; logic lives in `arbiter_kb/`.
- Git: `git init` in `arbiter-kb` on the Mac (the linked shell has no git-lfs; this repo needs none). Move code to the pod by git bundle as in `POD_RUNBOOK.md`.

## Output
- `arbiter_kb.ipynb` executed in smoke mode, saved with outputs as `executed/arbiter_kb_smoke.ipynb`, and the unexecuted source `arbiter_kb.ipynb`.
- `arbiter_kb/` package with docstrings and the tests above, all passing.
- `scripts/run_offline_eval.py` and `scripts/build_obs_cache.py` as thin CLIs (argument parsing + one call into the package). They may be stubs with `TODO` bodies if the build runs out of time; list any stubs in `README.md`.
- `figures/` and `results/` populated by the smoke run.
- `NOTES.md` with: verified facts (Materials, "Facts to verify"), every deviation from this file and why, and measured numbers.
- `README.md` with a 10-line status: what runs, what is stubbed, the exact commands for full mode on the pod.
- A final 5-line summary to the user: what was built, what passed, what was skipped and why, the next command to run.
