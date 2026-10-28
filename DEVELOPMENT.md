# ArbiterKB — Development Spec

Version 1.3 · 2026-10-28 · supersedes `INSTRUCTIONS.md` (kept for history). v1.1: after the M0 reviews, M1 is rewritten, phase G (the gate) is pulled forward, M2/M6 are adjusted. v1.2: after the M0.1 reviews, M1 shrinks to plumbing (no tuning before G), OverlapHaircut is added, and E5c is enforced through a test-read ledger. v1.3: after the M1 reviews, silent failure is always reported with coverage and AURC, M2 (c) compares at matched coverage and adds MarginHaircut, G records every pass (swapped or not) for an ROC, R5 skips mixed cubes.

This is the single source of truth for building ArbiterKB. Design rationale lives in the plan report ("ArbiterKB 项目规划报告") and in `scene-graph-robot/docs/` (`system-design-kb-2026-10-27.md`, `evidence-gaps-2026-10-28.md`). This file says **how**.

To build a phase, tell the coding agent: *"Build phase Mx of `arbiter-kb/DEVELOPMENT.md`. Read §0–§5 first, then §6.Mx. Stop when every check in Mx's 'Done when' list passes; then the user runs `/review-phase Mx` (§1.1)."*

---

## 0. Ground rules

1. **One phase at a time.** Each phase leaves the whole pipeline runnable end to end and adds one measurable change to the results table.
2. **Interfaces first, implementations later.** The contracts in §4 are fixed in M0. Changing one needs a `NOTES.md` entry ("contract change: what, why") and a bump of `arbiter_kb.CONTRACT_VERSION`.
3. **Never fake an output.** If something cannot run, print exactly what is missing (`SKIPPED: <reason>; to enable: <command>`) and continue.
4. **Don't modify `~/Code/active/scene-graph-robot/`. Copy, don't import.** The few functions ArbiterKB needs from `notebooks/pilot/sg_pilot.py` are copied verbatim into one file, `arbiter_kb/sg_compat.py`. Each copied function carries `# copied from sg_pilot.py:<line> @ e2e06f9`. The set is `Replay`, `load_replay`, `target_cube_nodes`, `ActiveTarget`, `decision_frame`, `cam_to_world`, `project_points`, `backproject_mask`, `oracle_graph`, `RuleParams`, `GraphTracker`, `wilson`, plus the `F_*` constants. This keeps arbiter-kb self-contained: its results cannot change because the other repo or branch changed. `tests/test_sg_compat.py` imports the original `sg_pilot` when it is present and asserts identical outputs on a synthetic input; otherwise it is skipped. Nothing else from scene-graph-robot is used.
5. **Compute.** M0–M4 run on the Mac CPU. GPU appears only in M6 (simulator rendering and perception), and only after the user says go. The LLM is reached through OpenRouter from M4 onward.
6. **The user is learning.** Docstrings say *what, why and units*. Every phase adds a short walkthrough notebook (≤ 15 cells) that calls the package and shows intermediate results.

---

## 1. Engineering evals: the quality bar for every phase

Each eval is a concrete check that `make check` runs. A phase is not done until all pass.

| # | Eval | What it means here | Check (automated unless noted) |
|---|---|---|---|
| E1 | **Minimal** | The smallest code that does the job; no frameworks, no speculative abstraction | `scripts/quality.py`: every module ≤ 250 lines, every function ≤ 40 lines, ≤ 1 level of class inheritance, dependencies ⊆ the whitelist in §5.3. A new abstraction needs ≥ 2 call sites (reviewed in NOTES) |
| E2 | **Debuggable** | Any number in any table can be traced back to the observations that produced it | Every episode writes a trace (§3.9). `python -m arbiter_kb.debug explain --run R --ep E --cube C --t T` prints the support chain down to observation ids. A test asserts that the chain exists for every row of the results |
| E3 | **Correct by construction** | Invariants hold at every frame, not just at the end | `KB.check()` asserts the invariants (§3.2) after every `apply()` when `ARBITER_DEBUG=1` (the default in tests). Each rule has a unit test on hand-made frames |
| E4 | **Reproducible** | Same inputs → same bytes | Run twice → identical `summary.csv` hash (test). Every results file records the commit, args, `params.json` hash, contract version and LLM cache hit/miss counts |
| E5 | **Honest evaluation** | No future information, no test leakage, the same treatment for every system | (a) Causality test: a `PoisonedSource` raises if any frame after t is requested before step(t) returns. (b) Systems never see `GroundTruth`. (c) `evaluate.run` refuses test episodes unless the Params was loaded from a `params.json` whose stored hash matches, has `frozen: true`, and has a FREEZE entry in the append-only ledger `results/test_reads.log` (`arbiter_kb/ledger.py`); every test read appends a READ entry. (d) Baselines run through the same `MemorySystem` interface and the same evaluation code |
| E6 | **Explainable outputs** | Every decision says why | Every `PlanStep` carries `support` (evidence ids) and every arbiter patch cites evidence ids. Test: no inferred / arbiter item has empty support |
| E7 | **Fast feedback** | Iterate in seconds, not minutes | `pytest` < 60 s; the M0 synthetic evaluation < 60 s; median KB update < 1 ms per frame (printed by `--selftest`) |
| E8 | **Cost-bounded** | LLM spend cannot surprise you | Every call is cached on disk; the hard cap is `ARBITER_MAX_USD` (default 2.00) per run, after which the backend raises. Calls, tokens, seconds and dollars are counted per episode |
| E9 | **Stable interfaces** | Later phases don't rewrite earlier ones | `tests/test_contracts.py` snapshots the field names of every contract dataclass. A change fails the test until `CONTRACT_VERSION` is bumped and NOTES updated |
| E10 | **Measured, not claimed** | Every claim in a README or NOTES points to a results file | Review item: each number in README/NOTES is followed by `(results/<file>)` |

### 1.1 Two independent reviewers: the phase gate

`make check` proves the code does what it says. It does not prove the code is right, or worth building. So every phase ends with two reviews by agents that did not build it. They are Claude Code subagents in `.claude/agents/`, with read-only tools, and each starts from a fresh context:

| Agent | Question | Anchor | Report | Blocking verdict |
|---|---|---|---|---|
| `code-reviewer` | Is the code correct, honest and minimal? | DEVELOPMENT.md §1–§3 | `reviews/<phase>_code.md` | BLOCK: wrong math, failing check, leakage or causality violation, faked or hard-coded result, non-reproducible |
| `goal-reviewer` | Does this phase move a claim, from first principles, without self-deception? | `docs/GOALS.md` | `reviews/<phase>_goal.md` | OFF-TRACK: moves no claim, headline result true by construction, test leakage |

**Procedure** (run in Claude Code inside `arbiter-kb/`):
0. One-time setup on the Mac (the agent files ship in `setup/claude/` because `.claude/` cannot be written remotely): `cd ~/Code/active/arbiter-kb && mkdir -p .claude && cp -R setup/claude/agents setup/claude/commands .claude/`
1. The builder finishes the phase: `make check` passes and NOTES is updated.
2. Run `/review-phase <phase>`. Both agents run in parallel. The builder must not pass them its own summary of the work.
3. The user reads both verdicts. Each BLOCK / OFF-TRACK finding is either fixed (then re-review) or explicitly overruled by the user, with the reason written in `NOTES.md`. MAJOR findings get a fix or a NOTES entry. MINOR findings are optional.
4. Tag `phase-<M>-done`. The next phase starts only after the tag.

The goal reviewer also answers "is the next phase still the best next step?". If it proposes a cheaper experiment that decides more, the user decides, and DEVELOPMENT.md §6 is updated before building.

Design patterns that make E1–E3 cheap (mandatory):
- **Rules are pure functions that return operations.** `rule(view, frame, params) -> list[Op]`. Only `KB.apply(ops)` mutates state, and `apply` logs every op to the trace. Rules are then trivially unit-testable, and the trace is complete.
- **One source of parameters.** All tunables live in one frozen dataclass per component, serialised together into `params.json`. No magic numbers in rule bodies.
- **Plain data over objects.** Contracts are dataclasses with numpy arrays and dicts; no hidden state outside `KB`.

---

## 2. Architecture (from the plan report)

The design follows Murphy's hybrid architecture as taught in the course: a knowledge base at the centre, a reactive fast loop, and deliberative plan / execute / monitor. It makes two changes: every fact carries a **belief**, and a small LLM acts as an **arbiter** at key moments.

```
                    ┌──────────────── every frame (cheap, deterministic) ────────────────┐
 Source ──FrameObs─►│ geometry/visibility ─► detectors ─► identity check ─► rules R1–R8  │
 (synthetic /       │          (L0/L1)        (evidence)     (evidence)       (ops)       │
  oracle replay /   └───────────────────────────────┬────────────────────────────────────┘
  corrupted /                                       ▼  KB.apply(ops)
  perceived)                    ┌──────────────────────────────────────────┐
                                │ KB: L0 geometry · L1 objects · L2 relation│◄──── monitor (R8, risk,
                                │ distributions · L3 events · L4 task        │      decisions)
                                │ + evidence log · beliefs · support (TMS)   │
                                └───────┬──────────────────────▲───────────┘
                     trigger (rare)     │ text view + keyframes │ validated patch → ops
                                        ▼                       │
                                 arbiter (small LLM via OpenRouter, JSON only, cites evidence)
                                        │
 decision point ──► planner: KB → (skill, target, conf, decision) ──► skills: target → 4 waypoints
                                                                     ──► executor (RoboMME waypoint,
                                                                          offline: evaluated vs truth)
```

| Report concept | Code |
|---|---|
| L0–L4 + belief layer + TMS | `arbiter_kb/kb.py`, `beliefs.py` |
| Perception input | `sources/*` (only sources know where observations come from) |
| Change detection, Hungarian identity check | `detectors.py`, `identity.py` (M2) |
| Rules R1–R8 | `rules.py` |
| Monitor, decision rule | `monitor.py` (M3) |
| Arbiter + validator | `arbiter.py`, `llm.py` (M4) |
| Planner + skills | `planner.py`, `skills.py`, `reach.py` (M5) |
| Baselines (HODOR frozen, bound; from M1 also BoundHaircut, OverlapHaircut, JumpDetector) | `systems/baselines.py` |
| Evaluation, metrics | `evaluate.py`, `metrics.py` |
| Debugging | `trace.py`, `debug.py` |

---

## 3. How the framework works

### 3.1 One episode

```
system.reset(meta)
for frame in source.frames(ep):          # strictly one frame at a time (E5a)
    system.step(frame)                   # fast loop; may call the arbiter on a trigger
    if frame.t == gt.t_star (evaluation only; the system is not told):
        record system.believe_inside(target), system.plan(target)
```

Inside `ArbiterKBSystem.step(frame)`, the order is fixed:
1. `ops = observe(view, frame)`: L0 poses and L1 visibility from the observations (§3.4).
2. `ops += detect(view, frame)`: new evidence (appear / disappear / move_start / move_stop / near_pass / gripper / track_drop). From M2: `+ identity(view, frame)` → `id_margin` evidence.
3. `ops += rules(view, new_evidence, frame)`: R1–R8 (§3.5).
4. `kb.apply(ops)` → trace.
5. `ops = monitor(view)` → risk evidence and downgrades (M3) → `kb.apply`.
6. If `trigger(view, new_evidence)` and the budget allows: `patch = arbiter.review(kb.text_view(), new_evidence, keyframes)` → `validate` → `patch_to_ops` → `kb.apply` (M4).

`view` is a read-only snapshot accessor on the KB, so rules cannot mutate.

### 3.2 KB data model

- `objects: dict[id, ObjectFrame]` (L0 + L1). The pose is a `Belief` whose value is an xyz array; visibility is a `Belief` over {visible, hidden, held, occluded_by_arm, out_of_view, unknown}.
- `relations: dict[(pred, subject), Distribution]` (L2), e.g. `("Inside", "cube_green") → {bin_1: .55, bin_2: .45, None: 0}`.
- `evidence: list[Evidence]` (append-only, ids `e1, e2, …`); `events: list[Event]` (L3, append-only, always cite evidence).
- `task: dict` (L4): goal, plan pointer, progress. Counts are derived by `count()` and never stored.
- `support` is a list of ids on every Belief, Distribution hypothesis and Event. `retract` scans all items (cheap at this size; a reverse index can come later if profiling asks for it).

Invariants (`KB.check()`): (I1) every Distribution sums to 1 ± 1e-9; (I2) no position inside any key; (I3) counts only via `count()`; (I4) every `inferred` / `arbiter` item has non-empty support, and all support ids exist; (I5) no arbiter-sourced numeric pose anywhere; (I6) at most one live `Inside` distribution per cube.

`retract(id)` (as implemented in M0.2): every item whose support contains `id` (transitively, through events) is downgraded, and every step is logged.
- A Belief gets `conf ← conf · 0.5` with the id removed from its support. Its source stays `inferred` while support remains, otherwise it becomes `prior` (keeps I4).
- A Distribution hypothesis backed by `id` loses its **current** mass, and the rest is renormalised; if nothing is left → `{None: 1}`. After a `swap_mix`, each hypothesis keeps its own evidence (a key that receives mass for the first time inherits the partner's), so this is an approximation. Exact per-origin mass tracking is due in M5, where R4 first retracts.
- Retracting an id that backs only the whole distribution (e.g. a near-pass) moves half of the mass to `None`; it does **not** undo the mix. To undo a mix (M4's `unlikely`), restore the pre-mix distribution from the trace.

Views: `ask(pattern)` (course `unify` / `satisfy`; returns `[(bindings, prob)]`); `text_view()` (one line per object, ≤ 40 tokens each); `som_image(frame_rgb)` (Set-of-Mark: ids + top-2 hypotheses drawn on the image; needs RGB, so replays only).

### 3.3 Belief math

`Distribution` over containers ∪ {None}. Operations (all renormalise):
- `from_distances(d: dict[id, float], sigma)`: p ∝ exp(−(d² − min d²)/2σ²). The shift prevents underflow; with no candidate → `{None: 1}`, citing the evidence that triggered it.
- `swap_mix(a, b, w)`: p'(a) = (1−w)·p(a) + w·p(b), p'(b) = (1−w)·p(b) + w·p(a). This means "with probability w the tracks a and b exchanged identity". It asserts 0 ≤ w ≤ 1.
- `flatten(f, plausible)`: p ← f·p + (1−f)·uniform(plausible). With f = λ^Δt this is R7, which replaces "conf × λ^Δt" for distributions. Δt = 0 (the distribution changed this frame) means no decay; an empty `plausible` is a no-op.
- `joint_exclusive(dists)`: R5. Over injective assignments π of hidden cubes → containers (N ≤ 3, enumerate ≤ 24), P(π) ∝ Π_h p_h(π(h)); return the marginals.
- `max_prob`, `argmax`, `entropy`.

Pose beliefs: `conf ← conf · λ_pose^Δt` while unconfirmed.

### 3.4 Observation → L0/L1

- Visible object: `pose = Belief(xyz, conf=track_q, t, "observed", [obs_id])`, `t_last_seen = t`.
- Visibility state machine (deferred until after G; VideoUnmaskSwap metrics read only visible / hidden / unknown): visible → `hidden` if a container is within `r_cover` of the last xy at the disappear frame; → `held` if the gripper is closed (width < `g_closed`) and the EE is within `d_grip`; → `occluded_by_arm` if the EE is within `d_occ` (no container) and the object reappears within `T_occ` frames; → `out_of_view` if the last uv is within `m_edge` px of the border; else `unknown`.
- Hidden cube display pose (R2): Σ_c p(c)·pose(c). Decisions use the distribution, never this pose.

### 3.5 Rules (exact form; parameters in `RuleParams`)

| Rule | Fires when | Ops | Phase |
|---|---|---|---|
| R1 cover | cube visible→not visible at t, and containers visible at t with xy distance d < `r_cover` to the cube's last xy | `SetDist(Inside(cube), from_scores(d, σ_cover))`; no candidate → `{None: 1}`; `AddEvent(cover)`. Covers the teleport case: the container appears in the same frame | M0 |
| R2 follow | each frame for hidden cubes | update display pose | M0 |
| R3 held | gripper closed near x | `Held(x)`; pose follows the EE | after G (no current metric needs it) |
| R4 reveal | container z > 0.15 and cube x observed at the container's former xy | `SetDist(Inside(x), {c: 1})`, `source="verified"`, plus `Retract` of contradicted hypotheses | M5 (needs the peek variant) |
| R5 exclusive | a cube was just covered (a `disappear` of a cube this frame) | `SetDist` = marginals of `joint_exclusive` over the hidden cubes whose beliefs were never mixed by R6 (after a mix they are correlated, and joint-normalising the marginals is overconfident). No consistent assignment → `AddEvidence(exclusive_conflict)` for M3's R8 | M1 |
| R6 swap | a `near_pass(a, b)` evidence closes (M0), or `id_margin < m_thr` (M2) | `MixDist(Inside(h), a, b, w)` for every **hidden** h with mass on a or b; `AddEvent(swap)` only when w > 0. M0: w = w_mix·(1 − closeness). M1: w = w_max·exp(−d/d0), with d the 3D closest-approach distance (the M0 image form was removed in M1.1: the detector always records d). M2: w from the identity margin and G's measured signature | M0 / M1 / M2 |
| R7 decay | each frame for unconfirmed items (Δt > 0) | `flatten`, pose conf decay | wired since M0 (λ = 1, off); λ is tuned in M2 |
| R8 consistency | two objects within `d_same`; a "hidden" cube observed; an expected container missing | `AddEvidence(risk)` + `MixDist` toward uniform with `w_risk` | M3 |

### 3.6 Monitor and decision

`decide(dist) → act | act_on_argmax | verify`, with thresholds `τ_hi`, `τ_lo` (M3 tunes them on train). They maximise expected utility U = +1 (act, correct), −`c_wrong` (act, wrong), −`c_verify` (verify), with defaults c_wrong = 1, c_verify = 0.3. On stock RoboMME, "verify" cannot execute (a wrong lift is an immediate failure), so it is scored as an outcome only.

### 3.7 Arbiter protocol (M4)

- **Triggers:** a `near_pass` closes; a cube disappears with ≥ 2 candidate containers; `track_drop`; a risk evidence; and every `K` frames while something is hidden. Budget: ≤ 6 calls per episode, at least 20 frames apart.
- **Input** (≤ 1,500 tokens): the system prompt (role: reconcile hypotheses; JSON only; cite evidence ids; never output coordinates or probabilities), `kb.text_view()`, the evidence lines since the last call, and ≤ 2 Set-of-Mark keyframes (max-overlap frame and min-margin frame; replays only).
- **Output (JSON schema, strict):**
  ```json
  {"events":[{"type":"swap|cover|none","args":["bin_1","bin_2"],"strength":"clear|ambiguous|unlikely","cites":["e19","e20"]}],
   "hypotheses":[{"cube":"cube_green","keep":["bin_1","bin_2"],"cites":["e12","e20"]}],
   "risks":[{"obj":"bin_2","reason":"<=80 chars","cites":["e21"]}]}
  ```
- **Validator:** schema; every id exists; every `cites` non-empty and existing; `keep` ⊆ containers that currently hold mass or are cited; no numeric field at all (strength is categorical). On failure: one retry with the error text, then fall back to RuleOnly for this trigger (logged).
- **Patch → ops:** swap `clear` → `MixDist(w = 1 − ε)` (a full exchange); `ambiguous` → `w = 0.5`; `unlikely` → `w = 0`, which undoes the rule's mix by restoring the pre-mix distribution from the trace (not via Retract, which only moves mass to None; §3.2). `keep` drops the mass of the other containers (renormalised). Each risk becomes `AddEvidence(risk)`. Numbers come from this fixed table (method a) or, as an ablation, from self-consistency over k = 5 samples (method b: the agreement rate becomes w).
- **What the arbiter can never do:** write a pose, create an object, or edit evidence.

### 3.8 Planner and skills

- `plan(view, cube) → PlanStep(skill, target, conf, decision, support, waypoints)`. The target is the argmax of `Inside(cube)`; `decision` comes from §3.6.
- `pick(c)`: four 7-D waypoints `[x, y, z, roll, pitch, yaw, gripper]` in the world frame: pre-grasp (xy_c, z = 0.20, open), descend (xy_c, z_c, keep), close (−1), lift (z = 0.20, keep). Orientation is top-down (π, 0, yaw_c): yaw 0 until M5 estimates it.
- M5 adds a **reach check** before any waypoint: a numpy damped-least-squares IK on Panda DH parameters, from the base at (−0.615, 0, 0). mplib has no macOS arm64 wheel, so the authoritative mplib check runs only on the pod.

### 3.9 Trace and debugging

- `results/runs/<run_id>/traces/<system>_<source>_<ep>.jsonl`, one line per frame: `{t, obs: {id: [visible, xyz, track_q]}, evidence: [...], ops: [{op, target, before, after, rule}], arbiter: {...} | null}`. Tests use gzip off; full runs gzip on.
- `debug explain`: walks support ids back to observations and prints a tree.
- `debug timeline`: plots one cube's distribution over time with evidence markers.
- `debug diff --a SYS1 --b SYS2` and `KB.restore()`: not built yet; add them when a debugging session needs them (YAGNI). `KB.snapshot()` exists.
- CLI: `python -m arbiter_kb.debug explain|timeline --run RUN_DIR --condition C --system S --ep N --cube ID [--t T]`. Trace files are `RUN_DIR/traces/<condition>_<system>_<ep>.jsonl`.

### 3.10 Evaluation definitions

- **Ground truth:** `cube_to_bin` (fixed at cover time; the cube rides with its container); `t_star` = synthetic: last frame; replays: `decision_frame(rep)`; the target from `task_goal` (`target_cube_nodes`). Multi-target episodes: evaluate each target at the frame it becomes active (`ActiveTarget`).
- **Correctness is physical:** the system's label whose believed position is nearest the true container at t* (≤ 5 cm). Under id corruption labels lie, so the id alone cannot be trusted (NOTES M0 deviation 2).
- **Per row:** `pred, p_max, correct, decision, entropy, grasp_err_m, moved, corrupted`.
- **Metrics:** accuracy; **silent-failure rate** = P(wrong ∧ p_max ≥ τ_hi), always reported with **coverage** = P(p_max ≥ τ_hi) (a system can reach 0 silent failures by lowering all confidences, so the two are only meaningful together) and **AURC** (area under the risk–coverage curve: rows answered from most to least confident; risk = error rate among rows at least that confident; ties enter together; threshold-free, lower is better), with a risk–coverage figure; detection rate = P(p_max < τ_hi ∨ risk | wrong); Brier = mean Σ_c (p_c − 1[c = true])²; ECE (10 equal-width bins on p_max); reliability diagram; decision outcomes; cost per episode (calls, tokens, s, $); grasp error. 95% Wilson intervals for rates. Split moved vs back-at-start.
- **Splits:** synthetic train seeds 0–999, test seeds 1000–1999; RoboMME train episodes for tuning, the 50 test episodes only after the freeze. A split is read from `EpisodeMeta.split`. Every test read is checked against, and appended to, `results/test_reads.log` (E5c). **Any new or changed parameter** (e.g. M1's w_max, d0, c, j_max) means re-freezing `params.json` (a new FREEZE line) before the next test read.

### 3.11 Corruption model (input side, never output side)

A corruption must look like a real tracker error, otherwise results are true by construction (docs/GOALS.md, Known traps).
- `id_swap` (M0, **teleport, sanity only**): at the closest approach of the target container's first swap, exchange the two ids for all later frames. The label jumps about 12 cm in one frame (0.116–0.138 m on train seeds, `results/m0_synthetic_stats.json`), which no real tracker does. Never used for tuning or claims.
- `id_swap_overlap` (M1): swaps happen only where the two containers' 3D centres come within `d_ov` (default 0.03 m) of each other, i.e. where the tracks really overlap. The synthetic generator needs `lane_offset` ≤ about 0.03 for this. The ids are exchanged at a frame inside the overlap, so the label jump is ≤ `d_ov`. p_swap is the probability that an overlapping pass swaps (episodes chosen by seeded hash, nested). The measured jump distribution is reported with every result.
- `id_swap_measured` (M2, after G): swap rate and signature (overlap, `track_q` drop, jump size, phase of the swap) taken from phase G's measurements of the real tracker, not chosen by us.
- `pos_noise` (M1): Gaussian xyz noise σ ∈ {5, 15} mm on visible objects.
- The old Q1/Q2 (`sg_pilot.corrupt`) overwrite the bound rule's output. They stay available only to reproduce old numbers for the baselines.

---

## 4. Code layout and contracts

```
arbiter-kb/
  .claude/agents/{code-reviewer,goal-reviewer}.md  .claude/commands/review-phase.md
  docs/GOALS.md   reviews/
  DEVELOPMENT.md  README.md  NOTES.md  Makefile  requirements.txt  .env (git-ignored)  .gitignore
  arbiter_kb/
    __init__.py          CONTRACT_VERSION = 1
    types.py             ObjObs, FrameObs, EpisodeMeta, GroundTruth, PlanStep
    beliefs.py           Belief, Distribution (§3.3)
    kb.py                ObjectFrame, Evidence, Event, Op types, KB (apply/check/ask/retract/count/views/snapshot)
    sg_compat.py         functions copied from sg_pilot.py (§0.4)
    ledger.py            append-only test-read ledger results/test_reads.log (E5c)
    params.py            RuleParams, DetectParams, MonitorParams, ArbiterParams, load/save/hash/freeze
    trace.py  debug.py
    sources/  __init__.py synthetic.py replay.py corrupt.py perceived.py(M6)
    systems/  __init__.py baselines.py arbiter_kb.py
    observe.py  detectors.py  identity.py(M2)  rules.py  monitor.py(M3)
    arbiter.py(M4)  llm.py(M4)  planner.py  skills.py  reach.py(M5)
    evaluate.py  metrics.py  plots.py
  scripts/  run.py  selftest.py  quality.py  make_notebook.py  measure_synthetic.py  copy_replays.sh(G)  measure_swaps.py(G)  tune.py(M2)
  stress/   stress_env.py  render.py  perceive.py  POD_RUN.md   (G; runs on the pod, outside the package)
  results/  test_reads.log (ledger)  latest/<run>/{summary.csv, meta.json} (committed, --keep only)  runs/ (git-ignored)
  tests/    one file per module + test_contracts.py test_causality.py test_repro.py
  notebooks/ m0_walkthrough.ipynb … (one per phase)   executed/
  results/  runs/<run_id>/{rows.csv, summary.csv, params.json, meta.json, traces/}  llm_cache/
  figures/
```

Contracts (`types.py`; fixed in M0):

```python
@dataclass
class ObjObs:  id: str; cls: str; visible: bool; xyz: np.ndarray | None; uv: np.ndarray | None; track_q: float
@dataclass
class FrameObs: t: int; objects: dict[str, ObjObs]; ee_xyz: np.ndarray; gripper_width: float; is_video: bool; rgb: np.ndarray | None = None
@dataclass
class EpisodeMeta: source: str; episode: int; split: str; task_goal: str; objects: list[tuple[str, str, dict]]; K: np.ndarray | None; E: np.ndarray | None
@dataclass
class GroundTruth: targets: list[str]; cube_to_bin: dict[str, str]; t_star: dict[str, int]; bin_xyz_at_t_star: dict[str, np.ndarray]; moved: dict[str, bool]
@dataclass
class PlanStep: skill: str; target: str | None; conf: float; decision: str; support: list[str]; waypoints: np.ndarray  # [4, 7]

class Source(Protocol):
    name: str
    def episodes(self, split: str) -> list[int]: ...
    def meta(self, ep: int) -> EpisodeMeta: ...
    def frames(self, ep: int) -> Iterator[FrameObs]: ...
    def ground_truth(self, ep: int) -> GroundTruth: ...      # evaluation only

class MemorySystem(Protocol):
    name: str
    def reset(self, meta: EpisodeMeta) -> None: ...
    def step(self, frame: FrameObs) -> None: ...
    def believe_inside(self, cube: str) -> Distribution: ...
    def pose(self, obj: str) -> np.ndarray | None: ...       # added in M0 (NOTES deviation 3)
    def plan(self, cube: str) -> PlanStep: ...
    def trace_rows(self) -> list[dict]: ...
```

---

## 5. Shared infrastructure

### 5.1 Commands (Makefile)

```
make test       # pytest -q
make quality    # python scripts/quality.py   (E1, E9)
make selftest   # python scripts/run.py --selftest   (phase "Done when" checks, PASS/FAIL lines)
make check      # test + quality + selftest   ← first gate for every phase
/review-phase M # (in Claude Code) code-reviewer + goal-reviewer   ← second gate (§1.1)
make eval ARGS="--source synthetic --corrupt none,id_swap --p 0.5 --split test --n 30"
python scripts/run.py --freeze [--force]   # untuned phases (M0, M1): params.json from defaults + FREEZE ledger line
python scripts/run.py --systems all --corrupt none,id_swap_overlap,pos_noise --lane mix --steps mix --n 60 --keep-as m1_table
python scripts/run.py --keep[-as NAME] ...  # also copy summary/meta to results/latest and figures to figures/, from a clean tree
make notebook   # regenerate + execute the phase walkthrough
```

### 5.2 OpenRouter (M4)

- **Models (chosen 2026-10-28 from OpenRouter's live model list; all open-weight, image input, structured outputs supported):**
  - **Primary:** `qwen/qwen3.6-35b-a3b` (MoE, 3B active; $0.10 / $0.90 per M tokens; runs locally on a 24 GB GPU at 4-bit).
  - **Small-end ablation:** `qwen/qwen3.5-9b` (dense 9B; $0.10–0.17 / $0.25; fits a 24 GB GPU in bf16).
  - **Cross-family check:** `google/gemma-4-26b-a4b-it` ($0.09 / $0.30).
- **Request:** `POST https://openrouter.ai/api/v1/chat/completions` with `Authorization: Bearer $OPENROUTER_API_KEY` and body:
  ```json
  {"model":"qwen/qwen3.6-35b-a3b","temperature":0,"seed":0,"max_tokens":500,
   "response_format":{"type":"json_schema","json_schema":{"name":"patch","strict":true,"schema":{...}}},
   "provider":{"order":["DeepInfra"],"allow_fallbacks":false,"require_parameters":true},
   "usage":{"include":true},
   "messages":[{"role":"system","content":"..."},{"role":"user","content":[{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]}
  ```
  DeepInfra serves all three models with structured outputs (fp8 / bf16). Pinning one provider keeps runs comparable. If it is down, fail loudly; don't silently fall back.
- **Cache:** key = sha256(model, provider, messages, schema, temperature, seed) → `results/llm_cache/<key>.json` (request + response + usage). A cache hit costs 0 and is counted.
- **Key:** `.env` (`OPENROUTER_API_KEY=…`, `ARBITER_MAX_USD=2.00`), loaded by `llm.py` without extra dependencies (parse the file with 10 lines of code). `.env` is in `.gitignore`. Never print the key.
- **HTTP:** `urllib.request` (stdlib); retries 2× with backoff on 429/5xx; timeout 60 s.

### 5.3 Dependency whitelist
numpy, scipy, matplotlib, pytest, nbformat, nbconvert, ipykernel. Nothing else without a NOTES entry. M6 adds whatever the existing perception worker needs, on the pod only.

### 5.4 Data
- **Synthetic:** generated on the fly (§6 M0).
- **RoboMME replays:** `ARBITER_REPLAY_DIR` (default `arbiter-kb/data/replays/`, git-ignored). `scripts/copy_replays.sh` (written in M0, run in M6 or whenever a pod is up) copies about 20 train + 50 test VideoUnmaskSwap replays plus `motion.json` from the pod volume (paths from `scene-graph-robot/notebooks/pilot/POD_RUNBOOK.md`), and prints the total size first. This needs a running pod but no GPU work.

---

## 6. Phases

Time estimates are for a coding agent with review. Each phase lists: **Goal · Build (how) · Params · Done when · Not in scope**.

**Order (v1.2):** M0 → M1 (plumbing) → **G** → M2 → M3 → M4 → M5 → M6. G needs a GPU pod and the user's go; its Mac-side code is written right after M1, or alongside it. Tuning starts only after G (M2), against `id_swap_measured`.

### M0 — Walking skeleton (1–2 days, Mac CPU)

**Goal.** The full path runs on synthetic data: Source → KB → rules (R1, R2, R6-minimal) → planner → evaluation table + figure + traces. Three systems behind one interface. The first datapoint of the research claim.

**Build.**
1. **Scaffold:** layout §4, `sg_compat.py` + `test_sg_compat.py` (§0.4), Makefile, `.gitignore` (`.env`, `data/`, `results/runs/`, `results/llm_cache/`), `git init`, `quality.py`, `test_contracts.py`.
2. **`types.py`, `beliefs.py`, `kb.py`** with Op types (`SetPose, SetVis, SetDist, MixDist, AddEvidence, AddEvent, Retract`), `apply`, `check`, `ask`, `retract`, `count`, `text_view`, `snapshot`. Tests: Distribution invariants under every op; `ask(("Inside","cube_green","?c"))` → `[({"?c":"bin_1"}, .55), ({"?c":"bin_2"}, .45)]`; a retract cascade three levels deep.
3. **`sources/synthetic.py`**, mirroring the real VideoUnmaskSwap (`evidence-gaps` §7):
   - 3 cubes (red / green / blue) and 3 or 4 containers in a triangle or line layout (±0.15 m), randomly rotated.
   - t 0–31: cubes visible, containers `visible=False`. t = 32: containers appear exactly over the cubes, and the cubes turn invisible in the same frame.
   - Swaps from t = 64, `swap_steps` (50) each, 1–3 swaps. The partner is the nearest container. Path = straight line ± `lane_offset·sin(πα)` along the normal, with smoothstep α.
   - Hold 10 frames; t* = the last frame.
   - Camera: eye (0.3, 0, 0.4), target (0, 0, −0.2), 256 × 256, fov 90°; a `look_at` helper builds K/E; uv by projection.
   - `ee_xyz` home, gripper open; `is_video` True except the last frame.
   - Knobs: `n_containers, n_swaps, swap_steps, lane_offset, seed`. The seed list covers both moved and back-at-start targets.
4. **`sources/replay.py`:** wraps `sg_compat.load_replay` + `oracle_graph` → FrameObs (node → `bin_k` / `cube_<color>`; `gripper_width = gripper_state.sum()`; t* = `decision_frame`). With no replays present, `episodes()` returns `[]` and prints SKIPPED plus the `copy_replays.sh` hint.
5. **`sources/corrupt.py`:** `id_swap` (§3.11).
6. **`systems/baselines.py`:** Frozen and Bound adapt `sg_compat.GraphTracker` (FrameObs → `[N, 8]` rows). `believe_inside` = one-hot on the container nearest to the tracked cube xyz.
7. **`observe.py`, `detectors.py`** (appear / disappear / move_start / move_stop / near_pass, with an image-distance threshold `d_px` and a 3D fallback), **`rules.py`** (R1, R2, R6-minimal; R7 wired with λ = 1).
8. **`systems/arbiter_kb.py`:** the step order of §3.1, with the monitor and arbiter as stubs (the trigger is logged; `RuleOnlyBackend` returns None; `validate()` is implemented and tested on one valid and two invalid hand-written patches).
9. **`planner.py` + `skills.py`:** `pick` waypoints with yaw 0 and `# TODO M5`.
10. **`evaluate.py`, `metrics.py`, `plots.py`, `trace.py`, `debug.py`** (explain + timeline).
11. **`scripts/run.py`** (eval + `--selftest`), `notebooks/m0_walkthrough.ipynb`, executed to `executed/`.

**Params (defaults, untuned):** r_cover 0.06 m, σ_cover 0.02 m, d_px 30 px (3D fallback 0.10 m), v_move 0.001 m/frame (was 0.005; see NOTES M0 deviation 1), w_mix 0.5, λ 1.0, τ_hi 0.8, τ_lo 0.5.

**Done when.**
- (a) `make check` passes.
- (b) Synthetic clean: ArbiterKB and Bound have 100% accuracy; Frozen's accuracy is lower on moved than on back-at-start.
- (c) Synthetic `id_swap` p = 1: Bound's silent-failure rate is 100%; ArbiterKB's is lower (both printed).
- (d) ArbiterKB's mean confidence when right, on clean input, is reported (it will be < 1: the cost of honesty that M1 tunes).
- (e) `debug explain` works on a corrupted episode.
- (f) Two identical runs give the same summary hash.
- (g) Replay source: real oracle numbers for Frozen and Bound reproduce `sg_pilot` within 0.5 px, or SKIPPED.

**Not in scope.** R3–R5, R8, Hungarian matching, thresholds, LLM, yaw, IK, real perception.

### M1 — Plumbing for realistic corruption and the trivial baselines (1–1.5 days, CPU)

**Goal.** Build what G and M2 need, without claiming anything. The M0.1 goal review showed that on any synthetic corruption we design, the best possible system is the one that copies our own choices (the overlap threshold and the swap rate). M1 therefore tunes nothing and makes no claim. It delivers the generator, the corruption, the baselines and R5, plus one table at default parameters.

**Build.**
1. **Synthetic generator:** `lane_offset` down to 0.0 (overlapping tracks), `swap_steps` 20–50, `pos_noise`. Record per pass the true minimum 3D centre distance (evaluation-only extras).
2. **`corrupt.py`:** `id_swap_overlap` (§3.11), with its label-jump histogram written to results. Keep `id_swap` labelled "teleport (sanity)".
3. **Baselines** (`systems/baselines.py`, same interface):
   - `BoundHaircut(c)`: after any near-pass involving the container that holds the cube, the answer becomes {pred: c, partner: 1 − c}.
   - `OverlapHaircut(c)`: the same, but only after a pass whose minimum 3D centre distance < `d_ov`. This is the Bayes-optimal rule for `id_swap_overlap`.
   - `JumpDetector(j_max)`: a container label that moves > j_max (0.03 m) in one frame sets its hypotheses to 0.5. By construction it should be useless on `id_swap_overlap`.
4. **R5** `joint_exclusive`, ε_none = 0.01. R5 is the one rule that can beat any haircut (3 cubes in 3 containers: pinning one cube informs the others).
5. **R6 v1.5:** w = w_max·exp(−d/d0) from the 3D closest-approach distance (image distance as fallback). Defaults, not tuned.
6. **Metrics:** ECE and a reliability diagram.
7. **Parameters (defaults, untuned; all in `params.py`):** w_max 0.5, d0 0.02 m, d_ov 0.03 m, c 0.5 for both haircuts (the Bayes-optimal value at p_swap 0.5, i.e. it copies our own choice on purpose; Bound's answer is listed first, so the 50/50 tie keeps it), j_max 0.03 m, ε_none 0.01. They change `params.json`, so re-freeze with a new FREEZE line before the test table. Corruption settings (p_swap, d_ov, pos_noise σ) are source arguments recorded in meta.json, not system parameters.
8. **Debuggability carried over from the M0.1 code review:** a test that `debug explain` reaches observation ids for **every** ArbiterKB row of a small evaluation (not just five episodes).
9. **CLI:** `scripts/run.py --systems` to choose systems, `--corrupt id_swap_overlap`, `--pos-noise`.

**Done when.**
- (a) `make check`.
- (b) One table on test seeds at the frozen default parameters: Frozen, Bound, BoundHaircut, OverlapHaircut, JumpDetector and ArbiterKB under {clean, `id_swap_overlap` p = 0.5, `pos_noise` 15 mm}, with Wilson intervals. The column that matters is **ArbiterKB minus OverlapHaircut**. NOTES must say that, at about 0, M1 shows plumbing only.
- (c) The label-jump histogram of `id_swap_overlap`: median ≤ `d_ov`, and JumpDetector ≈ Bound.
- (d) The R5 test: two hidden cubes, one pinned → the other's mass moves; plus one synthetic scenario where R5 beats OverlapHaircut (M1.1: `tests/test_r5_scenario.py`, an ambiguous cover; R5 never fires on stock VideoUnmaskSwap layouts).

**Deferred until G's measurements exist:** `scripts/tune.py`, the trade-off curve, R3 held and the full visibility state machine (no VideoUnmaskSwap metric reads them).

### G — Gate: how often and how does the real tracker swap ids? (Mac code ~1 day + one GPU pod session, about 3–5 h; needs the user's go)

**Goal.** Measure, on RoboMME VideoUnmaskSwap stock and stress variants, how often the real perception pipeline (Grounding DINO → SAM → Cutie) swaps container ids, and what a swap looks like. This decides whether claim 1 can be tested with real errors, and it defines `id_swap_measured`. It is the plan report's November gate, pulled forward.

**Build (Mac, no GPU).**
1. `stress/stress_env.py` (pod-side code, outside the `arbiter_kb` package): subclass `VideoUnmaskSwap` as `VideoUnmaskSwapStress`, with knobs `swap_steps`, `lane_offset`, `n_swaps ≤ 3`, `n_bins` and a static camera shift. Override `_refresh_swap_schedule`, `configs`, `_default_sensor_configs`, and pass `lane_offset` to the swap call (l.603–614; `evidence-gaps` §7). Settings:
   - stock (50, 0.07)
   - fast (20, 0.07)
   - close (50, 0.02)
   - fast + close (20, 0.02)
   - fast + close with 3 swaps
   - camera shifted ±5 cm
2. `stress/render.py`: render 20 episodes per setting into the existing `Replay` npz format (front RGB, depth, segmentation, E, obj_xyz). Print the total size first; it must stay under 8 GB on the volume.
3. `stress/perceive.py`: run the existing `PerceptionWorker` on each replay and save small per-frame outputs (track centroids, mask areas, scores).
4. `scripts/measure_swaps.py` (CPU, `arbiter_kb` side): match tracks to the oracle each frame. Report identity accuracy and swap events per episode. For **every pass** (two containers moving together, as in `realistic.passes`), swapped or not, record: swapped (0/1), min 3D distance, image IoU, `track_q` / mask-area change, label-jump size, Hungarian margin, and phase. Outputs: `results/g_passes.csv` plus a summary table. Recording clean passes too is what makes an ROC possible (M1 goal review).
5. `stress/POD_RUN.md`: the exact commands (tmux, environment, Vulkan fix from `POD_RUNBOOK.md`), a cost estimate, and "stop the pod" as the last step. In the same session, run `scripts/copy_replays.sh` to bring the stock train/test oracle replays to the Mac.

**Done when.**
- (a) A table of identity error and swap rate per setting.
- (b) Histograms of each feature for swapped vs clean passes, and the ROC / AUC of each single feature for "this pass swapped".
- (c) **The gate decision, written in NOTES:**
  - if swaps occur in < 2% of passes in every setting, claim 1 cannot be tested with real errors on RoboMME, so switch to the plan's fallback (failure-attribution paper) or choose a harder task;
  - otherwise, define `id_swap_measured` from the measured signature, and M2 uses it;
  - **and** if the best single feature separates swapped from clean passes almost perfectly (AUC ≥ 0.98), write in NOTES that a threshold rule on it solves claim 1's detection part, before M2 builds R6 v2; the belief layer must then show value through propagation (R5, retraction, arbiter), not detection.

### M2 — Evidence layer: change detectors + Hungarian identity check (2 days, CPU)

**Goal.** Ambiguity is measured, not assumed. R6 uses the identity margin instead of a fixed near-pass rule; corruption becomes probabilistic and leaves the traces a real tracker leaves.

**Build.**
1. `detectors.py`: hysteresis on speed; overlap via image distance and (replays) mask IoU; `track_drop` when `track_q` falls below `q_thr`; cap of 5 evidence items per frame.
2. `identity.py`: each frame (oracle) or every `N_id` frames (perceived), solve `linear_sum_assignment` on the cost = 3D distance between each container's predicted pose (constant velocity) and the observed tracks. Margin = (second-best total − best total) / best total. Emit `id_margin` evidence when the margin < `m_thr`.
3. R6 v2: `w = w_max·exp(−margin/m0)`.
4. `corrupt.py`: `id_swap_measured` from G's signature (no hand-chosen sigmoid), plus `pos_noise`.
5. `scripts/tune.py` (first tuning in the project): train seeds only, against `id_swap_measured`. Writes `params.json` and a FREEZE ledger entry. The trade-off curve (clean coverage vs silent failure) is drawn here.
6. `MarginHaircut` baseline: the GuardedBound threshold rule on the same margin feature R6 v2 uses (threshold tuned on train). Without it a win only shows that the margin is a better feature than 3D distance, not that a belief layer helps (M1 goal review).
7. Evaluate on fresh test seeds (synthetic ≥ 1060): seeds 1000–1059 have been read many times during M0/M1 (harmless while nothing was tuned).

**Done when.**
- (a) `make check`.
- (b) On synthetic, the minimum margin falls inside a swap window in ≥ 95% of episodes.
- (c) Under `id_swap_measured`, **at matched clean coverage**, ArbiterKB's silent-failure rate < that of Bound, OverlapHaircut and MarginHaircut, with non-overlapping Wilson intervals, and its AURC is lower; otherwise report honestly. (A fixed-τ_hi silent rate alone is not a criterion: any haircut below τ_hi scores 0.)
- (d) Clean accuracy holds (M1 criterion).

### M3 — Monitor and decisions (1 day, CPU)

**Goal.** Beliefs change behaviour.

**Build.**
1. R8 consistency checks → risk evidence + `MixDist` toward uniform (`w_risk`).
2. `decide()` with τ_hi / τ_lo tuned by expected utility on train (§3.6).
3. Decision outcome columns: act-correct / act-wrong / verify.

**Done when.**
- (a) `make check`.
- (b) The table shows decision outcomes for every system (the baselines always "act").
- (c) Under corruption, ArbiterKB converts a measurable share of act-wrong into verify; the number is reported with intervals.

### M4 — Arbiter (3 days, CPU + OpenRouter)

**Goal.** Test whether a small LLM adds value over RuleOnly, at a measured cost.

**Build.**
1. `llm.py`: the OpenRouter client (§5.2), `CachedBackend`, budget cap, usage accounting.
2. `arbiter.py`: triggers, keyframe selection, prompt builder, JSON schema, validator, `patch_to_ops`, retry-then-fallback. Strength-to-weight table (method a); self-consistency k = 5 (method b), behind a flag.
3. Construct the cases where rules are weakest. Synthetic: 3 containers moving at once, lane_offset 0.02, swap_steps 15, two hidden cubes. Report RuleOnly and +LLM on these and on the standard set.
4. Prompt iteration only on train episodes. Freeze the prompt text (hash in `params.json`) before test.

**Done when.**
- (a) `make check` (validator tests, including a patch with a coordinate and a patch with an invented evidence id).
- (b) 100% of applied patches validate; fallbacks counted.
- (c) A cache replay reproduces the KB exactly.
- (d) The table includes calls / tokens / s / $ per episode, next to the 2610.17350 row (9.5 calls, 18.3 s per call, $0.39 per episode; labelled "different system and setting").
- (e) RuleOnly vs +LLM: the difference in silent failure and Brier, with paired bootstrap intervals. If there is no difference, that is a reportable finding (the plan's falsifier 2).

### M5 — Planner, skills, reach check (2 days, CPU; optional GPU closed loop)

**Goal.** The KB's output is a usable pose.

**Build.**
1. Container yaw: replays use PCA of the container mask's back-projected points; synthetic uses the known yaw plus noise. Store it in the L0 pose belief.
2. `reach.py`: damped-least-squares IK on Panda DH (numpy), with joint limits; reject unreachable waypoints → `decision = "unreachable"` (never send one: a planning failure ends a RoboMME episode).
3. Grasp-error metric vs the true container pose at t*; project the waypoints onto the image (`figures/`).
4. Optional, with the user's go, on a GPU + Vulkan pod: closed loop with `make_env(..., action_space="waypoint")` on 5 episodes; report success, steps per waypoint, wall time. The authoritative mplib reach check runs there too.

**Done when.**
- (a) `make check`.
- (b) On clean oracle input the planned grasp is within 3 cm of the true container in ≥ 95% of episodes.
- (c) Reach-check agreement with mplib on ≥ 50 sampled poses (pod only, else SKIPPED).

### M6 — Full evaluation on real RoboMME data (2–3 days; GPU pod only for new renders)

**Goal.** The claims are made here: every system on the RoboMME oracle replays and on perceived input (stock and G's stress settings), with the frozen parameters.

**Build.**
1. Run every phase's evaluation on `OracleReplaySource` (train for tuning, test after the freeze).
2. `PerceivedSource` from G's cached perception outputs (and new renders if more episodes are needed).
3. The full table on perceived input, per stress setting.
4. **Peek variant** (optional): `failure_func=None` for the pick tasks plus R4 reveal → measure verify-then-act.

**Done when.**
- (a) `make check`.
- (b) The final table (oracle, perceived × stress settings) with Wilson intervals, cost row included.
- (c) NOTES maps each claim and falsifier in GOALS.md to the numbers that support or refute it.

---

## 7. Open questions (track in NOTES; answer by measurement)
1. Do the pod replay caches contain segmentation and depth? (Checked in G's pod session.)
2. Does Cutie expose a usable confidence? If not, `track_q` = mask-area change + identity margin (M2).
3. Is the rule planner enough on RoboMME? The LLM planner is out of scope unless M4 shows the arbiter helps.
4. ECoMEM's update parameters are unpublished: implement it as a structural baseline (log-odds + half-life + hysteresis) only if time allows after M4.
