# Handoff: orchestrator / project manager — 2026-10-30

For a new conversation that runs the project as orchestrator: it plans, briefs worker windows (or subagents),
proofreads their output and every paid run, keeps NOTES / HANDOFF true, and talks to the user. It writes little code
itself. Read with `HANDOFF.md` (state, where things are), `NOTES.md` (every number; newest at the bottom; sections
named below), `docs/HANDOFF_EXECUTION.md` (the execution gap: IK layers, options 1–5), `docs/TEST_PLAN_COMPONENTS.md`
(layer-by-layer tests against simulator truth). Answer the user in Chinese, concisely; docs in English.

## The project in five lines
- ArbiterKB: a belief-carrying KB as the memory of an LLM controller that outputs waypoints in 2610.17350's harness on
  RoboMME (ManiSkill, Franka). Arms differ only in memory: **H** = image history (3×3 frame grids), **R** = the raw
  tracker log, **K** = the KB view (+ the controller's own record, `self_memory`), **K2** = new layout of K (below).
- Claims: (1) a smaller model or (2) far fewer tokens at matched success; (3) the KB's beliefs are calibrated.
- Pipeline per step: perception (GroundingDINO → SAM → Cutie, `sources/online.py`) → KB → view → LLM returns one
  JSON (`waypoint` TCP pose, `gripper` −1 close / 1 open / 0 keep, `target`, `cites`) → RoboMME's planner (IK, screw,
  RRT*). `null` waypoint = "done".

## What the last two days established (NOTES sections in brackets)
1. **The harness works; the controller model was the bottleneck.** Qwen 3.6-35B val pilot 1 / 24 (stopped) vs
   GPT-6-Astra ["Astra VideoUnmaskSwap val pilot…"]: VideoUnmaskSwap val 0–9, **H 9/10, R 6/10, K 8/10**.
2. **Layer 1 ("where") is not the problem** ["Layer 1 on real scenes"]: with memory, the first call-out is right for
   both models (Astra 19/20, Qwen 12/15); memory arms copy the perceived position (error = perception's 0.3–2.9 cm).
   Astra H localises from images (10/10, 4.9 mm); Qwen H cannot (1/9). Failures come after a right first move.
3. **Astra's 7 failures** ["Astra's 7 VideoUnmaskSwap failures…"]: 6 of 7 on two-pick goals; losing progress (going
   back to the first bin, mostly R) and bad put-downs / low lifts; no premature "done" in that set.
4. **Cost: prompt caching** ["Why K's fewer tokens did not cost less"; "K's request layout and the provider's prompt
   cache"]: K sends 38% of H's tokens but paid the same, because H's growing conversation is 77% cached and K's fresh
   requests 0% (K's fixed prefix is 886 tokens, under OpenAI's 1,024 minimum; writes cost 1.25×, reads 0.1×).
5. **K two-pick run, dec1e03** ["Does K lose track on two-pick goals?"]: K 9/20, ~$12.5, **run with the gripper-0
   bug** ["Bug: keep recorded as open"], so it cannot answer its question. The user judged it wasted → the preflight
   rule below. Not yet read by its rule; results on the pod, not on the Mac.
6. **VideoPlaceOrder:** no target phrase passes the fixed rule ("circle." best but loses the named cube in ep 2), so its
   K arm is not run (user decision, "the target-phrase rule stands").
7. **Fixed in code:** gripper 0 recorded as open (038ddf1); episodes outside the split now raise (RoboMME silently built
   a default scene for 50–59); provider `cached_tokens` and per-step `description` / `target` are logged.
8. **Built, not run:** K2 (`view_k2.py`, user design: A initial state → B appended change log → C current state;
   cache-friendly; simulated cost 0.53–0.55 of K) and its K-vs-K2 offline test (rule in NOTES "K vs K2 offline").

## Open questions, in the order they block things
1. **Two free checks on real pilot K views (before any K / K2 run):** (a) after a cube is seen uncovered, does the
   Inside belief drop, or does the view say "visible" and "Inside 0.98" at once (seen in K2's simulated example)?
   (b) is tracker jitter logged as moves ("bin_0 moved 18 times"; K2's log mostly mm-scale moves)? Either may be a
   KB bug that changes K's and K2's text.
2. **A move threshold shared by K and K2**, from the task's physical scale, not from pilot results (if 1b holds).
3. **K2 offline test needs a noisy condition** (the earlier pos_noise stress), since clean synthetic data cannot show
   the log-noise risk; the pass rule must hold in both. Then run it (Qwen, ~$1–3, user go).
4. **Read the dec1e03 two-pick results by the rule** (download the tarball first; label as "with the bug").
5. **One package with every fix, preflighted, then the paid run:** K (fixed) **and H** on the same 20 two-pick
   episodes (H tells whether two-pick is hard for every arm); K2 too if it passed offline. The option-4 decision
   (progress facts) is taken on the fixed K.
6. Later: execution options 3 (skills layer, separate system claim) / 4; model ladder (option 1); claims 1–2 on more
   tasks; claim 3 on fresh episodes; upstream reports (docs/UPSTREAM.md); VideoPlaceOrder perception.

## Workers (other windows in this repo, all on main, local commits)
- **Two-pick / failure-reading window:** wrote the layer-1 scorer, the failure reading, the two-pick test, the
  gripper fix. Owns `callout.py`, `closed_loop.py` gripper logic, the two-pick NOTES section.
- **Cache / K2 window:** wrote the caching analysis, `tools/cache_sim.py`, `view_k2.py`, `scripts/k2_offline.py`.
  Owns the K2 NOTES section.
- **Pod session:** Claude in tmux `main` window 0 on the RunPod A40 (`/root/.local/bin/claude --continue`); runs only
  what is pasted, reports, never starts paid runs itself. Code in `/workspace/score` (dec1e03), never over
  `/workspace/arbiter-kb`. May be stopped (see HANDOFF.md "Pod").
- Coordination: one writer per file set; check `git log` before touching shared files (`closed_loop.py`, NOTES);
  no window rebuilds the pod upload while another's run uses it.

## Rules (the user's; all still hold)
- **Preflight before every paid run (new, 2026-10-30):** proofread the exact package (commit, md5, command, every
  flag, cap) against the NOTES rule; run a free no-LLM preflight on the pod (scripted actions from simulator truth
  through the closed loop, every K view printed and checked: gripper state vs fingers, held object vs `is_grasping`,
  Inside vs visibility, logged moves vs real moves); read its output; give the user the checklist; only then ask for
  go. Do not approve a paid run while an open question could change the model-facing text. The preflight script
  does not exist yet (a task).
- Rules fixed in NOTES (committed) before any paid run; honest negatives; no-LLM baselines; never pool versions
  (every model-facing change = new prompts hash, `python scripts/run.py --freeze --force`).
- Code-reviewer + goal-reviewer agents at the end of a phase; fix findings. Zoom out: what should memory give a
  controller, not only how to pass the next test.
- Commit to main, never push without the user. Never print or store the OpenRouter key. Stop the pod when idle;
  backups before stopping. Do not modify `~/Code/active/scene-graph-robot`. Upstream issues are filed by the user.
- Budget is not the constraint; stability first, speed second.
