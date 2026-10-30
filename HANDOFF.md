# ArbiterKB: handoff (2026-10-30, orchestrator round)

Read first: `README.md` (one page), `docs/GOALS.md` (claims, falsifiers), `DEVELOPMENT.md` (spec v1.12: §2 the loop and its
reasoning, §3.8 controller, §6 phases), `NOTES.md` (the log; newest at the bottom). Answer the user in Chinese,
concisely. Docs in English. For the execution-gap discussion (IK, controller design, models, landscape):
`docs/HANDOFF_EXECUTION.md`. Upstream bug reports to prepare: `docs/UPSTREAM.md`.

## Direction (NOTES "Direction v2")
- The KB is the memory module of an LLM controller that outputs waypoints itself, in the 2610.17350 harness.
- Claims: (1) a smaller model, or (2) far fewer tokens and images, at matched success; (3) the KB's beliefs are calibrated.

## Now (2026-10-30): orchestrated by a separate conversation
- **Start here:** `docs/HANDOFF_ORCHESTRATOR.md` (state, open questions in order, workers, rules). Execution gap:
  `docs/HANDOFF_EXECUTION.md`; layer tests: `docs/TEST_PLAN_COMPONENTS.md`.
- Closed loop code: `arbiter_kb/closed_loop.py`, `sources/online.py`, `stress/closed_loop_pod.py` (`--model`,
  `--provider auto`, `--max-usd-episode`, `--layout`, `--prompt-cache`), K2 view `arbiter_kb/view_k2.py`; runbook
  `stress/POD_CLOSED_LOOP.md`. Upload = `git archive` → `pod_upload_arbiter-kb.tar.gz` (git-ignored); the pod
  checks its md5 and extracts into its own directory (`/workspace/score`), never over a running job's code.
- **Results this round** (NOTES, bottom): Astra VideoUnmaskSwap val 0–9 H 9/10, R 6/10, K 8/10 (with the gripper-0 bug); K 38% of H's prompt
  tokens but the same cost (caching); first call-outs right with memory (layer 1 is not the problem); K two-pick 9/20
  but run with the gripper-0 bug (not usable for its question); VideoPlaceOrder K not run (no phrase passes).
- **Pod:** RunPod A40; results under `/workspace/cl/` (pilot, astra_pilot, phrase_check, callout, astra_k2pick,
  goals.jsonl). If stopped: `/workspace` survives, the container disk (Claude binary, `~/.claude`) does not; read
  `/workspace/POD_STATE.md` first if it exists.
- **Rule added 2026-10-30:** every paid run is proofread and preflighted for free first (HANDOFF_ORCHESTRATOR "Rules").

## Latest results (NOTES for every number and its file)
- **The LLM adds nothing offline:** on every QA set the KB's own top answer equals K (G Q1 0.846 = 0.846 train,
  0.809 = 0.81 test); on counting, a rule with no per-decision LLM scores 0.98–1.00. Claims 1–2 need the closed loop.
- **K vs the raw log (G perceived QA):** K 0.85 vs 0.55 / 0.69 on train; on G test +0.31 without thinking but
  +0.07 [−0.006, +0.143] with it (falsifier 3 nearly triggered). On tracker-swapped cubes R beats K (0.33–0.36 vs 0.09–0.12).
- **Claim 3 on G, no LLM** (NOTES "Claim 3, corrected read and attribution"; G test now spent for claim 3): the
  registered main test (the re-linking margin feature) is not supported; with distance, the KB's p beats SoftHaircut
  (AURC −0.034 [−0.069, −0.004], AUROC 0.74 vs 0.64). Attribution: R6 does most of the work, the cover kernel the
  rest, R5 nothing: a feature, not the belief layer. p ranks swapped cubes lower (AUROC 0.81) but stays high (median 0.92).
- **Memory organisation (overnight A / B / C):** A (provenance + alternatives in words) and A2 (numbers removed) do not
  improve the model's calibration, which copies the KB's numbers; B (`self_memory.py`: decisions, outcomes, changes
  since the last decision) works offline (0.984 / 0.000 / 0.019) and is in the closed loop's K arm; C (`memtool.py`,
  memory as a tool) is stopped.
- **Perception fixes (Mac side, on G's outputs):** occlusion step for covered cubes (recall 0.907); `relink.py`
  (container id consistency 0.64 → 0.73). The rest is lost container tracks (28% of frames): the tracker itself.
- **Earlier, still true:** synthetic permanence K = raw log with thinking; motion v2 on G test F1 0.81; the VLM cannot
  tell a pick-up from a put-down in 256-px front frames, so robot actions come from robot state.

## Next
See `docs/HANDOFF_ORCHESTRATOR.md` "Open questions": the two free checks on real K views (Inside after uncover, jitter
as moves), a noisy K2 offline test, the dec1e03 read, the preflight script, then one preflighted package for K (fixed)
+ H (+ K2) on the 20 two-pick episodes.

## Reminders for the pod
- Keep `/workspace/g/replays` (the multi-object Cutie re-run needs them). Never print or store the OpenRouter key.
- Stop the pod when a run is done.

## Where things are
- Repo: `~/Code/active/arbiter-kb` (GitHub `vinzppp/arbiter-kb`, private; not pushed since 2026-10-29). Do not modify
  `~/Code/active/scene-graph-robot`; copy pure functions verbatim into `arbiter_kb/sg_compat.py` instead.
- `archive/` holds superseded docs, the M0 / M1 notebooks and superseded results (`archive/README.md` maps old paths).
- Tags: `phase-M0-done`, `phase-M1-done` (bb50365).
- Gate per phase: `make check` (pytest + quality.py + selftest), then the code-reviewer and goal-reviewer agents
  (`/review-phase`, or spawned directly); reviews land in `reviews/`. Fix, record in NOTES, then tag.
- Frozen params: hash a810942859b9, prompts 946e17f79534 (`params.json` holds both; `load_params` refuses a mismatch).
  E5c: test-split reads need a FREEZE line in `results/test_reads.log`; any parameter or prompt change means
  `python scripts/run.py --freeze --force` before the next read.

## What is built
- **The KB (M0–M3):** beliefs as distributions, an evidence log, retraction, ops only via `KB.apply`; rules R1 cover,
  R2 follow, R3 grasp / place, R5 exclusivity, R6 swap; semantic actions from robot state (`actions.py`); motion v2;
  the L4 goal counter; views v4 (`views.py`, `narrative.py`); the controller's own record (`self_memory.py`).
- **Sources:** synthetic, oracle replay, corrupted, G perceived (`perceived.py`, with occlusion and `relink.py`),
  online perception for the closed loop (`online.py`), the RoboMME LeRobot dataset (`lerobot.py`).
- **Baselines and metrics:** Frozen, Bound, the haircuts, SoftHaircut; accuracy, silent failure, coverage, AURC,
  Brier, ECE, paired bootstraps.
- **Controller:** prompt (2610.17350's text and notes verbatim, plus the memory section per arm), parsing, checks;
  arms H / K / R / K0 / S / K-L3; the closed loop (`closed_loop.py`) and the pod runner.
- **G:** 180 perceived episodes (`data/g_perceived/`), the swap gate (swaps exist, best rule AUC 0.926),
  `id_swap_measured`. Still open: `/review-phase G` for `stress/`, and the `--vocab open` re-run on G's replays.
