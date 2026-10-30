# ArbiterKB: handoff (2026-10-30)

Read first: `README.md` (one page), `docs/GOALS.md` (claims, falsifiers), `DEVELOPMENT.md` (spec v1.12: §2 the loop and its
reasoning, §3.8 controller, §6 phases), `NOTES.md` (the log; newest at the bottom). Answer the user in Chinese,
concisely. Docs in English. For the execution-gap discussion (IK, controller design, models, landscape):
`docs/HANDOFF_EXECUTION.md`. Upstream bug reports to prepare: `docs/UPSTREAM.md`.

## Direction (NOTES "Direction v2")
- The KB is the memory module of an LLM controller that outputs waypoints itself, in the 2610.17350 harness.
- Claims: (1) a smaller model, or (2) far fewer tokens and images, at matched success; (3) the KB's beliefs are calibrated.

## Now: the closed loop runs on a RunPod A40 (NOTES "Closed loop", bottom)
- Code: `arbiter_kb/closed_loop.py` (2610.17350's loop), `arbiter_kb/sources/online.py`, `stress/closed_loop_pod.py`
  (`--model`, `--provider auto`, `--max-usd-episode`), runbook `stress/POD_CLOSED_LOOP.md`. Upload = `git archive` →
  `pod_upload_arbiter-kb.tar.gz` (repo root, git-ignored); the pod session checks its md5 before extracting.
- **Qwen 3.6-35B-A3B val pilot, stopped at 25 / 60:** 1 success; every arm near 0 (floor effect). Failures are
  execution and decisions (layers 2, 3, 5 in docs/HANDOFF_EXECUTION.md), runaway reasoning, and ManiSkill's zero-step bug.
- **GPT-6-Astra probe:** H 2 / 2 ($0.17, $0.65 per episode). The harness works; the controller model was the bottleneck.
- **Running on the pod:** (1) Astra VideoUnmaskSwap val pilot, H / R / K × 10 (`/workspace/cl/astra_pilot`, budget $20);
  (2) the VideoPlaceOrder target-phrase test (no LLM; rule fixed in NOTES before measuring). The pod's Claude runs
  as `/root/.local/bin/claude --continue` in tmux `main` (window 0); jobs in windows avus / afinish / phrase.
- Fixed this round (tests, reviews, re-frozen; prompts 7aade10cec0b): R's ids and cites, the `"table"` crash, 32k cap,
  targets are not containers (`task_items.place_word`, `PLACE_PHRASE`), robot-base / below-table detections dropped
  (`online.not_objects`), API defaults for non-Qwen models, unscored per-episode spend cap.

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
1. Read the Astra VideoUnmaskSwap pilot and the phrase test; look at K's failures (memory wrong vs read wrong vs
   premature "done"). Pick VideoPlaceOrder's target phrase by the NOTES rule; run that task only if it passes.
2. The execution-gap decisions in `docs/HANDOFF_EXECUTION.md` (model ladder with an oracle-memory arm, fair harness
   fixes, a skills layer as a separate system claim, the KB as completion verifier, RoboMME's `multi_choice`).
3. Upstream: a minimal repro of ManiSkill's `follow_path` zero-step crash (docs/UPSTREAM.md); the user files it.
4. Claim 3 on fresh episodes; M6 on test after the freeze; the multi-object Cutie on G's replays.

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
- Frozen params: hash a810942859b9, prompts c4b915da48c6 (`params.json` holds both; `load_params` refuses a mismatch).
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
