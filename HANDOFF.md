# ArbiterKB: handoff (2026-10-29)

Read first: `DEVELOPMENT.md` (spec **v1.12**: counting closed as a KB test; `PerceivedSource` with the occlusion step; `relink.py`; v1.11: robot actions from robot state; no periodic VLM review; the VLM lists objects at keyframes and answers one closed question per marked frame (`ask.py`); L4 goal counter; view v4; motion v2; §2 the loop and its reasoning, §3.7 the VLM calls, §3.8 controller, §6 phases), `docs/GOALS.md` (claims, falsifiers), `NOTES.md` (per-phase log; the newest sections are at the bottom). Answer the user in Chinese, concisely. Docs in English.

## Direction (NOTES "Direction v2")
- The KB is the memory module of an LLM controller that outputs waypoints itself, in the 2610.17350 harness.
- Claims: (1) a smaller model, or (2) far fewer tokens and images, at matched success; (3) the KB's beliefs are calibrated (G and M2).

## Latest results (NOTES for every number and its file)
- **Counting is closed as a test of the KB** (NOTES "Is the counting win ArbiterKB's, or the counter's?", qwen3.6-35b-a3b, off / default):
  - RULE (goal parsed once, no LLM per decision) 0.996 / 0.981; it never loses a decision to K;
  - K (v4 view + counter) 0.97 / 0.96 (31–60), 0.965 / 0.952 (61–100);
  - PC (raw log + the same counter line) = K with thinking, worse without (SwingXtimes 0.57–0.79 vs 0.88–0.92);
  - raw log 0.61 / 0.82. The win is robot-state events + the counter, not the KB's beliefs; claim 1 has no content on counting.
- **Real perception (G train, 90 perceived episodes; NOTES "Does the KB-in-words gain survive real perception?")**:
  - the tracker's cube masks climb onto the covering container instead of vanishing (244 / 270); the occlusion step in `sources/perceived.py` fixes that (recall 0.907);
  - Q1 (which container hides each cube): **K 0.78 / 0.77 vs raw log 0.49 / 0.61**, better at both efforts, stock 0.97, stress 0.74; the joined question 0.63 vs 0.21;
  - the worded view (KR) = K (v4 already joins the facts);
  - **claim 3 not supported:** on tracker-swapped cubes every arm is mostly wrong and the stated confidence does not separate right from wrong (AUROC 0.29–0.54); the KB printed 0.98 for swapped containers.
- **Re-linking by motion (`relink.py`, NOTES "Fixing the tracker's swaps")**: container id consistency 0.64 → 0.73; K's Q1 0.85 / 0.84 (+0.07), K − R unchanged. The rest is lost container tracks (28% of frames), which needs the tracker.
- **Earlier, still true:** synthetic permanence K = raw log with thinking (0.96), the KB saves tokens; ArbiterKB ≈ SoftHaircut on AURC (0.182 vs 0.180); motion v2 on G test F1 0.68 → 0.81; the VLM cannot tell a pick-up from a put-down in the front video (≤ 0.52), so robot actions come from robot state and `ask.py` asks one closed question (not yet measured with truth).

## Next
1. **Claim 3 on the Mac:** turn the re-linking's assignment ambiguity (cost margin) into `id_margin` evidence for R6; give SoftHaircut the same feature; measure the KB's own Q1 probability (AUROC on swapped cubes) and then the perceived QA again. Rule first in NOTES. If the KB does not beat SoftHaircut, report claim 3 as negative.
2. **Freeze** the occlusion and re-linking settings in `params.json` (`run.py --freeze --force`); `/review-phase` for the 2026-10-29 work (counting controls, `PerceivedSource`, `relink.py`).
3. **`harness.py`** (the 2610.17350 loop) and a dry run on the Mac; a cost estimate for the closed loop, shown to the user before any spend.
4. **One pod session (the user's go):** one multi-object Cutie on G's replays (lost tracks); `--vocab open`; perception of VideoPlaceOrder / VideoRepick; fresh RoboMME episodes as the final set; a closed-loop pilot with arms H / R / K on permanence and VideoPlaceOrder.
5. **M6** on the fresh episodes. Paused: task-progress compiler, DINOv2 re-ID, event-triggered calls, counting length curves and arm H on counting.
6. Push: the user approves each push.

## Reminders for the pod
- Tell the pod not to delete `/workspace/g/replays`: sizes, the open-vocabulary re-run (POD_RUN §6b) and the relation / ask-call truth checks need them.
- G's perception used the simulator's object list (privileged). Claims need `perceive.py --vocab open`.
- `/review-phase G` must review `stress/` and the sg_pilot worker. DINOv2 is computed per frame and never used.
- Stop the pod when a run is done.

## Where things are
- Repo: `~/Code/active/arbiter-kb` (GitHub `vinzppp/arbiter-kb`, private). Do not modify `~/Code/active/scene-graph-robot`; copy pure functions verbatim into `arbiter_kb/sg_compat.py` instead.
- Tags: `phase-M0-done`, `phase-M1-done` (bb50365). HEAD on main: the 2026-10-29 tests (counting controls, perceived QA, re-linking) and spec v1.12; not pushed.
- Gate per phase: `make check` (pytest + quality.py + selftest, which uses train seeds only), then the user runs `/review-phase Mx` in Claude Code (a code-reviewer and a goal-reviewer agent; reviews land in `reviews/`). Fix, record in NOTES, then tag.
- Current frozen params: hash 49a8577dc73f, prompts ba6ba13b8dbd (`params.json` holds both; `load_params` refuses a mismatch).
- E5c: test-split reads need a frozen `params.json` with a FREEZE line in `results/test_reads.log`. Any parameter change means `python scripts/run.py --freeze --force` before the next test read.
- Cowork session mechanics (if continuing from Claude.ai):
  - edit in the container copy, stage the changed files under `/mnt/user-data/outputs/<fresh dir>/`, then `device_commit_files` to the Mac (reuse of a staged path can serve a stale copy);
  - run `make check` on the Mac via device_bash;
  - commit with `git -c user.name=Claude -c user.email=noreply@anthropic.com`;
  - the Makefile is protected for commit_files, so edit it via device_bash;
  - the user pushes, because the device shell has no GitHub credentials.

## What is built
- **M0 / M1 / M1.1 / M1.2 (done):**
  - The KB: beliefs as distributions, an evidence log, retract, ops applied only via `KB.apply`.
  - Rules: R1 cover, R6 swap (w = 0.5·exp(−min_3d/0.02)), R5 exclusivity (same-frame covers only), R7 decay (off), R2 follow.
  - Synthetic RoboMME-like source with lane / steps "mix". Corruptions: `id_swap_overlap` and `pos_noise`.
  - Baselines: Frozen (HODOR-style), Bound, BoundHaircut, OverlapHaircut, JumpDetector.
  - Metrics: accuracy, silent failure, coverage, AURC, Brier, ECE, risk–coverage figure.
  - `debug explain` reaches observations for every scored row.
- **Honest status (two goal reviews):**
  - On every test row ArbiterKB's answer equals Bound's. Its only effect is graded confidence (R6's distance formula).
  - A KB-free `SoftHaircut` with the same formula would very likely tie. Nothing shows yet that the KB helps.
  - The KB's own contributions are propagation (R5 on ambiguous covers, retraction, containment as a defeasible default, §7 q5) and evidence for the LLM controller (memory arms K vs R / H, spec v1.4 §3.8).
  - **2026-10-29:** as evidence for the controller, the KB beats the raw log on real perception (G Q1 0.77–0.85 vs 0.49–0.68). Its probabilities do not yet track the tracker's swaps (AUROC ≈ 0.5); propagation and retraction remain untested on real data.
- **2026-10-29:** `sources/perceived.py` (PerceivedSource + occlusion step), `relink.py`, `scripts/perceived_qa.py`, PC / RULE arms in `scripts/count_test.py`, the target-count split in `scripts/count_report.py`.
- **G (Mac side, done; pod side in progress):**
  - `stress/stress_env.py`: `VideoUnmaskSwapStress`.
  - `stress/render.py`, `stress/perceive.py`.
  - `arbiter_kb/swapstats.py` + `scripts/measure_swaps.py`: every pass, six features, per-feature and ≤ 2-feature ROC, gate.
  - `stress/POD_RUN.md`.

## G status (NOTES "G — gate result and `id_swap_measured`")
- The full pod run is done: 180 perceived episodes in `data/g_perceived/` (git-ignored; the pod's own logs and results are in `data/g_pod/`). `results/g_passes.csv`, `g_summary.json` and `g_signature.json` are committed.
- Gate: **swaps exist** (stock never swaps; the stress settings swap at 18–82% of passes). No threshold suffices (best AUC 0.926), so M2's "frequent and hard to predict" branch applies.
- `id_swap_measured` is built (`MeasuredSwapSource`). First train-seed table: ArbiterKB = SoftHaircut exactly; both beat Bound on AURC (0.169 vs 0.427).
- **Still open:**
  1. `/review-phase G` (the pod's only code change, two settings in `stress/render.py`, is merged);
  2. the stock oracle replays (`scripts/copy_replays.sh`) and the optional `--vocab open` re-run (POD_RUN §6b) need a pod again;
  3. remind the user to STOP THE POD if it is still up.
- **Next for M2 (now Next item 1; `relink.py` is the Hungarian step):** the margin as evidence + R6 v2 (d0 re-tuned: real swaps happen at 5–9 cm, not ≤ 3 cm), tuned on train seeds against `id_swap_measured`; give SoftHaircut the same feature; freeze; one test read.

