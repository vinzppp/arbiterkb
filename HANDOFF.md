# ArbiterKB: handoff (2026-10-29)

Read first: `DEVELOPMENT.md` (spec **v1.12**: counting closed as a KB test; `PerceivedSource` with the occlusion step; `relink.py`; v1.11: robot actions from robot state; no periodic VLM review; the VLM lists objects at keyframes and answers one closed question per marked frame (`ask.py`); L4 goal counter; view v4; motion v2; §2 the loop and its reasoning, §3.7 the VLM calls, §3.8 controller, §6 phases), `docs/GOALS.md` (claims, falsifiers), `NOTES.md` (per-phase log; the newest sections are at the bottom). Answer the user in Chinese, concisely. Docs in English.

## Direction (NOTES "Direction v2")
- The KB is the memory module of an LLM controller that outputs waypoints itself, in the 2610.17350 harness.
- Claims: (1) a smaller model, or (2) far fewer tokens and images, at matched success; (3) the KB's beliefs are calibrated (G and M2).

## Latest results (NOTES for every number and its file)
**Read NOTES "Corrected tables after the code review"** (2026-10-29): it supersedes the earlier perceived-QA numbers, because the Q6 truth and the "swapped" flag were wrong.
- **Counting is closed as a test of the KB:** a rule with no per-decision LLM (RULE) scores 0.98–1.00 and never loses a decision to K; PC (raw log + counter) = K with thinking. The win is robot-state events + the counter.
- **G perceived QA (train only, re-linked tracks; simulator word list):**
  - Q1 (which container hides each cube): K 0.85, raw log 0.55 / 0.69 (off / default). **The KB's own top answer with no LLM is also 0.846.** So the LLM adds nothing on Q1; "K beats the raw log" is the KB's rules beating an LLM reading the raw log.
  - On tracker-swapped cubes every arm is mostly wrong (K 0.13).
  - Calibration: the KB's own p has AUROC 0.69; the model's stated confidence 0.67–0.68; the stock / stress bit 0.56.
- **Overnight A / B / C (memory organisation for the controller):**
  - **A** (how each fact is known + alternatives in words): no gain (AUROC KA − K 0.00). **A2** (numbers removed): the stated confidence spreads out but its AUROC falls (−0.07 / −0.11, n.s.). Better calibration needs better KB probabilities, not presentation.
  - **B** (the controller's decisions and outcomes, changes since the last decision, `self_memory.py`, `harness.py` skeleton): offline, outcomes are right for 0.984 of right intents; 0.000 "done" for wrong intents and 0.019 for premature ones (after a side / how fix). Its value can only be measured in the closed loop.
  - **C** (memory as a tool, `memtool.py`): a 220-token working memory keeps Q1 at 0.82 (it states the KB's answer, so this is by construction); queries do not recover the history questions. Stopped.
- **Fixes on G's perception (Mac side):** occlusion step for covered cubes (recall 0.907); `relink.py` (container id consistency 0.64 → 0.73). Lost container tracks (28% of frames) need the tracker itself.
- **Earlier, still true:** synthetic permanence K = raw log with thinking; ArbiterKB ≈ SoftHaircut on AURC; motion v2 on G test F1 0.81; the VLM cannot tell a pick-up from a put-down in the front video.

## Next
1. **Claim 3, on the Mac, without an LLM:** turn `relink.py`'s assignment cost margin into `id_margin` evidence for R6. Score the KB's p (AUROC / AURC on swapped vs clean cubes, within stress) against SoftHaircut with the same margin feature, the setting bit and the near-pass heuristic. Tune on G train; commit the rule; then **one frozen G test read**, adding R / K / the KB's top answer on Q1 (about $0.4). If the KB does not beat SoftHaircut, claim 3 is negative.
2. **The closed loop is the only place claims 1–2 can move:** port the 2610.17350 prompt skeleton and completion rules into `harness.py` (downloading their code needs the user's go). Then one pod session: a multi-object Cutie on G's replays, `--vocab open`, perception of VideoPlaceOrder / VideoRepick, fresh RoboMME episodes, and a closed-loop pilot with arms H / R / K (K with `self_memory`'s view first).
3. **M6** on the fresh episodes.
4. **Dropped or frozen:** C (`memtool.py`), more view arms (A, KR), the LLM's stated confidence as a claim-3 measure; B frozen until M5; counting.
5. Push: the user approves each push.

## Reminders for the pod
- Tell the pod not to delete `/workspace/g/replays`: sizes, the open-vocabulary re-run (POD_RUN §6b) and the relation / ask-call truth checks need them.
- G's perception used the simulator's object list (privileged). Claims need `perceive.py --vocab open`.
- `/review-phase G` must review `stress/` and the sg_pilot worker. DINOv2 is computed per frame and never used.
- Stop the pod when a run is done.

## Where things are
- Repo: `~/Code/active/arbiter-kb` (GitHub `vinzppp/arbiter-kb`, private). Do not modify `~/Code/active/scene-graph-robot`; copy pure functions verbatim into `arbiter_kb/sg_compat.py` instead.
- Tags: `phase-M0-done`, `phase-M1-done` (bb50365). HEAD on main: the 2026-10-29 work incl. the overnight A / B / C tests and both reviews (`reviews/overnight_2026-10-29_*.md`); not pushed.
- Gate per phase: `make check` (pytest + quality.py + selftest, which uses train seeds only), then the user runs `/review-phase Mx` in Claude Code (a code-reviewer and a goal-reviewer agent; reviews land in `reviews/`). Fix, record in NOTES, then tag.
- Current frozen params: hash 49a8577dc73f, prompts 0c77893f87f4 (`params.json` holds both; `load_params` refuses a mismatch).
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
- **2026-10-29:** `sources/perceived.py` (PerceivedSource + occlusion step), `relink.py`, `epistemic.py`, `memtool.py`, `self_memory.py`, `harness.py` (skeleton + ReplayEnv), `scripts/perceived_qa.py` + `qa_stats.py`, `scripts/check_self_memory.py`, PC / RULE arms in `scripts/count_test.py`, the target-count split in `scripts/count_report.py`.
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

