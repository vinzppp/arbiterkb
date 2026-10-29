# ArbiterKB: handoff (2026-10-29)

Read first: `DEVELOPMENT.md` (spec **v1.11**: robot actions from robot state; no periodic VLM review; the VLM lists objects at keyframes and answers one closed question per marked frame (`ask.py`); L4 goal counter; view v4; motion v2; §2 the loop and its reasoning, §3.7 the VLM calls, §3.8 controller, §6 phases), `docs/GOALS.md` (claims, falsifiers), `NOTES.md` (per-phase log; the newest sections are at the bottom). Answer the user in Chinese, concisely. Docs in English.

## Direction (NOTES "Direction v2")
- The KB is the memory module of an LLM controller that outputs waypoints itself, in the 2610.17350 harness.
- Claims: (1) a smaller model, or (2) far fewer tokens and images, at matched success; (3) the KB's beliefs are calibrated (G and M2).

## Latest results (NOTES for every number and its file)
- **Counting on RoboMME robot state** (qwen3.6-35b-a3b, off / default thinking):
  - raw log 0.61 / 0.82;
  - KB v3 view 0.83 / 0.94, the same as Sonnet 5.5 with the raw log (0.94) at a fifth of the cost;
  - **v4 with the goal counter 0.97 / 0.96** (episodes 31–60; the parse rules were written after errors there);
  - oracle 0.96 / 0.99.
- **Wording:**
  - the KB in words helps when it joins facts (synthetic questions: 0.66 → 0.91 without thinking);
  - "now" sentences hurt counting (0.83 → 0.70–0.74);
  - view v4 = v3 + the counter line + joined facts.
- **VLM:**
  - listing from the image alone 4 / 4 (with memory text 2 / 4, 0 / 4);
  - a periodic review confirmed 68% of planted false events;
  - telling a pick-up from a put-down in the front video ≤ 0.52, even for Sonnet.
  - → removed; `ask.py` asks one closed question with no memory text (not yet measured with truth).
- **Motion v2:** real tracker (G test) F1 0.68 → 0.81, spurious moves −43%. Fragments under 15 mm iid noise.
- **Permanence (synthetic, text):** with thinking the raw log is as good as the KB (0.96); the KB saves tokens.
- **Claim 3:** ArbiterKB ≈ SoftHaircut (AURC 0.182 vs 0.180). Not supported yet.

## Next
1. **The full rerun on final sets** (NOTES "Before the full rerun", rule first):
   - counting episodes 1–30 with v4 vs v3 vs raw log vs oracle, plus Sonnet with v4;
   - synthetic permanence and questions on test seeds (≥ 1000, recorded in the E5c ledger);
   - `scripts/count_report.py` makes the tables.
2. **Pod:**
   - the rules' relations against simulator truth;
   - the ask call with truth;
   - perceived-data runs;
   - the closed loop (M5);
   - VideoRepick tracking.
3. Push: the user approves each push.

## Reminders for the pod
- Tell the pod not to delete `/workspace/g/replays`: sizes, the open-vocabulary re-run (POD_RUN §6b) and the relation / ask-call truth checks need them.
- G's perception used the simulator's object list (privileged). Claims need `perceive.py --vocab open`.
- `/review-phase G` must review `stress/` and the sg_pilot worker. DINOv2 is computed per frame and never used.
- Stop the pod when a run is done.

## Where things are
- Repo: `~/Code/active/arbiter-kb` (GitHub `vinzppp/arbiter-kb`, private). Do not modify `~/Code/active/scene-graph-robot`; copy pure functions verbatim into `arbiter_kb/sg_compat.py` instead.
- Tags: `phase-M0-done`, `phase-M1-done` (bb50365). HEAD: G Mac-side code plus the POD_RUN fresh-volume path.
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
  - The KB's own contributions are propagation (R5 on ambiguous covers, retraction, containment as a defeasible default, §7 q5) and evidence for the LLM controller (memory arms K vs R / H, spec v1.4 §3.8). They are untested on real data.
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
- **Next for M2:** `identity.py` (Hungarian margin) + R6 v2 (d0 re-tuned: real swaps happen at 5–9 cm, not ≤ 3 cm), tuned on train seeds against `id_swap_measured`; give SoftHaircut the same feature; freeze; one test read.

## Parallel work that does not depend on G (candidates for the new window)
1. **M2 infrastructure: done** (NOTES "M2 infra", commit 73fab80 and the NOTES commit after it).
   - Built: `SoftHaircut`, Wilson CI on coverage, bootstrap / paired-bootstrap AURC, matched coverage (`scripts/compare.py`), `--first` for fresh seeds, and `--systems m2`.
   - Findings on train seeds: ArbiterKB = SoftHaircut exactly under id_swap_overlap.
   - Under pos_noise ArbiterKB was worse, because R1 returned a confident {None: 1}. Fixed (`eps_far`, re-frozen as f59aaea5e5b2): ArbiterKB now beats SoftHaircut under pos_noise, on held-out train seeds 450–749 as well.
   - That gain is a cover-distance feature, not the KB. In M2, give SoftHaircut the same check.
2. **M4 skeleton (now the controller harness, spec v1.4 §3.8 / M4):**
   - `llm.py` (OpenRouter `qwen/qwen3.6-35b-a3b`, DeepInfra pinned; key in the git-ignored `.env`, never printed), with a cache and a budget cap;
   - `harness.py`, ported from the 2610.17350 code;
   - `controller.py` with the memory arms H / R / K / K−L3 / K+A and the waypoint validator;
   - a replay-driven dry run on the Mac (no GPU).
   - M3's event log and the other KB layers can be built in parallel.
3. **Open question §7 q5 (containment as a defeasible default):** check whether RoboMME or other tasks have "container lifted, object left behind" cases before building anything synthetic.
4. **MINORs from the M1.1 code review** (listed at the end of NOTES M1.2): haircut-trigger trace, argmax tie order, and so on.
