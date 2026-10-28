# ArbiterKB: handoff (2026-10-28, 04:20 PT)

Read first: `DEVELOPMENT.md` (spec v1.3, §6 phases), `docs/GOALS.md` (claims, falsifiers), `NOTES.md` (per-phase log; the newest sections are at the bottom). Answer the user in Chinese, concisely. Docs in English.

## Where things are
- Repo: `~/Code/active/arbiter-kb` (GitHub `vinzppp/arbiter-kb`, private). Do not modify `~/Code/active/scene-graph-robot`; copy pure functions verbatim into `arbiter_kb/sg_compat.py` instead.
- Tags: `phase-M0-done`, `phase-M1-done` (bb50365). HEAD: G Mac-side code plus the POD_RUN fresh-volume path.
- Gate per phase: `make check` (pytest + quality.py + selftest, which uses train seeds only), then the user runs `/review-phase Mx` in Claude Code (a code-reviewer and a goal-reviewer agent; reviews land in `reviews/`). Fix, record in NOTES, then tag.
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
  - The KB's own contributions are propagation (R5 on ambiguous covers, retraction, containment as a defeasible default, §7 q5) and evidence for the LLM (M4 arm C vs D). They are untested on real data.
- **G (Mac side, done; pod side in progress):**
  - `stress/stress_env.py`: `VideoUnmaskSwapStress`.
  - `stress/render.py`, `stress/perceive.py`.
  - `arbiter_kb/swapstats.py` + `scripts/measure_swaps.py`: every pass, six features, per-feature and ≤ 2-feature ROC, gate.
  - `stress/POD_RUN.md`.

## Running right now (pod, user-operated)
- A new RunPod pod with a **fresh** network volume (the old volume's datacenter had no GPUs), GPU RTX 4000 Ada (or similar).
- The pod Claude session is following `stress/POD_RUN.md` §0F:
  - building `sg-main` (robomme) and `sg-percep` (GroundingDINO / SAM / Cutie);
  - downloading weights;
  - then the G smoke test (step 3), after which it stops.
- The pod code in `stress/` has never run against the simulator. Expect fixes there, and only there.
- **Next when the smoke output comes back:**
  1. review it: T, schedule, `pose_log_aligned`, seeds, per-episode time;
  2. re-estimate the cost;
  3. the user decides on the full run (POD_RUN steps 4–5);
  4. copy `g_perceived` to `data/g_perceived/`, then run `python scripts/measure_swaps.py`;
  5. write the gate decision in NOTES. It is one of three outcomes:
     - swaps under 2% of passes → fallback;
     - swaps exist → define `id_swap_measured`;
     - a ≤ 2-feature rule with AUC ≥ 0.98 → "a threshold suffices" (say so before M2).
  6. pull the pod's `stress/` fixes back into the repo;
  7. then `/review-phase G`.
- Remind the user to STOP THE POD.

## Parallel work that does not depend on G (candidates for the new window)
1. **M2 / M4 infrastructure (independent of G's numbers):**
   - `SoftHaircut(feature)` baseline (Bound's answer, confidence 1 − w_max·exp(−feature/scale));
   - Wilson CI on coverage;
   - bootstrap CIs for AURC;
   - matched-coverage comparison (sweep the haircut threshold on train);
   - support for n ≥ 300 fresh test seeds (≥ 1060);
   - drop Frozen / BoundHaircut / JumpDetector from M2 tables.
2. **M4 skeleton:**
   - `llm.py` (OpenRouter `qwen/qwen3.6-35b-a3b`, DeepInfra pinned; key in the git-ignored `.env`, never printed), with a cache and a budget cap;
   - the prompt builder for arm D (KB text view + cited evidence);
   - **arm C, an LLM on the raw tracker log without a KB** (same schema, same keyframes). C vs D is the main test of whether the KB helps.
3. **Open question §7 q5 (containment as a defeasible default):** check whether RoboMME or other tasks have "container lifted, object left behind" cases before building anything synthetic.
4. **MINORs from the M1.1 code review** (listed at the end of NOTES M1.2): haircut-trigger trace, argmax tie order, and so on.
