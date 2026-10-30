# ArbiterKB

A robot memory module for an LLM controller. A knowledge base (KB) whose facts carry beliefs (probability, time,
evidence) is built from perception and robot state, and a controller LLM reads it to output end-effector waypoints for
RoboMME's motion planner, in the harness of arXiv 2610.17350. The question: can a structured memory let a **smaller
model**, or the same model with **far fewer tokens and images**, match a VLM that reads the whole image history; and
are the KB's beliefs **calibrated** (`docs/GOALS.md`).

## Where things stand (2026-10-29)
- **Offline (done):** on every question-answering set the KB beats the raw log for a cheap model (G perceived episodes,
  Q1 0.85 vs 0.55–0.69), but the KB's own top answer is as good as the LLM reading it, and a rule with no LLM matches it
  on counting. Claims 1–2 can only move in the closed loop.
- **Claim 3 (no LLM, G test):** the registered main test is not supported; with the distance feature the KB's p ranks
  its errors better than SoftHaircut, and the lead comes from R6 on a cover-time distribution, not the belief layer.
- **Closed loop (in progress):** the 2610.17350 loop runs on a RunPod A40. Qwen as the controller hit a floor (1 / 25); GPT-6-Astra solved 2 / 2 probes; its H / R / K pilot on VideoUnmaskSwap is running. The execution gap: `docs/HANDOFF_EXECUTION.md`.
  running on a RunPod A40 (`stress/POD_CLOSED_LOOP.md`). Next: a val pilot with arms H / R / K.
- Details and every number: `HANDOFF.md` (the short version), `NOTES.md` (the log, newest at the bottom).

## Read in this order
| File | What it is |
|---|---|
| `HANDOFF.md` | the current state, latest results, next steps, where things are |
| `docs/GOALS.md` | claims, falsifiers, what is ours and what is not |
| `DEVELOPMENT.md` | the build spec (v1.12): the loop (§2), how it works (§3), contracts (§4), phases (§6) |
| `NOTES.md` | the per-phase log: every test, its rule fixed before running, its results and readings |
| `reviews/` | the independent code and goal reviews per phase |
| `stress/POD_CLOSED_LOOP.md`, `stress/POD_RUN.md` | the pod runbooks (closed-loop pilot; G) |
| `archive/` | superseded docs, notebooks and results (`archive/README.md`) |

## Layout
```
arbiter_kb/   the package: KB, beliefs, observe / detectors / rules, views, controller, closed_loop, sources/, systems/
scripts/      experiments and reports (each test's script; run.py, selftest.py, quality.py)
stress/       pod-side code and runbooks (rendering, perception, the closed-loop runner)
tests/        pytest, one file per module
results/      latest/<test>/ (committed results), test_reads.log (the E5c ledger); runs/ and llm_cache/ git-ignored
data/         git-ignored: G's perceived episodes, the RoboMME LeRobot dataset tables
figures/      committed figures   tools/   dataset conversion and frame extraction
setup/claude/ the review agents and /review-phase command, to copy into .claude/
```

## Quick start (Mac, CPU)
```
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
make check           # pytest + quality.py + the self-test (PASS / FAIL lines)
make eval ARGS="--corrupt none,id_swap --p 1.0"
python -m arbiter_kb.debug explain --run results/runs/<run> --trace <trace> --cube cube_green
```
LLM tests need `OPENROUTER_API_KEY` in the git-ignored `.env`. Test-split reads need a frozen `params.json`
(`python scripts/run.py --freeze --force`) and are logged in `results/test_reads.log`.
Reviews: one-time `mkdir -p .claude && cp -R setup/claude/agents setup/claude/commands .claude/`, then `/review-phase <phase>`.
