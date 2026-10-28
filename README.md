# ArbiterKB

A robot knowledge base whose facts carry beliefs (confidence, time, evidence), used as the memory module of an LLM controller that outputs waypoints for a motion planner (spec v1.4). The goal is to match a frontier VLM that reads the whole image history (arXiv 2610.17350), with a smaller model or far fewer tokens. Build spec: `DEVELOPMENT.md`. Goals: `docs/GOALS.md`. Log: `NOTES.md`.

**Status (2026-10-28):**
- M0–M2 infrastructure: KB, beliefs, baselines, metrics with CIs.
- G: running on a pod.
- M3, first part: events, relations, event log, controller view.
- M4, Mac side: OpenRouter client, controller, dry run. Not run yet.

It runs on synthetic RoboMME-like data on CPU. Real replays need `scripts/copy_replays.sh`. See `HANDOFF.md` and `NOTES.md` for details.

```
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
make check PY=python          # tests + quality + M0 self-test (PASS/FAIL lines)
make eval PY=python ARGS="--corrupt none,id_swap --p 1.0"
python -m arbiter_kb.debug explain --run results/runs/<run> --trace id_swap_arbiterkb_1025 --cube cube_green
make notebook PY=python       # notebooks/m{0,1}_walkthrough.ipynb -> executed/
```
Reviews: one-time `mkdir -p .claude && cp -R setup/claude/agents setup/claude/commands .claude/`, then `/review-phase M0` in Claude Code.
