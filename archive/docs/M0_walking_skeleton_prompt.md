# Build prompt — ArbiterKB phase M0 (walking skeleton)

Build phase **M0** of `~/Code/active/arbiter-kb/DEVELOPMENT.md`.

1. Read §0–§5 of DEVELOPMENT.md in full, then §6 M0. Also read `scene-graph-robot/docs/evidence-gaps-2026-10-28.md` §6–7 (how VideoUnmaskSwap really moves objects) and `scene-graph-robot/notebooks/pilot/sg_pilot.py` (the functions to copy into sg_compat.py, §0.4).
2. Work in small commits: scaffold + contracts → beliefs/KB + tests → synthetic source → baselines → ArbiterKB system → evaluation/trace/debug → CLI + notebook.
3. Respect the engineering evals E1–E10 (§1) and the mandatory patterns (pure rules returning ops; one params source; plain data).
4. Stop when every "Done when" item of M0 prints PASS in `make check`, or print SKIPPED with the reason.
5. Do not run the reviews yourself; the user runs `/review-phase M0` afterwards (DEVELOPMENT.md §1.1). Do read `docs/GOALS.md`, especially "Known traps".
6. Finish with a 5-line summary: what was built, which checks passed, what was skipped and why, the next command to run.

No GPU, no simulator, no LLM, no network. Don't modify `scene-graph-robot`.
