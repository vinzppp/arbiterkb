# Code review: 2026-10-29 work (commits 9001dd4^..fbd8d0e)

Reviewer: independent code-reviewer agent; written to file by the main session (the agent worked read-only).
Verdict: **BLOCK** (one truth bug changes reported numbers). `make check` passed (177 tests, quality PASS, selftest 16 / 0 / 1 skipped). Every stored perceived-QA report was recomputed from its rows and matched.

## Findings and what was done (NOTES "Overnight code review: fixes")
- **BLOCKER** `scripts/perceived_qa.py` Q6 truth: the container xy was taken at the cube's last visible frame (t = 31), where the oracle container is NaN, so Q6 was "no" in all 90 episodes. **Fixed** (first finite container position at or after the cover) and tested.
- **MAJOR** "tracker-swapped" counted `then is None` (no slot on the true container just after the cover) as a swap: 17 of 67 raw, 17 of 48 re-linked. **Fixed:** None means unknown, left out of the subset.
- **MAJOR** `epistemic.strip_probs` kept "(conf 0.99)" in event lines (303 times over 90 KN views), so A2's KN was not the registered arm. **Fixed** and tested; KN re-run.
- **MINOR** alternatives propagate back from the end rather than forward in time (11 / 225 sets differ; truth-in-set 0.960 vs 0.956). Not changed (registered design, negligible).
- **MINOR** `slot_on` took the lowest slot when two sit on the true container (about 0.4 pp). **Fixed:** any slot counts.
- **MINOR** `qa_stats.auroc_pair` resampled cubes, not episodes. **Fixed.**
- **MINOR** a last-turn query in C is scored as a wrong answer; a string "query" ends the loop. As documented; untested.
- **MINOR** the harness decides 8 frames into a subgoal; an early action can fall before the snapshot. Not observed (wrong intent 0.000); may explain some SwingXtimes hold misses.
- **MINOR** test gaps: `truth`, `strip_probs` (now tested), `alternatives`, `qa_stats`, `ReplayEnv`, `count_test.rule`, the KT loop.

## Checked and found correct
Truth and memory read the same arrays; occlusion hysteresis; relink (0 of 28,636 container observations dropped; F_ID / F_LABEL; cubes untouched); the KT loop's turn limits and token sums; RULE rows (0 usage, identical across efforts); `by_target`; ReplayEnv windows (contiguous, exclusive ends); `outcome`'s `until` boundary.

## Unused code
`self_memory.intent` (waypoint inference) is only used by tests; `harness.Env` has no implementation besides ReplayEnv (for the pod).
