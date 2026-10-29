# Code review — claim3_g (e5b30fe^..f489a23) — 2026-10-29
Verdict: BLOCK
make check: 179 passed; quality 103 files, 0 problems; selftest 16 PASS, 0 FAIL, 1 SKIPPED ((g) no replays)
Reproducibility: yes. Re-running the train grid rows gives the same AURC/acc as grid.json (bound 0.1295, kb/min_3d/0.02 0.0908, kb/min_q/0.05 0.0874, soft/min_q/0.05 0.0950, soft/min_3d/0.02 0.1127).

## Findings (most severe first)
- [BLOCK] scripts/claim3_g.py:60-63: `aurc` breaks ties by input row order (stable sort on -p, then per-row cumulative risk). The spec (DEVELOPMENT.md:444, "ties enter together") and arbiter_kb/metrics.py:30 do it differently. Row order is episode order, then cube order, so the metric depends on that order. SoftHaircut(min_q, q0 0.05) puts 206/241 train rows at p = 1.0, because `round(w, 6)` in soft_haircut.py:44 zeroes w when min_q is about 1. Bound puts all rows at 1.0. Train, claim3_g vs metrics.aurc: bound 0.1295 vs 0.1660; soft/min_q 0.0950 vs 0.1169; soft/min_3d 0.1127 vs 0.1214; kb/min_q 0.0874 vs 0.0950; kb/min_3d 0.0908 vs 0.0993. Random tie orders give soft/min_q anywhere from 0.070 to 0.167. On train the main contrast KB − Soft (min_q) goes from −0.008 to −0.022. The bootstrap at :79-80 reorders rows (by pick), so its tie order differs from the point estimate's. **Reported numbers change:** every AURC in grid.json and read.json, and the paired AURC CIs. The main reading ("min_q: not supported", CI [−0.088, +0.019]) may flip. The grid choice does not change under the spec AURC (I checked all 16 settings). Fix: use arbiter_kb.metrics.aurc, mapping p→p_max and right→correct. Re-run the read with the committed choice.json and log it in the ledger as a correction.
- [MAJOR] tests/: nothing in the range tests the new code. No tests for relink_margin (value, inf when only one identity is matched, pairwise index use), Detectors min_q (minimum over an open pass), r6_feature/q0 in r6_swap, SoftHaircut's min_q scale, or PerceivedSource track_q (container-only, cap, inf→1). A wrong index in relink.py:42-47 or a swapped scale in rules.py:103 would still pass make check.
- [MINOR] relink.py:42-47: the margin only counts competitors that were matched in that frame. A live identity that is not matched (container hidden or lost) is ignored, and so is a first-seen identity (assigned after the Hungarian step at :48-50). Their neighbours get margin inf, so track_q = 1, which is exactly where a re-link error is possible. Also, the minimum is over every matched identity, not the pass partner, so a third container can set (a,b)'s min_q. This does not change the read as frozen. It is a design limit to mention in NOTES.
- [MINOR] scripts/claim3_g.py:69-71,125: `auroc(...) or 0.5` maps a real AUROC of 0.0 (perfectly inverted) to 0.5. It does not affect the reported values (all are > 0.5 or are None→0.5 for Bound). Fix: `x if x is not None else 0.5`.
- [MINOR] arbiter_kb/systems/soft_haircut.py:44: `round(w, 6)` is not in R6. The KB keeps unrounded w, so tiny weights give KB distinct p values while SoftHaircut collapses to 1.0. Ranking is unchanged in principle, but combined with the tie bug it inflates SoftHaircut's lead over its true AURC. Drop the rounding, or round only for the trace.
- [MINOR] scripts/perceived_qa.py:207-218: with `--split test --print`, test prompts are built and printed without a ledger READ, and nothing stops `--r6 min_q:*` without `--relinked`, where track_q = 1, so min_q = 1 and w is about 0. The read as run was correct (meta split test, r6 min_q:0.05, relinked True, commit e91ee66 clean, READ a810942859b9 = base hash). Usage shows 8 cache hits out of 360 "new" calls: the source is not established (one READ line only). Check which prompts hit.
- [MINOR] perceived_qa summary meta.params_hash = 40e166310e73 (the replaced params) while the ledger records a810942859b9. That is correct but undocumented. Write both into meta.

## Checks requested
1. relink_margin: the indices are right. `i`/`i2` are cost rows (live positions), `col[i]` is the cost column (obs position), and `margin[t, live[i]]` is indexed by identity, which matches perceived.py:92 `margin[t, s]` on out-rows (= identities). Deltas are ≥ 0 at the Hungarian optimum (checked on 200 random matrices, N ≤ 4). relink output is bit-identical to the pre-change relink on all 180 G files. No NaN anywhere. inf (31% of entries) gives track_q = min(1, inf) = 1. Cubes always get 1. min_q is taken only over frames where both containers move and d < thr.
2. track_q leakage: observe.py:21-22 puts it into pose/visibility conf. Those feed views (views.py:50-67) and the held-object SetPose conf, not Inside. I re-ran kb/min_3d/0.02 on train with track_q forced to 1 (MARGIN_CAP = 1e-12) and got identical right/p on all 241 rows, so the KB-vs-SoftHaircut and min_3d rows are unaffected. For the K arm in perceived_qa, container confidences under 1 can now reach the view text, so K on test does not have exactly the train-era view (NOTES "train: +0.15" comparison).
3. claim3_g: excluding None from the top answer is the same for all systems (one `answers()`). Pairing is the same cubes in the same order (checked). make() replaces d0/q0/r6_feature in the rule. SoftHaircut takes the feature as an argument and the scale from the rule (consistent). TrackerSystem ignores both. The read parsed choice.json into exactly the committed settings (grid at e5b30fe clean, read at c865523 clean, READ after FREEZE a810942859b9). AURC tie handling: BLOCK above.
4. perceived_qa --r6: K's KB is built with with_r6(params) (memory() → ArbiterKBSystem(params)). The ledger line uses the base frozen hash. The cache key is sha256 of the body including messages (llm.py:102), so test prompts are new calls apart from the 8 unexplained hits. I made no calls.
5. Tests: none broken. The missing tests are listed above.

## E1–E10 (only those touched)
| Eval | Result | Evidence |
|---|---|---|
| E1 make check | PASS | 179 passed; selftest 16/0/1 SKIPPED |
| E2 explain chain | PASS (selftest) | "M1 E2: explain reaches an observation for every scored row: 1080 rows, 0 failing". Not exercised for claim3_g rows |
| E4 reproducible | PASS | train grid rows re-run match grid.json exactly |
| E5 test read after freeze | PASS | test_reads.log: FREEZE 13:57:15, READ 13:58:23 (claim3_g), READ 14:00:11 (perceived_qa); choice.json committed in c865523 before the read |
| E6 same eval path for baselines | FAIL | claim3_g.aurc ≠ arbiter_kb.metrics.aurc (spec), see BLOCK |
| E7 step time | PASS | 0.173 ms |
| E3, E8–E10 | not re-assessed | no change in this range |

## Unused code / one-call abstractions
- relink.relink is now a one-line wrapper over relink_margin. It is kept for tests/test_perceived.py:37 and is fine.
- perceived_qa.with_r6 / log_test_read have one call site each (main). Acceptable.
- No new dependencies.
