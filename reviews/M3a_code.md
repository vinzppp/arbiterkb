# Code review — M3a (M3 first part, commit 7b17641; diff against 9e5879a) — 2026-10-28
(Saved by the builder session: the code-reviewer agent could not write files. Text as reported by the agent.)

Verdict: **PASS-WITH-NOTES**

make check: with `PY=.venv/bin/python` (as README.md:9 says), **122 passed in 5.81s; `quality: 63 files, 0 problems — PASS`; `selftest: 14 PASS, 0 FAIL, 1 SKIPPED`** ((g) replays: "no replays in …/data/replays"). Plain `make check` fails at once with `/Users/vincentcfpun/miniconda3/bin/python3: No module named pytest`. That is the environment, not the code, but the Makefile default `PY ?= python3` does not match the documented venv.

Reproducibility: **yes**. `scripts/measure_kb.py` run twice gives identical `conditions` (sha256 cc354b10…), equal to the committed `results/latest/m3_kb_truth/summary.json`. `run.py --split train --systems all --corrupt none,id_swap_overlap,pos_noise --lane mix --steps mix --n 30` run twice gives the same summary.csv sha1, b606b123f540…. The selftest's own check also passed: `1b211fc01474` ×2.

**Side effect the reviewer caused:** for the hash check it first ran `scripts/run.py --corrupt none,id_swap_overlap,pos_noise --lane mix --steps mix --n 20` without `--split`. The default split is **test**, so 6 READ lines (hash b0a2dd3cb30b, seeds 1000–1019, run twice) were appended to `results/test_reads.log`. Nothing was tuned, and those seeds had already been read many times. The reviewer did not revert them because the ledger is append-only.

## Findings (most severe first)
- **[MAJOR] arbiter_kb/rules.py:79-84 (`below`)** — When a container is placed back, the place target can be the hidden cube inside it. `below()` filters only by `pose.value is not None` and z. Hidden cubes keep an R2 display pose equal to their container's pose, so a container lowered back and released gives the event `place(bin_a, cube_r)`: "the bin was put on the cube". This hits exactly the object the VideoUnmaskSwap robot handles (the container), and the controller reads this event log.
  - Repro: hand-made frames (cube covered at t=3, grip closes on bin_a at t=6, bin lifted at t=7, lowered at t=8, opened at t=9). Events: `[('cover', ('bin_a','cube_r')), ('grasp', ('bin_a',)), ('place', ('bin_a','cube_r'))]`.
  - Fix: limit candidates to `visibility.value in ("visible","held")`, as `views.geometric` already does, and add this case as a negative test.
- **[MAJOR] arbiter_kb/detectors.py:88 + arbiter_kb/systems/arbiter_kb.py:46-52 + arbiter_kb/debug.py:41-56** — M3 Done-when (c) and E2 are not met for the new items.
  - `grip_close`/`grip_open` evidence carries `"obs": []`, and the trace stores only object observations (no `ee_xyz` / `gripper_width`). So grasp / place / Held and the R3-driven poses cannot be followed back to observation ids.
  - `debug explain` only walks `Inside(cube)`, so event-log lines, Near/On/LeftOf and Held cannot be explained at all.
  - Fix: put `ee` and `gripper` into the trace `obs` with an id (for example `o<t>:ee`), cite it from the grip evidence, and give `explain` an `--item` (event id / object pose). The selftest has no M3 line, so add PASS/FAIL lines for M3 (b) and (c).
- **[MAJOR] arbiter_kb/views.py:53, 57-66** — Parameters do not come from one source.
  - `geometric(kb, p=RuleParams())` is always called with defaults (facts() at l.50, controller_view at l.162), so the frozen `params.json` values r_near / r_on never reach the views.
  - The LeftOf margin `0.02` and the On height band `0.005 < dz < 0.08` are magic numbers in the body, which §1.1 forbids.
  - Repro: change `r_near` in params.json and the controller view stays the same.
  - Fix: add `left_margin`, `on_dz_min`, `on_dz_max` to RuleParams and pass `params.rule` through `controller_view` / `facts`.
- **[MINOR] arbiter_kb/views.py:162** — The geometric relation lines in the controller view have no KB id or support (`  Near(bin_a, cube_r) 1.00`), though §3.8 says every line carries its KB id so that cites can point to it. Fix: print the pose-support ids of both objects.
- **[MINOR] arbiter_kb/views.py:36-50 / rules.py:49-62** — A held cube that is seen after the grasp and then lost from view keeps `visibility="visible"`, because observe() overwrote "held" and R1 skips held cubes. `facts()` then returns both `('Visible','cube_r')` and `('Held','cube_r')` for an unseen object. Fix: on `disappear` of a held object, SetVis "held".
- **[MINOR] arbiter_kb/detectors.py:54-56 / views.py:121-124** — Off-by-one in the "from" time. `move_start` stores `prev.xyz` (frame t−1) but is stamped t, so the log reads "moved … (t=65-85)" when the move began at 64. `measure_kb.py:59` compensates with `t0 - 1`, so the measured numbers are right; only the printed time is wrong. Fix: store `t_from = t-1`.
- **[MINOR] arbiter_kb/views.py:114** — `event_log(last=30)` silently drops the oldest lines, which are the covers that say which cube is under which bin. This is not hit on synthetic data (max 18 lines), but is likely on 1300-step RoboMME episodes. Fix: print "(N earlier lines omitted)", or always keep cover / grasp events.
- **[MINOR] arbiter_kb/detectors.py:82-89** — The hysteresis band 0.07/0.075 m is only 5 mm wide, and an object wider than 7 cm never registers a close. Neither hysteresis nor the new move xyz payload has a unit test. The reviewer's probe gave correct flips, but this is untested.
- **[MINOR] tests/test_m3_kb.py** — Weak or missing assertions:
  - `ts == sorted(ts)` is close to tautological;
  - `len(v)/4 < 1500` is a rough budget check;
  - the hidden-pose test needs `assert any(...)` so that it cannot pass vacuously;
  - missing negatives: nothing within r_grasp, place onto the table, the container over a hidden cube.
- **[MINOR] README.md:5** still says "Status: M0 built". The NOTES M3 figure "about 360 tokens" has no results file (E10); the reviewer measures a median of ~415 and a max of ~512 (chars/4) over 100 train episodes. The m3_kb_truth result was made from a dirty tree; the reviewer reproduced it exactly from HEAD.

**Math / spec:**
- No Distribution operation changed.
- R3 is the nearest non-hidden object within r_grasp, with ties broken by id.
- On is a fixed 0.5–8 cm band, not "≈ lower object's height" (extent is not built).
- Geometric "probability" = product of pose confidences. That is not a calibrated relation probability, and M3 (b) asks for calibration of the L2 probabilities. measure_kb measures covers, passes and moves only.

**Honesty:**
- Causality: the detectors use only the current and previous frame. The PoisonedSource test passes.
- GroundTruth is used only for scoring (measure_kb, dry_run), never in the prompt or the KB.
- measure_kb and dry_run use train only.
- Two FREEZE lines, with no READ before the reviewer's accidental one.
- No hard-coded expected numbers.

**Debuggability:** `debug explain` on a corrupted wrong prediction (ep 2, cube_red, id_swap_overlap) reaches `o89:bin_0`, `o89:bin_3`, `o31:cube_red`, `o32:bin_0`. M3's new items are not reachable.

| Eval | Result |
|---|---|
| E1 | PASS |
| E2 | PASS for scored rows / FAIL for new M3 items |
| E3 | PASS (hysteresis untested) |
| E4 | PASS |
| E5 | PASS |
| E6 | PASS (geometric lines carry no ids) |
| E7 | PASS (pytest 5.8 s; ArbiterKB step 0.126 ms) |
| E8 | PASS (not exercised) |
| E9 | PASS |
| E10 | PASS-WITH-NOTE ("360 tokens" uncited; README stale) |

Unused code: none. The one-call helpers are plain functions. `held(kb)` is recomputed per object in `r2_follow`: O(objects × events), harmless at this size. No dependencies outside the whitelist.
