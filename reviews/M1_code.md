# Code review — M1 — 2026-10-28
Verdict: PASS-WITH-NOTES
make check: PASS — pytest 63 passed in 3.45 s; quality "48 files, 0 problems — PASS"; selftest "12 PASS, 0 FAIL, 1 SKIPPED" ((g) real replays: no data). Wall time 28.7 s.
Reproducibility: same hash? y — I ran the M1 table command twice (`python3 scripts/run.py --systems all --corrupt none,id_swap_overlap,pos_noise --p 0.5 --lane mix --steps mix --n 60`). Both runs gave summary.csv sha256 e1568c20…3a68e, and the file is byte-identical to the committed results/latest/m1_table/summary.csv. Selftest M0 (f) also matches: cf778c860f2c twice.

## Findings (most severe first)

- [MAJOR] arbiter_kb/debug.py:58-67 and scripts/selftest.py:107 — The E2 "explain reaches an observation for every scored row: 1080 rows" check is a tautology for 5 of the 6 systems (900 of 1080 rows). `_explain_tracker` always prints the literal "rests on observation o{t}:{pred}" and never looks up the observation in the trace (GuardedBound rows have no `obs` field). The selftest only greps for the word "observation" and does not check for "NOT FOUND" (the pytest at tests/test_debug.py does). Reproduce: `python3 -m arbiter_kb.debug explain --run results/runs/synthetic_test_4e45e899 --condition none --system bound --ep 1050 --cube cube_nonexistent --t 143` prints "cube_nonexistent -> None … rests on observation o143:None" and the check passes. Fix: make the tracker explanation look up `obs[t][pred]` in its own trace rows (store `obs` in TrackerSystem/GuardedBound rows) and print NOT FOUND if it is missing; add the `"NOT FOUND" not in text` condition to selftest.py:107. Narrow the NOTES claim ("all 1080 rows, all six systems") until this is done.

- [MAJOR] arbiter_kb/rules.py:51-52 and arbiter_kb/systems/baselines.py (GuardedBound._triggers) — The two numbers that matter in M1 have no test that fails when they are wrong.
  - (1) The R6 v1.5 weight w = w_max·exp(−min_3d/d0) has no test. tests/test_rules.py:62-77 builds near_pass evidence without `min_3d`, so it only exercises the M0 fallback branch, which is now dead in production: Detectors always sets min_3d (detectors.py:64-71).
  - (2) The OverlapHaircut d_ov gate has no test, and neither do BoundHaircut, JumpDetector or the stale-doubt fix. No test file covers GuardedBound's behaviour; it only runs inside the E2 test.
  - Mutation evidence:
    - Replacing the v1.5 formula with the M0 closeness formula: pytest still 63 passed.
    - Replacing `e.data.get("min_3d",1e9) < b.d_ov` with `True` (so OverlapHaircut becomes BoundHaircut): pytest still 63 passed, and the selftest's "OverlapHaircut silent on corrupted < Bound's" would also still pass.
  - Fix: add tests/test_rules.py::test_r6_v15_weight (an evidence with min_3d = 0.02 gives w = 0.5·e⁻¹ ≈ 0.1839; min_3d = 0.12 gives ≈ 0.0012), a detector test that min_3d equals the true closest 3D distance, and tests/test_baselines.py on hand-made frames: haircut fires on any pass involving the cube's container; overlap_haircut does not fire for min_3d = 0.05 and does for 0.02; jump fires for a 0.04 m label step and not for 0.02; doubt is dropped once pred leaves the pair.

- [MAJOR] DEVELOPMENT.md §6 M1 "Done when" (d) — Half of this criterion is not done: "plus one synthetic scenario where R5 beats OverlapHaircut" exists nowhere (grep for R5/joint_exclusive in tests, scripts and notebooks finds only the three unit tests in test_rules.py:113-140). R5 emitted 0 ops in the table run (selftest INFO). NOTES explains why R5 is inert on VideoUnmaskSwap, but does not record that criterion (d) is unmet. Fix: add the scenario (e.g. two cubes covered by two adjacent containers within r_cover, so the cover marginals are ambiguous and one is pinned) as a test comparing Brier against GuardedBound("overlap_haircut"), or record in NOTES that (d)-part-2 was waived, with the user's sign-off.

- [MINOR] arbiter_kb/beliefs.py:107 + baselines.py:136 — When the haircut is 50/50 (c = 0.5), the tie is broken by the larger label string, not by the tracker's prediction. So the accuracy and grasp_err of BoundHaircut / OverlapHaircut are a label-ordering artifact. Example: OverlapHaircut is wrong on uncorrupted episodes 1006, 1007, 1050 and 1052 under id_swap_overlap (rows.csv of the run above). NOTES calls this a "coin toss"; it is deterministic. Brier and silent rate are not affected. Fix: break ties toward the tracker's pred (e.g. use `c` for pred and `1−c−1e-9` for the partner), or state the tie rule in NOTES.

- [MINOR] arbiter_kb/sources/realistic.py:114 — NoisySource writes `self.log[ep]` after the last frame. This is the same bug NOTES says was fixed for OverlapSwapSource. It is harmless only because the value is always False (run.py reads it with `.get`). Fix: write the log before the loop.

- [MINOR] tests/test_evaluate.py:27-28 — The PoisonedSource causality test does not include the three new GuardedBound systems. GuardedBound also has no `.t`, so `getattr(s,"t",f.t-1)` would pass trivially for it anyway. By inspection it only uses `self.prev` (frame t−1), so this is not a causality violation, but it is unchecked. Fix: set `self.t` in GuardedBound.step and add the three systems to `systems()`.

- [MINOR] arbiter_kb/rules.py:74-86 vs DEVELOPMENT.md:167 — The spec says R5 runs "after any change to an Inside"; the code runs it only on cube-cover frames. The reason is recorded in NOTES and backed by test_joint_exclusive_would_overconfide_after_a_mix, but §3.5 was not updated. Also, on a cover frame R5 joint-normalises over ALL hidden cubes, including ones already correlated by earlier R6 mixes. That is the same overconfidence the docstring warns about, once covers are staggered (not the case in synthetic data, where all covers happen at t = 32). Fix: update §3.5 and restrict the joint to the newly covered cube plus cubes whose distributions have not been mixed.

- [MINOR] arbiter_kb/beliefs.py:139 — If no assignment is feasible (e.g. two cubes both certain of the same bin, or 3 cubes over 2 bins with no None), joint_exclusive silently returns its input unchanged. Checked: `{"x":{b:1},"y":{b:1}}` comes back unchanged. That is an R8-style contradiction thrown away without evidence. Fix: return a flag or log it, so M3's R8 can turn it into a risk.

- [MINOR] arbiter_kb/beliefs.py:55-59 — The `None` hypothesis created by eps_none has empty hyp_support (`{'a': ['e1'], None: []}`), and the `none_support` argument is ignored whenever eps_none > 0. I4 allows this (kb.py:185 only needs some support on the distribution), but if the argmax is None the explanation shows no evidence for it. Fix: `hyp_support[None] = none_support`.

- [MINOR] Makefile `selftest` — `make check` appends 17 READ lines to the tracked results/test_reads.log on every run, so the tree is always dirty afterwards (`M results/test_reads.log`). A builder who runs `make check` and then `--keep` gets `dirty=true` unless they commit the ledger first. Fix: document it in NOTES, or have the selftest's repeat runs use n = 10 train seeds for the hash check.

- [MINOR] NOTES.md M1 — These numbers have no `(results/<file>)` pointer (E10): "min_3d = 0.020 m … final p_max 0.808" (only in the uncommitted runs/…/rows.csv), "R5 … emitted 0 ops" (selftest stdout only), "label jumps median 0.004 m, max 0.023 m" (in meta.json corruption_log, but not cited).

## E1–E10
| Eval | Result | Evidence |
|---|---|---|
| E1 Minimal | PASS | quality.py "48 files, 0 problems"; imports ⊆ whitelist (stdlib + numpy, matplotlib, nbformat, sibling scripts). Dead branch: rules.py:53-54 (M0 closeness fallback; `w_mix` now reachable only from there). |
| E2 Debuggable | PASS-WITH-NOTE | ArbiterKB chain verified by hand: ep 1046 cube_red (wrong, p 0.808) → MixDist R6 → e14 near_pass(min_3d 0.02) → o89:bin_0/o89:bin_2, and e9 disappear → o31:cube_red, o32:bin_2. Tracker-baseline explanations are hard-coded text (MAJOR 1). |
| E3 Correct by construction | PASS | KB.check after each apply in tests (ARBITER_DEBUG=1); R5 has 3 unit tests. My checks: joint_exclusive matches brute force for N = 3, 200 random cases (max err 3e-16); empty input → {}; all mass on None kept; swap_mix w = 0 / 1 exact; flatten f = 1 (Δt = 0) is a no-op; from_distances with no candidates → {None: 1}. |
| E4 Reproducible | PASS | identical hash over 2 runs and vs the committed table; meta.json has commit 3d5751a, dirty False, params_hash 41d4718f2eb1, contract_version, llm_cache. |
| E5 Honest evaluation | PASS | (a) PoisonedSource test passes (GuardedBound not covered, MINOR). (b) No system or rule references ground_truth (grep). (c) FREEZE 41d4718f2eb1 at 08:53:14Z comes before the first M1 READ at 08:53:38Z; no test reads between the M0 reads (08:31:54Z) and the M1 FREEZE; params.json values = spec defaults. (d) All 6 systems go through evaluate.run / score. The corruptor looks ahead (realistic.py:66), but only on the input side. |
| E6 Explainable outputs | PASS | ArbiterKB plans carry support (test_systems.py:25); baseline haircut distributions carry the tracker's support. |
| E7 Fast feedback | PASS | pytest 3.45 s; make check 28.7 s; median step 0.112 ms. |
| E8 Cost-bounded | PASS (N/A) | no LLM; llm_cache {hits: 0, misses: 0} recorded. |
| E9 Stable interfaces | PASS | no types.py change; BaselineParams added to Params only; test_contracts passes. |
| E10 Measured | PASS-WITH-NOTE | the table points to results/latest/m1_table/summary.csv; three side numbers have no pointer (MINOR). |

Math against spec:
- R6 v1.5 (rules.py:51-52) matches §3.5 M1: w(0.02) = 0.1839 and w(0.12) = 0.0012, consistent with the NOTES values.
- joint_exclusive matches §3.3 (injective over containers, None may repeat).
- eps_none renormalises correctly (beliefs.py:55-58).
- Detector min_3d is the true minimum over the open pass (detectors.py:64-67).
- The corruption swaps the labels currently tracking the two physical containers (realistic.py:76-80); the selftest measured label jumps of median 0.0044 m and max 0.0232 m, below d_ov.

## Unused code / one-call abstractions
- rules.py:53-54: the M0 image-closeness fallback in R6 is never reached, because min_3d is always present. `RuleParams.w_mix` survives only for it, and test_rules.py:62-77 tests only this dead path.
- beliefs.py `joint_exclusive(max_states=5000)` guard: unreachable at N ≤ 3 cubes × ≤ 5 keys (125 states); untested.
- realistic.py `u01`, `_moved`, `passes`: one call site each (all inside OverlapSwapSource). Acceptable as named helpers.
- `NoisySource.log`: exists only so run.py's `getattr(src,"log",{})` has something to read; always {"corrupted": False}.
- `GuardedBound._nearest_other` fallback `else k` (pairs a container with itself): unreachable with ≥ 2 containers.
- `BaselineParams`: one consumer (GuardedBound); justified as a params.json section.
