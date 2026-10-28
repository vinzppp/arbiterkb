# Code review — M0 — 2026-10-28
Verdict: BLOCK. There is one math bug in `Distribution.from_distances`. It is latent at the M0 defaults, so no M0 number changes, but it breaks the §3.3 formula at σ values M1 will tune over. The fix is about two lines. Everything else is PASS-WITH-NOTES.

make check: PASS. pytest 39 passed in 1.08 s. `quality: 42 files, 0 problems — PASS`. `selftest: 9 PASS, 0 FAIL, 1 SKIPPED` ((g): no replays in data/replays). Total wall time 5.6 s.

Reproducibility: y. I ran the default command `scripts/run.py --source synthetic --corrupt none,id_swap --p 0.5 --split test --n 30` twice, in separate processes with PYTHONHASHSEED=1 and 2. Both runs gave summary.csv sha256 `638e481e…`, which equals the committed `results/m0_summary_idswap_p0.5.csv`. The selftest's p=1 run gives `2704e58c…`, which equals the committed `results/m0_summary_idswap_p1.0.csv`.

## Findings (most severe first)

- **[BLOCK] arbiter_kb/beliefs.py:51 together with beliefs.py:110-111** — `from_distances` takes `exp(-d²/2σ²)` without normalising first. `_normalise` then drops every score ≤ 1e-12 on that raw scale. When d/σ > 7.4, a valid cover candidate (inside `r_cover`) gets zero mass, and the result can collapse to `{None: 1}`. That breaks §3.3 "p ∝ exp(−d²/2σ²)".
  - Repro: one container 4 cm from the vanished cube, `RuleParams(sigma_cover=0.005)` (r_cover stays 0.06). `r1_cover` returns `{None: 1.0}`, yet still sets visibility to `hidden`. With σ=0.02 it returns `{bin_0: 1.0}`.
  - Impact today: none. At the M0 defaults the smallest weight is exp(−4.5).
  - Why it matters: M1 `tune.py` searches σ_cover, and the bug would silently turn every cover into "elsewhere".
  - Fix: compute `exp(-(d² − min d²)/2σ²)` (log-sum-exp shift), or normalise before thresholding. Add a test with d/σ = 8.
- **[MAJOR] arbiter_kb/evaluate.py:50, params.py:48, scripts/run.py:55** — E5c is only nominal.
  - `run()` checks the in-memory `Params.frozen`, and that field defaults to `True`. Any freshly built `Params()`, including one tuned on test, therefore passes.
  - No canonical `params.json` is read, and there is no "written before first test read" timestamp check.
  - run.py writes `params.json` into the run dir it is about to fill, which proves nothing.
  - There is no leakage in M0 because nothing is tuned. M1 is where it starts to matter.
  - Fix: default `frozen=False`. `run(split="test")` should load a top-level `params.json`, require `frozen: true`, and log its mtime and the first test-read time in meta.json.
- **[MAJOR] tests/ (missing test_rules.py and test_detectors.py)** — E3 requires that each rule has a unit test on hand-made frames. None exist. R1, R2, R6 and the detectors are only exercised end-to-end on synthetic episodes. Missing negative cases:
  - R6 must not fire when min_px ≥ d_px.
  - R6's w equals `w_mix·(1−min/d_px)` (check a numeric value).
  - R1 gives `{None:1}` when no container is within r_cover.
  - R1 gives the closest candidate the highest mass when there are two candidates.
  - A near-pass must not open when only one container moves.
  - `test_causality.py` and `test_repro.py` (§4 layout) are folded into test_evaluate.py. That is fine.
- **[MAJOR] arbiter_kb/beliefs.py:65-67 and 81-88** — Retract semantics after a mix are wrong for §3.2 ("loses the mass contributed under that support").
  - `swap_mix` gives both a and b the union of their hyp_support.
  - `drop` then removes every hypothesis whose list contains the id.
  - So retracting the cover evidence of one container, after any mix, wipes the whole distribution.
  - Repro: `D({"a":.9,"b":.1},0,["e1","e2"],{"a":["e1"],"b":["e2"]}).swap_mix("a","b",0.1,1,"e3").drop("e1",2).probs` gives `{None: 1.0}`.
  - This is unused in the M0 pipeline (no Retract op is emitted), but R4 needs it in M5. Track mass per support id, or document the semantics in NOTES.
- [MINOR] arbiter_kb/rules.py:53 — R6 mixes every `Inside` distribution with mass on a or b. §3.5 says "every hidden h". Add the condition `kb.objects[cube].visibility.value == "hidden"`.
- [MINOR] arbiter_kb/rules.py:60-64 — R7 applies λ per call, not λ^Δt (§3.3). It also skips distributions whose argmax is None. `flatten` raises ZeroDivisionError when `plausible` is empty (beliefs.py:72, i.e. zero containers). All of this is inert at λ=1 but must be fixed before M1 sets λ<1.
- [MINOR] arbiter_kb/kb.py:179-195 — `check()` does not assert I4 for Distributions. R1 with no candidate creates `{None:1}` with `support=[]` (beliefs.py:50). Events with conf<0.5 may lack support (kb.py:192), which goes beyond the spec. I6 is only implicit in the dict key.
- [MINOR] arbiter_kb/evaluate.py:57 — `at = {t: c for c, t in gt.t_star.items()}` drops a target when two targets share a t*. This is possible on multi-target replays. Use `t -> [cubes]`.
- [MINOR] arbiter_kb/metrics.py:14 with evaluate.py:41 — when `truth is None`, `brier` scores a None prediction as perfect while `correct=False`. It never happens in the current results (0 rows with truth None), but the definitions are inconsistent.
- [MINOR] arbiter_kb/sources/synthetic.py:128 — `EpisodeMeta.E` has length T, so a system could read t* = T−1 from `meta`. No system reads it (I grepped systems/, rules, detectors, observe, kb, planner, arbiter, views for GroundTruth / t_star / .E: no hits). Document this, or don't hand the full-length E to systems.
- [MINOR] arbiter_kb/arbiter.py:62 — The validator does not enforce "keep ⊆ containers that currently hold mass or are cited" (§3.7). If `cites` is a string, it is iterated character by character (arbiter.py:49-52).
- [MINOR] arbiter_kb/detectors.py:57-71 — A near-pass closes only when one container stops moving, so the evidence lags the closest approach. In episode 1001, `t_min=77` but the evidence is emitted at t=113. This is harmless for R6 in M0, but it matters for M4 trigger timing. Record the delay in NOTES.
- [MINOR] arbiter_kb/sources/corrupt.py:52 and scripts/run.py:58 — `v_move=0.001` is duplicated instead of taken from `DetectParams` (§1 "one source of parameters").
- [MINOR] arbiter_kb/debug.py:3,89 — Three small gaps:
  - The CLI uses `--trace <stem>` instead of §1's `--ep E`, and the docstring example `arbiterkb_1003` does not match the real filenames `<cond>_arbiterkb_<ep>`.
  - Trace `obs` omits `track_q` (§3.9).
  - Baseline traces carry no support ops, so `explain` returns "no Inside(...) belief" for Frozen/Bound rows. E2 says "any number in any table".
- [MINOR] scripts/run.py:97 — Both `make eval` and the selftest overwrite `figures/m0_belief_timeline.png`, so the committed figure depends on which ran last. It is currently modified in the working tree. Put the condition and p in the filename.
- [MINOR] Weak tests:
  - tests/test_systems.py:32-44 calls the rules with `evidence[-3:]` at t=40, where R1 and R6 produce no ops. It would not catch a mutating R1 or R6. Feed a disappear and a near_pass.
  - tests/test_evaluate.py:36-37 (systems have no `gt` attribute) is close to tautological.
  - tests/test_evaluate.py:46-53 is in-process only. I checked the cross-process case by hand.
  - tests/test_contracts.py snapshots only types.py, not Belief / Distribution / Op / Evidence / Event.
- [MINOR] executed/m0_walkthrough.ipynb has 18 cells (§0.6 says ≤ 15). It has 0 errors.
- [MINOR] NOTES.md:9-10, 41 — Some numbers have no results file (E10): the speeds 0.0055–0.0058, the 21–36 px pass distances, the median step of 0.40 ms (selftest now prints 0.100 ms), and "38 passed, 1 skipped" (here: 39 passed, since sg_pilot is present).
- [MINOR] NOTES.md:21 — Deviation 9 says `target_error` is needed, but `sg_compat.target_error` (sg_compat.py:310) has zero callers.

Math checked against §3.3 / §3.10 by running snippets:
- Empty candidates give `{None:1}`.
- Mixing an all-None distribution is a no-op.
- `flatten(1.0)` is the identity (Δt=0 / λ=1).
- Ties break deterministically: (bin_1, bin_2) → bin_2, and a bin beats None.
- Three hidden cubes under one mix stay pairwise consistent.
- swap_mix matches the formula, and w outside [0,1] is clipped by `_normalise` only because negative values are dropped. R6 clips upstream.
- Brier: 0 for a perfect forecast, 0.5 for a uniform forecast over two bins.
- ECE: 0.5 for (1.0, right) plus (1.0, wrong), and 0 for 19/20 right at 0.95.
- Wilson bounds and the summary values match the rows.
- NOTES tables match the CSVs, including Frozen moved 0/14 and back 16/16.

Honesty checks:
- Causality: PoisonedSource would fail if the frames were materialised early. CorruptedSource looks ahead, which is source-side and allowed.
- No hard-coded expected numbers: the selftest compares computed values.
- (g) prints SKIPPED with the reason and the command.
- All three systems go through `evaluate.run`, so the evaluation path is shared.
- The corruption is placed at the same closest approach that R6 mixes on. NOTES already flags this as circular. It is a goal-review issue, not a code defect.

Debuggability:
- Corrupted episode: `python -m arbiter_kb.debug explain --run results/runs/synthetic_test_ebbdc3cc --trace id_swap_arbiterkb_1001 --cube cube_green`.
- The wrong prediction is in the same episode: pred bin_1, physical truth bin_2, p_max 0.983, silent failure.
- The chain is R6 MixDist → e15 / e20 near_pass(bin_1, bin_2) with closeness 0.982, then down to o77:bin_1, o77:bin_2, o127:*. R1's e9 disappear(cube_green) reaches o31:cube_green and o32:bin_1. Observation ids are reached, and the failure is explainable: closeness 0.98 gives w ≈ 0.009. `--t 100` and `timeline` also work.

## E1–E10
| Eval | Result | Evidence |
|---|---|---|
| E1 | PASS (note) | `quality: 42 files, 0 problems`; deps ⊆ whitelist. NOTES has no "≥ 2 call sites" abstraction review (see below). |
| E2 | PASS (note) | explain reaches observation ids (ep 1001 above; tests/test_debug.py:9-17 on 5 train eps; n_support>0 for all 60 ArbiterKB rows). Baseline rows are not explainable; CLI differs from spec. |
| E3 | FAIL (MAJOR) | `ARBITER_DEBUG=1` via conftest.py:5 and Makefile:2, and check() runs after every apply (kb.py:79). But there are no per-rule hand-made-frame unit tests, and check() omits I4 for distributions. |
| E4 | PASS | Cross-process hashes match (`638e481e…` twice, `2704e58c…`). meta.json records commit, args, params_hash, contract_version, llm_cache. |
| E5 | FAIL on (c) (MAJOR) | (a) PASS (test_evaluate.py:12-37); (b) PASS (grep finds no GT access in systems); (c) nominal only (evaluate.py:50, params.py:48); (d) PASS (one `run()` path). |
| E6 | PASS | PlanStep.support = d.support; 0 rows with n_support=0. The `{None:1}` path could give empty support (beliefs.py:50). |
| E7 | PASS | pytest 1.08 s; whole `make check` 5.6 s; median step 0.100 ms. |
| E8 | N/A (M0) | No LLM yet; meta has llm_cache 0/0. |
| E9 | PASS (note) | test_contracts.py snapshots the types.py fields and CONTRACT_VERSION==1; Op/belief dataclasses not covered. |
| E10 | FAIL (MINOR) | The tables point to results/*.csv and match them; several NOTES numbers have no file (NOTES.md:9-10, 41). |

## Unused code / one-call abstractions
- **Unused:**
  - `ObjectFrame.track_quality`, `.extent`, `.stationary` (kb.py:29-32)
  - `KB.task` (kb.py:69)
  - `beliefs.SOURCES` (never enforced)
  - `sg_compat.target_error` (sg_compat.py:310, 0 callers)
  - `Trigger`'s periodic branch has no test.
- **Test-only by spec, acceptable:** `views.ask` / `unify`, `KB.count`, `KB.snapshot`, `arbiter.validate`.
- **Dead at M0 defaults, wired by spec:** `FlattenDist` / `r7_decay` (λ=1), `eps_none`.
- **One call site:**
  - `Detectors` class (systems/arbiter_kb.py:31)
  - `Trigger` (systems/arbiter_kb.py:32)
  - `RuleOnlyBackend` (systems/arbiter_kb.py:27)
  - `skills.pick` (planner.py:23)
  - `find_target_swap` (corrupt.py:72)
  - `frame_to_rows` (baselines.py:45)
  - `physical_truth` (evaluate.py:33)
  - The first three are spec-mandated stubs. NOTES should list them per E1.
- **Dependencies outside the whitelist:** none. `ipykernel` is listed but not imported, which is fine.
