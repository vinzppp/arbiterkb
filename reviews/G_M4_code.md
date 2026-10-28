# Code review — G_M4 (G: stress/, swapstats, measure_swaps, MeasuredSwapSource / id_swap_measured; M4: llm.py, controller.py, annotate/som/clue_objects/task_items, dry_run, dry_report, Batch A; diff 3188325..89aa725) — 2026-10-28

Verdict: **PASS-WITH-NOTES**

make check: exit 0, 1:24 wall time. `148 passed in 11.18s`; `quality: 70 files, 0 problems — PASS`; `selftest: 16 PASS, 0 FAIL, 1 SKIPPED` (the skipped item is (g) real oracle replays: "no replays in …/data/replays"). The selftest has **no line for G, id_swap_measured or M4**.

Reproducibility: **yes**.
- `scripts/measure_swaps.py --out <scratch>`, run twice: byte-identical outputs, and both equal the committed files. `g_passes.csv` sha1 16c35ecb…, `g_summary.json` d4b20851…, `g_signature.json` f53e2fa8….
- `run.py --split train --systems m2 --corrupt none,id_swap_measured --lane mix --steps mix --first 150 --n 300`, run twice: `summary.csv` sha1 774db7dc… both times, and `rows.csv` 386cff75… both times.
- The selftest's own hash check also passed (1b211fc01474 twice).

## Findings (most severe first)

- **[MAJOR] arbiter_kb/sources/realistic.py:135-139 vs arbiter_kb/swapstats.py:79-83 — id_swap_measured always fakes a full two-way label exchange, but 41% of G's "swaps" are one-sided.**
  - `_outcome` marks a pass as swapped when `end[sa]==b OR end[sb]==a`.
  - Re-running the matching on `data/g_perceived`: of 130 swaps, only 77 are true exchanges. In 30, both slots end on the same container (a duplicated track). In 23, the partner track is lost.
  - `MeasuredSwapSource` turns every drawn swap into a clean permutation of labels. So the swap rate is G's, but the swap's shape is not, and §3.11 requires the shape to match.
  - Repro: classification over `data/g_perceived` with `swapstats.identities` / `_slot_on` gives `{'clean': 223, 'exchange': 77, 'one_sided': 53 (30 same-container, 23 partner lost), 'skipped': 51}`.
  - A hand-made merge probe (slot a copied onto b from t_min) gives `swapped 1 lost 0`, and no test covers this case.
  - Fix: record the outcome type (exchange / merge / lost) in g_passes.csv and g_signature.json; make MeasuredSwapSource draw the type at G's rates; add a test for the merge case.

- **[MAJOR] arbiter_kb/controller.py:163-170 (`_citable`) — any made-up observation id passes the cite check.**
  - `head[:1]=="o" and head[1:].isdigit() and (obj in known or obj in ("ee","image"))` accepts any frame number, including frames that never existed or are in the future, and `o<t>:image` on text-only arms where no image was shown.
  - Repro: in a dry run with a fake transport, an answer citing `"o99999:bin_0"` gives `problems: ""`.
  - E6 and M4 Done-when (a) ("an invented cite") are therefore only half-checked.
  - Fix: pass in the set of observation ids that actually appear in the view or trace up to t*, and reject anything else. Add a negative test with a future t.

- **[MAJOR] scripts/dry_run.py:109-112, 131 + arbiter_kb/llm.py:93-95 — a cached re-run gives different numbers, and spend is capped per call instead of per run.**
  - On a cache hit, `usage` adds nothing to tokens, seconds or $, so a replayed dry run writes `prompt_tokens 0, usd 0, seconds 0` in dry.csv; `dry_report` then prints wrong `in_tok` / `s_per_call`. dry.csv has no hits / misses column.
  - Repro: two dry_run passes with a fake transport; the second, from cache, wrote `prompt_tokens [0,0,0,0,1000,1000]`. Fails M4 Done-when (a) and E8.
  - `ask()` builds a new `LLM` per job, so `ARBITER_MAX_USD` never adds up across calls. With a $0.25 cap and $0.10 per call, the transport was called 4 times ($0.40) before `SystemExit`; the paid rows are lost because the exit happens before dry.csv is written. With `--workers N` the overspend grows to about N calls.
  - Fix: return cached `usage` on a hit and count it separately; one locked budget counter across threads; write rows as they arrive; test that a cache replay gives an identical dry.csv apart from a hit flag.

- **[MAJOR] scripts/dry_run.py:164-173, results/latest/m4_dry_text/meta.json, scripts/measure_swaps.py:77-81 — E4 provenance is missing for the new results.**
  - dry_run writes no meta.json (commit, args, params hash, contract version, cache counts). The committed `m4_dry_text/meta.json` was hand-written (`"code_commit": "f6ff02d (dirty …)"`).
  - `g_summary.json` / `g_signature.json` / `g_passes.csv` record no commit, args or input hash.
  - id_swap_measured runs (`run.py:126`) do not record the hash of `results/g_signature.json`.
  - Fix: write meta.json from dry_run and measure_swaps; add `sha256(g_signature.json)` to run meta when id_swap_measured is used.

- **[MAJOR] G Done-when (b) is only half met.** No per-feature histograms for swapped vs clean passes and no ROC curves; only AUC numbers in `g_summary.json`. Fix: have measure_swaps write them to `figures/g_*.png`.

- **[MINOR] arbiter_kb/swapstats.py:144-146 — passes are dropped silently** (51 of 404, 12.6%, probably ones where a track was already lost). Report `n_passes_unmatched` per setting.
- **[MINOR] arbiter_kb/annotate.py:50-59, 68-78 and arbiter_kb/kb.py:205-207 — "no numbers anywhere" leaks.** Numbers in description strings / predicates, subject == object, and tuples / ints / strings in facts all pass. No negative tests.
- **[MINOR] arbiter_kb/clue_objects.py:54-57 — phrase order disagrees with `task_items.py`** (task items first vs containers first, which sg_pilot's detector needs). One shared ordering function plus a test.
- **[MINOR] arbiter_kb/annotate.py:44 — `controller_view(kb)` uses default `RuleParams()`.** Pass `params.rule`.
- **[MINOR] arbiter_kb/controller.py:173 — `target_distance` is never logged by dry_run.** Add `target_dist_m`.
- **[MINOR] arbiter_kb/controller.py:142-143 — booleans pass as numbers / gripper values.** Exclude `bool`.
- **[MINOR] E6: `check()` accepts an empty `cites` list.** Flag `no cites`.
- **[MINOR] arbiter_kb/views.py:202-206 — `citable_ids` includes LeftOf ids the view never prints.**
- **[MINOR] scripts/dry_run.py:70 — K-L3 keeps the `Progress:` line counted from L3 events.** Document or drop.
- **[MINOR] scripts/dry_report.py:61-64 — empty pairing crashes; dry_run / dry_report have no tests** although dry_report carries the pre-registered decision rule.
- **[MINOR] scripts/measure_swaps.py:50 — FALLBACK text says "Claim 1"; since v1.4 it is claim 3.**
- **[MINOR] stress/stress_env.py:83 — IndexError on an empty swap schedule** (pod-only, not reached).
- **[MINOR] stress/render.py:6-8 — wrist RGB stored as 1x1 placeholders**, so G replays cannot feed arm H, `dry_run.images()` or the annotator on G episodes.
- **[MINOR] E10: NOTES:635 cites a git-ignored path; NOTES:634 and NOTES:692 numbers point to no file.**

Checked and fine: rank AUC with ties equals brute force; `logistic_fit` recovers planted weights; per-setting calibration matches NOTES:632; `p_swap` clamping and units OK; gate branches OK; causality OK; no new test-split reads.

## E1–E10
| Eval | Result | Evidence |
|---|---|---|
| E1 Minimal | PASS (notes) | quality 0 problems; `stress/` pod-only imports allowed by §5.3 |
| E2 Debuggable | PASS for KB rows; dry-run rows have no trace | `debug explain … --ep 151 --cube cube_blue --t 113` reaches `o71:bin_0`, `o76:bin_3`, `o31:cube_blue`, `o32:bin_0` |
| E3 Correct by construction | PASS (notes) | I5 test misses tuples, ints, strings |
| E4 Reproducible | PASS on hashes; FAIL on provenance | see above |
| E5 Honest evaluation | PASS | last test_reads entry FREEZE 2941773a1dd4 |
| E6 Explainable outputs | PARTIAL | invented `o<t>:` ids and empty cites pass |
| E7 Fast feedback | PASS | pytest 11.2 s; median step 0.142 ms |
| E8 Cost-bounded | PARTIAL | run cap overshoots; cache hits report 0 tokens |
| E9 Stable interfaces | PASS | test_contracts passes |
| E10 Measured, not claimed | PARTIAL | dry-run and G tables re-derive; three NOTES numbers lack a file |

Phase Done-when: **G** (a), (c) met, (b) missing figures. **M4** (a) partly; (b) synthetic text only; (c) not seen. harness.py and reach.py not built.

## Unused code / one-call abstractions
- Called only by tests: `annotate.*`, `som.*`, `clue_objects.*`, `controller.target_distance`, `ArbiterParams.p_clear/p_ambiguous/p_unlikely`.
- No tests: `scripts/dry_run.py`, `scripts/dry_report.py`.
- One call site: `annotate.strength_p`, swapstats private helpers (acceptable), `MeasuredSwapSource._plan`, `controller.object_table/one_sentence/raw_tracker_log`, `views.citable_ids`.
- Duplicated: `swapstats.oracle_passes` vs `realistic.passes` (different interval ends); `task_items.open_vocabulary` vs `clue_objects.phrases` (opposite order).
- Dependencies: none outside the whitelist.
