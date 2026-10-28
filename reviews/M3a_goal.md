# Goal review — M3a (M3, first part) — 2026-10-28
Verdict: ON-TRACK-WITH-CHANGES

## Claim movement
- **Claims 1–2 (smaller model / fewer tokens):** none. M3a builds the text that arm K will read. No controller has read it yet, and none of the new layers (R3 grasp/place, Held/On/Near/LeftOf, L4 progress) has a truth check: synthetic never manipulates, and no replays are on the Mac. The "~360 tokens, budget 1,500" result is true by construction. A text summary is always far smaller than 148 images, so token savings will never be the hard part of claim 2. Non-inferior success is the hard part.
- **Claim 3:** none. `measure_kb.py` (train 0–99) gives cover P/R 1.00/1.00 and move error 0.0 m on clean input. Observations equal the truth there, and R1 encodes the generator's own cover rule. NOTES already says this is plumbing.
- **The one real finding (useful, negative):** under 15 mm iid noise, 38 moves are logged instead of 380 and near-pass recall is 0.09. The event log, which is the thing K−L3 is meant to test, is only as good as a 1 mm/frame speed threshold. That is weak evidence against the KB's robustness, and NOTES reports it honestly.
- **M3 done-when (b) is not met:** there is no per-type P/R for relations, no L2 calibration and no oracle-replay check. (c) "every line resolves via `debug explain`" is not tested. Do not tag M3 as done.

## Risks of self-deception
1. **Arm K confounds memory with free 3D coordinates (the sharp objection).** The view gives world-frame centres from depth. Arm H gets RGB only, with no depth, as in 2610.17350 (NOTES M4 already says this). For a waypoint-emitting LLM, pixel-to-world localisation is probably the hardest part of the job. So "K beats H with a small model" could come entirely from handing over coordinates, with no memory involved. Arm R does not isolate this, because it also carries the full history. → **Cheapest control:** add arm **K0**: the current frame plus a current-frame object table (id, class, 3D centre), with no relations, no event log and no history. If K0 ≈ K, the claim is "give the LLM 3D coordinates", not "give it memory".
2. **The L2 "probabilities" are not relation probabilities.** p = the product of the two pose confidences, which is about 1.0 for visible objects. The controller will read `Near(a,b) 1.00` as calibrated. Either label it "pose conf", or report it as uncalibrated until (b) measures it.
3. **The log asserts unproved facts with certainty.** R6 emits a `swap` event with conf 1.0 on every near_pass (w > 0 always). The view then prints "exchanged places", and Progress counts "container exchanges". 6% of clean near_passes do not match a true pass, and under noise 67% do not. → Print "passed close", not "exchanged places", unless the place exchange is measured.
4. **The pass truth is defined with the detector's own `v_move`.** `passes(true_frames, v_move)` shares the threshold being evaluated, so detector and truth fail together. Define truth from the generator's swap schedule.

## Missing baseline(s)
- **Bound-one-liner arm:** current frame plus one sentence, "the target cube is under <argmax Inside>". On VideoUnmaskSwap, ArbiterKB's answer equalled Bound's on every M1 test row. If this arm matches K on permanence, the KB's structure adds nothing that the bound rule does not. That is a stronger and cheaper falsifier than arm R.
- **K0** (above). Add both to the §3.8 arms table.

## Cut or defer
- `LeftOf`: computed but filtered out of the controller view, and nothing scores it. Defer it until a task needs it.
- L4 "container exchanges" and "covers" counts: no goal or metric reads them. Wait for task-template parsing.
- `g_close`/`g_open` hysteresis and `r_grasp`: parameters with no data that exercises them. Do not freeze meaning into them before replays exist.
- README still says "M0 built". Update it or leave it, but do not let it drift into claims.

## Recommendation for the next phase
**Modify.** Before porting the closed-loop harness (the expensive M4/M5 path), run the small feasibility probe that the user skipped. It costs a few dollars and needs the user's go. On ~20 train episodes at the decision point, the small model gets arms {current frame only, K0, Bound-one-liner, K}, scored on whether `target` is the true container. If K0 or Bound-one-liner ties K, the permanence suite cannot show claims 1–2 for *memory*. The project should then move straight to a suite where history matters beyond one argmax (counting / imitation, which needs L3 working under real noise). In that case, fixing the motion detector on G's measured tracker noise comes before any controller spend. Also budget n: H with a frontier model scores 97% on permanence, so showing non-inferiority on 50 test episodes gives a Wilson interval of about ±6 pp. Decide now what margin counts as "matches".
