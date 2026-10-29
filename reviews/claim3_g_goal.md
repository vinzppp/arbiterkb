# Goal review — claim 3 on G (e5b30fe^..f489a23) — 2026-10-29
Verdict: ON-TRACK-WITH-CHANGES (the process is clean; the NOTES summary overclaims and must be rewritten)

**Sharp objection.** The KB's AUROC lead over SoftHaircut (+0.10 to +0.11) is not swap detection. The chosen d0 = 0.02 m gives R6 w = 0.5·e^(−0.05/0.02) ≈ 0.04 at the 5–9 cm passes where G's real swaps happen, so R6 is close to inert for real swaps. On the 33 tracker-swapped test cubes the KB's answer is right only 0.09–0.12 of the time (K = KB). The lead must come from R1's cover kernel (σ 0.02 inside r_cover 0.06), the eps_far = 0.5 "cover not seen" floor, and R5. Those are **cover-time ambiguity features that SoftHaircut is never given**. GOALS asks for a comparison "on the same evidence feature", and this one is not. The comparison is between a cover-distance softmax and a point mass.

## Claim movement
| Item | Claim / falsifier | Evidence | Strength | Verdict |
|---|---|---|---|---|
| Order | process | e5b30fe rule 13:57:15Z → c865523 grid (run at e5b30fe, clean) + choice.json 13:58:22Z → ledger READ 13:58:23Z → read.json e91ee66 → LLM READ 14:00:11Z → f489a23. The choice was frozen before the read, and test was read once per question | — | sound. Caveat: the 68 s from rule commit to grid commit suggests that code was exercised before registration. That is fine only if nobody looked at train AURC before committing |
| Main reading | claim 3 | KB − Soft (min_q) AURC −0.016 [−0.088, +0.019] | none | **failed, as registered: "not supported on real perception"** |
| Secondary | claim 3 | min_3d AURC −0.031 [−0.074, −0.004]; AUROC +0.10 [+0.03, +0.19] | weak | the upper CI is near 0, it is one of two tests, d0 sits at the grid edge, and the mechanism is unattributed (see above) |
| Margin | new evidence | KB min_q − min_3d +0.009 [+0.002, +0.020]; q0 at the grid edge | negative | sound: drop the margin |
| LLM rows | falsifier 3 (raw log = KB) | K − R +0.31 off; **+0.067 [−0.006, +0.143] default** with R at 1.65k vs K at 2.5k prompt tokens | the falsifier is **nearly triggered** out of sample with thinking | NOTES does not name falsifier 3 here. It should |
| LLM rows | claims 1–2 | K = KB argmax (0.809) | none | QA, not control |

**(2) Honest summary.** "Claim 3 is partly supported" reads the secondary as a rescue. What may be claimed:
- "Registered main test: not supported. The margin evidence overfitted.
- Secondary (min_3d): the KB's p ranks its own errors better than a swap-score haircut does (AURC −0.031 [−0.074, −0.004]; AUROC 0.74 vs 0.64), at Bound's accuracy. The source is unattributed and is probably cover-time ambiguity, not swap detection.
- The KB does not catch tracker swaps (0.09–0.12 right on swapped cubes).
- G, privileged word list, QA only."

Silent failure at matched clean coverage, which is GOALS' headline measure, is not reported.

## Risks of self-deception
1. **Lead from an unshared feature** (above). Cheapest control (no LLM, minutes): register on G train, then on test, labelled post-hoc attribution:
   - (a) KB with w_max = 0;
   - (b) KB with σ_cover → 0 and eps_far = eps_none;
   - (c) KB with R5 off;
   - (d) **CoverHaircut**: R1's distribution at the cover, carried forward, with no KB, no R5 and no R6.

   If (a) ≈ KB and (d) ≈ KB, the "belief layer" is one cover feature. Falsifier 6 then applies to calibration as well as to detection.
2. **The swapped-vs-clean AUROC was asked for and not reported.** `claim3_g.py` collects `swapped` per row and never scores it. NOTES' "its p does not drop enough there" is an assertion, not a number. Fix: report the AUROC of the KB's p for swapped vs clean, and the KB's p on swapped cubes. This is free from the same rows.
3. **Baselines the HANDOFF required are missing from the read:** the stock/stress bit and "near pass since the cover". Add them to the same table.
4. **G test is now spent for claim 3.** Any further tuning on it (d0, σ) is leakage. A confirmatory claim needs fresh episodes.

## Missing baseline(s)
CoverHaircut (d); the setting bit; the near-pass heuristic. All three use the same rows, so the cost is nil.

## Cut or defer
- The `min_q` / `q0` path and `track_q` from the margin: keep the code but default to off, and add nothing to it.
- The ArbiterKB(min_q) rows.
- No more R6 variants until (a) shows that R6 contributes anything.

## (4) LLM rows: meaning
- The LLM adds nothing over the KB's argmax, out of sample too. Claims 1–2 cannot move outside the closed loop.
- R beats K on swapped cubes (0.33–0.36 vs 0.09–0.12, n = 33, CIs overlap only slightly). So the KB's text erases the positional evidence a model can use to undo a tracker swap.
- Design implication, worth one arm in the closed loop, developed on train only: **K plus the raw positions in a window around each near pass, whenever the KB's p for an affected cube falls below τ.** This is the concrete form of claim 3's "uncertain → look first". It only helps if p actually drops on swapped cubes, which is why risk 2 has to be checked first.

## Recommendation for the next phase
**Modify.** Before the closed loop, spend half an hour on risks 1–3 (no LLM, $0): the four ablations plus CoverHaircut and the swapped-vs-clean AUROC, registered first, train then test, with a ledger line. This decides what claim 3 is: belief layer, or a cover feature.

Then rewrite NOTES' "Honest summary" as in (2). Update HANDOFF "Latest results" as follows: claim 3's main test failed; the KB's calibration lead is unattributed; G test is spent; falsifier 3 is nearly triggered with thinking. HANDOFF Next 1 becomes the ablation, and Next 2 (the closed loop H / R / K, plus the K+raw-window arm) stays the only route to claims 1–2. Add a line to DEVELOPMENT M2's "Done when": "(c) met: ArbiterKB does not beat SoftHaircut on the registered feature; a calibrated cover feature may suffice (pending ablation)". Also add the requirement that any confirmatory claim-3 number comes from fresh episodes.
