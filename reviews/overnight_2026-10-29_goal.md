# Goal review — overnight 2026-10-29 (9001dd4^..fbd8d0e) — 2026-10-29
Verdict: ON-TRACK-WITH-CHANGES

**Sharp objection.** On Q1 the LLM adds nothing on top of the KB's argmax. The KB's top container is right for 0.83 of cubes (A free check). K with the LLM gets 0.84–0.85. So "K beats the raw log on real perception" means "the cover-follow rules (R1+R2, i.e. Bound, which GOALS lists as not ours) beat an LLM that tracks from a 10th-frame log". It is the counting/RULE finding again. Calibration is weaker than it looks: a 1-bit feature (stock vs stress setting) gives AUROC **0.641** on the same 241 cubes (computed from `perceived_qa_A/summary.json` rows). The model's stated confidence gets 0.675–0.709 and the KB's p gets 0.731.

## Claim movement
| Item | Claim / falsifier | Evidence | Strength | Verdict |
|---|---|---|---|---|
| Count controls | claim 1 on counting | RULE 0.98–1.00, never loses to K; PC ≈ K with thinking | strong (negative: counting is not a KB test) | sound. Caveat: registration and results are in one commit (9001dd4), so the order cannot be checked |
| Perceived QA | falsifier 3 (raw log = KB) | K − R +0.16 to +0.29, CI excludes 0 | moderate vs the raw log; **none vs Bound/argmax** | sound with caveat |
| Relink | input fix | +0.07 [+0.01, +0.14], tuned and evaluated on the same G train; no readings fixed in advance | weak | sound with caveat |
| A / A2 | claim 3, via the LLM | KA − K 0.000; KN − K −0.055 / −0.103, CIs include 0 | none. "Comes from the KB's numbers" is a plausible hypothesis, not a measured result | sound with caveat |
| C / C2 | claim 2 | KT keeps Q1 (0.81) at 0.21× tokens; Q4 0.06 | none (by construction, see below) | unsound as a claim-2 reading |
| B | none yet | done 0.984 / wrong 0.000 | none (plumbing, as NOTES says) | sound |

## Risks of self-deception
1. **Pre-registration cannot be checked for count controls, perceived QA, relink and C2.** Each was registered and run in the same commit. Only A (8dbe079 → f35f63f) and A2 (f35f63f → fbd8d0e) are ordered in git. The perceived-QA heading still says "DRAFT, not yet approved". → Commit each rule before its run, and fix the heading.
2. **G train is used three times.** The occlusion grid picked the best of 9 to pass a 0.90 gate (0.907, narrowly). Relink was tuned on id consistency. A's d = 0.10 was chosen with truth coverage. Every QA row is scored on those same 90 episodes. Truth is physical (the slot on the oracle container at the end), so it is not circular, but it is still in-sample. → One frozen read on G test with arms R and K (~$0.4).
3. **C's Q1 result is true by construction.** The working memory *contains* each cube's most likely container. "Keeps most of Q1 at a fifth of the tokens" just restates argmax 0.83. Questions that need history collapse (Q4 0.06–0.21).
4. **"The token case for KT needs long episodes" is overstated.** At 4.4 tok/item, K is 0.5–1.3k tokens. Claim 2 is measured against 16.5k tokens plus 148 images, so K vs KT is a second-order saving.
5. **"Real perception" overstates it.** G used the simulator's object list (privileged vocab, HANDOFF reminder). It is also one task family, QA rather than control, and the R arm reads a 10th-frame log. → Write "perceived tracks, privileged vocab, G train" wherever the headline appears.
6. **B's wrong-intent control is too easy.** A different action type almost never happens in that window, so 0.000 is near-guaranteed. → If B is kept, the control should be premature completion (declare the next subgoal's action).
7. **AUROC with ties.** 75% of K's confidences are ≥ 0.97, so A/A2's AUROC differences are tie-dominated and noisy.

## Missing baseline(s)
- **KB argmax / Bound text with no LLM** for Q1 (0.83, already computed). Report it as a row next to K and R.
- **1-bit setting feature** (AUROC 0.641) and **"any near pass < d since the cover → low confidence"** for calibration. Also SoftHaircut on the same cubes. Report the KB p AUROC **within stress cubes only**; otherwise the 0.731 is mostly stock-vs-stress.
- R + cover events (the PC analogue for permanence): does the gain need beliefs, or only the cover line?

## Cut or defer
- C/KT (`memtool.py`) and C2: stop. The KB view is already small, and query policy is not a claim.
- A's epistemic section (`epistemic.py`) and KR (`narrative.py`) for QA: freeze, no more arms.
- LLM-stated confidence as the claim-3 metric: drop. Claim 3 is about the KB's p, and the controller can read p directly.
- B / `self_memory.py`: keep as plumbing, no further work until M5.

## Recommendation for the next phase
**Modify.** HANDOFF Next 1 (relink cost margin as `id_margin` → R6 v2) is the right next step, but measure it **without any LLM**. Score the AUROC / AURC of the KB's p on tracker-swapped vs clean cubes, within stress. Compare against SoftHaircut with the same margin feature, the setting bit and the near-pass heuristic. Tune on G train, then do one frozen G test read, with the rule committed first. If the KB does not beat SoftHaircut, report claim 3 as negative and stop the presentation work. In the same G test read, run R / K / argmax on Q1 so that falsifier 3 rests on held-out data. After that, the only experiment that can move claims 1–2 is the closed-loop pilot (H / R / K) on permanence and VideoPlaceOrder.

**Stale docs.** HANDOFF does not mention A/A2/B/C/C2 at all. HANDOFF Next 3 still says "build harness.py", but the skeleton and ReplayEnv now exist. HANDOFF lists prompts hash ba6ba13b8dbd, but it is now 2f2213558b45. The HANDOFF "HEAD on main" line is out of date. The DEVELOPMENT v1.12 header and §2.7 code map are missing `epistemic.py`, `memtool.py` and `self_memory.py`, and do not record the A/A2/C results. The v1.12 header and HANDOFF state "K beats the raw log on real perception" without the train-only, privileged-vocab and argmax caveats.
