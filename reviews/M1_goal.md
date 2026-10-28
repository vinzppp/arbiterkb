# Goal review — M1 — 2026-10-28
Verdict: ON-TRACK-WITH-CHANGES

M1 delivers what v1.2 asked for: the overlap corruption, three baselines, R5 and one untuned table. NOTES and the notebook correctly say the table shows plumbing only. My one sharp objection: **the metric and baselines, as now set up, cannot tell ArbiterKB from a threshold heuristic, including on G's data.** M2's done-when (c) cannot be met as written.

## Claim movement
- **Claim 1: none.** The corruption is synthetic and we designed it. Under `id_swap_overlap` (`results/latest/m1_table/summary.csv`, run 233ed15c):
  - silent failure: ArbiterKB 2/60 (0.03, CI 0.01–0.11), OverlapHaircut 0/60, Bound 6/60;
  - on corrupted episodes: n = 13, 6 of them wrong;
  - every interval overlaps.
- **Clean cost (part of claim 1):** accuracy is unchanged (1.00). But ArbiterKB drops below τ_hi on 6/60 clean rows (conf_right 0.92, Brier 0.042, ECE 0.078).
- **Falsifiers and claim 2: not touched.** Bound's 0.10 silent rate is set by our p_swap = 0.5, so it says nothing about the falsifier "deterministic memory is rarely confidently wrong".

## Risks of self-deception
1. **Silent failure at a fixed τ_hi can be gamed, and the claim's clean guard (accuracy) does not stop it.**
   - Any haircut to a confidence c with 0.5 < c < τ_hi keeps Bound's argmax and therefore Bound's accuracy, while scoring zero silent failures.
   - M2 (c) asks for ArbiterKB < OverlapHaircut "with non-overlapping Wilson intervals". OverlapHaircut is already at 0/60, so that cannot be reached.
   - The real trade-off shows only in a number nobody reports, **confident coverage** (rows with p_max ≥ τ_hi), from rows.csv:

     | system | clean: confident / 60 | id_swap_overlap: wrong rows caught |
     |---|---|---|
     | ArbiterKB | 54 | 4/6 |
     | OverlapHaircut | 48 | 6/6 |

     These are two operating points of the same 3D-distance threshold. Neither system dominates.
   - **Control:** report a risk–coverage pair for every system: clean confident coverage, and silent rate on corrupted rows. Restate M2 (c) as "lower silent rate at matched clean coverage" (or AURC). Add coverage to GOALS claim 1 next to "no loss of accuracy".
2. **ArbiterKB is structurally a soft OverlapHaircut.**
   - R6 v1.5 sets w from min 3D distance, the same feature OverlapHaircut thresholds. R6 v2 (M2) swaps that feature for the Hungarian margin.
   - If M2 compares against a haircut on distance, any win only shows that the margin is a better feature. It does not show that a belief layer helps.
   - **Control:** in M2, add `MarginHaircut`, the same threshold rule on the margin feature R6 v2 uses.
   - Beyond a haircut on the same feature, the belief layer can only win through propagation, i.e. R5 or retraction.
3. **R5 is the propagation mechanism, and as built it cannot fire where it would matter.**
   - It ran 0 ops.
   - `test_joint_exclusive_would_overconfide_after_a_mix` shows that applying it after a mix is wrong, so it is restricted to cover frames, where the marginals are already exclusive.
   - Done-when (d)'s "one synthetic scenario where R5 beats OverlapHaircut" was not delivered. Say so in NOTES rather than "stays for layouts where covers are ambiguous".
4. **The haircut adapters are crippled.**
   - With c = 0.5, `Distribution.argmax` breaks the tie by label string (`beliefs.py:107`). That costs OverlapHaircut 9/60 clean accuracy (0.85) and BoundHaircut 21/60.
   - This is a property of the adapter, not of the idea. It makes the table's accuracy column flatter ArbiterKB.
   - **Control:** use c = 0.75 (below τ_hi, keeps the argmax) or break the tie in favour of Bound's prediction.

## Missing baseline(s)
- OverlapHaircut with c ∈ (0.5, τ_hi). On this table its clean accuracy would be 1.00 and its silent rate 0.
- MarginHaircut for M2 (see risk 2).

## Cut or defer
- R5 in the step loop (inert): keep the function and its test, and stop citing R5 as the edge over haircuts.
- BoundHaircut: dominated in every column.
- JumpDetector: keep it as a one-line check only if G shows real swaps jump.
- The teleport `id_swap` path and selftest (c). The planner/skills, `validate()` and R7 at λ = 1 are carried over and still unused.
- Housekeeping:
  - There are 31 READ lines on the same test seeds. Run 4e45e899 is a dirty-tree test read after commit 58a6320.
  - It is harmless while nothing is tuned. Still, use fresh synthetic seeds (≥ 1060) for anything M2 reports.

## Recommendation for the next phase
**Keep G, with one change to `measure_swaps.py`.** As specified, it records the signature (min 3D distance, IoU, `track_q` drop, jump, margin) only for swaps.
- **Record the same features for every non-swapping pass too.** Whether any system can beat a haircut depends on how well these features separate swapped from clean passes, i.e. their ROC. Swaps alone cannot give that.
- **Add one more gate output:** if the best single feature separates swaps nearly perfectly, a haircut on that feature solves the problem. Claim 1 then reduces to "a threshold rule suffices", and NOTES should say so before M2 builds R6 v2.
- **Before G:** apply the c fix and add the coverage column. That costs about an hour, and it changes how M2 (c) and GOALS claim 1 are worded.
