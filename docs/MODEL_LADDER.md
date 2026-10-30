# Model ladder for the closed-loop controller: candidates, roles, screening (2026-10-30)

Research and plan only: no code, no paid run. Answers option 1 of `docs/HANDOFF_EXECUTION.md` ("model ladder on the
current interface"). Every number found online carries its source and the date it was read (all read **2026-10-30**);
"unverified" means a secondary source or no source. Our own numbers come from NOTES ("Layer 1 on real scenes", "Astra
VideoUnmaskSwap val pilot…", "Does K lose track on two-pick goals?").

## 1. What the ladder has to show

- The interface is fixed: each step the model reads the 2610.17350 system prompt, the arm's memory (H images / R raw
  log / K KB view), a 256-px front and wrist image and the robot state, and returns one JSON (`waypoint` = TCP
  `[x, y, z, roll, pitch, yaw]` in the world frame, `gripper` −1/0/1, `target`, `cites`). RoboMME's planner executes.
  Every model must work **through this interface unchanged**; any adapter is a different arm (section 3).
- What we know: with memory, layer 1 (where) is solved by copying the view (Astra 19/20, Qwen 12/15 first call-outs
  right, error = perception's). Failures are in layers 2/3/5: grasp pose, lift height, "done" judgement, progress.
  Astra H 9/10, R 6/10, K 8/10 on VideoUnmaskSwap val 0–9; Qwen 3.6-35B-A3B pilot 1/24, H localises 1/9.
- **Claim 1** ("a smaller model with the KB reaches the same success") needs, from the ladder:
  1. several sizes **within one family**, so a difference is attributable to size, not to the family;
  2. a **weak end** where H fails and K may succeed (the claim lives in that cell), and a **strong reference**;
  3. an **oracle-memory diagnostic arm (K-O)**: simulator truth written into K's view, same layout. K-O fails →
     the model cannot execute (layers 2/3/5), no memory can help; K-O succeeds but K fails → the gap is perception /
     the KB. K-O is an upper bound, never a claim (GOALS "Oracle ≠ perceived").
- **Why not just the strongest spatial model:** K already hands the model the 3D positions, so strong spatial
  perception mostly helps H, i.e. it *narrows* the gap claim 1 needs. What a K-arm controller needs is layers 2/3/5:
  choosing a grasp height and orientation, sequencing, judging completion. No public benchmark measures that on this
  interface (HANDOFF_EXECUTION "Literature found"; a search on 2026-10-30 again found no model comparison for
  "VLM outputs 3D waypoints to a planner"). Spatial benchmark scores below are therefore context, not selection
  criteria; the screen in section 5 is the selection criterion.

## 2. Candidate table

Prices are USD per 1M tokens, input / output, OpenRouter list prices from its public models API
(`https://openrouter.ai/api/v1/models`, read 2026-10-30) unless marked. "Cache" = read price; write price in
brackets where the provider charges one. Benchmarks: **benchmark scores (QA / pointing accuracy), not task success**,
unless the column says success.

### 2a. General VLMs that can emit our JSON directly (text out; no adapter)

| model | access | price in / out | cache read [write] | ctx | spatial / embodied evidence (benchmark score, source) | notes |
|---|---|---|---|---|---|---|
| openai/gpt-6-astra | OpenRouter, OpenAI API | 10 / 50 | 1.0 [12.5] | 1.05M | ERQA / VSI: **none found** (unverified). Our harness: H 9/10, K 8/10, R 6/10 task success (NOTES). 2610.17350: 92.1% RoboMME (their run) | reference; released 2026-09-03 (computingforgeeks, unverified) |
| openai/gpt-6-sol | OpenRouter, OpenAI API | 2 / 10 | 0.2 [2.5] | 1.05M | none found; OSWorld 2.0 64.4% at max effort (computer use, not spatial; secondary) | same generation as Astra; released 2026-09-22 (secondary) |
| openai/gpt-6-luna | OpenRouter, OpenAI API | 0.1 / 0.5 | 0.01 [0.125] | 1.05M | none found; OSWorld 2.0 52.7% (secondary) | same generation; 1/100 of Astra's price |
| openai/gpt-6.1-terra | OpenRouter | 1 / 5 | 0.05 [1.25] | 1.05M | none found | 6.1 generation, not 6.0: not in the family ladder |
| qwen/qwen3.6-35b-a3b | OpenRouter; open weights (Apache-2.0) | 0.1 / 0.9 | 0.025 (provider-dependent) | 262k | ERQA 61.8, EmbSpatial 84.3, RefSpatial 64.3 (HF card Qwen3.6-27B table); our pilot 1/24 success, H localises 1/9 | GOALS' primary small model; 3B active |
| qwen/qwen3.6-27b | OpenRouter; open weights (Apache-2.0) | 0.32 / 3.2 | none listed | 262k | ERQA 62.5, EmbSpatial 84.6, RefSpatial 70.0, CountBench 97.8 (HF card) | dense 27B (card has no expert fields), same generation as 35B-A3B |
| qwen/qwen3.5-9b / -27b / -397b-a17b | OpenRouter; open weights | 0.17/0.25; 0.26/2.6; 0.5/3.6 | 0 / 0 / 0.3 | 262k | ERQA 55.5 / 60.5 / 67.5; EmbSpatial 83.0 / 84.5 / 84.5; RefSpatial 58.5 / 67.7 / 73.6 (HF cards 3.5-9B, 3.6-27B, 3.5-397B) | cleanest *parameter* ladder with published scores in matching tables; previous generation |
| qwen/qwen3.8-27b | OpenRouter (image input listed) | 0.15 / 1.875 | 0.037 | 1M | none found; open-weight status and vision of the HF release **unverified** (secondary sources conflict) | |
| google/gemini-3.5-flash | OpenRouter, Google API | 1.5 / 9 | 0.15 | 1M | none found for this version | base model of Robotics-ER 2 per Google's docs (ja page; unverified in the English page) |
| google/gemini-4-argon | OpenRouter, Google API | 2 / 10 (doubles 2026-12-01, Google pricing page) | 0.1 | 1M | none found | Google's current flagship |
| anthropic/claude-haiku-5.5 / sonnet-5.5 / opus-5.5 | OpenRouter, Anthropic API | 0.1/0.5; 2/10; 4/20 | 0.01; 0.1; 0.2 [1.25× in] | 1M | none found for 5.5; Claude 4.5 Opus ERQA 46.8 (Qwen3.5-397B card) | a full same-generation 3-tier family; **512-token cache minimum** (below) |

### 2b. Embodied / spatial specialists

| model | access | price | interface fit | evidence | notes |
|---|---|---|---|---|---|
| Gemini Robotics-ER 2 (`gemini-robotics-er-2-preview`) | **Google API only** (not on OpenRouter list, 2026-10-30); needs a Google API key | 1 / 5 until 2026-12-31, then 2 / 10; cache read 0.10 (implicit only) (ai.google.dev pricing) | Text-out LLM: it can be prompted for our JSON with no adapter. Its *trained* spatial formats are 2D `[y, x]` points 0–1000, boxes, trajectories; no 3D mentioned (robotics-overview docs). Whether it writes good 6-D poses is unknown → the screen decides | ERQA 78.5 vs ER 1.6 72.5 and Gemini 3.6 Flash 73.0 (eesel.ai summary of DeepMind charts, **unverified**); progress classification 57.4%, moment finding 91.3% (Google blog); no task success reported | input limit 131,072 tokens (H's longest requests must fit); thinking level settable; ER 1.6 deprecated 2026-08-31 |
| Embodied-R1.5-8B (`IffYuan/Embodied-R1.5`, base Qwen3-VL-8B) | open weights; self-host (8B bf16 ≈ 17 GB, fits the A40) | GPU time only | **Needs an adapter**: outputs 2D points normalised 0–1000 and 2D/3D visual traces, not world-frame 6-D poses | ERQA 46.0, VSI-Bench 56.1, RoboSpatial 69.7, Where2Place 74.0, RefSpatial 54.2 vs its base Qwen3-VL-8B 41.8 / 55.1 / 65.4 / 64.0 / 42.2 (arXiv 2606.11324v2 Tables 2–3); zero-shot real robot (PGC framework, 6 trials): pick&place 6/6, door open 1/6 (Table 12) | gains over its base are mostly pointing; ERQA below GPT-5.4 (50.5) |
| MolmoER / MolmoAct2 (AI2, arXiv 2605.02881) | open weights | GPU time | MolmoER: VLM backbone (output formats not read); MolmoAct2: trained action expert (a VLA, different interface) | "surpasses GPT-5 and Gemini Robotics-ER 1.5 across 13 benchmarks" (abstract; numbers not read) | **unverified** beyond the abstract |

**Adapter note.** A 2D-point model in our loop needs: point in the front image → depth lookup → back-projection with
the camera pose → 3D target, plus a fixed rule for orientation, lift height and gripper sequence. That rule *is* the
skills layer (option 3): it takes layers 2/3 away from the model. Results from such an arm cannot be compared with
H/R/K under claims 1–2; they belong to the separate system claim. So specialists that only point are **out of the
ladder**; Robotics-ER 2 stays a candidate only because it is a text-out LLM that can be asked for our JSON.

### 2c. Request mechanics that change cost or validity

| | OpenAI GPT-6 | Anthropic 5.5 | Gemini (3.6+ / 4 / ER 2) | Qwen via OpenRouter |
|---|---|---|---|---|
| image tokens, one 256 × 256 frame | 77 (32-px patches × 1.2; developers.openai.com images guide, astra) | ≈ 87 (w·h/750; formula **unverified** this round) | 258 per image ≤ 384 px (ai.google.dev image-understanding) | ≈ 64–66 (our code comment, measured) |
| H's 768 × 768 grid image | 692 | ≈ 786 (unverified) | tiles of 258; ≈ 4 tiles (rounding rule unstated) | ≈ 576 |
| images per request | 1,500 | unverified | 3,600 | provider-dependent, unverified |
| cache | automatic, or explicit `prompt_cache_breakpoint`; **min 1,024 tokens**; write 1.25× (30 min) / 2.5× (12 h); read 0.1× (0.05× on 6.1 Sol/Terra) (OpenAI prompt-caching guide) | up to 4 `cache_control` breakpoints; **min 512 tokens** for all 5.x models; write 1.25× (5 min) / 2× (1 h); read 0.1×, 0.05× on Sonnet/Opus 5.5 (Claude docs) | implicit only on Gemini 4 / ER 2; **min 4,096 tokens** (3.6+/4); on Gemini 4 a multi-turn request is cached only if it adds ≥ 2,050 tokens (ai.google.dev caching) | Alibaba: explicit `cache_control` only, 5 min, write 1.25×, read 0.1× (OpenRouter prompt-caching docs); other providers vary |
| routing | — | — | — | pin `provider.order` for reproducibility (providers differ in quantisation); pinning disables OpenRouter's sticky cache routing (OpenRouter docs) |

Consequence for claim 2: K's fixed prefix is 886 tokens (NOTES). It misses OpenAI's 1,024 and Gemini's 4,096
minimum, but **would be cached by Anthropic (512)**. Any cost comparison must state the provider's cache rule; a
per-family cost table is needed, not one number.

JSON stability: no source measures it for our schema. All models above list `response_format` / `structured_outputs`
on OpenRouter, but the harness parses free text (as 2610.17350 does); turning structured outputs on for some models
only would be an unequal harness. The parse rate is measured in screen S0 instead.

Self-hosting on the A40 (48 GB), weights only, **estimates** (params × bytes, not tested): Qwen3.6-27B bf16 ≈ 54 GB
(no), 8-bit ≈ 28 GB (yes), 4-bit ≈ 16 GB (yes); Qwen3.6-35B-A3B bf16 ≈ 70 GB (no), 4-bit ≈ 20 GB (yes);
Embodied-R1.5-8B bf16 ≈ 17 GB (yes). A self-hosted quantised model is a **different model** from the OpenRouter one:
own id, never pooled.

## 3. Recommended ladder (5 models, two families)

Closed models do not publish sizes: within GPT-6, "smaller" means a lower tier (and price), not a known parameter
count. Only an open family gives parameter counts. So the ladder has one closed family anchored at the reference and
one open family that carries the "smaller model" wording.

| # | model | role | why |
|---|---|---|---|
| 1 | **gpt-6-astra** | strong reference (ceiling) | the 2610.17350 model; we already have H 9/10, K 8/10. New runs: K-O (execution ceiling) and the main episodes |
| 2 | **gpt-6-sol** | same-family middle | 1/5 of Astra's price, same generation (6.0). Shows whether the H–K gap opens as the tier drops |
| 3 | **gpt-6-luna** | same-family weak end; first claim-1 candidate | 1/100 of Astra's price. Claim 1 in this family = Luna-K ≈ Astra-H while Luna-H is clearly lower. Its role is fixed by the S0 rule below (it must fail H localisation to be the weak end) |
| 4 | **qwen3.6-27b** (dense) | open-weight claim-1 candidate | same generation as #5, 27B dense vs 3B active; best Qwen3.6 spatial scores (ERQA 62.5, RefSpatial 70.0); Apache-2.0, self-hostable at 8/4-bit |
| 5 | **qwen3.6-35b-a3b** | open weak end (GOALS' primary) | already measured (pilot 1/24, H 1/9 localisation); re-screened because its pilot ran before the gripper-0 fix and other harness fixes |

Optional probe, outside the ladder (needs a Google key): **Gemini Robotics-ER 2 vs gemini-3.5-flash**, same base
model by Google's docs, so the pair isolates "embodied fine-tuning" at our interface: does it help layers 2/3/5 when
K gives the positions? A probe, not a claim-1 rung (different family from the rest).

Not chosen, with the reason: Embodied-R1.5 / MolmoER (need an adapter; section 2b); gpt-6.1-terra (other
generation); Gemini 4 Argon (one more family without a size ladder); Claude 5.5 family (a clean 3-tier family and
cache-friendly for K, but no spatial evidence for 5.5; the backup family if GPT-6 fails S0); Qwen3.5 9B/27B/397B
(cleanest parameter ladder, but previous generation; the backup if the user wants a 3-size open ladder).

## 4. Arms per model

- Every rung: **K-O** (diagnostic), **K**, **H**. R only for Astra (already have it) unless the user asks.
- Same harness for every model: same prompt bytes (one prompts hash), same step limit, same `--max-tokens`; the
  per-model settings that cannot be equal (reasoning effort names, Qwen's sampling) are fixed per model before the
  screen and recorded in NOTES; not tuned after.
- K-O does not exist in code yet: it needs the simulator-truth view (same layout as K, only the source changes),
  built by the window that owns `closed_loop.py` / views, and the preflight rule applies.

## 5. Screening, with rules fixed before any run

All screening is on **val** episodes; test episodes are untouched (E5c freeze rule). VideoUnmaskSwap val:
one-pick 0, 2, 4, 5, 6; two-pick 1, 3, 7, 8, 9 (NOTES "Astra's 7 failures").

### S0: replay screen (format + layer 1), ~20 calls per model, no simulator
- **Input:** the exact first-step request bytes already paid for in the Astra pilot (VideoUnmaskSwap val 0–9, arms K
  and H; from the LLM cache), sent once to each candidate with its fixed settings. Needs a small replay script
  (not written; a task).
- **Measured:** valid JSON with all fields; first waypoint xy; first close dz if any; output tokens; hitting the
  `max_tokens` cap.
- **K rule (gate):** out if valid JSON < 9/10, or the first waypoint is more than 2 cm (xy) from **Astra K's first
  waypoint on the same request** in more than 1/10 (Astra K copies the view; this is a copy test, so a KB error in
  the view is not held against the model), or the cap is hit in ≥ 2/20 calls.
- **H rule (role, not a gate):** H "localises" if right ≥ 8/10 by the layer-1 rule (NOTES "Layer 1 on real
  scenes"), "fails" if right ≤ 3/10, otherwise "partial". A model can be the claim-1 weak end only if H fails or is
  partial. If Luna's H localises, Luna becomes a middle rung and Qwen3.6-35B-A3B (H 1/9 already) is the weak end.

### S1: oracle-memory K on 6 episodes (execution ability)
- **Episodes:** val 0, 2, 4 (one-pick) and 1, 3, 7 (two-pick); K-O only.
- **Rule:** pass if K-O ≥ 4/6; out if ≤ 1/6; at 2–3/6 run val 5, 6, 8, 9 and pass if ≥ 6/10. Also out if
  > 20% of calls are unparsable or any episode ends at the cost cap.
- **Why this threshold:** Astra reached 8/10 on *perceived* K; a model below a majority with perfect memory cannot
  reach "matched success" with any memory, so it cannot carry claim 1.
- **Same 6 episodes, H**, for models that pass (cheap except for Astra / Sol): gives the H–K-O direction before the
  main run; it is a screen, not a result (n = 6).
- Astra runs S1 too: its K-O is the ceiling the other rungs are read against.

### After screening
The main run (episodes, arms, n, analysis rule) is a separate NOTES rule written before it, on episodes not used in
S0/S1, through the full preflight. Not designed here.

## 6. Cost estimates (list prices; no caching assumed, so an upper bound for H on OpenAI)

Per-episode token model from Astra's pilot: K ≈ 19.6k prompt + ≈ 1.7k output tokens (median; output derived from
$0.279 − prompt cost); H ≈ 51.7k prompt + ≈ 2.4k output. "High" = 3× calls and 10× output (Qwen's pilot repeated
waypoints and reasoned to the 32k cap): K ≈ 59k + 17k.

| model | K $/ep central | K $/ep high | H $/ep central (uncached) | S0 (20 calls) | S1 (6 K-O + 6 H, high) |
|---|---|---|---|---|---|
| gpt-6-astra | 0.28 | 1.44 | 0.64 (observed 0.28 with caching) | ≈ 4 | ≈ 13 |
| gpt-6-sol | 0.056 | 0.29 | 0.13 | ≈ 0.8 | ≈ 2.6 |
| gpt-6-luna | 0.003 | 0.014 | 0.006 | < 0.1 | < 0.2 |
| qwen3.6-27b | 0.012 | 0.073 | 0.024 | < 0.2 | ≈ 0.6 |
| qwen3.6-35b-a3b | 0.004 | 0.021 | 0.007 | < 0.1 | < 0.2 |
| robotics-er-2 (probe) | 0.028 | 0.14 | 0.064 | ≈ 0.3 | ≈ 1.3 |
| gemini-3.5-flash (probe) | 0.045 | 0.24 | 0.10 | ≈ 0.5 | ≈ 2.1 |

Screening the five rungs: ≈ $5 (S0) + ≈ $17 (S1, high case; Astra is most of it); + ≈ $4 for the Google probe.
A main run of 20 episodes × 3 arms × 5 rungs: ≈ $23 central, ≈ $90 high, about three-quarters of it Astra. These are
estimates from one task's token counts; `--max-usd-episode` caps each episode.

## 7. Open questions (the user's decisions)

1. **Google API key:** add the Robotics-ER 2 / 3.5-flash probe? (ER 2's price doubles on 2027-01-01.)
2. **Self-host or OpenRouter for Qwen:** OpenRouter (pinned provider) keeps the model id identical to the
   published weights' hosted version; self-hosting quantised on the A40 is cheaper per run but is a different model.
3. **Primary small model:** keep qwen3.6-35b-a3b as GOALS' primary, or let the screen choose between it and
   qwen3.6-27b? (Changing GOALS' primary is a direction change and goes in NOTES.)
4. **Closed family:** GPT-6 (anchored at Astra; K's prefix not cacheable) or Claude 5.5 (cacheable K prefix, no
   spatial evidence)? Recommended GPT-6.
5. **Reasoning settings** per model (effort level, Qwen sampling), fixed before S0.
6. **Build order:** S0 needs a replay script; S1 needs the K-O view. Both are code tasks for the owning windows,
   after the open questions in HANDOFF_ORCHESTRATOR that can change K's text (Inside after uncover, jitter as moves).

## Sources (all read 2026-10-30)

- OpenRouter models API (prices, context, modalities, supported parameters): https://openrouter.ai/api/v1/models
- OpenRouter prompt caching: https://openrouter.ai/docs/features/prompt-caching
- OpenAI images / vision (patch tokens, limits): https://developers.openai.com/api/docs/guides/images-vision
- OpenAI prompt caching: https://developers.openai.com/api/docs/guides/prompt-caching
- Anthropic prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Gemini pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini robotics overview (ER 2 ids, formats, limits): https://ai.google.dev/gemini-api/docs/robotics-overview ;
  base model statement: https://ai.google.dev/gemini-api/docs/robotics-overview?hl=ja
- Gemini image understanding: https://ai.google.dev/gemini-api/docs/image-understanding
- Gemini caching: https://ai.google.dev/gemini-api/docs/caching
- Gemini Robotics ER 2 launch: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/
- ER 2 ERQA numbers (secondary, unverified): https://www.eesel.ai/blog/gemini-robotics-2
- Embodied-R1.5: https://arxiv.org/html/2606.11324v2 ; https://embodied-r.github.io/
- MolmoAct2 / MolmoER: https://arxiv.org/abs/2605.02881
- Qwen cards: https://huggingface.co/Qwen/Qwen3.6-35B-A3B , https://huggingface.co/Qwen/Qwen3.6-27B ,
  https://huggingface.co/Qwen/Qwen3.5-9B , https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- Qwen3.8 open-weight status (conflicting, unverified): https://github.com/QwenLM/Qwen3.8
- GPT-6 release dates and OSWorld numbers (secondary): https://computingforgeeks.com/gpt-6-sol-luna-released-features-benchmarks/
