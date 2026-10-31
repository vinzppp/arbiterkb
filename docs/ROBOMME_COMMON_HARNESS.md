# RoboMME common harness v1

This is the executable protocol for the frozen MEMI-versus-ArbiterKB Plan
comparison. It deliberately does not expand the rule set.

## Frozen comparison

- Source branch: `research/perception-belief-robomme` (the runner refuses
  `main`, `master`, and any branch not named in the manifest).
- Manifest: `experiments/robomme_common_v1.json`.
- Arms: official FullHist, official MEMI, soft identity repair, full ArbiterKB.
- All arms use the same RoboMME `multi_choice` executor, episode, reset,
  semantic capture, model, effort, token budget, and decision budget.
- FullHist and MEMI are imported from the pinned official MEMI checkout.
  The structured arms use the same FullHist action scaffold, but replace the
  conversation history with a deterministic snapshot built from the same
  visible semantic records.
- Simulator actor names are used only after a decision to score belief truth.
  The structured policies receive anonymous geometric tracks and never actor
  identity. The per-episode `reset_digest` and resolved benchmark seed must
  match across all four arms.

The frozen order is:

1. no-LLM, one-action executor preflight on val episode 0;
2. four-arm clean smoke test on val episode 0;
3. val episodes 1--9 under clean, 15% record dropout, and 4 px centroid jitter;
4. only after the preceding gates pass, test episodes 0--49 with the already
   frozen manifest.

No test result may be used to alter observations, prompts, parameters, rules,
tasks, or contrasts. A change requires a new manifest version and a new val
cycle. `VideoRepick` is marked optional because older RoboMME releases may not
ship it; the pinned release used here does.

## Exact environment

The native macOS path cannot import pinned RoboMME because its pinned
ManiSkill requires `mplib==0.1.1`, which has no macOS wheel. The reproducible
fallback is the Linux/amd64 CPU image:

```sh
docker build --platform linux/amd64 \
  -f stress/Dockerfile.robomme-cpu \
  -t arbiterkb-robomme-cpu:bb6c37c .

docker run --rm --platform linux/amd64 \
  -v "$PWD:/workspace/ArbiterKB" \
  arbiterkb-robomme-cpu:bb6c37c
```

The second command resets all four frozen tasks without an LLM. On Apple
Silicon this uses x86 translation and CPU rendering; it is valid but slow.

Run a stage with a fresh container per episode/arm:

```sh
docker run --rm --platform linux/amd64 \
  -v "$PWD:/workspace/ArbiterKB" \
  -e OPENAI_API_KEY \
  -e OPENAI_API_URL \
  arbiterkb-robomme-cpu:bb6c37c \
  python stress/common_harness_pod.py \
    --stage smoke --memi-root /opt/memi \
    --out /workspace/ArbiterKB/results/robomme_common
```

Do not pass, print, copy, or serialize an OpenRouter key. The official MEMI
client used here reads `OPENAI_API_URL` (the public OpenAI chat-completions
endpoint when unset). A model run should use a deliberately supplied
compatible credential and endpoint; absent that, stop after the no-LLM
preflight rather than inventing scores.

## Outputs and scoring

Each row records the manifest and source SHAs, ArbiterKB commit/dirty state,
parameters, prompt hash, resolved episode seed/difficulty, reset digest,
observational degradation, decisions, outcome, model usage, latency, and wall
time. API failure produces no scored row.

```sh
.venv/bin/python scripts/common_harness_report.py \
  results/robomme_common/smoke \
  --out results/robomme_common/smoke/summary.json
```

The report includes Wilson success intervals, exact paired McNemar tests with
Holm adjustment, paired bootstrap intervals for cost/latency/calls, belief
accuracy/Brier/ECE where truth is observable, incomplete-arm and reset-digest
audits, dirty-tree rows, and harness errors. A no-LLM preflight outcome is an
executor diagnostic, not a closed-loop policy score.

## Validation status (2026-10-31)

The transport-only OpenRouter amendment is frozen in
`experiments/robomme_common_v1_openrouter.json` (SHA `507cbdd26bf429e7`).
The clean validation produced 15/16 scored rows. `ButtonUnmaskSwap/soft` did
not return after 16:24 and was stopped without a result, so the audit correctly
fails for an incomplete arm set. Degradation and held-out stages have not run.
See `paper/ROBOMME_COMMON_HARNESS_REPORT_ZH.md` and
`results/robomme_common/openrouter_v1/reproducibility_manifest.json`.

## R2 watchdog v2.1

The process-level watchdog protocol is frozen in
`experiments/robomme_common_v2_1_watchdog_openrouter.json` (SHA
`a839ccc592f8aa55`). It hash-checks and inherits the full v2 manifest. The
first v2 hash was retired before any cell because its reset ceiling was below
the observed v1 VideoRepick preflight reset proxy.

Each v2.1 cell runs in a fresh process group and writes fsync'd JSONL phase
events. The supervisor can terminate a blocked reset, LLM call, executor
command, or whole episode, escalate from SIGTERM to SIGKILL, and write a
terminal result without waiting for the blocked call to return. Timeouts are
primary failures; infrastructure errors and incomplete cells are separate and
fail the gate. Loop detection is diagnostic only.

The four-task no-LLM simulator preflight passed 4/4 with complete result and
event traces. The 16-cell API smoke is not started because no newly authorized
credential is configured; the runner fails closed before creating a run or
making a paid request. See `paper/ROBOMME_WATCHDOG_V2_REPORT_ZH.md`.
