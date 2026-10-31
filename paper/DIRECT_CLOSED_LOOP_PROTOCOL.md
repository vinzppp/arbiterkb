# Frozen protocol for the missing direct RoboMME comparison

This protocol is the smallest experiment that can support a direct MEMI-versus-ArbiterKB conclusion. It is **not yet
run**. Do not merge its future outputs with the validation pilots or the gripper-bug run.

## Environment and pinned sources

- Linux GPU host with NVIDIA, Vulkan, and RoboMME rendering preflight passing.
- MEMI `bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0`.
- MEMI's RoboMME submodule `fae9f7300347f1146c2ae4844faa4eef7e38098e`.
- This research branch, with an exact commit recorded in every result.
- No `.env` file in an archive. Keys remain environment variables and are never printed or serialized.

## Factorial comparison

Run two observation regimes and four memory policies. Every cell uses the same planner model, prompts outside the
memory block, decision menu, executor, step limit, and test episodes.

Observation regimes:

1. **privileged semantic:** MEMI's simulator-segmentation text, to isolate memory organization;
2. **perceived:** one shared perception stream; MEMI and baselines receive its text serialization, while ArbiterKB
   receives the same detections as structured observations. No method sees simulator names, poses, or segmentation.

Memory policies:

1. FullHist;
2. released final MEMI schema/policy prompt;
3. causal re-linking plus frozen detector-only soft repair;
4. ArbiterKB with the same re-linker and frozen belief parameters.

Use GPT-6 Luna first because the released MEMI result leaves substantial headroom and its VideoUnmaskSwap gain is
large. A second model is confirmatory only after the first matrix completes. The low-level executor is RoboMME
`multi_choice` for every arm; the model selects a skill and image point, and the executor is unchanged.

## Tasks and split

Primary tasks:

- VideoUnmaskSwap: hidden identity and container crossing;
- ButtonUnmaskSwap: a cue appears during execution;
- PickHighlight: another cue-during-execution task where action logs alone are inadequate.

Use validation episodes only for smoke tests and integration fixes. Freeze code, prompts, perception weights, memory
parameters, and the complete run manifest before reading test. The primary result uses all 50 public test episodes per
task, paired across arms. Any harness error is rerun with identical code; policy failures are never rerun.

## Endpoints and tests

Primary endpoint: episode success. Secondary endpoints: policy calls, input/cached/output tokens, dollars, wall time,
invalid decisions, executor failures, and memory calibration at every hidden-target decision.

- Report successes with Wilson 95% intervals.
- Compare paired success with exact McNemar tests and Holm correction across the six planned pairwise contrasts.
- Bootstrap cost and call-count differences by episode (10,000 paired draws).
- Report AURC, Brier, ECE, coverage, and silent failure for belief-bearing methods.
- Stratify by observation regime and task. Do not pool privileged and perceived observations.
- Failure analysis is descriptive and performed only after the primary table is locked.

## Gates before any paid call

1. `make check` passes at the archived commit.
2. `stress/pod_checks.py` passes actors, grasp, state, pick, and solver checks.
3. A no-LLM scripted episode passes for every task and observation regime.
4. The episode-count guard reports exactly 50 test episodes.
5. The manifest records Git SHAs, container image, GPU, driver, CUDA/Vulkan, model/provider, every prompt hash, every
   data hash, caps, and commands.
6. A dry-run proofreads that no result path or log can contain an API key.

## Stopping rule

There is no efficacy stopping. Stop only for a shared harness defect, exhausted predeclared budget, or infrastructure
failure. Preserve partial data, mark it incomplete, and do not compute a headline comparison until every paired arm
for that episode is valid.
