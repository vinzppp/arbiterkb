# Closed-loop pilot on RoboMME: pod session

Needs the user's go (GPU and API spend). A pilot, not the claim: development on the **val** split, to see what the
closed loop breaks that the offline tests could not show. Claims (M6) come later from one frozen **test** read.

**What runs:** the 2610.17350 loop (`arbiter_kb/closed_loop.py`, their endings, prompt notes and frame stride) with
three arms that differ only in memory: **H** (the image history, their memory), **R** (the raw tracker log as text +
current frames), **K** (the KB view, the controller's own decisions and outcomes first, + current frames). One model for
all arms: `qwen/qwen3.6-35b-a3b` (takes images, OpenRouter). The KB is filled by the real perception pipeline on every
3rd front frame (the frames H sees), with the open word list (no simulator scene).

**Rough cost:** RTX 4090 at about $0.4–0.7/h, about 5–6 h: **$3–4**. LLM: about 10 calls per episode, 60 episodes,
qwen at $0.10 per million input tokens: **under $1** (H's final requests carry ~150 images). Re-estimate from the
smoke run (step 3) before the pilot.

## 0. On the Mac
```bash
cd ~/Code/active/arbiter-kb && git archive -o pod_upload_arbiter-kb.tar.gz HEAD
```
If the volume is fresh, also the scene-graph-robot archive (POD_RUN.md §0F). Do not upload `.env`: the OpenRouter key is
typed into the pod's shell by the user (`export OPENROUTER_API_KEY=...`), never written into a file on the volume.

## 1. Pod and environment
- Same template and settings as G (`stress/POD_RUN.md` §0F / §1): RTX 4090, `NVIDIA_DRIVER_CAPABILITIES=all`, Vulkan
  check, venvs `sg-main` and `sg-percep`, weights.
- **RoboMME at commit `856bc3a`** (2610.17350's benchmark version; identical environments to main as of 2026-10-09).
  If the volume has G's `53673b3`, note both in NOTES and prefer `856bc3a` for this run.
- Hand-off prompt for the pod Claude session:
```
Set up this pod for the arbiter-kb closed-loop pilot (stress/POD_CLOSED_LOOP.md), then run its unit check (step 2) and
smoke test (step 3), and stop there with a report: per-frame perception time, model-call time, $ per episode per arm,
the first K prompt (text part only), and any errors.
If /workspace/envs/sg-main or sg-percep is missing, build them first exactly as stress/POD_RUN.md §0F says (Vulkan,
both venvs, weights, the checks), with two changes: install robomme at commit 856bc3a (not 53673b3), and also install
matplotlib and pytest into sg-main. If sg-percep exists but its GroundingDINO extension was built for another GPU
(BUILD_NOTES: sm_89 on the 4090), rebuild it with TORCH_CUDA_ARCH_LIST for THIS GPU (nvidia-smi; an A40 is 8.6).
Write what you installed to /workspace/CL_SETUP.md.
The code in stress/ and arbiter_kb/closed_loop.py has never run against the simulator: fix only what the smoke test
shows is broken, say what you changed and why. Do not run the pilot (step 4) until the user says so.
The OpenRouter key is in the environment variable OPENROUTER_API_KEY: never print it or write it to a file.
```

## 2. Unit check on the pod
```bash
cd /workspace/arbiter-kb && /workspace/envs/sg-main/bin/python -m pytest -q tests/test_closed_loop.py tests/test_perceived.py
```

## 3. Smoke test (one episode per arm, val)
```bash
cd /workspace/arbiter-kb && source /workspace/env/pod_env.sh
/workspace/envs/sg-main/bin/python stress/closed_loop_pod.py --tasks VideoUnmaskSwap --episodes 0 --arms H,R,K \
    --dataset val --out /workspace/cl/smoke --max-usd 0.5
```
Check, and write into NOTES "Closed loop":
- the env exposes `front_depth_list`, `front_camera_extrinsic_list` and `info["front_camera_intrinsic"]` (the
  perception needs them; the model never sees depth);
- `env.reset()` twice on one env gives the same episode (`closed_loop_pod.episode` resets once for the goal);
- the K prompt's memory lists the containers and cubes, and after the first action starts with "Since your last decision";
- seconds per perceived frame, per model call, per episode; $ per episode per arm.

## 4. Pilot (val, after the user's go)
```bash
tmux new -s cl
/workspace/envs/sg-main/bin/python stress/closed_loop_pod.py --tasks VideoUnmaskSwap,VideoPlaceOrder \
    --episodes $(seq 0 9) --arms H,R,K --dataset val --out /workspace/cl/pilot --max-usd 5
```
Resumable: finished episodes are skipped. Two tasks can run in two tmux windows (perception ~7 GB GPU each).

## 5. Bring back and stop
```bash
cd /workspace && tar czf cl_pilot.tar.gz cl/
```
Download `cl_pilot.tar.gz` to `data/cl_pod/` on the Mac (git-ignored). Keep `/workspace/g/replays` (the multi-object
Cutie re-run needs them). **STOP THE POD.**

## Later, not in this session
- One multi-object Cutie (spec §2.5) to cut lost container tracks (28% of G frames), re-run on G's replays.
- The frozen test read for M6: fresh episodes of the same tasks, `--dataset test`, after the freeze, in the ledger.
