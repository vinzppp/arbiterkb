# Phase G pod session: measure how the real tracker swaps container ids

Needs the user's go (GPU spend). Uses the existing RunPod volume from scene-graph-robot (`/workspace`: venvs `sg-main` and `sg-percep`, model weights, Vulkan fix). See `scene-graph-robot/notebooks/pilot/POD_RUNBOOK.md` and `BUILD_NOTES.md` ("new pod" section) for the pod itself.

**Rough cost:** RTX 4090 at about $0.4–0.7/h. Rendering 7 settings × 20 episodes takes about 0.5–1 h, perception about 1 min per episode (about 2–2.5 h), plus copying. Total about 3–4 h, i.e. **about $2–3**. Run the smoke test (step 3) first and re-estimate from its timings before the full run.

## 0. On the Mac
```bash
cd ~/Code/active/arbiter-kb && git push && git archive -o ~/Downloads/arbiter-kb.tar.gz HEAD
```
Upload `~/Downloads/arbiter-kb.tar.gz` to `/workspace` with Jupyter Lab's upload button.

## 1. Start the pod (same template, volume and `NVIDIA_DRIVER_CAPABILITIES=all` as before)
```bash
bash /workspace/setup_new_pod.sh
source /workspace/env/pod_env.sh
mkdir -p /workspace/arbiter-kb && tar -xzf /workspace/arbiter-kb.tar.gz -C /workspace/arbiter-kb --no-same-owner
cd /workspace/arbiter-kb && source /workspace/envs/sg-main/bin/activate
export SG_PILOT_DIR=/workspace/sg/notebooks/pilot
```
PASS: `python -c "import robomme, mani_skill, sg_pilot" ` runs with `PYTHONPATH=$SG_PILOT_DIR`.

## 2. Hand-off prompt for the pod Claude session (`cd /workspace/arbiter-kb && claude`)
```
Run phase G of arbiter-kb/DEVELOPMENT.md (§6 G) following stress/POD_RUN.md steps 3-7. The code in stress/ was written on a Mac without a simulator and has never run: expect small API mismatches with robomme @ 53673b3 / mani_skill 3.0.0b21. Fix only files in stress/ and say what you changed and why. Never modify /workspace/sg. Run the smoke test (step 3) first and stop to show me its output before the full run.
```

## 3. Smoke test (one episode, about 5 min)
```bash
python stress/render.py --out /workspace/g/replays --n 1 --settings stock,fast_close
python stress/perceive.py --replays /workspace/g/replays --out /workspace/g/perceived --limit 2
python scripts/measure_swaps.py --dir /workspace/g/perceived --out /workspace/g/results
```
Check in the printed JSON lines:
- `T` is about 64 + n_swaps · swap_steps + a few frames;
- `schedule` matches the setting (stock: 50-step windows from 64; fast_close: 20-step windows);
- `pose_log_aligned` is true (if false, the oracle still works from segmentation; note it);
- the rendered frames show the swap with the expected lane (open one with `python -c "import numpy as np, imageio; z=np.load('...'); imageio.mimsave('/workspace/g/smoke.gif', z['front_rgb'][::2])"` and download it);
- perceive prints `seeds` > 0 and a per-episode time (use it to re-estimate the full run).

## 4. Full run (tmux)
```bash
tmux new -s g
python stress/render.py --out /workspace/g/replays --n 20 2>&1 | tee /workspace/g/render.log
python stress/perceive.py --replays /workspace/g/replays --out /workspace/g/perceived 2>&1 | tee /workspace/g/perceive.log
python scripts/measure_swaps.py --dir /workspace/g/perceived --out /workspace/g/results | tee /workspace/g/measure.log
```
Both scripts skip finished episodes, so a crash or restart resumes where it stopped. `render.py` stops after the first episode if the projected size exceeds 8 GB.

## 5. Bring results to the Mac
The perception outputs are small (no images). Pack and download them via Jupyter Lab:
```bash
tar -czf /workspace/g_perceived.tar.gz -C /workspace/g perceived results render.log perceive.log measure.log
```
On the Mac: `mkdir -p data/g_perceived && tar -xzf ~/Downloads/g_perceived.tar.gz -C data/ && mv data/perceived/* data/g_perceived/`, then `python scripts/measure_swaps.py` (writes `results/g_passes.csv`, `results/g_summary.json`).

## 6. Same session: stock oracle replays for M6 (no GPU work)
On the Mac, with the pod's SSH host/port: `HOST=... PORT=... bash scripts/copy_replays.sh` (prints the size first).

## 7. STOP THE POD
In the RunPod console. The volume keeps `/workspace` (the replays stay there for later).
