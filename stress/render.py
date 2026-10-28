"""Render VideoUnmaskSwapStress episodes into sg_pilot Replay npz files (phase G). ON THE POD, sg-main venv.

  python stress/render.py --out /workspace/g/replays --n 20                 # all settings
  python stress/render.py --out /workspace/g/replays --n 1 --settings stock  # smoke test first

Only the conditioning video is rendered (env.reset plays it through RoboMME's DemonstrationWrapper);
no policy acts. Wrist-camera arrays are stored as 1x1 placeholders to halve the size (the oracle uses
the front camera; a container is never visible in the wrist camera only during the video).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SG_PILOT = Path(os.environ.get("SG_PILOT_DIR", "/workspace/sg/notebooks/pilot"))
sys.path.insert(0, str(SG_PILOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SETTINGS = {  # name: (swap_steps, lane_offset, n_swaps, cam_shift)
    "stock": (50, 0.07, None, (0, 0, 0)),
    "fast": (20, 0.07, None, (0, 0, 0)),
    "close": (50, 0.02, None, (0, 0, 0)),
    "fast_close": (20, 0.02, None, (0, 0, 0)),
    "fast_close_3swaps": (20, 0.02, 3, (0, 0, 0)),
    "cam_left": (50, 0.07, None, (0, 0.05, 0)),
    "cam_right": (50, 0.07, None, (0, -0.05, 0)),
}
INCLUDE = dict(include_maniskill_obs=True, include_front_depth=True, include_wrist_depth=True,
               include_front_camera_extrinsic=True, include_wrist_camera_extrinsic=True,
               include_front_camera_intrinsic=True, include_wrist_camera_intrinsic=True)


def make_env(setting: str, seed: int):
    import gymnasium as gym
    import stress_env  # noqa: F401  (registers VideoUnmaskSwapStress)
    from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper

    steps, lane, n_swaps, cam = SETTINGS[setting]
    env = gym.make("VideoUnmaskSwapStress", obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
                   render_mode="rgb_array", reward_mode="dense", seed=seed, swap_steps=steps, lane_offset=lane,
                   n_swaps=n_swaps, cam_shift=cam)
    return DemonstrationWrapper(env, max_steps_without_demonstration=2000, gui_render=False, **INCLUDE)


def render_episode(setting: str, seed: int):
    import sg_pilot as sg

    env = make_env(setting, seed)
    keys = ["front_rgb", "wrist_rgb", "front_depth", "wrist_depth", "E_front", "E_wrist", "joint_state",
            "gripper_state", "ee_xyz", "front_seg", "wrist_seg"]
    store = {k: [] for k in keys}
    obs, info = env.reset(seed=seed)
    T = sg._collect(obs, store)
    scene = sg.scene_objects(env)
    base = env.unwrapped
    poses = np.asarray(base.pose_log, np.float32)
    aligned = len(poses) == T
    obj_xyz = poses if aligned else np.full((T, len(scene["objects"]), 3), np.nan, np.float32)
    meta_extra = {"setting": setting, "seed": seed, "swap_steps": base.swap_steps, "lane_offset": base.lane_offset,
                  "n_swaps": base.swap_times, "difficulty": base.difficulty, "pose_log_aligned": aligned,
                  "pose_log_len": len(poses), "schedule": [(int(s), int(e)) for _, _, s, e in base.swap_schedule]}
    env.close()
    st = {k: np.stack(v) for k, v in store.items()}
    tiny = lambda a: np.zeros((T, 1, 1) + a.shape[3:], a.dtype)  # noqa: E731
    scene_static = {k: v for k, v in scene.items() if k != "id_map"}
    scene_static["objects"] = [{k: v for k, v in o.items() if k != "actor"} for o in scene["objects"]]
    scene_static["stress"] = meta_extra
    goal = info.get("task_goal", "")
    rep = sg.Replay(task="VideoUnmaskSwapStress", episode=seed, front_rgb=st["front_rgb"], wrist_rgb=tiny(st["wrist_rgb"]),
                    front_depth_m=st["front_depth"], wrist_depth_m=tiny(st["wrist_depth"]), front_seg=st["front_seg"],
                    wrist_seg=tiny(st["wrist_seg"]), K_front=np.asarray(info["front_camera_intrinsic"], np.float32).reshape(-1, 3, 3)[-1],
                    K_wrist=np.eye(3, dtype=np.float32), E_front=st["E_front"], E_wrist=st["E_wrist"],
                    joint_state=st["joint_state"], gripper_state=st["gripper_state"], ee_xyz=st["ee_xyz"],
                    is_video=np.ones(T, bool), policy_frame=np.zeros(0, int), actions=np.zeros((0, 8), np.float32),
                    task_goal=str(goal[0] if isinstance(goal, (list, tuple)) and goal else goal), scene=scene_static,
                    status="video_only", obj_xyz=obj_xyz)
    return rep, meta_extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/g/replays")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--settings", default="all", help=f"comma list or 'all' of {list(SETTINGS)}")
    ap.add_argument("--max-gb", type=float, default=8.0)
    args = ap.parse_args()
    import sg_pilot as sg

    names = list(SETTINGS) if args.settings == "all" else args.settings.split(",")
    out, total, done = Path(args.out), len(names) * args.n, 0
    for name in names:
        for seed in range(args.n):
            path = out / f"{name}_ep{seed:03d}_replay.npz"
            if path.exists():
                done += 1
                continue
            t0 = time.time()
            rep, meta = render_episode(name, seed)
            sg.save_replay(rep, path)
            done += 1
            mb = path.stat().st_size / 1e6
            print(json.dumps({"file": path.name, "T": len(rep.front_rgb), "MB": round(mb, 1),
                              "s": round(time.time() - t0, 1), **meta}), flush=True)
            if done == 1 and mb * total / 1e3 > args.max_gb:
                sys.exit(f"projected {mb * total / 1e3:.1f} GB > {args.max_gb} GB: lower --n or --settings")
    print(f"done: {done} replays in {out}")


if __name__ == "__main__":
    main()
