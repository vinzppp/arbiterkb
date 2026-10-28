"""Run the real perception pipeline (GroundingDINO -> SAM -> Cutie, sg_pilot's PerceptionWorker) on the
stress replays and save small per-episode outputs for scripts/measure_swaps.py. ON THE POD, sg-main venv
(the worker itself runs in sg-percep).

  python stress/perceive.py --replays /workspace/g/replays --out /workspace/g/perceived [--limit 1]
  python stress/perceive.py --vocab open --replays /workspace/g/replays --out /workspace/g/perceived_open

--vocab scene (G's run): the detector's word list comes from the simulator scene (how many containers,
which cube colours). That is privileged information a real robot does not have. --vocab open (spec v1.5):
generic words with fixed slot counts plus the phrases the TASK INSTRUCTION names
(arbiter_kb.task_items); the scene is used only to store the evaluation truth in meta.

Per episode: perceived [T, S, 8] (sg_pilot slot rows), bbox [T, S, 4] (normalised row/col extents,
zeros when absent: mask-size and overlap features), oracle [T, N, 8] (sg_pilot.oracle_graph from the
simulator's segmentation + depth), obj_xyz [T, N, 3] (actor poses; NaN if not aligned) and meta.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # arbiter_kb (task_items: pure Python)


def open_config(sg, goal: str) -> tuple[dict, list[dict]]:
    """sg_pilot's RoboMME config with the open word list: no scene counts, task phrases marked."""
    from arbiter_kb.task_items import GENERIC_SLOTS, open_vocabulary, parse_task_items

    items = parse_task_items(goal)
    cfg = sg.make_robomme_perception_config(GENERIC_SLOTS["container"], [])  # containers first, area ranges set
    classes = open_vocabulary(items)
    cfg["task"].update(classes=classes, num_objects=len(classes), slot_by_class={i: i + 1 for i in range(len(classes))})
    cfg["perception"]["num_processors"] = len(classes)
    return cfg, items


def perceive(rep, handle, episode_num: int, scale: int = 2):
    """sg_pilot.perceived_graph, also keeping each slot's bbox extents (perceived_graph drops them)."""
    import cv2
    import sg_pilot as sg

    T, S = len(rep.front_rgb), handle.num_classes
    g = np.full((T, S, 8), np.nan, np.float32)
    box = np.zeros((T, S, 4), np.float32)
    n_seed = 0
    for t in range(T):
        rgb = rep.front_rgb[t]
        big = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        out = handle.process_frame(hand_bgr_rotated=np.ascontiguousarray(cv2.cvtColor(big, cv2.COLOR_RGB2BGR)),
                                   frame_num=t, episode_num=episode_num, depth=rep.front_depth_m[t],
                                   hand_rgb_unrotated=rgb, intrinsics=rep.K_front,
                                   extrinsics=sg.cam_to_world(rep.E_front[t]))
        n_seed += len(out.seed_events)
        for s in range(S):
            bb = out.bboxes[s + 1]
            visible = not np.allclose(bb[:4], 0.0)
            g[t, s, sg.F_ID], g[t, s, sg.F_LABEL] = s, 0 if handle.classes[s].startswith("container") else 1
            g[t, s, sg.F_HIDDEN] = 0 if visible else 1
            if visible:
                box[t, s] = bb[:4]
                if np.any(bb[4:7] != 0):
                    g[t, s, sg.F_X:sg.F_Z + 1] = bb[4:7]
                    g[t, s, sg.F_U:sg.F_V + 1] = sg.project_points(bb[4:7], rep.K_front, rep.E_front[t])[0]
    return g, box, n_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="/workspace/g/replays")
    ap.add_argument("--out", default="/workspace/g/perceived")
    ap.add_argument("--limit", type=int, default=0, help="only the first N replays (smoke test)")
    ap.add_argument("--vocab", choices=["scene", "open"], default="scene",
                    help="scene: word list from the simulator (privileged, G's run); open: generic + task items")
    args = ap.parse_args()
    import sg_pilot as sg

    files = sorted(Path(args.replays).glob("*_replay.npz"))[: args.limit or None]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(files):
        dest = out / f.name.replace("_replay.npz", ".npz")
        if dest.exists():
            continue
        t0 = time.time()
        rep = sg.load_replay(f)
        sc = rep.scene
        items = []
        if args.vocab == "open":
            cfg, items = open_config(sg, rep.task_goal)
        else:
            cfg = sg.make_robomme_perception_config(sc["n_bins"], sc["color_names"])
        handle = sg.make_robomme_processor(cfg)
        per, box, n_seed = perceive(rep, handle, episode_num=k)
        oracle, _ = sg.oracle_graph(rep)
        meta = {**sc["stress"], "vocab": args.vocab, "task_items": items, "goal": rep.task_goal,
                "classes": handle.classes, "n_bins": sc["n_bins"], "colors": sc["color_names"],  # truth, for scoring
                "cube_to_bin": {str(a): b for a, b in sc["cube_to_bin"].items()}, "T": len(rep.front_rgb),
                "seed_events": n_seed, "seconds": round(time.time() - t0, 1)}
        np.savez_compressed(dest, meta=json.dumps(meta), perceived=per, bbox=box, oracle=oracle,
                            obj_xyz=rep.obj_xyz if rep.obj_xyz is not None else np.zeros(0))
        print(json.dumps({"file": dest.name, "T": meta["T"], "s": meta["seconds"], "seeds": n_seed}), flush=True)
    sg.close_perception_worker()
    print(f"done: {len(files)} episodes in {out}")


if __name__ == "__main__":
    main()
