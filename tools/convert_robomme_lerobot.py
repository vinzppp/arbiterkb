"""Convert the RoboMME LeRobot dataset tables (data/robomme_lerobot, downloaded from
huggingface.co/datasets/lerobot/robomme) to one small .npz per episode for the proprioception-only tests.

Runs with a Python that has pyarrow (not in the project's dependency whitelist, so this lives in tools/, outside
scripts/quality.py; NOTES records it). The project reads only the .npz files (numpy).

  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -I tools/convert_robomme_lerobot.py [--tasks ...]

Per episode: state [T, 8] (7 joint angles + finger position), is_demo [T], step_idx [T], task (instruction),
simple / grounded subgoal per frame (strings), family (task name).
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "robomme_lerobot")
FAMILIES = [("first target it was previously", "VideoPlaceOrder"), ("second target", "VideoPlaceOrder"),
            ("third target", "VideoPlaceOrder"), ("fourth target", "VideoPlaceOrder"),
            ("same block that was previously", "VideoRepick"), ("place it on the target", "PickXtimes"),
            ("right-side target", "SwingXtimes"), ("into the bin", "BinFill"), ("same manner", "MoveCube"),
            ("button was pressed", "VideoPlaceButton"), ("stop the cube", "StopCube")]


def family(task: str) -> str:
    return next((f for k, f in FAMILIES if k in task), "other")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="VideoPlaceOrder,VideoRepick,PickXtimes")
    a = ap.parse_args()
    want = set(a.tasks.split(","))
    tasks = {r["task_index"]: r["task"] for r in pq.read_table(os.path.join(ROOT, "meta/tasks.parquet")).to_pylist()}
    cols = ["episode_index", "task_index", "is_demo", "state", "step_idx", "simple_subgoal", "grounded_subgoal"]
    eps = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "data/**/*.parquet"), recursive=True)):
        for r in pq.read_table(f, columns=cols).to_pylist():
            if family(tasks[r["task_index"]]) in want:
                eps[r["episode_index"]].append(r)
    out = os.path.join(ROOT, "npz")
    os.makedirs(out, exist_ok=True)
    count = collections.Counter()
    for e, rows in eps.items():
        rows.sort(key=lambda r: r["step_idx"])
        task = tasks[rows[0]["task_index"]]
        fam = family(task)
        np.savez_compressed(os.path.join(out, f"{fam}_{e:04d}.npz"),
                            state=np.array([r["state"] for r in rows], np.float32),
                            is_demo=np.array([bool(r["is_demo"]) for r in rows]),
                            step_idx=np.array([r["step_idx"] for r in rows], np.int32),
                            simple=np.array([r["simple_subgoal"] or "" for r in rows]),
                            grounded=np.array([r["grounded_subgoal"] or "" for r in rows]),
                            meta=json.dumps({"episode": e, "task": task, "family": fam}))
        count[fam] += 1
    print(dict(count), "->", out)


if __name__ == "__main__":
    main()
