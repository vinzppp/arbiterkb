"""Layer 1 on real scenes (NOTES "Layer 1 on real scenes"; arbiter_kb/callout.py). ON THE POD, sg-main venv. No LLM:
scores the first waypoint and the first close of closed-loop episodes already run, against the simulator's poses after
one reset of a fresh env per (task, episode, dataset).

  python stress/score_callout.py --list --runs /workspace/cl/pilot=qwen/qwen3.6-35b-a3b \\
         /workspace/cl/astra_pilot=openai/gpt-6-astra
  python stress/score_callout.py --runs ... --out /workspace/cl/callout

--list prints each episode's objects and right object without scoring (read it first: actor names decide the right
object). Otherwise writes <out>/callout_rows.csv (one row per episode) and <out>/callout_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "stress")]

from arbiter_kb.callout import objects, right_object, score, summarize  # noqa: E402


def episodes(runs: list[str]):
    """(result row, model) for every result.json under each DIR of DIR=MODEL; the row's own model wins."""
    for spec in runs:
        d, _, model = spec.partition("=")
        for f in sorted(Path(d).glob("*/result.json")):
            row = json.loads(f.read_text())
            if "task" in row and "episode" in row:
                yield row, row.get("model") or model or "?", str(f.parent)


def scene(task: str, ep: int, dataset: str, cache: dict) -> tuple[dict, str]:
    """(objects, goal) after one reset of a fresh env (only the first reset gives the episode's scene)."""
    key = (task, ep, dataset)
    if key not in cache:
        from closed_loop_pod import make_env
        from pod_checks import truth
        env = make_env(task, ep, dataset)
        _, info = env.reset()
        goal = info["task_goal"][0] if isinstance(info["task_goal"], (list, tuple)) else info["task_goal"]
        cache[key] = (objects(truth(env.unwrapped)), str(goal))
        env.close()
    return cache[key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="DIR=MODEL (the model when a row has none)")
    ap.add_argument("--out", default="/workspace/cl/callout")
    ap.add_argument("--list", action="store_true", help="print objects and the right object only")
    a = ap.parse_args()
    cache, rows = {}, []
    for row, model, where in episodes(a.runs):
        task, ep, dataset = row["task"], int(row["episode"]), row.get("dataset", "val")
        objs, goal = scene(task, ep, dataset, cache)
        right, how = right_object(goal, objs)
        if a.list:
            if (task, ep, dataset) not in {(r["task"], r["episode"], r["dataset"]) for r in rows}:
                print(json.dumps({"task": task, "episode": ep, "goal": goal, "right": right, "how": how,
                                  "objects": {n: [round(v, 3) for v in p] for n, p in objs.items()}}), flush=True)
            rows.append({"task": task, "episode": ep, "dataset": dataset})
            continue
        r = {"task": task, "episode": ep, "dataset": dataset, "arm": row.get("arm"), "model": model,
             "status": row.get("status"), "end_reason": row.get("end_reason"), "run": where, "how": how,
             **score(row.get("steps") or [], objs, right)}
        rows.append(r)
        print(json.dumps({k: r.get(k) for k in ("task", "episode", "arm", "model", "category", "nearest", "d1",
                                                "d_right", "close_dz")}, default=str), flush=True)
    if a.list:
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("callout_rows.csv", rows), ("callout_summary.csv", summarize(rows))):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
    for s in summarize(rows):
        print(json.dumps(s), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
