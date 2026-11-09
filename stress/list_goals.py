"""The goals of a task's episodes, to choose episodes by a rule fixed before any paid run (NOTES "Does K lose track on
two-pick goals?"). ON THE POD, sg-main venv. Free: no LLM, no perception; one reset of a fresh env per episode.

  python stress/list_goals.py --task VideoUnmaskSwap --dataset val --first 10 --last 59 --out /workspace/cl/goals.jsonl

Prints one JSON line per episode (episode, goal, the goal's coloured cubes in order); stops at the first episode the
split does not have.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "stress")]

from arbiter_kb.task_items import parse_task_items  # noqa: E402


def main() -> None:
    from pod_checks import make_env
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="VideoUnmaskSwap")
    ap.add_argument("--dataset", choices=["val", "test"], default="val")
    ap.add_argument("--first", type=int, default=10)
    ap.add_argument("--last", type=int, default=59)
    ap.add_argument("--out", default="/workspace/cl/goals.jsonl")
    a = ap.parse_args()
    if a.dataset == "test":
        sys.exit("goals of the test split are not listed here (a test read needs the freeze and the ledger)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for ep in range(a.first, a.last + 1):
            try:
                env = make_env(a.task, ep, a.dataset)
                _, info = env.reset()
                env.close()
            except Exception as ex:  # noqa: BLE001 (the split has no such episode, or the env failed: stop and say)
                print(json.dumps({"episode": ep, "stopped": repr(ex)[:300]}), flush=True)
                break
            g = info["task_goal"]
            goal = str(g[0] if isinstance(g, (list, tuple)) else g)
            cubes = [it["color"] for it in parse_task_items(goal) if it["noun"] == "cube" and it["color"]]
            line = json.dumps({"task": a.task, "dataset": a.dataset, "episode": ep, "goal": goal, "cubes": cubes})
            print(line, flush=True)
            f.write(line + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
