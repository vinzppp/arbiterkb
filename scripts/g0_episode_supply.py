"""G0 episode supply for the "memory or policy?" study (main NOTES 7327cf1, "Episodes").

Inputs (free, metadata only): MEMI's ``data/decisions/<Task>.jsonl`` at bb6c37c, RoboMME's
``env_metadata/{train,val,test}`` at fae9f73 (episode index, seed, difficulty), this
branch's result rows, and the main repository's recorded uses (listed below with their
source).  No test episode is reset or run and no test result is read; test metadata is
read only for the index, seed and difficulty columns.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ("RouteStick", "SwingXtimes", "VideoUnmaskSwap", "StopCube", "VideoRepick")
# Main repository (read-only), RoboMME val episodes run or read; source in the note.
MAIN_USED = {
    ("VideoUnmaskSwap", "val", "run"): (list(range(10)) + [11, 15, 16, 17, 19, 23, 24, 25, 27, 29, 31, 32, 35, 37,
                                                           39, 40, 41, 43, 44, 45],
                                        "main NOTES 'GPT-6-Astra VideoUnmaskSwap val pilot' (ep 0-9) and "
                                        "data/cl_pod/pod_backup_2026-10-30/POD_STATE.md 'Astra K on two-pick goals'"),
    ("VideoUnmaskSwap", "val", "goal_read"): (list(range(10, 50)),
                                              "main data/cl_pod/pod_backup_2026-10-30/cl/goals.jsonl (ep 10-59 reset, "
                                              "goal text listed; 50-59 do not exist in val)"),
    ("VideoPlaceOrder", "val", "run"): (list(range(5)), "main NOTES 'Perception check' / phrase test (not a study task)"),
}


def memi_episodes(decisions: Path) -> dict:
    out = {}
    for path in sorted(decisions.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        episodes = sorted(int(r["episode"]) for r in rows)
        out[path.stem] = {"episodes": len(episodes), "min": episodes[0], "max": episodes[-1],
                          "contiguous": episodes == list(range(episodes[0], episodes[-1] + 1)),
                          "duplicates": len(episodes) - len(set(episodes)),
                          "memi_split": dict(Counter(r.get("split") for r in rows))}
    return out


def metadata(root: Path) -> dict:
    out = defaultdict(dict)
    for split in ("train", "val", "test"):
        for path in sorted((root / split).glob("record_dataset_*_metadata.json")):
            records = json.loads(path.read_text())["records"]
            task = path.name[len("record_dataset_"):-len("_metadata.json")]
            out[task][split] = {int(r["episode"]): {"seed": r.get("seed"), "difficulty": r.get("difficulty")}
                                for r in records}
    return out


def branch_used(results: Path) -> dict:
    used = defaultdict(set)
    for path in list(results.rglob("result.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        task, split, ep = row.get("task"), row.get("split"), row.get("episode")
        if task and split and isinstance(ep, int):
            used[(task, split)].add(ep)
    return used


def split_audit(meta: dict) -> dict:
    sizes, overlap = {}, {}
    for task, splits in sorted(meta.items()):
        sizes[task] = {s: len(v) for s, v in splits.items()}
        seeds = {s: {r["seed"] for r in v.values()} for s, v in splits.items()}
        overlap[task] = {"train_val": len(seeds["train"] & seeds["val"]),
                         "train_test": len(seeds["train"] & seeds["test"]), "val_test": len(seeds["val"] & seeds["test"])}
    beyond = {t: sorted(e for e in meta[t]["train"] if e > 99) for t in meta}
    return {"split_sizes": sizes, "seed_overlap_across_splits": overlap,
            "train_beyond_0_99": {t: v for t, v in beyond.items() if v}}


def task_supply(task: str, splits: dict, branch: dict, preflight: set) -> dict:
    val = sorted(splits["val"])
    runs = set(branch.get((task, "val"), set())) | set(MAIN_USED.get((task, "val", "run"), ([], ""))[0])
    goal_read = set(MAIN_USED.get((task, "val", "goal_read"), ([], ""))[0]) - runs
    strict = [e for e in val if e not in runs | goal_read | preflight]
    lenient = [e for e in val if e not in runs | preflight]
    test = splits["test"]
    return {
        "val_total": len(val), "val_run_before": sorted(runs), "val_goal_text_read_only": sorted(goal_read),
        "val_used_by_g0_preflight": sorted(preflight),
        "val_unused_strict": strict, "val_unused_if_goal_reads_allowed": lenient,
        "pilot_g1_strict": strict[:5], "pilot_g1_lenient": lenient[:5],
        "test_0_49_metadata_complete": sorted(test) == list(range(50))
        and all(r["seed"] is not None for r in test.values()),
        "test_run_or_read_before": sorted(branch.get((task, "test"), set())),
        "test_difficulty": dict(Counter(r["difficulty"] for r in test.values())),
        "val_difficulty": dict(Counter(r["difficulty"] for r in splits["val"].values())),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memi-decisions", type=Path, required=True)
    ap.add_argument("--env-metadata", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--g0-preflight", type=json.loads, default={},
                    help='episodes consumed by the G0 preflight, e.g. {"RouteStick": [49]}')
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta, branch = metadata(a.env_metadata), branch_used(a.results)
    report = {"memi_decisions": memi_episodes(a.memi_decisions), **split_audit(meta),
              "tasks": {t: task_supply(t, meta[t], branch, set(a.g0_preflight.get(t, []))) for t in TASKS}}
    report["branch_val_runs_all_tasks"] = {f"{t}/{s}": sorted(v) for (t, s), v in sorted(branch.items())}
    report["main_used_sources"] = {f"{t}/{s}/{k}": src for (t, s, k), (_, src) in MAIN_USED.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({t: {"strict": len(v["val_unused_strict"]), "lenient": len(v["val_unused_if_goal_reads_allowed"]),
                          "test_ok": v["test_0_49_metadata_complete"]} for t, v in report["tasks"].items()}))


if __name__ == "__main__":
    main()
