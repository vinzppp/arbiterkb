"""Collect the G0 outputs into results/memory_or_policy/g0/summary.json and compute run hours.

Hours model (stated in the report): per cell = process start + reset (measured, model-free
preflight, Mac Docker) + calls x seconds per call.  Seconds per call = (executor + LLM
phase) / calls of the v2.1 validation (144 cells, same image and Luna endpoint).  Calls per
episode = MEMI's expert decision steps per task (data/decisions, metadata only); the
pessimistic row doubles them and lets 5% of cells run to the 30-min backstop.
Hours = 9 arms x n episodes x mean cell seconds / parallel width.  Free; reads files only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ("RouteStick", "SwingXtimes", "VideoUnmaskSwap", "StopCube", "VideoRepick")
ARMS = 9
V21 = {"cells": 144, "calls": 577, "executor_s_per_cell": 108.75, "llm_s_per_cell": 13.34,
       "source": "paper/ROBOMME_CANONICAL_MEMORY_IDENTIFIABILITY_ZH.md §6; validation report (577 calls)"}
BACKSTOP_S = 1800


def preflight_rows(root: Path) -> dict:
    rows = {}
    for path in sorted(root.glob("*/g0_preflight.json")):
        r = json.loads(path.read_text())
        phases = {p["phase"]: p for p in r["phase_timings"]}
        rows[r["task"]] = {
            "episode": f"{r['split']} {r['episode']}", "outcome": r["outcome"], "decisions": r["decisions"],
            "decision_results": r["decision_results"], "episode_wall_s": r["episode_wall_s"],
            "episode_cpu_s": r["episode_cpu_s"], "cpu_cores": r["cpu_utilisation_cores"],
            "start_s": phases["start"]["wall_s"], "reset_s": phases["reset"]["wall_s"],
            "executor_s": r["decision_seconds"], "capture_object_types": r["capture_object_types"],
            "bridge_classes": r["bridge_classes_visible_frames"],
            "dropped_by_bridge": sorted(set(r["capture_object_types"]) - {"None", "container", "cube", "target"}
                                        - {k for k in r["capture_object_types"] if k.endswith(" cube")}),
            "prompt_chars": {k: v["chars"] for k, v in r["prompts"].items()},
            "kb_raw_progress": r["kb_raw_progress_lines_per_decision"]}
    return rows


def hours(cell_s: dict, n_per_task: int, width: int) -> float:
    return ARMS * n_per_task * sum(cell_s.values()) / width / 3600


def estimate(pre: dict, expert_calls: dict) -> dict:
    per_call = (V21["executor_s_per_cell"] + V21["llm_s_per_cell"]) * V21["cells"] / V21["calls"]
    base = {t: pre[t]["start_s"] + pre[t]["reset_s"] + expert_calls[t] * per_call for t in TASKS}
    pess = {t: 0.95 * (pre[t]["start_s"] + pre[t]["reset_s"] + 2 * expert_calls[t] * per_call)
            + 0.05 * BACKSTOP_S for t in TASKS}
    table = {}
    for name, cells in (("expected", base), ("pessimistic", pess)):
        table[name] = {"cell_s": {t: round(v, 1) for t, v in cells.items()},
                       "mean_cell_s": round(sum(cells.values()) / len(cells), 1),
                       "G1_225_cells_h": {f"width_{w}": round(hours(cells, 5, w), 2) for w in (1, 3, 6, 8, 12)},
                       "confirmation_n250_h": {f"width_{w}": round(hours(cells, 50, w), 1) for w in (1, 3, 6, 8, 12)}}
    return {"seconds_per_call": round(per_call, 1), "v21": V21, "expert_calls": expert_calls, **table}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--g0", type=Path, default=Path("results/memory_or_policy/g0"))
    ap.add_argument("--expert-calls", type=json.loads, required=True, help="task -> MEMI expert decisions/episode")
    ap.add_argument("--parallel", type=json.loads, default={}, help="Mac parallel measurement")
    ap.add_argument("--status", type=json.loads, required=True, help="item -> PASS / FAIL / OPEN with reason")
    a = ap.parse_args()
    pre = preflight_rows(a.g0 / "preflight")
    summary = {
        "gate": "G0, main NOTES 7327cf1; nothing paid, no LLM, no test content",
        "status": a.status,
        "episode_supply": json.loads((a.g0 / "episode_supply.json").read_text()),
        "preflight_serial": pre,
        "preflight_parallel3": preflight_rows(a.g0 / "preflight_parallel3"),
        "mac_parallel": a.parallel,
        "memi_split": json.loads((a.g0 / "memi_split/split_summary.json").read_text()),
        "g2_snapping": json.loads((a.g0 / "g2_snapping.json").read_text()),
        "compute": estimate(pre, a.expert_calls),
    }
    (a.g0 / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["compute"], indent=1))


if __name__ == "__main__":
    main()
