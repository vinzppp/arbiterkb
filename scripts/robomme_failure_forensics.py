"""Offline, read-only forensics for the frozen RoboMME v2.1 validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common_harness_report import (  # noqa: E402
    _paired,
    load_rows,
    paired_mean_ci,
    primary_eligible,
    primary_success,
    terminal_status,
)
from arbiter_kb.paper_stats import holm_adjust, mcnemar_exact  # noqa: E402

ARMS = ("fullhist", "memi", "soft", "arbiter")
CONTRASTS = (
    ("memi", "fullhist"), ("soft", "fullhist"),
    ("arbiter", "fullhist"), ("arbiter", "memi"),
    ("arbiter", "soft"), ("memi", "soft"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_repick(goal: str) -> tuple[int, str]:
    if "three times" in goal:
        cycles = 3
    elif "two times" in goal:
        cycles = 2
    elif "again" in goal:
        cycles = 1
    else:
        raise ValueError(f"unparsed VideoRepick goal: {goal}")
    return cycles, "AB" * cycles + "C"


def action_path(row: dict) -> str:
    return "".join(str(d.get("command", {}).get("choice", "?")).upper()
                   for d in row.get("decisions") or [])


def first_divergence(row: dict, expected: str) -> dict | None:
    decisions = row.get("decisions") or []
    for index, decision in enumerate(decisions):
        actual = str(decision.get("command", {}).get("choice", "?")).upper()
        if index >= len(expected) or actual != expected[index]:
            return {"step": index, "expected": expected[index] if index < len(expected) else None,
                    "actual": actual, "basis": "choice"}
        if index == 0 and decision.get("status") != "ongoing":
            return {"step": 0, "expected": "A on demonstrated target", "actual": "A rejected",
                    "basis": "environment_status"}
    if len(decisions) < len(expected):
        return {"step": len(decisions), "expected": expected[len(decisions)], "actual": None,
                "basis": "trajectory_ended"}
    return None


def progress_lines(event_rows: list[dict]) -> list[dict]:
    out = []
    for event in event_rows:
        if event.get("event") != "phase_start" or event.get("phase") != "llm":
            continue
        text = ((event.get("memory_before") or {}).get("text") or "")
        match = re.search(r"^Progress: (.+)$", text, re.MULTILINE)
        if match:
            out.append({"decision": event.get("decision"), "progress": match.group(1)})
    return out


def memory_trace(row: dict, event_rows: list[dict]) -> list[dict]:
    if row["arm"] == "memi":
        return [{"decision": call.get("index"), "before": call.get("memory_before"),
                 "after": call.get("memory_after")}
                for call in (row.get("policy") or {}).get("log") or []]
    if row["arm"] == "arbiter":
        return progress_lines(event_rows)
    return []


def repick_forensics(rows: list[dict]) -> list[dict]:
    out = []
    for row in sorted((r for r in rows if r["task"] == "VideoRepick"),
                      key=lambda r: (r["episode"], ARMS.index(r["arm"]))):
        result_path = Path(row["_path"])
        event_path = result_path.with_name("events.jsonl")
        evs = events(event_path)
        cycles, expected = expected_repick(row["goal"])
        decisions = row.get("decisions") or []
        first = decisions[0] if decisions else {}
        divergence = first_divergence(row, expected)
        if row.get("success"):
            diagnosis = "success"
        elif first.get("status") != "ongoing":
            diagnosis = "initial_target_selection_or_grounding"
        elif divergence:
            diagnosis = "task_progress_or_termination_policy"
        else:
            diagnosis = "unresolved_from_trace"
        out.append({
            "episode": row["episode"], "arm": row["arm"], "goal": row["goal"],
            "required_execution_cycles": cycles, "expected_choices": expected,
            "actual_choices": action_path(row), "success": bool(row.get("success")),
            "terminal": row.get("terminal"), "first_action_point": first.get("command", {}).get("point"),
            "first_action_environment_status": first.get("status"),
            "all_decisions_executed": bool(decisions) and all(d.get("result") == "executed" for d in decisions),
            "divergence": divergence, "trace_supported_diagnosis": diagnosis,
            "memory_trace": memory_trace(row, evs),
            "evidence": {"result": str(result_path.relative_to(ROOT)),
                         "events": str(event_path.relative_to(ROOT))},
        })
    return out


def success_matrix(rows: list[dict]) -> list[dict]:
    keyed = {(r["task"], r["episode"], r["arm"]): primary_success(r) for r in rows}
    return [{"task": task, "episode": episode,
             **{arm: bool(keyed[(task, episode, arm)]) for arm in ARMS}}
            for task, episode in sorted({(r["task"], r["episode"]) for r in rows})]


def paired(rows: list[dict]) -> dict:
    tests, raw = {}, {}
    for a, b in CONTRASTS:
        av, bv, keys = _paired(rows, a, b)
        name = f"{a}_vs_{b}"
        tests[name] = {**mcnemar_exact(av, bv), "n_paired": len(keys),
                       "risk_difference_ci95": paired_mean_ci(rows, a, b, primary_success)}
        tests[name]["both_success"] = sum(x and y for x, y in zip(av, bv))
        tests[name]["both_fail"] = sum(not x and not y for x, y in zip(av, bv))
        tests[name]["a_only_cells"] = [list(k) for k, x, y in zip(keys, av, bv) if x and not y]
        tests[name]["b_only_cells"] = [list(k) for k, x, y in zip(keys, av, bv) if y and not x]
        raw[name] = tests[name]["p_exact"]
    adjusted = holm_adjust(raw)
    for name, value in adjusted.items():
        tests[name]["p_holm_six_contrasts"] = value
    return tests


def behavior(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        sample = [r for r in rows if r["arm"] == arm]
        decisions = [d for r in sample for d in (r.get("decisions") or [])]
        calls = [c for r in sample for c in (r.get("policy") or {}).get("log") or []]
        out[arm] = {
            "episodes": len(sample), "policy_calls": len(calls), "completed_decisions": len(decisions),
            "executed_decisions": sum(d.get("result") == "executed" for d in decisions),
            "decision_results": dict(Counter(d.get("result", "missing") for d in decisions)),
            "environment_statuses": dict(Counter(d.get("status", "missing") for d in decisions)),
            "terminal_statuses": dict(Counter(terminal_status(r) for r in sample)),
            "calls_per_episode": dict(sorted(Counter(len((r.get("policy") or {}).get("log") or [])
                                                     for r in sample).items())),
        }
    return out


def loops(rows: list[dict]) -> dict:
    cells, kinds = Counter(), Counter()
    episode_cells = set()
    for row in rows:
        path = Path(row["_path"]).with_name("events.jsonl")
        found = [e for e in events(path) if e.get("event") == "loop_detected"]
        if found:
            key = f'{row["task"]}/{row["episode"]}/{row["arm"]}'
            episode_cells.add(key)
            cells[key] += len(found)
            kinds.update(e.get("kind", "unknown") for e in found)
    return {"diagnostic_only": True, "episodes_with_loop": len(episode_cells),
            "loop_events": sum(cells.values()), "by_kind": dict(kinds), "by_cell": dict(cells)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.root = args.root.resolve()
    args.out = args.out.resolve()
    rows = load_rows(args.root)
    rows = [r for r in rows if primary_eligible(r)]
    source_files = sorted(p for p in args.root.rglob("*") if p.is_file())
    result = {
        "analysis": "frozen RoboMME v2.1 offline failure forensics",
        "source_root": str(args.root.relative_to(ROOT)), "n_cells": len(rows),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        "success_matrix": success_matrix(rows), "paired_tests": paired(rows),
        "behavior": behavior(rows), "loops": loops(rows),
        "video_repick": repick_forensics(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
