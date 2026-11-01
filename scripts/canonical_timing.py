"""Reconcile recorded factorial and v2.1 validation timings without reruns."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def phase_times(events: list[dict]) -> dict[str, float]:
    starts, totals = {}, {"reset": 0.0, "llm": 0.0, "executor": 0.0}
    for event in events:
        key = (event.get("phase"), event.get("decision"))
        if event.get("event") == "phase_start":
            starts[key] = float(event["elapsed_s"])
        elif event.get("event") == "phase_end" and key in starts:
            phase = str(event.get("phase"))
            if phase in totals:
                totals[phase] += float(event["elapsed_s"]) - starts[key]
    return totals


def memory_after_executor(events: list[dict]) -> float:
    ended, total = {}, 0.0
    for event in events:
        if event.get("event") == "phase_end" and event.get("phase") == "executor":
            ended[event.get("decision")] = float(event["elapsed_s"])
        elif event.get("event") == "memory_transition" and event.get("decision") in ended:
            total += float(event["elapsed_s"]) - ended[event.get("decision")]
    return total


def summarize(name: str, base: Path) -> dict:
    files = sorted(base.glob("**/events.jsonl"))
    cells, utc = [], []
    for path in files:
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not events:
            continue
        times = [parse_utc(e["utc"]) for e in events if e.get("utc")]
        utc.extend(times)
        start = next((e for e in events if e.get("event") == "episode_start"), None)
        end = next((e for e in reversed(events) if e.get("event") == "episode_end"), None)
        result_path = path.with_name("result.json")
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        episode_s = result.get("seconds")
        if episode_s is None and end is not None:
            episode_s = end.get("seconds")
        phases = phase_times(events)
        cells.append({
            "cell": path.parent.name,
            "initialization_proxy_s": float(start["elapsed_s"]) if start else None,
            "episode_s": float(episode_s) if episode_s is not None else None,
            **{f"{k}_s": v for k, v in phases.items()},
            "post_executor_observation_and_memory_s": memory_after_executor(events),
        })
    def stats(key: str) -> dict:
        vals = np.asarray([c[key] for c in cells if c.get(key) is not None], float)
        return {"n": int(len(vals)), "sum": float(vals.sum()), "mean": float(vals.mean()),
                "median": float(np.median(vals)), "p90": float(np.percentile(vals, 90))}
    wall = (max(utc) - min(utc)).total_seconds() if utc else None
    return {
        "name": name, "cells": len(cells),
        "recorded_utc_first": min(utc).isoformat() if utc else None,
        "recorded_utc_last": max(utc).isoformat() if utc else None,
        "experiment_event_span_s": wall,
        "components": {key: stats(key) for key in (
            "initialization_proxy_s", "reset_s", "llm_s", "executor_s",
            "post_executor_observation_and_memory_s", "episode_s")},
        "definitions": {
            "initialization_proxy_s": "worker/EventSink start to episode_start; includes imports, simulator/env construction and process overhead",
            "reset_s": "phase_start(reset) to phase_end(reset); includes env reset, demo capture, observation conversion, and initial memory feed",
            "llm_s": "recorded LLM phase; close to API latency but also includes prompt/response handling",
            "executor_s": "env.step(multi_choice), including privileged motion planning and execution",
            "post_executor_observation_and_memory_s": "executor phase end to memory_transition; combines capture, degradation, conversion, and memory update",
            "episode_s": "episode log seconds after environment construction; excludes initialization proxy",
            "experiment_event_span_s": "first to last recorded event; includes sequential gaps but omits pre-first-event startup",
        },
        "cells_detail": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    factorial = ROOT / "results/robomme_common/factorial_v1/pilot_20261101T0453Z/pilot/clean"
    validation = ROOT / "results/robomme_common/watchdog_v2_1/validation_20261031T150046Z/validation/clean"
    result = {
        "schema": 1,
        "source": "recorded events only; no reruns",
        "factorial": summarize("factorial_24_cell", factorial),
        "validation": summarize("v2_1_144_cell", validation),
        "interpretation_limits": [
            "reset does not isolate simulator reset from initial memory processing",
            "post-executor memory timing combines observation capture/conversion with memory update",
            "no GPU utilization or planner-internal timing was logged",
            "event span is not a scheduler-independent benchmark",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "factorial_cells": result["factorial"]["cells"],
        "factorial_event_span_s": result["factorial"]["experiment_event_span_s"],
        "validation_cells": result["validation"]["cells"],
        "validation_event_span_s": result["validation"]["experiment_event_span_s"],
        "factorial_episode_sum_s": result["factorial"]["components"]["episode_s"]["sum"],
        "validation_episode_sum_s": result["validation"]["components"]["episode_s"]["sum"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
