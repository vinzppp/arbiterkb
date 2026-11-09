"""Process-level watchdog and durable events for RoboMME common-harness cells."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WatchdogSpec:
    episode_wall_s: float
    reset_wall_s: float
    llm_call_wall_s: float
    executor_command_wall_s: float
    poll_interval_s: float
    terminate_grace_s: float
    kill_grace_s: float
    max_high_level_actions: int
    same_action_repeats: int
    alternating_cycles: int

    @classmethod
    def from_manifest(cls, manifest: dict) -> "WatchdogSpec":
        raw = manifest["watchdog"]
        loop = raw["loop_detection"]
        return cls(
            episode_wall_s=float(raw["episode_wall_s"]),
            reset_wall_s=float(raw["reset_wall_s"]),
            llm_call_wall_s=float(raw["llm_call_wall_s"]),
            executor_command_wall_s=float(raw["executor_command_wall_s"]),
            poll_interval_s=float(raw["poll_interval_s"]),
            terminate_grace_s=float(raw["terminate_grace_s"]),
            kill_grace_s=float(raw["kill_grace_s"]),
            max_high_level_actions=int(raw["max_high_level_actions"]),
            same_action_repeats=int(loop["same_action_repeats"]),
            alternating_cycles=int(loop["alternating_cycles"]),
        )

    def phase_limit(self, phase: str) -> float | None:
        return {
            "reset": self.reset_wall_s,
            "llm": self.llm_call_wall_s,
            "executor": self.executor_command_wall_s,
        }.get(phase)


class EventSink:
    """Append fsync'd JSON rows so the supervisor retains pre-timeout state."""

    def __init__(self, path: Path | None):
        self.path = path
        self.started = time.monotonic()
        self.sequence = 0

    def emit(self, event: str, **fields: Any) -> dict:
        self.sequence += 1
        row = {
            "seq": self.sequence,
            "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_s": round(time.monotonic() - self.started, 6),
            "event": event,
            **fields,
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
        return row


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def normalized_command(command: Any) -> str:
    if not isinstance(command, dict):
        return json.dumps(command, sort_keys=True, default=str)
    choice = str(command.get("choice", "")).strip().lower()
    point = command.get("point")
    if isinstance(point, (list, tuple)):
        point = [round(float(value), 2) for value in point]
    return json.dumps({"choice": choice, "point": point}, sort_keys=True)


def detect_loop(commands: list[Any], same_repeats: int, alternating_cycles: int) -> dict | None:
    keys = [normalized_command(command) for command in commands]
    if len(keys) >= same_repeats and len(set(keys[-same_repeats:])) == 1:
        return {"kind": "same_action", "span": same_repeats, "pattern": [keys[-1]]}
    span = alternating_cycles * 2
    if len(keys) >= span:
        tail = keys[-span:]
        if tail[0] != tail[1] and all(value == tail[index % 2] for index, value in enumerate(tail)):
            return {"kind": "alternating_actions", "span": span, "pattern": tail[:2]}
    return None


def terminal_record(status: str, reason: str, phase: str, elapsed: float) -> dict:
    primary_eligible = status in {"success", "task_failure", "timeout"}
    return {
        "status": status,
        "reason": reason,
        "phase": phase,
        "primary_eligible": primary_eligible,
        "primary_success": status == "success" if primary_eligible else None,
        "elapsed_s": round(elapsed, 3),
    }


def terminal_for_outcome(outcome: str, elapsed: float) -> dict:
    if outcome == "success":
        return terminal_record("success", "environment_success", "environment", elapsed)
    if outcome == "decision_budget":
        return terminal_record("timeout", "high_level_action_budget", "episode", elapsed)
    if outcome in {"harness_error"}:
        return terminal_record("infrastructure_error", outcome, "worker", elapsed)
    return terminal_record("task_failure", outcome or "unknown_task_failure", "environment", elapsed)
