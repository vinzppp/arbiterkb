"""External process supervisor for one common-harness cell."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

from arbiter_kb.common_watchdog import WatchdogSpec, read_events, terminal_record


def _atomic_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=1, ensure_ascii=False, default=str))
    temporary.replace(path)


def _stop_group(proc: subprocess.Popen, spec: WatchdogSpec) -> str:
    if proc.poll() is not None:
        return "already_exited"
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=spec.terminate_grace_s)
        return "sigterm"
    except ProcessLookupError:
        return "already_exited"
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=spec.kill_grace_s)
        return "sigkill"
    except ProcessLookupError:
        return "sigterm_race"
    except subprocess.TimeoutExpired:
        return "sigkill_unreaped"


def _phase(events: list[dict], started: float) -> tuple[str, float] | None:
    active = None
    for event in events:
        kind, name = event.get("event"), str(event.get("phase", ""))
        if kind == "phase_start":
            active = (name, started + float(event.get("elapsed_s", 0)))
        elif kind == "phase_end" and active and active[0] == name:
            active = None
    return active


def _wait(proc: subprocess.Popen, event_path: Path, started: float,
          spec: WatchdogSpec) -> tuple[str, float, float] | None:
    while proc.poll() is None:
        now = time.monotonic()
        episode_elapsed = now - started
        if episode_elapsed > spec.episode_wall_s:
            return "episode", episode_elapsed, spec.episode_wall_s
        active = _phase(read_events(event_path), started)
        if active:
            phase, phase_started = active
            limit = spec.phase_limit(phase)
            if limit is not None and now - phase_started > limit:
                return phase, now - phase_started, limit
        time.sleep(spec.poll_interval_s)
    return None


def _partial_policy(base: dict, events: list[dict]) -> dict:
    policy = dict(base.get("policy") or {})
    calls = [event["policy_call"] for event in events
             if event.get("event") == "phase_end" and event.get("phase") == "llm"
             and isinstance(event.get("policy_call"), dict)]
    fields = ("input_tokens", "cached_tokens", "cache_write_tokens",
              "output_tokens", "reasoning_tokens")
    usage = {field: sum(int(call.get(field) or 0) for call in calls) for field in fields}
    usage["latency"] = round(sum(float(call.get("latency") or 0) for call in calls), 3)
    usage["retries"] = sum(int(call.get("retries") or 0) for call in calls)
    policy.update({"log": calls, "calls": {"decide": len(calls)}, "usage": usage,
                   "statuses": dict(Counter(str(call.get("status", "unknown")) for call in calls)),
                   "models_returned": sorted({str(call.get("model_returned")) for call in calls
                                              if call.get("model_returned")})})
    return policy


def _timeout_row(base: dict, events: list[dict], timeout: tuple, cleanup: str,
                 elapsed: float, event_path: Path) -> dict:
    phase, observed, limit = timeout
    status = "infrastructure_error" if phase in {"reset", "llm", "executor"} else "timeout"
    reason = f"{phase}_wall_timeout: observed>{limit:.3f}s"
    return {**base, "policy": _partial_policy(base, events), "outcome": status,
            "success": False, "seconds": round(elapsed, 1),
            "terminal": terminal_record(status, reason, phase, elapsed),
            "watchdog": {"triggered": True, "phase": phase, "limit_s": limit,
                         "observed_s": round(observed, 3), "cleanup": cleanup},
            "partial_events": events, "event_log": str(event_path),
            "usage_completeness": "unknown_inflight_call" if phase == "llm" else "completed_calls"}


def _exit_row(base: dict, events: list[dict], returncode: int,
              elapsed: float, event_path: Path) -> dict:
    failure = next((event for event in reversed(events)
                    if event.get("event") == "terminal_error"), {})
    if returncode < 0:
        status, reason = "infrastructure_error", f"container_crash_worker_signal_{-returncode}"
    else:
        status = "infrastructure_error"
        reason = str(failure.get("reason") or f"container_crash_worker_exit_{returncode}_without_result")
    return {**base, "policy": _partial_policy(base, events), "outcome": status,
            "success": False, "seconds": round(elapsed, 1),
            "terminal": terminal_record(status, reason, str(failure.get("phase", "worker")), elapsed),
            "watchdog": {"triggered": False, "cleanup": "worker_exited"},
            "partial_events": events, "event_log": str(event_path),
            "usage_completeness": "partial_worker_trace"}


def supervise_cell(argv: list[str], *, cwd: Path, env: dict[str, str], event_path: Path,
                   result_path: Path, base_row: dict, spec: WatchdogSpec) -> int:
    """Run one cell and kill its process group when a frozen limit expires."""
    if event_path.exists() or result_path.exists():
        raise FileExistsError(f"fresh cell required under {result_path.parent}")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, start_new_session=True)
    timeout = _wait(proc, event_path, started, spec)
    if timeout:
        cleanup = _stop_group(proc, spec)
        events, elapsed = read_events(event_path), time.monotonic() - started
        _atomic_json(result_path, _timeout_row(base_row, events, timeout, cleanup,
                                               elapsed, event_path))
        return 0
    if result_path.exists():
        return int(proc.returncode or 0)
    events, elapsed = read_events(event_path), time.monotonic() - started
    _atomic_json(result_path, _exit_row(base_row, events, int(proc.returncode or 0),
                                        elapsed, event_path))
    return 0
