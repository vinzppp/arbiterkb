import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from arbiter_kb.common_supervisor import supervise_cell
from arbiter_kb.common_watchdog import WatchdogSpec, detect_loop

HARNESS_SPEC = importlib.util.spec_from_file_location(
    "common_harness_pod", Path(__file__).resolve().parents[1] / "stress/common_harness_pod.py")
HARNESS = importlib.util.module_from_spec(HARNESS_SPEC)
HARNESS_SPEC.loader.exec_module(HARNESS)


def spec(**overrides):
    values = {
        "episode_wall_s": 1.0,
        "reset_wall_s": 0.15,
        "llm_call_wall_s": 0.15,
        "executor_command_wall_s": 0.15,
        "poll_interval_s": 0.01,
        "terminate_grace_s": 0.05,
        "kill_grace_s": 0.2,
        "max_high_level_actions": 20,
        "same_action_repeats": 3,
        "alternating_cycles": 3,
    }
    values.update(overrides)
    return WatchdogSpec(**values)


def hanging_worker(event_path, phase=None):
    code = ["import json,sys,time"]
    if phase:
        code.append(
            "open(sys.argv[1], 'w').write(json.dumps("
            f"{{'event':'phase_start','phase':'{phase}','elapsed_s':0.0}}) + '\\n')"
        )
    code.append("time.sleep(30)")
    return [sys.executable, "-c", ";".join(code), str(event_path)]


@pytest.mark.parametrize(
    ("phase", "terminal_status", "primary_eligible"),
    [("llm", "infrastructure_error", False), ("executor", "timeout", True)],
)
def test_phase_watchdog_kills_blocking_worker(tmp_path, phase, terminal_status, primary_eligible):
    events = tmp_path / phase / "events.jsonl"
    result = events.with_name("result.json")
    started = time.monotonic()
    rc = supervise_cell(
        hanging_worker(events, phase),
        cwd=tmp_path,
        env={},
        event_path=events,
        result_path=result,
        base_row={"task": "Task", "arm": "arm"},
        spec=spec(),
    )
    row = json.loads(result.read_text())
    assert rc == 0
    assert time.monotonic() - started < 2
    assert row["terminal"]["status"] == terminal_status
    assert row["terminal"]["phase"] == phase
    assert row["terminal"]["primary_eligible"] is primary_eligible
    assert row["watchdog"]["cleanup"] in {"sigterm", "sigkill", "sigterm_race"}


def test_episode_watchdog_kills_worker_without_phase_event(tmp_path):
    events = tmp_path / "events.jsonl"
    result = tmp_path / "result.json"
    supervise_cell(
        hanging_worker(events),
        cwd=tmp_path,
        env={},
        event_path=events,
        result_path=result,
        base_row={"task": "Task", "arm": "arm"},
        spec=spec(episode_wall_s=0.15),
    )
    row = json.loads(result.read_text())
    assert row["terminal"]["status"] == "timeout"
    assert row["terminal"]["phase"] == "episode"
    assert row["terminal"]["primary_success"] is False


def test_loop_detection_is_diagnostic_for_same_and_alternating_actions():
    a = {"choice": "A", "point": None}
    b = {"choice": "B", "point": None}
    same = detect_loop([a, a, a], same_repeats=3, alternating_cycles=3)
    alternating = detect_loop([a, b, a, b, a, b], same_repeats=3, alternating_cycles=3)
    assert same == {"kind": "same_action", "span": 3, "pattern": same["pattern"]}
    assert alternating["kind"] == "alternating_actions"
    assert alternating["span"] == 6


def test_validation_gate_is_forwarded_to_worker(tmp_path):
    gate = tmp_path / "smoke-summary.json"
    args = SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        stage="validation",
        memi_root=tmp_path / "memi",
        out=tmp_path / "out",
        gate_file=gate,
        run_id="validation-test",
    )
    child = HARNESS._child_args(
        args, "Task", 1, "fullhist", {"mode": "clean"}, tmp_path / "events.jsonl")
    assert child[child.index("--gate-file") + 1] == str(gate)
    assert child[child.index("--run-id") + 1] == "validation-test"
