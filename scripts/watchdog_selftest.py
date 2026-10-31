"""Dependency-free process-kill self-test for the frozen common watchdog."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.common_supervisor import supervise_cell  # noqa: E402
from arbiter_kb.common_watchdog import WatchdogSpec, detect_loop  # noqa: E402


def frozen_test_spec(**changes):
    values = {"episode_wall_s": 1.0, "reset_wall_s": .15, "llm_call_wall_s": .15,
              "executor_command_wall_s": .15, "poll_interval_s": .01,
              "terminate_grace_s": .05, "kill_grace_s": .2,
              "max_high_level_actions": 20, "same_action_repeats": 3,
              "alternating_cycles": 3}
    values.update(changes)
    return WatchdogSpec(**values)


def hanging_worker(events: Path, phase: str | None):
    code = ["import json,sys,time"]
    if phase:
        row = json.dumps({"event": "phase_start", "phase": phase, "elapsed_s": 0.0})
        code.append(f"open(sys.argv[1],'w').write({row!r}+'\\n')")
    code.append("time.sleep(30)")
    return [sys.executable, "-c", ";".join(code), str(events)]


def check_timeout(root: Path, phase: str | None, want: str):
    name = phase or "episode"
    events, result = root / name / "events.jsonl", root / name / "result.json"
    spec = frozen_test_spec(episode_wall_s=.15) if phase is None else frozen_test_spec()
    supervise_cell(hanging_worker(events, phase), cwd=root, env={}, event_path=events,
                   result_path=result, base_row={"task": "selftest", "arm": "selftest"}, spec=spec)
    row = json.loads(result.read_text())
    assert row["terminal"]["status"] == want
    assert row["terminal"]["phase"] == name
    assert row["watchdog"]["cleanup"] in {"sigterm", "sigkill", "sigterm_race"}
    return {"phase": name, "status": want, "cleanup": row["watchdog"]["cleanup"]}


def main():
    with tempfile.TemporaryDirectory(prefix="arbiter-watchdog-") as directory:
        root = Path(directory)
        checks = [check_timeout(root, "llm", "infrastructure_error"),
                  check_timeout(root, "executor", "timeout"),
                  check_timeout(root, None, "timeout")]
    a, b = {"choice": "a", "point": None}, {"choice": "b", "point": None}
    loop = detect_loop([a, b, a, b, a, b], same_repeats=3, alternating_cycles=3)
    assert loop and loop["kind"] == "alternating_actions"
    print(json.dumps({"status": "PASS", "process_kill_checks": checks,
                      "loop_detection": loop["kind"]}))


if __name__ == "__main__":
    main()
