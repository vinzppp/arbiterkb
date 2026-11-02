import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arbiter_kb.confirmation_retry import (
    execute_with_retries,
    infrastructure_reason,
    mark_existing_pair_rows,
    pair_signature,
    record_pair_drop,
    retry_identity,
)


MODEL = "openai/gpt-6-luna"
ARMS = ["fh_g", "fh_m", "ld_g", "ld_m", "mm_g", "mm_m", "kb_g", "kb_m", "memi"]
HARNESS_SPEC = importlib.util.spec_from_file_location(
    "confirmation_common_harness", Path(__file__).resolve().parents[1] / "stress/common_harness_pod.py")
HARNESS = importlib.util.module_from_spec(HARNESS_SPEC)
HARNESS_SPEC.loader.exec_module(HARNESS)


def row(status="success", reason="environment_success", phase="environment", *, model=MODEL):
    value = {
        "task": "Task", "episode": 7, "split": "test", "arm": "fh_g",
        "observation": {"mode": "clean", "seed": 73301},
        "policy": {"model": MODEL, "effort": "low", "models_returned": [model], "log": []},
        "provenance": {
            "manifest_sha": "manifest", "arbiter_commit": "commit", "arbiter_dirty": False,
            "sources": {"memi": "m", "robomme": "r"}, "params_sha": "params",
            "prompt_sha": "prompt", "episode": {"seed": 123, "difficulty": "easy", "episode_count": 50},
            "factorial": {"x": 1},
        },
        "reset_digest": "reset",
        "terminal": {"status": status, "reason": reason, "phase": phase,
                     "primary_eligible": status in {"success", "task_failure", "timeout"}},
    }
    value["retry_identity"] = retry_identity(value)
    return value


def writer(result: Path, rows: list[dict]):
    calls = {"n": 0}

    def run():
        value = rows[calls["n"]]
        calls["n"] += 1
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps(value))

    return calls, run


def test_successful_infrastructure_retry(tmp_path):
    result, events = tmp_path / "result.json", tmp_path / "events.jsonl"
    failed = row("infrastructure_error", "CallFailed: retries exhausted", "llm")
    succeeded = row()
    calls, run = writer(result, [failed, succeeded])
    outcome = execute_with_retries(
        result_path=result, event_path=events, expected_model=MODEL,
        expected_identity=failed["retry_identity"], reference=None, max_reruns=2, run_once=run)
    assert calls["n"] == 2
    assert outcome.exhausted_reason is None
    assert outcome.reruns == 1
    assert outcome.row["terminal"]["status"] == "success"
    assert (tmp_path / "infrastructure_attempts/attempt_01/result.json").exists()


def test_infrastructure_failure_exhausts_after_two_reruns(tmp_path):
    result, events = tmp_path / "result.json", tmp_path / "events.jsonl"
    failures = [row("infrastructure_error", "executor_wall_timeout: observed>300.000s", "executor")
                for _ in range(3)]
    calls, run = writer(result, failures)
    outcome = execute_with_retries(
        result_path=result, event_path=events, expected_model=MODEL,
        expected_identity=failures[0]["retry_identity"], reference=None, max_reruns=2, run_once=run)
    assert calls["n"] == 3
    assert outcome.reruns == 2
    assert outcome.exhausted_reason == "executor_timeout"
    assert (tmp_path / "infrastructure_attempts/attempt_01/result.json").exists()
    assert (tmp_path / "infrastructure_attempts/attempt_02/result.json").exists()


def test_first_arm_retry_rejects_changed_reset_digest(tmp_path):
    result, events = tmp_path / "result.json", tmp_path / "events.jsonl"
    first = row("infrastructure_error", "CallFailed: retries exhausted", "llm")
    changed = row()
    changed["reset_digest"] = "different-reset"
    calls, run = writer(result, [first, changed])
    with pytest.raises(RuntimeError, match="pair signature changed across reruns"):
        execute_with_retries(
            result_path=result, event_path=events, expected_model=MODEL,
            expected_identity=first["retry_identity"], reference=None,
            max_reruns=2, run_once=run)
    history = json.loads((tmp_path / "infrastructure_retry_history.json").read_text())
    assert calls["n"] == 2
    assert history["first_attempt_pair_signature"] == pair_signature(first)


def test_exhaustion_records_all_arm_pair_drop(tmp_path):
    failed = row("infrastructure_error", "CallFailed: retries exhausted", "llm")
    failed["infrastructure_retry"] = {"reruns_used": 2}
    from arbiter_kb.confirmation_retry import RetryOutcome
    exhausted = RetryOutcome(failed, 2, "api_failure_after_client_retries")
    drop = tmp_path / "_pair_drops/Task_ep7.json"
    paths = [tmp_path / f"Task_ep7_{arm}/result.json" for arm in ARMS]
    for path in paths[:2]:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row()))
    record = record_pair_drop(drop, task="Task", episode=7, arms=ARMS,
                              failed_arm="fh_g", exhausted=exhausted,
                              reference=pair_signature(failed))
    mark_existing_pair_rows(paths, drop, record)
    assert json.loads(drop.read_text())["all_arms_excluded"] == ARMS
    assert all(json.loads(path.read_text())["pair_exclusion"]["all_arms_excluded"] == ARMS
               for path in paths[:2])
    assert all(not path.exists() for path in paths[2:])


@pytest.mark.parametrize(
    ("status", "reason", "phase"),
    [
        ("task_failure", "api_error", "environment"),
        ("task_failure", "fail", "environment"),
        ("timeout", "high_level_action_budget", "episode"),
        ("timeout", "episode_wall_timeout: observed>1800.000s", "episode"),
    ],
)
def test_policy_task_and_episode_limits_are_never_retried(tmp_path, status, reason, phase):
    result, events = tmp_path / "result.json", tmp_path / "events.jsonl"
    terminal = row(status, reason, phase)
    calls, run = writer(result, [terminal])
    outcome = execute_with_retries(
        result_path=result, event_path=events, expected_model=MODEL,
        expected_identity=terminal["retry_identity"], reference=None, max_reruns=2, run_once=run)
    assert calls["n"] == 1
    assert outcome.reruns == 0
    assert outcome.exhausted_reason is None


def test_wrong_model_and_pair_mismatches_are_infrastructure():
    assert infrastructure_reason(row(model="wrong/model"), MODEL) == "wrong_model_id"
    reference = pair_signature(row())
    reset_bad = row()
    reset_bad["reset_digest"] = "different"
    assert infrastructure_reason(reset_bad, MODEL, reference) == "reset_mismatch"
    seed_bad = row()
    seed_bad["provenance"]["episode"]["seed"] = 999
    assert infrastructure_reason(seed_bad, MODEL, reference) == "seed_mismatch"


def test_runner_drops_exhausted_pair_and_resume_skips_every_arm(tmp_path, monkeypatch):
    stage = {
        "backend": "api", "split": "test", "episodes": [7],
        "observations": [{"mode": "clean"}],
    }
    spec = {
        "arms": ARMS, "tasks": ["Task"], "model": MODEL,
        "infrastructure_retry": {"enabled_stages": ["confirmation"],
                                 "max_reruns_per_cell": 2},
    }
    args = SimpleNamespace(out=tmp_path, stage="confirmation")
    calls = {arm: 0 for arm in ARMS}

    def fake_base(_a, _spec, _stage, _task, _episode, arm, cfg, _sources):
        value = row()
        value["arm"], value["observation"] = arm, dict(cfg)
        value["retry_identity"] = retry_identity(value)
        return value

    def fake_run(_a, _spec, _stage, _sources, task, episode, arm, cfg):
        calls[arm] += 1
        value = fake_base(_a, _spec, _stage, task, episode, arm, cfg, _sources)
        if arm == "ld_g":
            value["terminal"] = {
                "status": "infrastructure_error", "reason": "CallFailed: retries exhausted",
                "phase": "llm", "primary_eligible": False,
            }
        path = HARNESS.result_path(_a.out, _a.stage, HARNESS.cell_name(cfg), task, episode, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return 0

    monkeypatch.setattr(HARNESS, "base_result", fake_base)
    monkeypatch.setattr(HARNESS, "run_supervised_one", fake_run)
    assert HARNESS.run_parent(args, spec, stage, {}) == 0
    assert calls == {"fh_g": 1, "fh_m": 1, "ld_g": 3,
                     "ld_m": 0, "mm_g": 0, "mm_m": 0,
                     "kb_g": 0, "kb_m": 0, "memi": 0}
    drop = tmp_path / "confirmation/clean/_pair_drops/Task_ep7.json"
    assert json.loads(drop.read_text())["all_arms_excluded"] == ARMS
    for arm in ("fh_g", "fh_m", "ld_g"):
        path = HARNESS.result_path(tmp_path, "confirmation", "clean", "Task", 7, arm)
        assert json.loads(path.read_text())["pair_exclusion"]["all_arms_excluded"] == ARMS
    for arm in ARMS[3:]:
        assert not HARNESS.result_path(
            tmp_path, "confirmation", "clean", "Task", 7, arm).exists()
    assert HARNESS.run_parent(args, spec, stage, {}) == 0
    assert sum(calls.values()) == 5
