import json
from pathlib import Path

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
