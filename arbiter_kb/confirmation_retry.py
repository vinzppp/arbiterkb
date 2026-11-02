"""Strict infrastructure retries for the memory-or-policy confirmation run.

Only the preregistered failure classes are retryable.  Raw failed attempts are
kept beside the final cell result, and an exhausted failure excludes the whole
(task, episode) pair rather than only the arm that happened to fail.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


RETRYABLE_REASONS = {
    "api_failure_after_client_retries",
    "wrong_model_id",
    "container_crash",
    "executor_timeout",
    "reset_mismatch",
    "seed_mismatch",
}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def retry_identity(row: dict) -> str:
    """Hash every code, seed and input field that must stay fixed on a retry."""
    provenance = row.get("provenance") or {}
    payload = {
        "task": row.get("task"),
        "episode": row.get("episode"),
        "split": row.get("split"),
        "arm": row.get("arm"),
        "observation": row.get("observation"),
        "policy": {key: (row.get("policy") or {}).get(key) for key in ("model", "effort")},
        "manifest_sha": provenance.get("manifest_sha"),
        "arbiter_commit": provenance.get("arbiter_commit"),
        "arbiter_dirty": provenance.get("arbiter_dirty"),
        "sources": provenance.get("sources"),
        "params_sha": provenance.get("params_sha"),
        "prompt_sha": provenance.get("prompt_sha"),
        "episode_seed": provenance.get("episode"),
        "factorial": provenance.get("factorial"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reset_digest(row: dict) -> str:
    if row.get("reset_digest"):
        return str(row["reset_digest"])
    for event in row.get("partial_events") or []:
        if event.get("event") == "phase_end" and event.get("phase") == "reset":
            return str(event.get("reset_digest") or "")
    return ""


def pair_signature(row: dict) -> dict:
    episode = ((row.get("provenance") or {}).get("episode") or {})
    return {
        "seed": episode.get("seed"),
        "difficulty": episode.get("difficulty"),
        "episode_count": episode.get("episode_count"),
        "reset_digest": _reset_digest(row),
    }


def _returned_models(row: dict) -> set[str]:
    policy = row.get("policy") or {}
    models = {str(value) for value in policy.get("models_returned") or [] if value}
    models.update(str(call.get("model_returned")) for call in policy.get("log") or []
                  if isinstance(call, dict) and call.get("model_returned"))
    return models


def infrastructure_reason(row: dict, expected_model: str,
                          reference: dict | None = None) -> str | None:
    """Return one of the six preregistered retry reasons, else ``None``.

    Policy failures, environment/task failures, the 50-call cap and the whole
    episode timeout deliberately fall through as non-retryable.
    """
    returned = _returned_models(row)
    if returned and returned != {expected_model}:
        return "wrong_model_id"

    current = pair_signature(row)
    if reference:
        seed_fields = ("seed", "difficulty", "episode_count")
        if any(reference.get(key) != current.get(key) for key in seed_fields):
            return "seed_mismatch"
        if reference.get("reset_digest") and current.get("reset_digest") \
                and reference["reset_digest"] != current["reset_digest"]:
            return "reset_mismatch"

    terminal = row.get("terminal") or {}
    status = str(terminal.get("status") or "")
    phase = str(terminal.get("phase") or "")
    reason = str(terminal.get("reason") or "")
    if status == "infrastructure_error" and phase == "llm" \
            and (reason.startswith("CallFailed:") or reason.startswith("api_failure_after_client_retries")):
        return "api_failure_after_client_retries"
    if status == "infrastructure_error" and phase == "executor" \
            and reason.startswith("executor_wall_timeout"):
        return "executor_timeout"
    if status == "infrastructure_error" and reason.startswith("container_crash"):
        return "container_crash"
    return None


@dataclass(frozen=True)
class RetryOutcome:
    row: dict
    reruns: int
    exhausted_reason: str | None = None


def _history_path(result_path: Path) -> Path:
    return result_path.parent / "infrastructure_retry_history.json"


def _load_history(result_path: Path) -> dict:
    path = _history_path(result_path)
    if path.exists():
        return json.loads(path.read_text())
    return {"max_reruns": None, "failed_attempts": []}


def _archive_attempt(result_path: Path, event_path: Path, attempt: int) -> None:
    target = result_path.parent / "infrastructure_attempts" / f"attempt_{attempt:02d}"
    target.mkdir(parents=True, exist_ok=False)
    result_path.replace(target / "result.json")
    if event_path.exists():
        event_path.replace(target / "events.jsonl")


def _annotate_result(result_path: Path, row: dict, history: dict,
                     exhausted_reason: str | None = None) -> dict:
    annotated = dict(row)
    annotated["infrastructure_retry"] = {
        "max_reruns": history["max_reruns"],
        "reruns_used": len(history["failed_attempts"]),
        "failed_attempts": list(history["failed_attempts"]),
        "exhausted_reason": exhausted_reason,
    }
    _atomic_json(result_path, annotated)
    return annotated


def execute_with_retries(*, result_path: Path, event_path: Path,
                         expected_model: str, expected_identity: str,
                         reference: dict | None, max_reruns: int,
                         run_once: Callable[[], None]) -> RetryOutcome:
    """Run or resume one cell, retrying only strict infrastructure failures."""
    history = _load_history(result_path)
    history["max_reruns"] = int(max_reruns)
    while True:
        if not result_path.exists():
            run_once()
        if not result_path.exists():
            raise RuntimeError(f"cell attempt produced no result: {result_path}")
        row = json.loads(result_path.read_text())
        found_identity = str(row.get("retry_identity") or retry_identity(row))
        if found_identity != expected_identity:
            raise RuntimeError(
                f"retry identity changed for {result_path}: {found_identity} != {expected_identity}")
        reason = infrastructure_reason(row, expected_model, reference)
        if reason is None:
            return RetryOutcome(_annotate_result(result_path, row, history),
                                len(history["failed_attempts"]))
        if reason not in RETRYABLE_REASONS:
            raise AssertionError(reason)
        if len(history["failed_attempts"]) >= max_reruns:
            annotated = _annotate_result(result_path, row, history, reason)
            return RetryOutcome(annotated, len(history["failed_attempts"]), reason)
        attempt = len(history["failed_attempts"]) + 1
        history["failed_attempts"].append({
            "attempt": attempt,
            "reason": reason,
            "retry_identity": found_identity,
            "terminal": row.get("terminal"),
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        })
        _atomic_json(_history_path(result_path), history)
        _archive_attempt(result_path, event_path, attempt)


def pair_drop_path(run_root: Path, stage: str, cell: str, task: str, episode: int) -> Path:
    return run_root / stage / cell / "_pair_drops" / f"{task}_ep{episode}.json"


def record_pair_drop(path: Path, *, task: str, episode: int, arms: list[str],
                     failed_arm: str, exhausted: RetryOutcome,
                     reference: dict | None) -> dict:
    record = {
        "status": "pair_dropped",
        "task": task,
        "episode": episode,
        "all_arms_excluded": list(arms),
        "failed_arm": failed_arm,
        "reason": exhausted.exhausted_reason,
        "reruns_used": exhausted.reruns,
        "pair_reference": reference,
        "failed_cell_retry_identity": exhausted.row.get("retry_identity"),
        "rule": "one cell failed after initial attempt plus two identical reruns; exclude (task, episode) from every arm",
    }
    _atomic_json(path, record)
    return record


def mark_existing_pair_rows(result_paths: list[Path], drop_path: Path, record: dict) -> None:
    """Keep raw outcomes but make the all-arm analysis exclusion explicit."""
    for path in result_paths:
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        row["pair_exclusion"] = {
            "status": "pair_dropped",
            "record": str(drop_path),
            "reason": record["reason"],
            "all_arms_excluded": record["all_arms_excluded"],
        }
        _atomic_json(path, row)
