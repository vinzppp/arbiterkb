"""Integrity audit helpers for a frozen confirmation result package."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from scripts.common_harness_report import terminal_status

ARBITER_COMMIT = "797894d86cd7559d494b9535fccb5623302b27be"


def find_pair_drops(observation_root: Path) -> list[Path]:
    """Return the preregistered all-arm exclusion records, if any."""
    return sorted(observation_root.rglob("_pair_drops/*.json"))


def _row_issues(row: dict, spec: dict, manifest_sha: str) -> list[str]:
    provenance = row.get("provenance") or {}
    policy = row.get("policy") or {}
    path = row["_path"]
    expected_prompt = spec["memory_policy"]["memi_prompts_sha256"][:16]
    checks = [
        (provenance.get("manifest_sha") == manifest_sha, "manifest mismatch"),
        (provenance.get("arbiter_commit") == ARBITER_COMMIT, "commit mismatch"),
        (provenance.get("arbiter_dirty") is False, "dirty provenance"),
        (provenance.get("sources") == spec["sources"], "source mismatch"),
        (provenance.get("params_sha") == spec["params_sha"], "params mismatch"),
        (provenance.get("prompt_sha") == expected_prompt, "prompt mismatch"),
        (policy.get("model") == spec["model"] and policy.get("effort") == spec["effort"],
         "requested model/effort mismatch"),
        (terminal_status(row) in {"success", "task_failure", "timeout", "policy_failure"},
         "invalid terminal"),
    ]
    returned = {model for model in policy.get("models_returned", []) if model}
    if returned and returned != {spec["model"]}:
        checks.append((False, "returned model mismatch"))
    return [f"{message}: {path}" for passed, message in checks if not passed]


def _group_issues(key: tuple[str, int], group: list[dict], expected_arms: set[str]) -> list[str]:
    issues = []
    if {row["arm"] for row in group} != expected_arms:
        issues.append(f"{key}: incomplete arms")
    digests = {row.get("reset_digest") for row in group}
    if len(digests) != 1 or not group[0].get("reset_digest"):
        issues.append(f"{key}: reset digest missing or differs")
    episode = {json.dumps((row.get("provenance") or {}).get("episode"), sort_keys=True)
               for row in group}
    if len(episode) != 1:
        issues.append(f"{key}: seed/difficulty differs")
    return issues


def audit(rows: list[dict], spec: dict, manifest_sha: str) -> dict:
    """Check cells and frozen provenance before any statistical interpretation."""
    groups = defaultdict(list)
    issues = []
    for row in rows:
        groups[(row["task"], int(row["episode"]))].append(row)
        issues.extend(_row_issues(row, spec, manifest_sha))
    for key, group in groups.items():
        issues.extend(_group_issues(key, group, set(spec["arms"])))
    expected_pairs = {(task, episode) for task in spec["tasks"]
                      for episode in spec["stages"]["confirmation"]["episodes"]}
    if set(groups) != expected_pairs:
        issues.append("task/episode set differs from the frozen manifest")
    root = Path(rows[0]["_path"]).parents[3] if rows else None
    histories = [json.loads(path.read_text())
                 for path in sorted(root.rglob("infrastructure_retry_history.json"))] if root else []
    failed_attempts = sum(len(item.get("failed_attempts") or []) for item in histories)
    pair_drops = find_pair_drops(root) if root else []
    passed = len(rows) == 2250 and len(groups) == 250 and not issues and not pair_drops
    return {"passed": passed, "rows": len(rows), "pairs": len(groups), "issues": issues,
            "infrastructure_failed_attempts": failed_attempts,
            "pair_drop_records": len(pair_drops)}
