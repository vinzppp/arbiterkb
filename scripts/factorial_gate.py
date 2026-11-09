"""Fail-closed preflight and fairness gate for the frozen 24-cell pilot."""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "stress")]

from arbiter_kb.experiment_manifest import load_manifest  # noqa: E402
POLICY_SPEC = importlib.util.spec_from_file_location("common_policy", ROOT / "stress/common_policy.py")
POLICY = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY)
factorial_policy_sha = POLICY.factorial_policy_sha


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def source_dirty() -> bool:
    lines = git("status", "--porcelain", "--untracked-files=all").splitlines()
    return any(not line[3:].startswith("results/robomme_common/") for line in lines)


def terminal(row: dict) -> str:
    return str((row.get("terminal") or {}).get("status") or "missing")


def source_hashes() -> dict[str, str]:
    paths = ["arbiter_kb/factorial_memory.py", "stress/common_policy.py",
             "stress/common_harness_pod.py", "stress/common_episode.py",
             "paper/ROBOMME_MEMORY_POLICY_FACTORIAL_PROTOCOL_ZH.md", "params.json"]
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def audit(spec: dict, probe: dict, rows: list[dict]) -> list[str]:
    issues = []
    expected_arms = ["ledger_generic", "ledger_matched", "arbiter_generic", "arbiter_matched"]
    if spec["arms"] != expected_arms or spec["tasks"] != ["VideoRepick", "VideoUnmaskSwap"]:
        issues.append("factorial arms/tasks differ from preregistration")
    stage = spec["stages"].get("pilot") or {}
    cells = len(spec["tasks"]) * len(stage.get("episodes", [])) * len(spec["arms"])
    if cells != 24 or stage.get("split") != "val" or stage.get("backend") != "api":
        issues.append("pilot is not exactly 24 validation/API cells")
    if set(spec["stages"]) != {"executor_preflight", "pilot"}:
        issues.append("held-out or degradation stage is present")
    if spec["model"] != "openai/gpt-6-luna" or not probe.get("passed"):
        issues.append("authorized exact-model probe did not pass")
    if probe.get("returned_model") != spec["model"]:
        issues.append("probe returned a different model")
    if len(rows) != 2 or {r.get("task") for r in rows} != set(spec["tasks"]):
        issues.append("scripted preflight is incomplete")
    if any(terminal(r) in {"missing", "infrastructure_error", "incomplete"} for r in rows):
        issues.append("scripted preflight has a terminal blocker")
    expected = spec["factorial"]["policy_sha256"]
    if factorial_policy_sha("VideoRepick", "generic") != expected["generic"]:
        issues.append("generic policy hash changed")
    for task in spec["tasks"]:
        if factorial_policy_sha(task, "matched") != expected[f"{task}/matched"]:
            issues.append(f"{task} matched policy hash changed")
    if float(spec["validation_analysis_plan"]["api_cost_stop_usd"]) != .10:
        issues.append("cost stop is not $0.10")
    if git("branch", "--show-current") != spec["research_branch"] or source_dirty():
        issues.append("source branch is wrong or source tree is dirty")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec, manifest_sha = load_manifest(a.manifest)
    probe = json.loads(a.probe.read_text())
    rows = [json.loads(p.read_text()) for p in sorted(a.preflight.rglob("result.json"))]
    issues = audit(spec, probe, rows)
    result = {"manifest_sha": manifest_sha, "arbiter_commit": git("rev-parse", "HEAD"),
              "audit": {"passed": not issues, "issues": issues},
              "gate": {"decision": "GO" if not issues else "NO_GO",
                       "approved_stages": ["pilot"] if not issues else []},
              "probe": {"returned_model": probe.get("returned_model"),
                        "cost_usd": probe.get("cost_usd")},
              "source_sha256": source_hashes(),
              "preflight_terminal_statuses": [terminal(row) for row in rows]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["gate"]))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
