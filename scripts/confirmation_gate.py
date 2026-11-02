"""Fail-closed gate for the memory-or-policy held-out confirmation run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "stress"), str(ROOT / "scripts")]

from arbiter_kb.experiment_manifest import load_manifest  # noqa: E402

ARMS = ["fh_g", "fh_m", "ld_g", "ld_m", "mm_g", "mm_m", "kb_g", "kb_m", "memi"]
TASKS = ["RouteStick", "SwingXtimes", "VideoUnmaskSwap", "StopCube", "VideoRepick"]
EPISODES = list(range(50))
PREFLIGHT_EPISODES = {
    "RouteStick": [49], "SwingXtimes": [49], "VideoUnmaskSwap": [0],
    "StopCube": [49], "VideoRepick": [0],
}
SOURCES = [
    "stress/mop_policy.py", "arbiter_kb/mop_memory.py", "stress/common_scene.py",
    "stress/common_episode.py", "stress/common_policy.py", "stress/common_harness_pod.py",
    "arbiter_kb/common_supervisor.py", "arbiter_kb/common_watchdog.py",
    "arbiter_kb/confirmation_retry.py", "arbiter_kb/common_observation.py",
    "params.json", "experiments/memory_or_policy_confirmation_v1.json",
    "paper/STOPCUBE_EXECUTOR_AUDIT.json",
]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def terminal(row: dict) -> str:
    return str((row.get("terminal") or {}).get("status") or "missing")


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route_memory_issues(probe_dir: Path) -> list[str]:
    mop_fairness = load_local("confirmation_mop_fairness", ROOT / "scripts/mop_fairness.py")
    paths = sorted(probe_dir.glob("RouteStick_*.json"))
    if len(paths) != 1:
        return ["RouteStick fairness probe is missing or ambiguous"]
    probe = json.loads(paths[0].read_text())
    issues = []
    for arm in ("ld_g", "ld_m", "kb_g", "kb_m"):
        calls = (probe.get("prompts") or {}).get(arm) or []
        payload = mop_fairness.parts(arm, calls[0])["payload"] if calls else ""
        if len(payload.strip()) < 100 or not any(word in payload.lower() for word in ("cube", "stick")):
            issues.append(f"RouteStick {arm} memory payload is empty or lacks task objects")
    return issues


def prompt_source_issues(spec: dict) -> list[str]:
    mop_policy = load_local("confirmation_mop_policy", ROOT / "stress/mop_policy.py")
    issues = []
    if hashlib.sha256(mop_policy.GENERIC_POLICY.encode()).hexdigest() \
            != spec["memory_policy"]["generic_policy_sha256"]:
        issues.append("generic policy text changed")
    for task, want in spec["memory_policy"]["matched_policy_sha256"].items():
        path = ROOT / f"results/memory_or_policy/g0/memi_split/{task}/policy_matched.txt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != want:
            issues.append(f"{task} matched policy text changed")
    return issues


def expected_preflight_keys(spec: dict) -> list[tuple[str, int, str, str]]:
    arm = spec["arms"][0]
    return [(task, episode, "val", arm)
            for task in TASKS for episode in PREFLIGHT_EPISODES[task]]


def _model_call_evidence(policy: dict) -> bool:
    if policy.get("log") or policy.get("models_returned") or policy.get("calls"):
        return True
    usage = policy.get("usage") or {}
    return any(value not in (None, 0, 0.0, "", [], {}) for value in usage.values())


def preflight_provenance_issues(spec: dict, manifest_sha: str,
                                arbiter_commit: str, rows: list[dict]) -> list[str]:
    expected = expected_preflight_keys(spec)
    found = [(row.get("task"), row.get("episode"), row.get("split"), row.get("arm"))
             for row in rows]
    issues = []
    stage = spec.get("stages", {}).get("executor_preflight") or {}
    if stage.get("split") != "val" or stage.get("backend") != "scripted" \
            or stage.get("episodes") != [] or stage.get("episodes_by_task") != PREFLIGHT_EPISODES:
        issues.append("manifest executor preflight is not the exact five scheduled val rows")
    if Counter(found) != Counter(expected):
        issues.append(f"preflight rows differ from the exact scheduled five: {found!r}")
    if len(found) != len(set(found)):
        issues.append("preflight contains duplicate task/episode/split/arm rows")
    prompt_sha = spec["memory_policy"]["memi_prompts_sha256"][:16]
    for index, row in enumerate(rows):
        label = "/".join(map(str, found[index]))
        provenance, policy = row.get("provenance") or {}, row.get("policy") or {}
        checks = {
            "manifest SHA": provenance.get("manifest_sha") == manifest_sha,
            "Arbiter commit": provenance.get("arbiter_commit") == arbiter_commit,
            "clean flag": provenance.get("arbiter_dirty") is False,
            "pinned sources": provenance.get("sources") == spec.get("sources"),
            "params hash": provenance.get("params_sha") == spec.get("params_sha"),
            "prompt hash": provenance.get("prompt_sha") == prompt_sha,
            "zero model calls": policy.get("model") == "none"
                                and policy.get("effort") == "none"
                                and not _model_call_evidence(policy),
            "nonempty reset digest": bool(str(row.get("reset_digest") or "").strip()),
        }
        issues.extend(f"preflight {label} has invalid {name}"
                      for name, passed in checks.items() if not passed)
        if terminal(row) in {"missing", "infrastructure_error", "incomplete"}:
            issues.append(f"preflight {label} has blocked terminal status")
    return issues


def audit(spec: dict, probe: dict, fairness: dict, rows: list[dict],
          probe_dir: Path, stopcube: dict, manifest_sha: str,
          arbiter_commit: str) -> list[str]:
    issues, stage = [], spec.get("stages", {}).get("confirmation") or {}
    retry = spec.get("infrastructure_retry") or {}
    plan = spec.get("validation_analysis_plan") or {}
    if not spec.get("frozen"):
        issues.append("manifest is still a draft; freeze and ledger-record it only after review")
    if "PENDING" in str(spec.get("execution_host")) or "PENDING" in str(plan.get("environment_selection")):
        issues.append("execution host and host-specific flags remain undecided")
    if spec.get("arms") != ARMS or spec.get("tasks") != TASKS:
        issues.append("arms or tasks differ from the confirmation plan")
    if stage.get("split") != "test" or stage.get("backend") != "api" or stage.get("episodes") != EPISODES:
        issues.append("confirmation is not exactly test 0-49 with the API backend")
    if set(spec.get("stages") or {}) != {"executor_preflight", "confirmation"}:
        issues.append("unexpected stage in confirmation manifest")
    if int(plan.get("cells", -1)) != 2250 or float(plan.get("api_cost_stop_usd", -1)) != 6.0:
        issues.append("cell count or $6 cost stop differs")
    if spec.get("model") != "openai/gpt-6-luna" or spec.get("effort") != "low" \
            or int(spec.get("watchdog", {}).get("max_high_level_actions", -1)) != 50:
        issues.append("model, effort or 50-call cap differs")
    if retry.get("enabled_stages") != ["confirmation"] or retry.get("max_reruns_per_cell") != 2:
        issues.append("strict two-rerun confirmation policy is absent")
    if not probe.get("passed") or probe.get("returned_model") != spec.get("model"):
        issues.append("endpoint probe did not return the exact model")
    fair_tasks = {row.get("task") for row in fairness.get("tasks", [])}
    if not fairness.get("passed") or fair_tasks != set(TASKS):
        issues.append("real-prompt fairness did not pass all five tasks")
    issues += preflight_provenance_issues(spec, manifest_sha, arbiter_commit, rows)
    if stopcube.get("conclusion") != "official_no_op_matches_harness" \
            or stopcube.get("official_source", {}).get("memi_commit") != spec.get("sources", {}).get("memi"):
        issues.append("StopCube audit is missing or does not match the pinned MEMI source")
    issues += route_memory_issues(probe_dir)
    issues += prompt_source_issues(spec)
    dirty = [line for line in git("status", "--porcelain", "--untracked-files=all").splitlines()
             if not line[3:].startswith("results/robomme_common/")]
    if git("branch", "--show-current") != spec.get("research_branch") or dirty:
        issues.append("wrong branch or dirty source tree")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "experiments/memory_or_policy_confirmation_v1.json")
    ap.add_argument("--probe", type=Path, required=True, help="scripts/factorial_probe.py output")
    ap.add_argument("--probe-dir", type=Path, required=True, help="stress/mop_probe.py outputs")
    ap.add_argument("--fairness", type=Path, required=True, help="scripts/mop_fairness.py output")
    ap.add_argument("--preflight", type=Path, required=True, help="executor_preflight run directory")
    ap.add_argument("--stopcube-audit", type=Path,
                    default=ROOT / "paper/STOPCUBE_EXECUTOR_AUDIT.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec, manifest_sha = load_manifest(args.manifest)
    arbiter_commit = git("rev-parse", "HEAD")
    rows = [json.loads(path.read_text()) for path in sorted(args.preflight.rglob("result.json"))]
    issues = audit(spec, json.loads(args.probe.read_text()), json.loads(args.fairness.read_text()),
                   rows, args.probe_dir, json.loads(args.stopcube_audit.read_text()),
                   manifest_sha, arbiter_commit)
    result = {
        "manifest_sha": manifest_sha,
        "arbiter_commit": arbiter_commit,
        "audit": {"passed": not issues, "issues": issues},
        "gate": {"decision": "GO" if not issues else "NO_GO",
                 "approved_stages": ["confirmation"] if not issues else []},
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES},
        "preflight_terminal_statuses": {f"{row.get('task')}/{row.get('episode')}": terminal(row)
                                        for row in rows},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["gate"]))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
