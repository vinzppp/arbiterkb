"""Fail-closed gate for the G1 pilot of the "memory or policy?" study (225 cells, val, $2 stop).

Writes the --gate-file that stress/common_harness_pod.py requires for --stage pilot.  Passes only if:
the manifest is frozen with the preregistered arms, tasks and per-task episodes; the scripted
executor preflight finished for all 5 tasks; the fairness audit passed and was made from this commit's
policy texts; the endpoint probe returned the exact model; the tree is clean on the research branch.
Free; reads files and git only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "stress")]

from arbiter_kb.experiment_manifest import load_manifest  # noqa: E402

ARMS = ["fh_g", "fh_m", "ld_g", "ld_m", "mm_g", "mm_m", "kb_g", "kb_m", "memi"]
TASKS = ["RouteStick", "SwingXtimes", "VideoUnmaskSwap", "StopCube", "VideoRepick"]
EPISODES = {"RouteStick": [0, 1, 2, 3, 4], "SwingXtimes": [0, 1, 2, 3, 4], "VideoUnmaskSwap": [10, 12, 13, 14, 18],
            "StopCube": [0, 1, 2, 3, 4], "VideoRepick": [10, 11, 12, 13, 14]}
SOURCES = ["stress/mop_policy.py", "arbiter_kb/mop_memory.py", "stress/common_scene.py", "stress/common_episode.py",
           "stress/common_policy.py", "stress/common_harness_pod.py", "arbiter_kb/common_observation.py",
           "arbiter_kb/factorial_memory.py", "params.json"]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def terminal(row: dict) -> str:
    return str((row.get("terminal") or {}).get("status") or "missing")


def matched_shas(spec: dict) -> list[str]:
    issues = []
    spec_mop = importlib.util.spec_from_file_location("mop_policy", ROOT / "stress/mop_policy.py")
    mop = importlib.util.module_from_spec(spec_mop)
    spec_mop.loader.exec_module(mop)
    if hashlib.sha256(mop.GENERIC_POLICY.encode()).hexdigest() != spec["memory_policy"]["generic_policy_sha256"]:
        issues.append("generic policy text changed")
    for task, want in spec["memory_policy"]["matched_policy_sha256"].items():
        text = (ROOT / f"results/memory_or_policy/g0/memi_split/{task}/policy_matched.txt").read_bytes()
        if hashlib.sha256(text).hexdigest() != want:
            issues.append(f"{task} matched policy text changed")
    return issues


def audit(spec: dict, probe: dict, fairness: dict, rows: list[dict]) -> list[str]:
    issues = []
    stage = spec["stages"].get("pilot") or {}
    if not spec.get("frozen"):
        issues.append("manifest is not frozen")
    if spec["arms"] != ARMS or spec["tasks"] != TASKS or stage.get("episodes_by_task") != EPISODES:
        issues.append("arms, tasks or episodes differ from the G1 plan")
    if stage.get("split") != "val" or stage.get("backend") != "api" or set(spec["stages"]) != {"executor_preflight", "pilot"}:
        issues.append("pilot is not val/API only, or another stage is present")
    if sum(len(v) for v in EPISODES.values()) * len(ARMS) != 225:
        issues.append("cell count is not 225")
    if float(spec["validation_analysis_plan"]["api_cost_stop_usd"]) > 2.0:
        issues.append("cost stop above $2")
    if spec["model"] != "openai/gpt-6-luna" or not probe.get("passed") or probe.get("returned_model") != spec["model"]:
        issues.append("endpoint probe did not return the exact model")
    if not fairness.get("passed") or {t["task"] for t in fairness.get("tasks", [])} != set(TASKS):
        issues.append("fairness audit did not pass on all 5 tasks")
    for row in fairness.get("tasks", []):
        want = spec["memory_policy"]["matched_policy_sha256"].get(row["task"], "")
        if row["policy_sha"]["m"] != want[:16]:
            issues.append(f"fairness audit of {row['task']} used a different matched policy text")
        if row["policy_sha"]["g"] != spec["memory_policy"]["generic_policy_sha256"][:16]:
            issues.append(f"fairness audit of {row['task']} used a different generic policy text")
    if {r.get("task") for r in rows} != set(TASKS) or any(terminal(r) in {"missing", "infrastructure_error",
                                                                          "incomplete"} for r in rows):
        issues.append("scripted executor preflight incomplete or blocked")
    issues += matched_shas(spec)
    dirty = [x for x in git("status", "--porcelain", "--untracked-files=all").splitlines()
             if not x[3:].startswith("results/robomme_common/")]
    if git("branch", "--show-current") != spec["research_branch"] or dirty:
        issues.append("wrong branch or dirty source tree")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=ROOT / "experiments/memory_or_policy_g1_v1.json")
    ap.add_argument("--probe", type=Path, required=True, help="scripts/factorial_probe.py output")
    ap.add_argument("--fairness", type=Path, default=ROOT / "results/memory_or_policy/g1_prep/fairness.json")
    ap.add_argument("--preflight", type=Path, required=True, help="executor_preflight run directory")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec, manifest_sha = load_manifest(a.manifest)
    rows = [json.loads(p.read_text()) for p in sorted(a.preflight.rglob("result.json"))]
    issues = audit(spec, json.loads(a.probe.read_text()), json.loads(a.fairness.read_text()), rows)
    result = {"manifest_sha": manifest_sha, "arbiter_commit": git("rev-parse", "HEAD"),
              "audit": {"passed": not issues, "issues": issues},
              "gate": {"decision": "GO" if not issues else "NO_GO", "approved_stages": [] if issues else ["pilot"]},
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
              "preflight_terminal_statuses": {f"{r.get('task')}/{r.get('episode')}": terminal(r) for r in rows}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["gate"]))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
