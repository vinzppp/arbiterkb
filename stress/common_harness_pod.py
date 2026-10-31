"""Run the frozen four-arm MEMI/ArbiterKB RoboMME Plan comparison.

Requires an official MEMI checkout at the pinned commit with both submodules.
Each arm/episode runs in a fresh process because SAPIEN cannot safely reuse an
environment. API secrets are read only by MEMI's client and never serialized.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]

from arbiter_kb import ledger  # noqa: E402
from arbiter_kb.experiment_manifest import digest, load_manifest  # noqa: E402
from arbiter_kb.params import Params  # noqa: E402
from arbiter_kb.common_supervisor import supervise_cell  # noqa: E402
from arbiter_kb.common_watchdog import (  # noqa: E402
    EventSink,
    WatchdogSpec,
    read_events,
    terminal_for_outcome,
    terminal_record,
)


def manifest(path: Path) -> tuple[dict, str]:
    return load_manifest(path)


def git(root: Path, *args) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def source_dirty() -> bool:
    """Generated common-harness results do not make the source tree dirty."""
    lines = git(ROOT, "status", "--porcelain", "--untracked-files=all").splitlines()
    return any(not line[3:].startswith("results/robomme_common/") for line in lines)


def verify_sources(memi_root: Path, spec: dict) -> dict:
    roots = {"memi": memi_root, "robomme": memi_root / "third_party/robomme_benchmark"}
    found = {name: git(path, "rev-parse", "HEAD") for name, path in roots.items()}
    for name, want in spec["sources"].items():
        if name in found and found[name] != want:
            raise SystemExit(f"{name} is {found[name]}, expected {want}")
    module = importlib.util.find_spec("mani_skill")
    if module and module.origin:
        try:
            ms_root = Path(git(Path(module.origin).parent, "rev-parse", "--show-toplevel"))
            found["maniskill"] = git(ms_root, "rev-parse", "HEAD")
        except subprocess.CalledProcessError:
            found["maniskill"] = "installed-wheel-unverifiable"
    if found.get("maniskill") != spec["sources"]["maniskill"]:
        raise SystemExit(f"maniskill is {found.get('maniskill')}, expected pinned source")
    return found


def result_path(out: Path, stage: str, cell: str, task: str, episode: int, arm: str) -> Path:
    return out / stage / cell / f"{task}_ep{episode}_{arm}" / "result.json"


def event_path(out: Path, stage: str, cell: str, task: str, episode: int, arm: str) -> Path:
    return result_path(out, stage, cell, task, episode, arm).with_name("events.jsonl")


def cell_name(cfg: dict) -> str:
    return cfg["mode"] if cfg["mode"] == "clean" else (
        f"dropout{cfg['dropout']:.2f}" if cfg["mode"] == "dropout" else f"jitter{cfg['jitter_px']:.1f}px")


def _child_args(a, task, episode, arm, cfg, events=None):
    args = [sys.executable, __file__, "--manifest", str(a.manifest), "--stage", a.stage,
            "--memi-root", str(a.memi_root), "--out", str(a.out), "--one", task,
            str(episode), arm, "--cell", json.dumps(cfg, separators=(",", ":"))]
    if a.gate_file:
        args.extend(["--gate-file", str(a.gate_file)])
    if events is not None:
        args.extend(["--worker", "--events", str(events)])
        if a.run_id:
            args.extend(["--run-id", a.run_id])
    return args


def row_provenance(a, spec, stage, task, episode, sources):
    params = Params.load(ROOT / "params.json")
    if params.hash() != spec["params_sha"]:
        raise SystemExit(f"params are {params.hash()}, expected {spec['params_sha']}")
    prompts = a.memi_root / spec["prompts"]
    return {
        "manifest_sha": digest(a.manifest),
        "arbiter_commit": git(ROOT, "rev-parse", "HEAD"),
        "arbiter_dirty": source_dirty(),
        "sources": sources,
        "params_sha": params.hash(),
        "prompt_sha": digest(prompts),
        "episode": episode_seed(task, episode, stage["split"]),
    }


def base_result(a, spec, stage, task, episode, arm, cfg, sources):
    return {
        "run_id": a.run_id,
        "task": task,
        "episode": episode,
        "split": stage["split"],
        "arm": arm,
        "observation": dict(cfg),
        "policy": {"model": spec["model"], "effort": spec["effort"], "log": []},
        "provenance": row_provenance(a, spec, stage, task, episode, sources),
    }


def _write_json(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=1, default=str))
    temporary.replace(path)


def _row_cost(row: dict, prices: dict) -> float:
    usage = ((row.get("policy") or {}).get("usage") or {})
    total, cached = int(usage.get("input_tokens", 0)), int(usage.get("cached_tokens", 0))
    written = int(usage.get("cache_write_tokens", 0))
    output = int(usage.get("output_tokens", 0))
    value = ((total - cached - written) * prices["fresh_input"]
             + cached * prices["cached_input"] + written * prices["cache_write"]
             + output * prices["output"])
    return value / 1e6


def enforce_cost_budget(out: Path, spec: dict, stage: dict):
    if stage["backend"] != "api":
        return
    limit = (spec.get("validation_analysis_plan") or {}).get("api_cost_stop_usd")
    if limit is None:
        return
    rows = [json.loads(path.read_text()) for path in out.rglob("result.json")]
    cost = sum(_row_cost(row, spec["prices_per_million"]) for row in rows)
    if cost >= float(limit):
        raise SystemExit(f"recorded API cost ${cost:.6f} reached frozen stop ${float(limit):.6f}")


def run_supervised_one(a, spec, stage, sources, task, episode, arm, cfg):
    cell = cell_name(cfg)
    result = result_path(a.out, a.stage, cell, task, episode, arm)
    events = event_path(a.out, a.stage, cell, task, episode, arm)
    base = base_result(a, spec, stage, task, episode, arm, cfg, sources)
    if result.exists():
        return 0
    enforce_cost_budget(a.out, spec, stage)
    if events.exists():
        partial = read_events(events)
        row = {
            **base,
            "outcome": "incomplete",
            "success": False,
            "seconds": float(partial[-1].get("elapsed_s", 0)) if partial else 0.0,
            "terminal": terminal_record(
                "incomplete", "stale_event_log_without_result", "supervisor", 0.0),
            "watchdog": {"triggered": False, "cleanup": "supervisor_was_interrupted"},
            "partial_events": partial,
            "event_log": str(events),
        }
        _write_json(result, row)
        return 0
    return supervise_cell(
        _child_args(a, task, episode, arm, cfg, events),
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        event_path=events,
        result_path=result,
        base_row=base,
        spec=WatchdogSpec.from_manifest(spec),
    )


def run_parent(a, spec, stage, sources):
    finished = 0
    arms = spec["arms"][:1] if stage["backend"] == "scripted" else spec["arms"]
    for cfg in stage["observations"]:
        cell = cell_name(cfg)
        for task in spec["tasks"]:
            for episode in stage["episodes"]:
                for arm in arms:
                    path = result_path(a.out, a.stage, cell, task, episode, arm)
                    if path.exists():
                        if spec.get("watchdog"):
                            finished += 1
                            continue
                        try:
                            prior = json.loads(path.read_text())
                        except (OSError, json.JSONDecodeError):
                            prior = {}
                        if prior.get("outcome") != "harness_error":
                            finished += 1
                            continue
                        print(f"retrying harness_error: {task} ep{episode} {arm}", flush=True)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if spec.get("watchdog"):
                        rc = run_supervised_one(a, spec, stage, sources, task, episode, arm, cfg)
                    else:
                        rc = subprocess.run(
                            _child_args(a, task, episode, arm, cfg), cwd=ROOT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
                    if rc == 4:
                        return rc
                    if spec.get("watchdog") and path.exists():
                        terminal = (json.loads(path.read_text()).get("terminal") or {}).get("status")
                        if terminal in {"infrastructure_error", "incomplete"}:
                            print(f"stage stopped on {terminal}: {task} ep{episode} {arm}", flush=True)
                            return 4
                    if rc and task not in spec.get("optional_tasks", []):
                        print(f"child failed rc={rc}: {task} ep{episode} {arm}", flush=True)
    print(f"complete or already present: {finished} prior rows; output {a.out / a.stage}")
    return 0


def episode_seed(task: str, episode: int, split: str):
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(env_id=task, dataset=split, action_space="multi_choice", gui_render=False)
    n = int(builder.get_episode_num())
    if episode >= n:
        raise IndexError(f"{task} {split} has {n} episodes, not {episode + 1}")
    seed, difficulty = builder.resolve_episode(episode)
    return {"seed": seed, "difficulty": difficulty, "episode_count": n}


def run_one(a, spec, stage, manifest_sha, sources):
    task, episode, arm = a.one[0], int(a.one[1]), a.one[2]
    cfg = json.loads(a.cell)
    events = EventSink(a.events)
    sys.path.insert(0, str(a.memi_root))
    from memi.llm import client as llm_client
    from memi.plan.rollout.policies import PolicySettings
    from common_episode import run_episode
    from common_policy import make_policy

    params = Params.load(ROOT / "params.json")
    if params.hash() != spec["params_sha"]:
        raise SystemExit(f"params are {params.hash()}, expected {spec['params_sha']}")
    llm_policy = spec.get("llm_policy", {})
    llm_client.RETRIES = max(int(llm_policy.get("max_attempts", 31)) - 1, 0)
    llm_client.BACKOFF_CAP = float(llm_policy.get("backoff_cap_s", 120))
    settings = PolicySettings(spec["model"], spec["effort"], max_tokens=spec["max_tokens"],
                              timeout=float(llm_policy.get("request_timeout_s", spec["timeout_s"])))
    prompts = str(a.memi_root / spec["prompts"])
    policy = make_policy(arm, stage["backend"], task, episode, settings, prompts, params)
    try:
        watchdog = spec.get("watchdog", {})
        log = run_episode(
            task,
            episode,
            stage["split"],
            policy,
            cfg,
            event_sink=events,
            max_decisions=watchdog.get("max_high_level_actions"),
            loop_detection=watchdog.get("loop_detection"),
        )
    except Exception as exc:
        from memi.llm import CallFailed
        if isinstance(exc, CallFailed):
            events.emit("terminal_error", phase="llm", reason=f"CallFailed: {exc}"[:400])
            print(f"API call failed; stopping without a scored row: {exc}", flush=True)
            return 4
        events.emit("terminal_error", phase="worker",
                    reason=f"{type(exc).__name__}: {exc}"[:400])
        raise
    row = log.__dict__
    row["run_id"] = a.run_id
    row["provenance"] = {
        "manifest_sha": manifest_sha,
        "arbiter_commit": git(ROOT, "rev-parse", "HEAD"),
        "arbiter_dirty": source_dirty(),
        "sources": sources,
        "params_sha": params.hash(),
        "prompt_sha": digest(Path(prompts)),
        "episode": episode_seed(task, episode, stage["split"]),
    }
    row["terminal"] = terminal_for_outcome(log.outcome, log.seconds)
    row["watchdog"] = {"triggered": False}
    row["event_log"] = str(a.events) if a.events else ""
    path = result_path(a.out, a.stage, cell_name(cfg), task, episode, arm)
    _write_json(path, row)
    print(json.dumps({"task": task, "episode": episode, "arm": arm, "outcome": log.outcome,
                      "calls": len((row.get("policy") or {}).get("log") or []), "seconds": log.seconds}), flush=True)
    return 0


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "experiments/robomme_common_v1.json")
    ap.add_argument("--stage", required=True,
                    choices=("executor_preflight", "smoke", "validation", "degradation", "heldout"))
    ap.add_argument("--memi-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/workspace/robomme_common"))
    ap.add_argument("--run-id")
    ap.add_argument("--one", nargs=3, metavar=("TASK", "EPISODE", "ARM"), help=argparse.SUPPRESS)
    ap.add_argument("--cell", help=argparse.SUPPRESS)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--events", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--gate-file", type=Path)
    return ap.parse_args()


def prepare_run(a, spec, manifest_sha):
    if not a.run_id:
        raise SystemExit("v2 watchdog runs require a fresh explicit --run-id")
    if "/" in a.run_id or ".." in a.run_id:
        raise SystemExit("run id must be one path-safe component")
    a.out = a.out / a.run_id
    metadata = a.out / "run.json"
    expected = {"run_id": a.run_id, "manifest_sha": manifest_sha, "model": spec["model"]}
    if metadata.exists():
        found = json.loads(metadata.read_text())
        if any(found.get(key) != value for key, value in expected.items()):
            raise SystemExit(f"run metadata mismatch at {metadata}")
        return
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"refusing non-empty run directory without metadata: {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    expected["created_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    expected["research_branch"] = spec["research_branch"]
    _write_json(metadata, expected)


def verify_api_transport(spec, stage):
    if stage["backend"] != "api":
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is absent; refusing paid/API stage")
    transport = spec.get("transport") or {}
    expected = transport.get("api_url")
    actual = os.environ.get("OPENAI_API_URL")
    if expected and actual != expected:
        raise SystemExit(f"OPENAI_API_URL must exactly match frozen transport {expected!r}")


def verify_stage_gate(a, stage, manifest_sha):
    required = stage.get("requires_gate")
    if not required:
        return
    if not a.gate_file or not a.gate_file.exists():
        raise SystemExit(f"stage requires {required!r}; provide its frozen --gate-file")
    gate = json.loads(a.gate_file.read_text())
    approved = (gate.get("gate") or {}).get("approved_stages") or []
    if gate.get("manifest_sha") != manifest_sha or not (gate.get("audit") or {}).get("passed"):
        raise SystemExit("gate manifest/audit does not authorize this stage")
    if a.stage not in approved:
        raise SystemExit(f"gate does not approve stage {a.stage!r}")


def main():
    a = args()
    spec, manifest_sha = manifest(a.manifest)
    branch = git(ROOT, "branch", "--show-current")
    if branch in ("main", "master") or branch != spec["research_branch"]:
        raise SystemExit(f"refusing branch {branch!r}; expected {spec['research_branch']!r}")
    stage, sources = spec["stages"][a.stage], verify_sources(a.memi_root, spec)
    os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cpu")
    verify_stage_gate(a, stage, manifest_sha)
    verify_api_transport(spec, stage)
    if spec.get("watchdog") and not a.worker:
        prepare_run(a, spec, manifest_sha)
    if stage["split"] == "test":
        if not spec.get("frozen") or not ledger.was_frozen(manifest_sha):
            raise SystemExit(f"heldout refused: freeze manifest {manifest_sha} in the ledger first")
        if not a.one:
            ledger.record("READ", manifest_sha, f"RoboMME common harness {a.stage}")
    if a.one:
        if spec.get("watchdog") and not a.worker:
            task, episode, arm = a.one[0], int(a.one[1]), a.one[2]
            return run_supervised_one(
                a, spec, stage, sources, task, episode, arm, json.loads(a.cell))
        return run_one(a, spec, stage, manifest_sha, sources)
    return run_parent(a, spec, stage, sources)


if __name__ == "__main__":
    raise SystemExit(main())
