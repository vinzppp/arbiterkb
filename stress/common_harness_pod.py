"""Run the frozen four-arm MEMI/ArbiterKB RoboMME Plan comparison.

Requires an official MEMI checkout at the pinned commit with both submodules.
Each arm/episode runs in a fresh process because SAPIEN cannot safely reuse an
environment. API secrets are read only by MEMI's client and never serialized.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]

from arbiter_kb import ledger  # noqa: E402
from arbiter_kb.params import Params  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def manifest(path: Path) -> tuple[dict, str]:
    data = json.loads(path.read_text())
    return data, digest(path)


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


def cell_name(cfg: dict) -> str:
    return cfg["mode"] if cfg["mode"] == "clean" else (
        f"dropout{cfg['dropout']:.2f}" if cfg["mode"] == "dropout" else f"jitter{cfg['jitter_px']:.1f}px")


def _child_args(a, task, episode, arm, cfg):
    return [sys.executable, __file__, "--manifest", str(a.manifest), "--stage", a.stage,
            "--memi-root", str(a.memi_root), "--out", str(a.out), "--one", task, str(episode), arm,
            "--cell", json.dumps(cfg, separators=(",", ":"))]


def run_parent(a, spec, stage):
    finished = 0
    arms = spec["arms"][:1] if stage["backend"] == "scripted" else spec["arms"]
    for cfg in stage["observations"]:
        cell = cell_name(cfg)
        for task in spec["tasks"]:
            for episode in stage["episodes"]:
                for arm in arms:
                    path = result_path(a.out, a.stage, cell, task, episode, arm)
                    if path.exists():
                        finished += 1
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rc = subprocess.run(_child_args(a, task, episode, arm, cfg), cwd=ROOT,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
                    if rc == 4:
                        return rc
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
    sys.path.insert(0, str(a.memi_root))
    from memi.plan.rollout.episode import dumps
    from memi.plan.rollout.policies import PolicySettings
    from common_episode import run_episode
    from common_policy import make_policy

    params = Params.load(ROOT / "params.json")
    settings = PolicySettings(spec["model"], spec["effort"], max_tokens=spec["max_tokens"],
                              timeout=spec["timeout_s"])
    prompts = str(a.memi_root / spec["prompts"])
    policy = make_policy(arm, stage["backend"], task, episode, settings, prompts, params)
    try:
        log = run_episode(task, episode, stage["split"], policy, cfg)
    except Exception as exc:
        from memi.llm import CallFailed
        if isinstance(exc, CallFailed):
            print(f"API call failed; stopping without a scored row: {exc}", flush=True)
            return 4
        raise
    row = log.__dict__
    row["provenance"] = {"manifest_sha": manifest_sha, "arbiter_commit": git(ROOT, "rev-parse", "HEAD"),
                         "arbiter_dirty": source_dirty(), "sources": sources,
                         "params_sha": params.hash(), "prompt_sha": digest(Path(prompts)),
                         "episode": episode_seed(task, episode, stage["split"])}
    path = result_path(a.out, a.stage, cell_name(cfg), task, episode, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1, default=str))
    print(json.dumps({"task": task, "episode": episode, "arm": arm, "outcome": log.outcome,
                      "calls": len((row.get("policy") or {}).get("log") or []), "seconds": log.seconds}), flush=True)
    return 0


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "experiments/robomme_common_v1.json")
    ap.add_argument("--stage", required=True, choices=("executor_preflight", "smoke", "degradation", "heldout"))
    ap.add_argument("--memi-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/workspace/robomme_common"))
    ap.add_argument("--one", nargs=3, metavar=("TASK", "EPISODE", "ARM"), help=argparse.SUPPRESS)
    ap.add_argument("--cell", help=argparse.SUPPRESS)
    return ap.parse_args()


def main():
    a = args()
    spec, manifest_sha = manifest(a.manifest)
    branch = git(ROOT, "branch", "--show-current")
    if branch in ("main", "master") or branch != spec["research_branch"]:
        raise SystemExit(f"refusing branch {branch!r}; expected {spec['research_branch']!r}")
    stage, sources = spec["stages"][a.stage], verify_sources(a.memi_root, spec)
    os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cpu")
    if stage["split"] == "test":
        if not spec.get("frozen") or not ledger.was_frozen(manifest_sha):
            raise SystemExit(f"heldout refused: freeze manifest {manifest_sha} in the ledger first")
        if not a.one:
            ledger.record("READ", manifest_sha, f"RoboMME common harness {a.stage}")
    return run_one(a, spec, stage, manifest_sha, sources) if a.one else run_parent(a, spec, stage)


if __name__ == "__main__":
    raise SystemExit(main())
