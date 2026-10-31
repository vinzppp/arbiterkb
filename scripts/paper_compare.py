"""Reproduce the defensible comparison for the perception/belief paper branch.

Local panel: held-out G VideoUnmaskSwap perception traces, no LLM.  The frozen
repair ladder is raw Bound, causal identity re-linking, detector-only belief
haircut, and ArbiterKB.  MEMI panel: released closed-loop test logs from its
public repository.  The panels are deliberately not pooled: their observations,
models, and executors differ.

  python scripts/paper_compare.py --memi-root /path/to/memi
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from perceived_qa import OCCLUSION, truth  # noqa: E402
from arbiter_kb import ledger  # noqa: E402
from arbiter_kb.paper_stats import (belief_summary, mcnemar_exact, paired_bootstrap,  # noqa: E402
                                    paired_correct, rate_interval)
from arbiter_kb.sources.perceived import PerceivedSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.systems.baselines import TrackerSystem  # noqa: E402
from arbiter_kb.systems.soft_haircut import SoftHaircut  # noqa: E402

SOFT_SCALE = 0.02  # selected on G train and frozen in results/latest/claim3_g/choice.json
POLICIES = ("astra_plan", "luna", "fable", "haiku", "astra_act")
METHODS = ("fullhist", "init", "memi")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def brier_set(probs: dict, truth_ids: list[str], bins: list[str]) -> float:
    target = 1 / len(truth_ids)
    return sum((float(probs.get(k, 0.0)) - (target if k in truth_ids else 0.0)) ** 2 for k in bins)


def system_rows(src: PerceivedSource, make_system, split: str) -> list[dict]:
    rows = []
    for ep in src.episodes(split):
        data, meta = src.load(ep), src.meta(ep)
        system, target = make_system(), truth(data)
        system.reset(meta)
        for frame in src.frames(ep):
            system.step(frame)
        bins = [oid for oid, cls, _ in meta.objects if cls == "container"]
        for question, want in target.items():
            if not question.startswith("q1_"):
                continue
            cube = question[3:]
            dist = system.believe_inside("cube_" + cube)
            pred, p_max = dist.argmax()
            rows.append({"ep": data["name"], "cube": cube, "correct": pred in want,
                         "pred": pred, "truth": want, "p_max": float(p_max),
                         "brier": brier_set(dist.probs, want, bins)})
    return rows


def local_panel(params, split: str) -> tuple[dict, dict]:
    if params.rule.r6_feature != "min_3d" or not np.isclose(params.rule.d0, SOFT_SCALE):
        raise RuntimeError("paper comparison requires the frozen min_3d/0.02 choice")
    relinked = PerceivedSource(**OCCLUSION, relinked=True)
    configs = {
        "raw_bound": (PerceivedSource(**OCCLUSION), lambda: TrackerSystem("bound", params)),
        "relink_bound": (relinked, lambda: TrackerSystem("bound", params)),
        "relink_soft": (relinked, lambda: SoftHaircut(params, "min_3d")),
        "arbiterkb": (relinked, lambda: ArbiterKBSystem(params)),
    }
    rows = {name: system_rows(src, factory, split) for name, (src, factory) in configs.items()}
    table = {name: belief_summary(rs, params.monitor.tau_hi) for name, rs in rows.items()}
    pairs = {}
    for base in ("raw_bound", "relink_bound", "relink_soft"):
        a, b = paired_correct(rows["arbiterkb"], rows[base])
        pairs[f"arbiterkb-{base}"] = {
            "accuracy": paired_bootstrap(rows["arbiterkb"], rows[base], "accuracy"),
            "aurc": paired_bootstrap(rows["arbiterkb"], rows[base], "aurc"),
            "brier": paired_bootstrap(rows["arbiterkb"], rows[base], "brier"),
            "mcnemar": mcnemar_exact(a, b),
        }
    return table, pairs


def plan_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def memi_records(root: Path, policy: str, method: str) -> list[dict]:
    base = root / "results" / policy / method
    if policy != "astra_act":
        return plan_records(base / "episodes.jsonl.gz")
    return [json.loads(p.read_text()) for p in sorted(base.glob("*/result.json"))]


def memi_success(row: dict) -> bool:
    return bool(row.get("success")) if "success" in row else row.get("outcome") == "success"


def memi_summary(rows: list[dict], task: str | None = None) -> dict:
    use = [r for r in rows if task is None or r["task"] == task]
    success = [memi_success(r) for r in use]
    tokens = [(r.get("policy") or {}).get("usage", {}).get("input_tokens") for r in use]
    costs = [r.get("cost_usd") for r in use if r.get("cost_usd") is not None]
    return {"success": rate_interval(success),
            "input_tokens_median": None if not any(t is not None for t in tokens) else
                int(np.median([t for t in tokens if t is not None])),
            "cost_usd_median": None if not costs else float(np.median(costs))}


def memi_panel(root: Path) -> dict:
    rows = {(p, m): memi_records(root, p, m) for p in POLICIES for m in METHODS}
    table = {f"{p}/{m}": {"all": memi_summary(rows[p, m]),
                            "VideoUnmaskSwap": memi_summary(rows[p, m], "VideoUnmaskSwap")}
             for p in POLICIES for m in METHODS}
    tests = {}
    for policy in POLICIES:
        mem, full = rows[policy, "memi"], rows[policy, "fullhist"]
        by = lambda xs: {(r["task"], int(r["episode"])): memi_success(r) for r in xs}  # noqa: E731
        a, b = by(mem), by(full)
        keys = sorted(set(a) & set(b))
        tests[policy] = mcnemar_exact([a[k] for k in keys], [b[k] for k in keys])
    return {"table": table, "memi_vs_fullhist": tests}


def verify_memi(root: Path) -> dict:
    proc = subprocess.run([sys.executable, "-m", "memi.report"], cwd=root, text=True,
                          capture_output=True, check=False)
    return {"exit_code": proc.returncode, "all_match": proc.returncode == 0 and "ALL MATCH" in proc.stdout,
            "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest()}


def data_manifest(split: str) -> dict:
    src = PerceivedSource(**OCCLUSION)
    files = [src.files[i] for i in src.episodes(split)]
    return {"count": len(files), "sha256": {p.name: sha256(p) for p in files}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memi-root", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--out", type=Path, default=ROOT / "results/latest/paper_comparison/summary.json")
    args = ap.parse_args()
    params, state = load_params(), git_state()
    if args.split == "test":
        if not ledger.was_frozen(params.hash()):
            raise RuntimeError("test comparison requires a frozen params hash")
        ledger.record("READ", params.hash(), "G perceived test, paper_compare.py post-hoc repair ladder; no tuning")
    table, pairs = local_panel(params, args.split)
    commit = subprocess.run(["git", "-C", str(args.memi_root), "rev-parse", "HEAD"], text=True,
                            capture_output=True, check=True).stdout.strip()
    result = {"schema": 1, "split": args.split, "local": {"table": table, "pairs": pairs},
              "memi": memi_panel(args.memi_root), "memi_verification": verify_memi(args.memi_root),
              "provenance": {"arbiter_commit": state[0], "arbiter_dirty": state[1],
                             "params_hash": params.hash(), "soft_scale_m": SOFT_SCALE,
                             "choice_sha256": sha256(ROOT / "results/latest/claim3_g/choice.json"),
                             "memi_commit": commit, "data": data_manifest(args.split)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({"out": str(args.out), "local": table,
                      "memi_verified": result["memi_verification"]["all_match"]}, indent=1))


if __name__ == "__main__":
    main()
