"""Run an evaluation, or the phase self-test.

  python scripts/run.py --source synthetic --corrupt none,id_swap --p 0.5 --split test --n 30
  python scripts/run.py --source oracle --split train
  python scripts/run.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb import CONTRACT_VERSION  # noqa: E402
from arbiter_kb.evaluate import run, save_run  # noqa: E402
from arbiter_kb.params import Params  # noqa: E402
from arbiter_kb.plots import belief_timeline  # noqa: E402
from arbiter_kb.sources.corrupt import CorruptedSource  # noqa: E402
from arbiter_kb.sources.replay import OracleReplaySource  # noqa: E402
from arbiter_kb.sources.synthetic import COVER_T, SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.systems.baselines import TrackerSystem  # noqa: E402

COLS = ["condition", "system", "n", "acc", "silent", "silent_lo", "silent_hi", "brier", "ece", "conf_right",
        "conf_wrong", "acc_moved", "acc_back", "n_corrupted", "silent_on_corrupted", "grasp_err_m"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "no-git"


PARAMS_FILE = ROOT / "params.json"


def load_params() -> Params:
    """The one parameter set evaluations may use; the test split requires it to be frozen (E5c)."""
    if not PARAMS_FILE.exists():
        sys.exit(f"{PARAMS_FILE.name} missing: run `python scripts/run.py --freeze` (M0) or scripts/tune.py (M1+)")
    return Params.load(PARAMS_FILE)


def freeze(force: bool = False) -> None:
    if PARAMS_FILE.exists() and not force:
        sys.exit(f"{PARAMS_FILE.name} exists; refusing to overwrite without --force")
    Params(frozen=True, note="M0 defaults, untuned, frozen before any test read").save(PARAMS_FILE)
    print("wrote", PARAMS_FILE)


def systems(params: Params) -> list:
    return [TrackerSystem("frozen", params), TrackerSystem("bound", params), ArbiterKBSystem(params)]


def evaluate(args, params: Params) -> tuple[Path, list[dict], dict]:
    base = SyntheticSource() if args.source == "synthetic" else OracleReplaySource()
    kw = {"n": args.n}
    eps = base.episodes(args.split, **kw)
    if not eps:
        print(f"SKIPPED {args.source} source: no episodes. To copy replays from the pod: scripts/copy_replays.sh")
        return None, [], {}
    tag = hashlib.sha256(json.dumps(vars(args), sort_keys=True).encode()).hexdigest()[:8]
    run_dir = ROOT / "results" / "runs" / f"{args.source}_{args.split}_{tag}"
    params.save(run_dir / "params.json")  # copy of the parameters this run used
    first_read = time.strftime("%Y-%m-%d %H:%M:%S")
    rows, traces, step_s, logs = [], {}, [], {}
    for cond in args.corrupt.split(","):
        src = base if cond == "none" else CorruptedSource(base, cond, args.p, v_move=params.detect.v_move)
        log = getattr(src, "log", {})
        r, tr, st = run(systems(params), src, eps, params, cond, lambda ep, lg=log: lg.get(ep, {}).get("corrupted", False))
        rows += r
        step_s += st
        traces.update({(f"{cond}_{s}", ep): v for (s, ep), v in tr.items()})
        logs[cond] = log
    meta = {"commit": git_commit(), "args": vars(args), "params_hash": params.hash(),
            "contract_version": CONTRACT_VERSION,
            "params_file_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(PARAMS_FILE.stat().st_mtime))
            if PARAMS_FILE.exists() else None, "first_data_read": first_read, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corruption_log": logs, "median_step_ms": 1000 * sorted(step_s)[len(step_s) // 2] if step_s else None,
            "llm_cache": {"hits": 0, "misses": 0}}
    summary = save_run(run_dir, rows, params, meta, traces)
    keep = ROOT / "results" / "latest" / f"{args.source}_{args.split}_n{args.n}_p{args.p}"  # committed (E10)
    keep.mkdir(parents=True, exist_ok=True)
    for f in ("summary.csv", "meta.json"):
        (keep / f).write_bytes((run_dir / f).read_bytes())
    print_table(summary)
    print(f"run dir: {run_dir.relative_to(ROOT)}")
    return run_dir, summary, {"traces": traces, "logs": logs, "rows": rows, "step_s": step_s,
                              "source": args.source, "p": args.p}


def print_table(summary: list[dict]) -> None:
    print(" | ".join(f"{c:>10}" for c in COLS))
    for r in summary:
        print(" | ".join(f"{(r[c] if not isinstance(r[c], float) else round(r[c], 3))!s:>10}" for c in COLS))


def figure(run_dir: Path, info: dict, base) -> Path | None:
    """Belief timeline for a corrupted episode, preferring one where a near-pass lowered confidence."""
    log = info["logs"].get("id_swap", {})
    corrupted = [e for e, v in sorted(log.items()) if v.get("corrupted")]
    if not corrupted:
        return None
    pm = {r["episode"]: r["p_max"] for r in info["rows"] if r["condition"] == "id_swap" and r["system"] == "arbiterkb"}
    ep = min(corrupted, key=lambda e: (pm.get(e, 1.0), e))
    gt, meta = base.ground_truth(ep), base.meta(ep)
    tr = info["traces"]
    cube = gt.targets[0]
    bins = [o for o, c, _ in meta.objects if c == "container"]
    a, b = log[ep]["ids"]
    title = (f"episode {ep}: the true container is labelled {a} until t={log[ep]['t']}, then {b} "
             f"(the tracker swapped the two ids)")
    marks = {"cover": COVER_T, "corruption": log[ep]["t"], "t*": gt.t_star[cube]}
    out = ROOT / "figures" / f"m0_belief_timeline_{info['source']}_p{info['p']}.png"
    return belief_timeline(tr[("id_swap_arbiterkb", ep)], {"bound": tr[("id_swap_bound", ep)],
                           "frozen": tr[("id_swap_frozen", ep)]}, cube, a, bins, marks, out, title)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "oracle"])
    ap.add_argument("--corrupt", default="none,id_swap")
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--freeze", action="store_true", help="write params.json from defaults (M0 only)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.freeze:
        return freeze(args.force)
    if args.selftest:
        from selftest import selftest  # scripts/selftest.py

        sys.exit(selftest())
    run_dir, summary, info = evaluate(args, load_params())
    if run_dir is not None and args.source == "synthetic":
        print("figure:", figure(run_dir, info, SyntheticSource()))


if __name__ == "__main__":
    main()
