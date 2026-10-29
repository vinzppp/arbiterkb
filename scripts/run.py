"""Run an evaluation, or the phase self-test.

  python scripts/run.py --corrupt none,id_swap --p 0.5 --n 30                       # M0 table
  python scripts/run.py --systems all --corrupt none,id_swap_overlap,pos_noise \
         --lane mix --steps mix --n 60 --keep-as m1_table                            # M1 table
  python scripts/run.py --systems m2 --split test --first 60 --n 300 ...            # M2: fresh seeds >= 1060
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

from arbiter_kb import CONTRACT_VERSION, ledger  # noqa: E402
from arbiter_kb.evaluate import run, save_run  # noqa: E402
from arbiter_kb.params import Params  # noqa: E402
from arbiter_kb.sources.corrupt import CorruptedSource  # noqa: E402
from arbiter_kb.sources.realistic import SIGNATURE, MeasuredSwapSource, NoisySource, OverlapSwapSource  # noqa: E402
from arbiter_kb.sources.replay import OracleReplaySource  # noqa: E402
from arbiter_kb.sources.synthetic import SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.systems.baselines import GuardedBound, TrackerSystem  # noqa: E402
from arbiter_kb.systems.soft_haircut import SoftHaircut  # noqa: E402

COLS = ["condition", "system", "n", "acc", "silent", "silent_lo", "silent_hi", "coverage", "cov_lo", "cov_hi", "aurc", "brier", "ece",
        "detect_among_wrong", "conf_right", "conf_wrong", "n_corrupted", "silent_on_corrupted", "grasp_err_m"]
SYSTEMS = {  # ArbiterKB last: evaluate.run times the last system (E7)
    "frozen": lambda p: TrackerSystem("frozen", p), "bound": lambda p: TrackerSystem("bound", p),
    "bound_haircut": lambda p: GuardedBound("haircut", p),
    "overlap_haircut": lambda p: GuardedBound("overlap_haircut", p),
    "jump_detector": lambda p: GuardedBound("jump", p), "soft_haircut": lambda p: SoftHaircut(p),
    "arbiterkb": lambda p: ArbiterKBSystem(p),
}
PRESETS = {"all": [k for k in SYSTEMS if k != "soft_haircut"],  # the M1 table, unchanged
           "m2": ["bound", "soft_haircut", "arbiterkb"]}  # claim 3's table since v1.6 (DEVELOPMENT §6 M2)
CONDITIONS = ("none", "id_swap", "id_swap_overlap", "id_swap_measured", "pos_noise")
PARAMS_FILE = ROOT / "params.json"
FREEZE_NOTE = ("M1.1 defaults plus R1 eps_far 0.5 (M2 infra) and M3's r_grasp / r_on / r_near / left_margin / on_dz / g_close / g_open, untuned; R7 (lam) removed in v1.6; hand-pause detector (v_pause, n_pause); semantic actions (z_low, z_drop, z_hold, press_rise, y_side). "
               "Nothing has been tuned on test seeds.")


def git_state() -> tuple[str, bool | None]:
    """(commit, dirty): dirty = the working tree differs from that commit (E4 provenance)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout
        return commit, bool(status.strip())
    except Exception:
        return "no-git", None


PROMPT_FILES = ("controller", "views", "action_view", "narrative", "goal_counter", "keyframe", "ask", "epistemic", "memtool", "self_memory", "closed_loop")


def prompts_hash() -> str:
    """sha256 over the code that writes what a model reads (prompts, views, the goal parse): any edit to these
    modules changes it, so a run can only read the prompts that were frozen (E5c, NOTES "Before the full rerun")."""
    h = hashlib.sha256()
    for name in PROMPT_FILES:
        h.update((ROOT / "arbiter_kb" / f"{name}.py").read_bytes())
    return h.hexdigest()[:12]


def load_params() -> Params:
    """The one parameter set evaluations may use; the test split requires it to be frozen (E5c), and so do the
    prompts and views (prompts_hash)."""
    if not PARAMS_FILE.exists():
        sys.exit(f"{PARAMS_FILE.name} missing: run `python scripts/run.py --freeze`")
    stored = json.loads(PARAMS_FILE.read_text()).get("prompts_hash")
    if stored != prompts_hash():
        sys.exit(f"prompts / views changed since the freeze ({stored} != {prompts_hash()}): "
                 f"run `python scripts/run.py --freeze --force`")
    return Params.load(PARAMS_FILE)


def freeze(force: bool = False) -> None:
    """Write params.json from the code defaults (untuned phases only) and log FREEZE (E5c)."""
    if PARAMS_FILE.exists() and not force:
        sys.exit(f"{PARAMS_FILE.name} exists; refusing to overwrite without --force")
    p = Params(frozen=True, note=FREEZE_NOTE)
    p.save(PARAMS_FILE)
    PARAMS_FILE.write_text(json.dumps({**json.loads(PARAMS_FILE.read_text()), "prompts_hash": prompts_hash()},
                                      indent=2, sort_keys=True))
    ledger.record("FREEZE", p.hash(), f"{PARAMS_FILE.name} via run.py --freeze; prompts {prompts_hash()}")
    print("wrote", PARAMS_FILE, "hash", p.hash(), "prompts", prompts_hash())


def _num_or_mix(v):
    return v if v == "mix" else float(v)


def base_source(args):
    if args.source == "oracle":
        return OracleReplaySource()
    steps = args.steps if args.steps == "mix" else int(args.steps)
    return SyntheticSource(lane_offset=_num_or_mix(args.lane), swap_steps=steps)


def make_source(cond: str, base, args, params: Params):
    if cond == "none":
        return base
    if cond == "id_swap":
        return CorruptedSource(base, cond, args.p, v_move=params.detect.v_move)
    if cond == "id_swap_overlap":
        return OverlapSwapSource(base, args.p, args.d_ov, v_move=params.detect.v_move)
    if cond == "id_swap_measured":
        return MeasuredSwapSource(base, v_move=params.detect.v_move)
    if cond == "pos_noise":
        return NoisySource(base, args.noise)
    raise SystemExit(f"unknown condition {cond!r}; one of {CONDITIONS}")


def evaluate(args, params: Params) -> tuple[Path | None, list[dict], dict]:
    commit, dirty = git_state()  # before any read: the run itself appends to the ledger
    base = base_source(args)
    if args.source == "synthetic" and args.first + args.n > 1000:  # train seeds 0-999, test 1000-1999
        sys.exit(f"--first {args.first} + --n {args.n} > 1000 would leave the {args.split} seed range")
    eps = base.episodes(args.split, n=args.n + args.first)[args.first:]
    if not eps:
        print(f"SKIPPED {args.source} source: no episodes. To copy replays from the pod: scripts/copy_replays.sh")
        return None, [], {}
    names = PRESETS.get(args.systems) or args.systems.split(",")
    names = sorted(names, key=list(SYSTEMS).index)
    tag = hashlib.sha256(json.dumps(vars(args), sort_keys=True).encode()).hexdigest()[:8]
    run_dir = ROOT / "results" / "runs" / f"{args.source}_{args.split}_{tag}"
    rows, traces, step_s, logs = [], {}, [], {}
    for cond in args.corrupt.split(","):
        src = make_source(cond, base, args, params)
        log = getattr(src, "log", {})
        r, tr, st = run([SYSTEMS[n](params) for n in names], src, eps, params, cond,
                        lambda ep, lg=log: lg.get(ep, {}).get("corrupted", False))
        rows, step_s, logs[cond] = rows + r, step_s + st, log
        traces.update({(f"{cond}_{s}", ep): v for (s, ep), v in tr.items()})
    meta = {"commit": commit, "dirty": dirty, "args": vars(args), "params_hash": params.hash(),
            "contract_version": CONTRACT_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": {ep: base.settings(ep) for ep in eps} if hasattr(base, "settings") else None,
            "corruption_log": logs, "median_step_ms": 1000 * sorted(step_s)[len(step_s) // 2] if step_s else None,
            "llm_cache": {"hits": 0, "misses": 0},
            "g_signature_sha256": hashlib.sha256(SIGNATURE.read_bytes()).hexdigest()
            if "id_swap_measured" in args.corrupt.split(",") else None}
    summary = save_run(run_dir, rows, params, meta, traces)
    keep = keep_dir(args)
    if keep is not None:  # results/latest is committed: only on request, from a clean tree
        keep.mkdir(parents=True, exist_ok=True)
        for f in ("summary.csv", "meta.json"):
            (keep / f).write_bytes((run_dir / f).read_bytes())
    print_table(summary)
    print(f"run dir: {run_dir.relative_to(ROOT)}")
    return run_dir, summary, {"traces": traces, "logs": logs, "rows": rows, "step_s": step_s, "keep": keep}


def keep_dir(args) -> Path | None:
    if getattr(args, "keep_as", ""):
        return ROOT / "results" / "latest" / args.keep_as
    if getattr(args, "keep", False):
        first = f"_f{args.first}" if args.first else ""
        return ROOT / "results" / "latest" / f"{args.source}_{args.split}{first}_n{args.n}_p{args.p}"
    return None


def print_table(summary: list[dict]) -> None:
    print(" | ".join(f"{c:>10}" for c in COLS))
    for r in summary:
        print(" | ".join(f"{(r[c] if not isinstance(r[c], float) else round(r[c], 3))!s:>10}" for c in COLS))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "oracle"])
    ap.add_argument("--systems", default="frozen,bound,arbiterkb", help=f"comma list of {list(SYSTEMS)}, or a preset {list(PRESETS)}")
    ap.add_argument("--corrupt", default="none,id_swap", help=f"comma list of {CONDITIONS}")
    ap.add_argument("--p", type=float, default=0.5, help="id_swap: share of episodes; id_swap_overlap: p_swap")
    ap.add_argument("--d-ov", dest="d_ov", type=float, default=0.03, help="id_swap_overlap distance (m)")
    ap.add_argument("--pos-noise", dest="noise", type=float, default=0.015, help="pos_noise sigma (m)")
    ap.add_argument("--lane", default="0.07", help="synthetic lane_offset (m) or 'mix'")
    ap.add_argument("--steps", default="50", help="synthetic swap_steps or 'mix'")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--first", type=int, default=0, help="skip the split's first seeds (M2: --first 60 -> test seeds >= 1060)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--freeze", action="store_true", help="write params.json from the code defaults")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep", action="store_true", help="copy summary + meta to results/latest (committed)")
    ap.add_argument("--keep-as", dest="keep_as", default="", help="like --keep, under this name")
    return ap


def parse(argv: list[str]) -> argparse.Namespace:
    return parser().parse_args(argv)


def main() -> None:
    args = parser().parse_args()
    if args.freeze:
        return freeze(args.force)
    if args.selftest:
        from selftest import selftest  # scripts/selftest.py

        sys.exit(selftest())
    run_dir, summary, info = evaluate(args, load_params())
    if run_dir is not None and args.source == "synthetic":
        from figures import all_figures  # scripts/figures.py

        for f in all_figures(run_dir, info, base_source(args), args):
            print("figure:", f)


if __name__ == "__main__":
    main()
