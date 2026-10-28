"""M3 (b): the KB's events against simulator truth (synthetic VideoUnmaskSwap; train seeds only).

  python scripts/measure_kb.py [--n 100] [--noise 0.015] [--keep-as m3_kb_truth]

Per condition (clean, pos_noise):
- cover: each cover(container, cube) event is right if the cube really is under that container (precision);
  recall = cubes that got a right cover event;
- passes: a near_pass evidence matches a true pass (realistic.passes) of the same pair whose closest frame
  lies in the evidence's open interval (precision / recall);
- moves: the xy error of each logged 'from -> to' against the true positions at those frames (median, max).
Grasp / place have no synthetic truth yet (the generator never manipulates); they are unit-tested only.
Observations equal the truth on clean synthetic input, so the clean numbers check plumbing, not perception.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from arbiter_kb.sources.realistic import NoisySource, passes  # noqa: E402
from arbiter_kb.sources.synthetic import SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402


def episode(src, ep: int, params) -> tuple[dict, list, list]:
    s = ArbiterKBSystem(params)
    s.reset(src.meta(ep))
    frames = []
    for f in src.frames(ep):
        s.step(f)
        frames.append(f)
    return src.ground_truth(ep), s.kb, frames


def covers(gt, kb) -> tuple[int, int, int]:
    """(right cover events, all cover events, cubes with a right one)."""
    cv = [e.args for e in kb.events if e.type == "cover"]
    right = [c for c in cv if gt.cube_to_bin.get(c[1]) == c[0]]
    return len(right), len(cv), len({c[1] for c in right})


def pass_match(kb, true_frames, v_move) -> tuple[int, int, int]:
    """(matched evidence, all near_pass evidence, true passes found)."""
    truth = passes(true_frames, v_move)
    ev = [e for e in kb.evidence if e.kind == "near_pass"]
    hit = lambda e, p: {p["a"], p["b"]} == set(e.args) and e.data["t_open"] <= p["t_min"] <= e.t  # noqa: E731
    return sum(any(hit(e, p) for p in truth) for e in ev), len(ev), sum(any(hit(e, p) for e in ev) for p in truth)


def move_errors(kb, true_frames) -> list[float]:
    starts, errs = {}, []
    for e in kb.evidence:
        if e.kind == "move_start":
            starts[e.args[0]] = (e.t, e.data["xyz"])
        elif e.kind == "move_stop" and e.args[0] in starts:
            t0, xyz0 = starts.pop(e.args[0])
            true0, true1 = true_frames[t0 - 1].objects[e.args[0]].xyz, true_frames[e.t].objects[e.args[0]].xyz
            errs += [float(np.linalg.norm(np.subtract(xyz0, true0)[:2])), float(np.linalg.norm(np.subtract(e.data["xyz"], true1)[:2]))]
    return errs


def measure(cond: str, n: int, noise: float, params) -> dict:
    base = SyntheticSource(lane_offset="mix", swap_steps="mix")
    src = base if cond == "none" else NoisySource(base, noise)
    tot = np.zeros(7)
    errs: list[float] = []
    for ep in base.episodes("train", n):
        gt, kb, _ = episode(src, ep, params)
        true_frames = list(base.frames(ep))
        c, pm = covers(gt, kb), pass_match(kb, true_frames, params.detect.v_move)
        tot += [*c, len(gt.cube_to_bin), *pm]
        errs += move_errors(kb, true_frames)
    n_true = sum(len(passes(list(base.frames(ep)), params.detect.v_move)) for ep in base.episodes("train", n))
    r = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"cover_precision": r(tot[0], tot[1]), "cover_recall": r(tot[2], tot[3]),
            "pass_precision": r(tot[4], tot[5]), "pass_recall": r(tot[6], n_true), "n_true_passes": n_true,
            "move_err_median_m": round(float(np.median(errs)), 4) if errs else None,
            "move_err_max_m": round(float(np.max(errs)), 4) if errs else None, "n_moves": len(errs) // 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.015)
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    commit, dirty = git_state()
    params = load_params()
    out = {"conditions": {c: measure(c, a.n, a.noise, params) for c in ("none", "pos_noise")},
           "meta": {"commit": commit, "dirty": dirty, "params_hash": params.hash(), "split": "train", "n": a.n,
                    "pos_noise_m": a.noise, "created": time.strftime("%Y-%m-%d %H:%M:%S")}}
    print(json.dumps(out, indent=2))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
