"""M3 (b): the KB's events against simulator truth (synthetic VideoUnmaskSwap; train seeds only).

  python scripts/measure_kb.py [--n 100] [--noise 0.015] [--keep-as m3_kb_truth]

Per condition (clean, pos_noise):
- cover: each cover(container, cube) event is right if the cube really is under that container (precision);
  recall = cubes that got a right cover event;
- passes: a true pass = a local minimum, below TRUE_D, of the TRUE 3D distance between two containers that
  are both moving (any motion); it uses none of the detector's thresholds, so the two cannot fail together.
  A near_pass evidence is right if a true pass of the same pair lies in its open interval (precision);
  recall = true passes with a matching evidence;
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
from arbiter_kb.sources.realistic import NoisySource  # noqa: E402
from arbiter_kb.sources.synthetic import SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.views import controller_view  # noqa: E402

TRUE_D = 0.15  # m: a closest approach under this counts as a pass (generous; the detector's d_px is ~6 cm)


def true_passes(frames: list) -> list[dict]:
    """Local minima of the true centre distance of each container pair while both move."""
    bins = sorted(k for k, o in frames[0].objects.items() if o.cls == "container")
    out = []
    for i, a in enumerate(bins):
        for b in bins[i + 1:]:
            d = np.array([np.linalg.norm(f.objects[a].xyz - f.objects[b].xyz) if f.objects[a].visible
                          and f.objects[b].visible else np.inf for f in frames])
            mv = [t > 0 and frames[t].objects[a].visible and frames[t - 1].objects[a].visible
                  and frames[t].objects[b].visible and frames[t - 1].objects[b].visible
                  and np.linalg.norm(frames[t].objects[a].xyz - frames[t - 1].objects[a].xyz) > 0
                  and np.linalg.norm(frames[t].objects[b].xyz - frames[t - 1].objects[b].xyz) > 0 for t in range(len(frames))]
            out += [{"a": a, "b": b, "t_min": t} for t in range(1, len(frames) - 1)
                    if mv[t] and d[t] < TRUE_D and d[t] <= d[t - 1] and d[t] < d[t + 1]]
    return out


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


def pass_match(kb, true_frames) -> tuple[int, int, int, int]:
    """(matched evidence, all near_pass evidence, true passes found, all true passes)."""
    truth = true_passes(true_frames)
    ev = [e for e in kb.evidence if e.kind == "near_pass"]
    hit = lambda e, p: {p["a"], p["b"]} == set(e.args) and e.data["t_open"] <= p["t_min"] <= e.t  # noqa: E731
    return (sum(any(hit(e, p) for p in truth) for e in ev), len(ev), sum(any(hit(e, p) for e in ev) for p in truth),
            len(truth))


def move_errors(kb, true_frames) -> list[float]:
    starts, errs = {}, []
    for e in kb.evidence:
        if e.kind == "move_start":
            starts[e.args[0]] = (e.data["t_at"], e.data["xyz"])
        elif e.kind == "move_stop" and e.args[0] in starts:
            t0, xyz0 = starts.pop(e.args[0])
            true0, true1 = true_frames[t0].objects[e.args[0]].xyz, true_frames[e.t].objects[e.args[0]].xyz
            errs += [float(np.linalg.norm(np.subtract(xyz0, true0)[:2])), float(np.linalg.norm(np.subtract(e.data["xyz"], true1)[:2]))]
    return errs


def measure(cond: str, n: int, noise: float, params) -> dict:
    base = SyntheticSource(lane_offset="mix", swap_steps="mix")
    src = base if cond == "none" else NoisySource(base, noise)
    tot = np.zeros(8)
    errs: list[float] = []
    tokens: list[int] = []
    for ep in base.episodes("train", n):
        gt, kb, _ = episode(src, ep, params)
        true_frames = list(base.frames(ep))
        c, pm = covers(gt, kb), pass_match(kb, true_frames)
        tot += [*c, len(gt.cube_to_bin), *pm]
        errs += move_errors(kb, true_frames)
        tokens.append(len(controller_view(kb, params.rule)) // 4)  # rough: 4 characters per token
    r = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"cover_precision": r(tot[0], tot[1]), "cover_recall": r(tot[2], tot[3]),
            "pass_precision": r(tot[4], tot[5]), "pass_recall": r(tot[6], tot[7]), "n_true_passes": int(tot[7]),
            "move_err_median_m": round(float(np.median(errs)), 4) if errs else None,
            "move_err_max_m": round(float(np.max(errs)), 4) if errs else None, "n_moves": len(errs) // 2,
            "view_tokens_median": int(np.median(tokens)), "view_tokens_max": int(max(tokens))}


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
