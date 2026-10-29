"""Motion detector v2 (arbiter_kb/motion.py) on G's perceived episodes and on synthetic data (NOTES "Motion detector
v2": the rule was fixed before the run). Truth = the swap intervals (two containers move in each); no identity needed.

  python scripts/measure_motion.py --grid                # G training episodes 0-9 + synthetic clean: pick a setting
  python scripts/measure_motion.py --read KEY [--keep-as motion_v2]   # KEY = m_win/k_lag/v_start/n_off/valley

A segment (move_start .. move_stop of one track) is matched when it overlaps exactly one swap for at least half its
length, merged when it overlaps two or more swaps, spurious when it overlaps none. recall = mean over swaps of
min(matched segments, 2) / 2; precision = matched / segments. Pass recall = swaps with a near_pass evidence from
their start to 10 frames after their end.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from arbiter_kb.detectors import Detectors  # noqa: E402
from arbiter_kb.sg_compat import F_HIDDEN, F_LABEL, F_U, F_V, F_X, F_Z  # noqa: E402
from arbiter_kb.sources.realistic import NoisySource  # noqa: E402
from arbiter_kb.sources.synthetic import HOLD, SWAP_T0, SyntheticSource  # noqa: E402
from arbiter_kb.types import FrameObs, ObjObs  # noqa: E402

G_DIR = ROOT / "data" / "g_perceived"
GRID = {"m_win": (1, 3, 5, 7), "k_lag": (1, 3), "v_start": (0.001, 0.002, 0.003, 0.005), "n_off": (2, 4),
        "valley": (0.0, 0.2, 0.35, 0.5)}


def g_episodes(split: str) -> list[tuple[list, list]]:
    """[(frames, swap intervals)] from G's perceived slots; train = episodes 0-9 of each setting, test = 10-19."""
    out = []
    for f in sorted(G_DIR.glob("*.npz")):
        if (int(f.stem.split("_ep")[1]) < 10) != (split == "train"):
            continue
        z = np.load(f, allow_pickle=True)
        p, meta = z["perceived"], json.loads(str(z["meta"]))
        frames = []
        for t in range(p.shape[0]):
            objs = {}
            for s in range(p.shape[1]):
                row = p[t, s]
                vis = bool(row[F_HIDDEN] == 0 and np.all(np.isfinite(row[F_U:F_Z + 1])))
                cls = "container" if np.nanmax(p[:, s, F_LABEL]) == 0 else "cube"
                objs[f"s{s}"] = ObjObs(f"s{s}", cls, vis, row[F_X:F_Z + 1].astype(float) if vis else None,
                                       row[F_U:F_V + 1].astype(float) if vis else None, 1.0)
            frames.append(FrameObs(t, objs, np.array([0.0, 0.0, 0.45]), 0.08, True))
        out.append((frames, [tuple(s) for s in meta["schedule"]]))
    return out


def synthetic(noise: float, n: int = 100) -> list[tuple[list, list]]:
    src = SyntheticSource(lane_offset="mix", swap_steps="mix")
    noisy = NoisySource(src, noise) if noise else src
    out = []
    for ep in src.episodes("train", n):
        frames, steps = list(noisy.frames(ep)), src.settings(ep)[1]
        out.append((frames, [(SWAP_T0 + k * steps, SWAP_T0 + (k + 1) * steps)
                             for k in range((len(frames) - SWAP_T0 - HOLD) // steps)]))
    return out


def run(frames: list, p) -> tuple[list, list]:
    det, segs, start, passes = Detectors(p), [], {}, []
    for f in frames:
        for e in det.step(f):
            if e.kind == "move_start":
                start[e.args[0]] = e.data["t_at"]
            elif e.kind == "move_stop" and e.args[0] in start:
                segs.append((start.pop(e.args[0]), e.t))
            elif e.kind == "near_pass":
                passes.append(e.t)
    return segs + [(t0, frames[-1].t) for t0 in start.values()], passes


def score(episodes: list, p) -> dict:
    n_seg = matched = merged = spurious = 0
    rec, prec_pass = [], []
    for frames, swaps in episodes:
        segs, passes = run(frames, p)
        hit = {s: 0 for s in swaps}
        for a, b in segs:
            over = [s for s in swaps if min(b, s[1]) - max(a, s[0]) > 0]
            n_seg += 1
            if len(over) >= 2:
                merged += 1
            elif not over:
                spurious += 1
            elif min(b, over[0][1]) - max(a, over[0][0]) >= 0.5 * max(b - a, 1):
                matched += 1
                hit[over[0]] += 1
        rec += [min(h, 2) / 2 for h in hit.values()]
        prec_pass += [any(s[0] <= t <= s[1] + 10 for t in passes) for s in swaps]
    r, pr = float(np.mean(rec)), matched / max(n_seg, 1)
    return {"recall": round(r, 3), "precision": round(pr, 3), "f1": round(2 * r * pr / max(r + pr, 1e-9), 3),
            "segments": n_seg, "merged": merged, "spurious": spurious, "pass_recall": round(float(np.mean(prec_pass)), 3)}


def setting(key: str, base):
    m, k, v, off, val = key.split("/")
    return replace(base, m_win=int(m), k_lag=int(k), v_start=float(v), v_stop=float(v) / 2, n_on=2, n_off=int(off),
                   valley=float(val), rev_cos=-0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--read", default="", help="the chosen key m/k/v_start/n_off, read once on G test")
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    base, (commit, dirty) = load_params().detect, git_state()  # the grid varies the motion fields of the frozen set
    out = {"meta": {"commit": commit, "dirty": dirty}, "v1": {}}
    if a.grid:
        g, syn = g_episodes("train"), synthetic(0.0)
        out["v1"] = {"g_train": score(g, base), "synthetic_clean": score(syn, base)}
        keys = ["/".join(map(str, c)) for c in itertools.product(*GRID.values())]
        out["grid"] = {k: {"g_train": score(g, setting(k, base)), "synthetic_clean": score(syn, setting(k, base))}
                       for k in keys}
        floor = min(0.95, out["v1"]["synthetic_clean"]["recall"])  # v1 itself is below 0.95 (NOTES)
        ok = [k for k in keys if out["grid"][k]["synthetic_clean"]["recall"] >= floor]
        out["floor"] = floor
        out["chosen"] = max(ok, key=lambda k: (out["grid"][k]["g_train"]["f1"], -int(k.split("/")[0])))
    if a.read:
        p = setting(a.read, base)
        out["read"] = {"key": a.read, "g_test": score(g_episodes("test"), p), "synthetic_clean": score(synthetic(0.0), p),
                       "synthetic_noise_15mm": score(synthetic(0.015), p)}
        v1 = replace(base, m_win=1, k_lag=1, v_start=0.001, v_stop=0.001, n_on=1, n_off=1, valley=0.0, rev_cos=-2.0)
        out["v1"] = {"g_test": score(g_episodes("test"), v1), "synthetic_noise_15mm": score(synthetic(0.015), v1)}
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / ("grid.json" if a.grid else "read.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
