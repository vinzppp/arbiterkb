"""Corruptions that look like real tracker errors (DEVELOPMENT.md §3.11, M1).

OverlapSwapSource: ids may swap only where two containers' 3D centres come within d_ov of each other
during a pass (the tracks really overlap), each such pass swapping with probability p_swap. The
label therefore jumps at most about d_ov. MeasuredSwapSource: swaps at the rate and with the shape
phase G measured on the real tracker. NoisySource: Gaussian noise on visible positions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from .. import sg_compat as sg
from ..params import DetectParams
from ..types import FrameObs


def u01(*key) -> float:
    return int(hashlib.sha256(":".join(map(str, key)).encode()).hexdigest()[:8], 16) / 16**8


def passes(frames: list[FrameObs], v_move: float) -> list[dict]:
    """Intervals [t_open, t_close) where two containers move at the same time, with their closest 3D
    approach (t_min, d_min)."""
    bins = sorted(k for k, o in frames[0].objects.items() if o.cls == "container")
    moving = {b: [False] + [_moved(frames[t - 1].objects[b], frames[t].objects[b], v_move)
                            for t in range(1, len(frames))] for b in bins}
    out = []
    for a, b in combinations(bins, 2):
        cur = None
        for t in range(len(frames)):
            if moving[a][t] and moving[b][t]:
                d = float(np.linalg.norm(frames[t].objects[a].xyz - frames[t].objects[b].xyz))
                if cur is None or d < cur["d_min"]:
                    cur = {"a": a, "b": b, "t_min": t, "d_min": d, "t_open": cur["t_open"] if cur else t}
            elif cur is not None:
                out.append({**cur, "t_close": t})
                cur = None
        if cur is not None:
            out.append({**cur, "t_close": len(frames)})
    return sorted(out, key=lambda p: p["t_min"])


def _moved(o0, o1, v_move) -> bool:
    return o0.visible and o1.visible and float(np.linalg.norm(o1.xyz[:2] - o0.xyz[:2])) > v_move


class OverlapSwapSource:
    def __init__(self, inner, p_swap: float = 0.5, d_ov: float = 0.03, seed: int = 0,
                 v_move: float = DetectParams().v_move):
        self.inner, self.p_swap, self.d_ov, self.seed, self.v_move = inner, p_swap, d_ov, seed, v_move
        self.name = f"{inner.name}+id_swap_overlap@{p_swap}"
        self.log: dict[int, dict] = {}

    def episodes(self, split, *a, **kw):
        return self.inner.episodes(split, *a, **kw)

    def meta(self, ep):
        return self.inner.meta(ep)

    def ground_truth(self, ep):
        return self.inner.ground_truth(ep)

    def frames(self, ep: int):
        frames = list(self.inner.frames(ep))  # the corruptor (not the system) may look ahead
        ps = passes(frames, self.v_move)
        swaps = [p for p in ps if p["d_min"] < self.d_ov and u01(self.seed, ep, p["a"], p["b"], p["t_min"]) < self.p_swap]
        jumps: list[float] = []  # filled while frames are yielded
        self.log[ep] = {"corrupted": bool(swaps), "n_overlap_passes": sum(p["d_min"] < self.d_ov for p in ps),
                        "n_passes": len(ps), "swaps": [[s["t_min"], s["a"], s["b"], round(s["d_min"], 4)] for s in swaps],
                        "label_jumps_m": jumps}  # written before the first frame: scoring at t* reads it
        perm = {k: k for k in frames[0].objects}  # label -> physical object it now reports
        for f in frames:
            for s in swaps:  # the two PHYSICAL containers overlap: whichever labels track them swap
                if s["t_min"] == f.t:
                    la, lb = (lab for lab, phys in perm.items() if phys in (s["a"], s["b"]))
                    perm[la], perm[lb] = perm[lb], perm[la]
                    s["labels"] = (la, lb)
            objs = {lab: replace(f.objects[phys], id=lab) for lab, phys in perm.items()}
            for s in swaps:
                if s["t_min"] == f.t and f.t > 0:
                    jumps += [round(float(np.linalg.norm(objs[k].xyz - self._prev[k].xyz)), 4) for k in s["labels"]]
            self._prev = objs
            yield replace(f, objects=objs)


SIGNATURE = Path(__file__).resolve().parents[2] / "results" / "g_signature.json"


class MeasuredSwapSource:
    """id_swap_measured (§3.11, M2): the rate and shape of swaps are G's, measured on the real tracker
    (results/g_signature.json, written by scripts/measure_swaps.py), not chosen by us.
    - Each pass swaps with p = logistic(min_3d, speed). min_3d is clamped to the range G saw: real
      containers never come closer than about 5 cm, while synthetic lanes <= 0.02 do.
    - A swapped pass exchanges the ids at t_min + an offset drawn from G's, right after one of the two
      tracks has been absent for a drawn number of frames (G: most swaps follow a mask dropout).
    - A clean pass drops a track for a length drawn from G's clean passes (often 0).
    Every draw is seeded per pass, so results are reproducible and nested in nothing."""

    def __init__(self, inner, sig: dict | None = None, seed: int = 0, v_move: float = DetectParams().v_move):
        self.inner, self.seed, self.v_move = inner, seed, v_move
        self.sig = sig or json.loads(SIGNATURE.read_text())
        self.name = f"{inner.name}+id_swap_measured"
        self.log: dict[int, dict] = {}

    def episodes(self, split, *a, **kw):
        return self.inner.episodes(split, *a, **kw)

    def meta(self, ep):
        return self.inner.meta(ep)

    def ground_truth(self, ep):
        return self.inner.ground_truth(ep)

    def p_swap(self, d: float, speed: float) -> float:
        lo, hi = self.sig["min_3d_range"]
        (wd, ws), b = self.sig["w"], self.sig["b"]
        return float(1 / (1 + np.exp(-(wd * min(max(d, lo), hi) + ws * speed + b))))

    def _plan(self, ep: int, frames: list[FrameObs]) -> tuple[list[dict], set]:
        """(swaps, hidden (t, physical container) pairs) for one episode."""
        T, swaps, hidden = len(frames), [], set()
        for p in passes(frames, self.v_move):
            key = (self.seed, ep, p["a"], p["b"], p["t_min"])
            draw = lambda xs, tag: xs[int(u01(*key, tag) * len(xs))] if xs else 0  # noqa: E731
            xy = np.array([frames[t].objects[p["a"]].xyz[:2] for t in range(p["t_open"], p["t_close"])])
            speed = float(np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0
            pr = self.p_swap(p["d_min"], speed)
            who = p["a"] if u01(*key, "who") < 0.5 else p["b"]
            if u01(*key, "swap") < pr:
                t_sw = int(np.clip(p["t_min"] + draw(self.sig["swap_dt"], "dt"), 1, T - 1))
                n = draw(self.sig["dropout_swapped"], "drop")
                swaps.append({**p, "t_sw": t_sw, "p": pr, "dropout": n})
                t0 = t_sw - n
            else:
                n = draw(self.sig["dropout_clean"], "drop")
                t0 = p["t_min"] - n // 2
            hidden |= {(t, who) for t in range(max(0, t0), min(T, t0 + n))}
        return swaps, hidden

    def frames(self, ep: int):
        frames = list(self.inner.frames(ep))  # the corruptor (not the system) may look ahead
        swaps, hidden = self._plan(ep, frames)
        lo = self.sig["min_3d_range"][0]
        jumps: list[float] = []  # filled while frames are yielded
        self.log[ep] = {"corrupted": bool(swaps), "n_passes": len(passes(frames, self.v_move)),
                        "swaps": [[s["t_sw"], s["a"], s["b"], round(s["d_min"], 4), round(s["p"], 3), s["dropout"]]
                                  for s in swaps],
                        "n_clamped": sum(p["d_min"] < lo for p in passes(frames, self.v_move)),
                        "hidden_frames": len(hidden), "label_jumps_m": jumps}  # before the first frame
        perm = {k: k for k in frames[0].objects}  # label -> physical object it now reports
        last: dict[str, np.ndarray] = {}  # label -> last reported position
        for f in frames:
            for s in swaps:
                if s["t_sw"] == f.t:
                    la, lb = (lab for lab, phys in perm.items() if phys in (s["a"], s["b"]))
                    perm[la], perm[lb] = perm[lb], perm[la]
                    s["labels"] = (la, lb)
            objs = {}
            for lab, phys in perm.items():
                o = replace(f.objects[phys], id=lab)
                objs[lab] = replace(o, visible=False, xyz=None, uv=None) if (f.t, phys) in hidden else o
            for s in swaps:
                if s["t_sw"] == f.t:
                    jumps += [round(float(np.linalg.norm(objs[k].xyz - last[k])), 4)
                              for k in s["labels"] if objs[k].visible and k in last]
            last.update({k: o.xyz for k, o in objs.items() if o.visible})
            yield replace(f, objects=objs)


class NoisySource:
    def __init__(self, inner, sigma: float = 0.015, seed: int = 0):
        self.inner, self.sigma, self.seed = inner, sigma, seed
        self.name = f"{inner.name}+pos_noise@{sigma}"
        self.log: dict[int, dict] = {}

    def episodes(self, split, *a, **kw):
        return self.inner.episodes(split, *a, **kw)

    def meta(self, ep):
        return self.inner.meta(ep)

    def ground_truth(self, ep):
        return self.inner.ground_truth(ep)

    def frames(self, ep: int):
        rng, meta = np.random.default_rng([self.seed, ep, 7]), self.inner.meta(ep)
        self.log[ep] = {"corrupted": False}  # before the first frame, like OverlapSwapSource
        for f in self.inner.frames(ep):
            objs = {}
            for k, o in f.objects.items():
                if o.visible:
                    xyz = o.xyz + rng.normal(0, self.sigma, 3) * np.array([1.0, 1.0, 0.3])
                    uv = sg.project_points(xyz, meta.K, meta.E[f.t])[0] if meta.K is not None else o.uv
                    o = replace(o, xyz=xyz, uv=uv)
                objs[k] = o
            yield replace(f, objects=objs)
