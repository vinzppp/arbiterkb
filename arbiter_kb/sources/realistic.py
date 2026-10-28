"""Corruptions that look like real tracker errors (DEVELOPMENT.md §3.11, M1).

OverlapSwapSource: ids may swap only where two containers' 3D centres come within d_ov of each other
during a pass (the tracks really overlap), each such pass swapping with probability p_swap. The
label therefore jumps at most about d_ov. NoisySource: Gaussian noise on visible positions.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from itertools import combinations

import numpy as np

from .. import sg_compat as sg
from ..params import DetectParams
from ..types import FrameObs


def u01(*key) -> float:
    return int(hashlib.sha256(":".join(map(str, key)).encode()).hexdigest()[:8], 16) / 16**8


def passes(frames: list[FrameObs], v_move: float) -> list[dict]:
    """Intervals where two containers move at the same time, with their closest 3D approach."""
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
                out.append(cur)
                cur = None
        if cur is not None:
            out.append(cur)
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
