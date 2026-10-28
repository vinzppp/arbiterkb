"""Corrupt the OBSERVATIONS the way trackers fail, never the answer (DEVELOPMENT.md §3.11).

id_swap: at the closest approach of the first swap that moves the target's container, the two
containers' track ids are exchanged for every later frame. Positions stay truthful; only the labels
lie. Episodes are chosen by a seeded hash, nested across p (the p = 0.25 set ⊂ the p = 0.5 set).
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from ..params import DetectParams
from ..types import FrameObs


def chosen(ep: int, p: float, seed: int = 0) -> bool:
    u = int(hashlib.sha256(f"{seed}:{ep}".encode()).hexdigest()[:8], 16) / 16**8
    return u < p


def find_target_swap(frames: list[FrameObs], target_bin: str, v_move: float) -> tuple[int, str] | None:
    """(frame of closest approach, partner id) for the first swap that moves `target_bin`."""
    bins = [k for k, o in frames[0].objects.items() if o.cls == "container"]
    best = None
    for t in range(1, len(frames)):
        a0, a1 = frames[t - 1].objects[target_bin], frames[t].objects[target_bin]
        if not (a0.visible and a1.visible) or np.linalg.norm(a1.xyz[:2] - a0.xyz[:2]) <= v_move:
            if best is not None:
                return best[1], best[2]  # the first motion interval of the target container ended
            continue
        for b in bins:
            o0, o1 = frames[t - 1].objects[b], frames[t].objects[b]
            if b == target_bin or not (o0.visible and o1.visible):
                continue
            if np.linalg.norm(o1.xyz[:2] - o0.xyz[:2]) > v_move:
                d = _dist(a1, o1)
                if best is None or d < best[0]:
                    best = (d, t, b)
    return None if best is None else (best[1], best[2])


def _dist(a, b) -> float:
    if a.uv is not None and b.uv is not None:
        return float(np.linalg.norm(a.uv - b.uv))
    return float(np.linalg.norm(a.xyz - b.xyz))


class CorruptedSource:
    """Wraps another source. `log[ep]` records what was corrupted (frame, swapped ids) or why not."""

    def __init__(self, inner, kind: str = "id_swap", p: float = 0.5, seed: int = 0, v_move: float = DetectParams().v_move):
        assert kind == "id_swap", "M0 implements id_swap only"
        self.inner, self.kind, self.p, self.seed, self.v_move = inner, kind, p, seed, v_move
        self.name = f"{inner.name}+{kind}@{p}"
        self.log: dict[int, dict] = {}

    def episodes(self, split: str, *a, **kw) -> list[int]:
        return self.inner.episodes(split, *a, **kw)

    def meta(self, ep):
        return self.inner.meta(ep)

    def ground_truth(self, ep):
        return self.inner.ground_truth(ep)

    def frames(self, ep: int):
        frames = list(self.inner.frames(ep))  # the corruptor (not the system) may look ahead
        gt = self.inner.ground_truth(ep)
        plan = None
        if chosen(ep, self.p, self.seed):
            plan = find_target_swap(frames, gt.cube_to_bin[gt.targets[0]], self.v_move)
        self.log[ep] = {"selected": chosen(ep, self.p, self.seed), "corrupted": plan is not None,
                        "t": None if plan is None else plan[0],
                        "ids": None if plan is None else [gt.cube_to_bin[gt.targets[0]], plan[1]]}
        for f in frames:
            if plan is not None and f.t >= plan[0]:
                a, b = self.log[ep]["ids"]
                objs = dict(f.objects)
                objs[a], objs[b] = replace(f.objects[b], id=a), replace(f.objects[a], id=b)
                f = replace(f, objects=objs)
            yield f
