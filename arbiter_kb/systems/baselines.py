"""Deterministic baselines: HODOR-style 'frozen' and the 'bound' (follow the cover) rule.

Thin adapters around sg_compat.GraphTracker (copied from sg_pilot.py). Their answer about a hidden
cube is the container nearest to the tracked cube position, with confidence 1: the one-hot answer
these memories implicitly give.
"""
from __future__ import annotations

import numpy as np

from .. import sg_compat as sg
from ..beliefs import Distribution
from ..params import Params
from ..planner import plan_pick
from ..types import EpisodeMeta, FrameObs, PlanStep


def frame_to_rows(frame: FrameObs, order: list[str]) -> np.ndarray:
    """FrameObs -> GraphTracker's [N, 8] rows (id, label, u, v, x, y, z, hidden)."""
    rows = np.full((len(order), 8), np.nan, np.float32)
    for n, oid in enumerate(order):
        o = frame.objects[oid]
        rows[n, sg.F_ID], rows[n, sg.F_LABEL] = n, 0 if o.cls == "container" else 1
        rows[n, sg.F_HIDDEN] = 0 if o.visible else 1
        if o.visible:
            rows[n, sg.F_X:sg.F_Z + 1] = o.xyz
            if o.uv is not None:
                rows[n, sg.F_U:sg.F_V + 1] = o.uv
    return rows


class TrackerSystem:
    def __init__(self, rule: str, params: Params | None = None):
        assert rule in ("frozen", "bound")
        self.name, self.rule, self.params = rule, rule, params or Params()

    def reset(self, meta: EpisodeMeta) -> None:
        self.order = [oid for oid, _, _ in meta.objects]
        self.cls = {oid: cls for oid, cls, _ in meta.objects}
        self.tracker = sg.GraphTracker(len(self.order), self.rule)
        self.out, self.t, self.rows = None, -1, []

    def step(self, frame: FrameObs) -> None:
        self.t = frame.t
        self.out = self.tracker.update(frame_to_rows(frame, self.order), frame.gripper_width, frame.ee_xyz)
        cubes = [o for o in self.order if self.cls[o] == "cube"]
        self.rows.append({"t": frame.t, "pred": {c: self.believe_inside(c).argmax()[0] for c in cubes}})

    def xyz(self, oid: str) -> np.ndarray | None:
        v = self.out[self.order.index(oid), sg.F_X:sg.F_Z + 1].astype(float)
        return v if np.all(np.isfinite(v)) else None

    pose = xyz  # MemorySystem.pose

    def believe_inside(self, cube: str) -> Distribution:
        c = self.xyz(cube)
        bins = [(b, self.xyz(b)) for b in self.order if self.cls[b] == "container"]
        bins = [(b, p) for b, p in bins if p is not None]
        if c is None or not bins:
            return Distribution.point(None, self.t, [])
        best = min(bins, key=lambda bp: np.linalg.norm(bp[1][:2] - c[:2]))[0]
        return Distribution.point(best, self.t, [f"o{self.t}:{best}"])

    def plan(self, cube: str) -> PlanStep:
        d = self.believe_inside(cube)
        return plan_pick(d, self.xyz, self.params.monitor, d.support)

    def trace_rows(self) -> list[dict]:
        return self.rows
