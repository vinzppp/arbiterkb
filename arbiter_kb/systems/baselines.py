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


NAMES = {"haircut": "bound_haircut", "overlap_haircut": "overlap_haircut", "jump": "jump_detector"}


class GuardedBound:
    """Bound plus one cheap doubt signal (M1 baselines, DEVELOPMENT §6 M1):
    - haircut: after any near-pass involving the cube's container, answer {pred: c, partner: 1 - c};
    - overlap_haircut: the same, only for passes whose closest 3D distance < d_ov (Bayes-optimal
      under id_swap_overlap, which we designed: it shows what copying our own choices achieves);
    - jump: a container label that moves > j_max in one frame -> {pred: 0.5, nearest other: 0.5}."""

    def __init__(self, mode: str, params: Params | None = None):
        assert mode in NAMES
        self.mode, self.name, self.params = mode, NAMES[mode], params or Params()
        self.inner = TrackerSystem("bound", self.params)

    def reset(self, meta: EpisodeMeta) -> None:
        from ..detectors import Detectors

        self.inner.reset(meta)
        self.det, self.doubt, self.prev, self.rows = Detectors(self.params.detect), {}, None, []

    def step(self, frame: FrameObs) -> None:
        self.inner.step(frame)
        cubes = [c for c in self.inner.order if self.inner.cls[c] == "cube"]
        pairs = self._triggers(frame)
        for cube in cubes:
            pred = self.inner.believe_inside(cube).argmax()[0]
            for a, b in pairs:
                if pred in (a, b):
                    self.doubt[cube] = (a, b, frame.t)
        self.prev = frame
        self.rows.append({"t": frame.t, "pred": {c: self.believe_inside(c).argmax()[0] for c in cubes},
                          "p_max": {c: round(self.believe_inside(c).argmax()[1], 4) for c in cubes},
                          "doubt": {c: list(v) for c, v in self.doubt.items()}})

    def _triggers(self, frame: FrameObs) -> list[tuple[str, str]]:
        b = self.params.baseline
        if self.mode != "jump":
            evs = [e for e in self.det.step(frame) if e.kind == "near_pass"]
            return [tuple(e.args) for e in evs if self.mode == "haircut" or e.data.get("min_3d", 1e9) < b.d_ov]
        out = []
        for k, o in frame.objects.items():
            p = self.prev.objects.get(k) if self.prev else None
            if o.cls == "container" and o.visible and p is not None and p.visible \
                    and float(np.linalg.norm(o.xyz - p.xyz)) > b.j_max:
                out.append((k, self._nearest_other(k)))
        return out

    def _nearest_other(self, k: str) -> str:
        me = self.inner.xyz(k)
        others = [(float(np.linalg.norm(x[:2] - me[:2])), o) for o in self.inner.order
                  if self.inner.cls[o] == "container" and o != k and (x := self.inner.xyz(o)) is not None]
        return min(others)[1] if others else k

    def pose(self, oid: str):
        return self.inner.xyz(oid)

    def believe_inside(self, cube: str) -> Distribution:
        d = self.inner.believe_inside(cube)
        pred, pair = d.argmax()[0], self.doubt.get(cube)
        if pred is None or pair is None or pred not in pair[:2]:
            return d
        partner = pair[1] if pred == pair[0] else pair[0]
        c = 0.5 if self.mode == "jump" else self.params.baseline.c
        return Distribution({pred: c, partner: 1 - c}, d.t, d.support, {pred: d.support, partner: d.support})

    def plan(self, cube: str) -> PlanStep:
        d = self.believe_inside(cube)
        return plan_pick(d, self.inner.xyz, self.params.monitor, d.support)

    def trace_rows(self) -> list[dict]:
        return self.rows
