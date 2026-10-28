"""Change detectors: observations -> numbered evidence (appear, disappear, move_start, move_stop,
near_pass). Task-agnostic and causal: they only use frames up to t.

near_pass(a, b): two containers both moving come closer than d_px in the image (d_3d metres without a
camera); it is emitted when the pass ends, with closeness = min distance / threshold in [0, 1) and
min_3d = the smallest 3D centre distance seen while the pass was open.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .kb import AddEvidence
from .observe import obs_id
from .params import DetectParams
from .types import FrameObs


class Detectors:
    def __init__(self, p: DetectParams):
        self.p = p
        self.prev: dict = {}  # id -> last ObjObs
        self.moving: dict[str, bool] = {}
        self.passes: dict[tuple, dict] = {}  # open near-passes

    def step(self, frame: FrameObs) -> list[AddEvidence]:
        t, ops = frame.t, []
        for oid, o in frame.objects.items():
            prev = self.prev.get(oid)
            if o.visible and (prev is None or not prev.visible):
                ops.append(AddEvidence(t, "appear", (oid,), {"obs": [obs_id(t, oid)]}))
            if prev is not None and prev.visible and not o.visible:
                ops.append(AddEvidence(t, "disappear", (oid,), {"last_xyz": [float(v) for v in prev.xyz],
                                                                "obs": [obs_id(t - 1, oid)]}))
        ops += self._motion(frame)
        ops += self._near_passes(frame)
        self.prev = dict(frame.objects)
        return ops

    def _motion(self, frame: FrameObs) -> list[AddEvidence]:
        ops = []
        for oid, o in frame.objects.items():
            prev = self.prev.get(oid)
            if o.cls != "container" or prev is None or not (o.visible and prev.visible):
                self.moving[oid] = False
                continue
            mv = float(np.linalg.norm(o.xyz[:2] - prev.xyz[:2])) > self.p.v_move
            if mv != self.moving.get(oid, False):
                kind = "move_start" if mv else "move_stop"
                ops.append(AddEvidence(frame.t, kind, (oid,), {"obs": [obs_id(frame.t, oid)]}))
            self.moving[oid] = mv
        return ops

    def _near_passes(self, frame: FrameObs) -> list[AddEvidence]:
        ops, t = [], frame.t
        bins = sorted(k for k, o in frame.objects.items() if o.cls == "container")
        for a, b in combinations(bins, 2):
            both = self.moving.get(a) and self.moving.get(b)
            if both:
                d, thr = self._distance(frame.objects[a], frame.objects[b])
                if d < thr:
                    cur = self.passes.get((a, b))
                    d3 = float(np.linalg.norm(frame.objects[a].xyz - frame.objects[b].xyz))
                    min_3d = min(d3, cur["min_3d"]) if cur else d3
                    if cur is None or d < cur["min_d"]:
                        cur = {"min_d": d, "thr": thr, "t_min": t, "t_open": cur["t_open"] if cur else t}
                    self.passes[(a, b)] = {**cur, "min_3d": min_3d}
            elif (a, b) in self.passes:
                ps = self.passes.pop((a, b))
                data = {"closeness": ps["min_d"] / ps["thr"], "min_d": ps["min_d"], "thr": ps["thr"], "min_3d": ps["min_3d"],
                        "t_min": ps["t_min"], "t_open": ps["t_open"],
                        "obs": [obs_id(ps["t_min"], a), obs_id(ps["t_min"], b)]}
                ops.append(AddEvidence(t, "near_pass", (a, b), data))
        return ops

    def _distance(self, oa, ob) -> tuple[float, float]:
        if oa.uv is not None and ob.uv is not None:
            return float(np.linalg.norm(oa.uv - ob.uv)), self.p.d_px
        return float(np.linalg.norm(oa.xyz - ob.xyz)), self.p.d_3d
