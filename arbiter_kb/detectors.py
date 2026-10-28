"""Change detectors: observations -> numbered evidence (appear, disappear, move_start, move_stop,
near_pass, grip_close, grip_open). Task-agnostic and causal: they only use frames up to t.
move_start / move_stop carry the object's xyz, so the event log can say from where to where (M3);
grip_close / grip_open carry the end-effector xyz and the gripper width (hysteresis g_close < g_open).

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
        self.closed = False  # gripper state, with hysteresis
        self.ee_prev, self.still, self.paused = None, 0, True  # hand-pause state (starts at rest)

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
        ops += self._gripper(frame)
        ops += self._hand_pause(frame)
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
                kind, at, t_at = ("move_start", prev.xyz, frame.t - 1) if mv else ("move_stop", o.xyz, frame.t)
                ops.append(AddEvidence(frame.t, kind, (oid,), {"xyz": [round(float(v), 4) for v in at], "t_at": t_at,
                                                               "obs": [obs_id(t_at, oid)]}))
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

    def _gripper(self, frame: FrameObs) -> list[AddEvidence]:
        w = float(frame.gripper_width)
        flip = (not self.closed and w < self.p.g_close) or (self.closed and w > self.p.g_open)
        if not flip:
            return []
        self.closed = not self.closed
        data = {"ee": [round(float(v), 4) for v in frame.ee_xyz], "width": round(w, 4), "obs": [obs_id(frame.t, "ee")]}
        return [AddEvidence(frame.t, "grip_close" if self.closed else "grip_open", (), data)]

    def _hand_pause(self, frame: FrameObs) -> list[AddEvidence]:
        """`hand_pause` once the end effector has been still for n_pause frames after moving: where the robot
        stopped (the robot's own actions, no vision needed; M3 robot-state memory)."""
        ee = np.asarray(frame.ee_xyz, float)
        moved = self.ee_prev is not None and float(np.linalg.norm(ee - self.ee_prev)) > self.p.v_pause
        self.ee_prev = ee
        if moved:
            self.still, self.paused = 0, False
            return []
        self.still += 1
        if self.paused or self.still < self.p.n_pause:
            return []
        self.paused = True
        data = {"ee": [round(float(v), 4) for v in ee], "closed": self.closed, "obs": [obs_id(frame.t, "ee")]}
        return [AddEvidence(frame.t, "hand_pause", (), data)]

    def _distance(self, oa, ob) -> tuple[float, float]:
        if oa.uv is not None and ob.uv is not None:
            return float(np.linalg.norm(oa.uv - ob.uv)), self.p.d_px
        return float(np.linalg.norm(oa.xyz - ob.xyz)), self.p.d_3d
