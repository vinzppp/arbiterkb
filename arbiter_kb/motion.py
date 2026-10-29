"""Container motion: move_start / move_stop evidence (detectors.py), v2 (NOTES "Motion detector v2"). Causal.

Real tracker positions jump: in G's perceived episodes 26% of still frames move more than 1 mm and 35% of
frames inside a swap move less, so a one-frame speed test starts and stops at random. v2:
- the position is the median of the last m_win frames (spikes go);
- the speed is the smoothed displacement over k_lag frames, per frame;
- a move starts after n_on frames above v_start and stops after n_off frames below v_stop (hysteresis);
- while moving, a speed valley splits the move in two: the speed falls below `valley` x the move's peak and then
  rises above twice that again; the split is at the valley. Back-to-back swaps were one move in v1 (every
  scripted swap slows to almost zero at its end); a reversal (rev_cos) also splits, but most consecutive
  swaps do not reverse (the container goes on to a third place).
m_win = k_lag = n_on = n_off = 1, v_stop = v_start, valley = 0 and rev_cos = -2 reproduce v1.
"""
from __future__ import annotations

import numpy as np

from .kb import AddEvidence
from .observe import obs_id
from .params import DetectParams
from .types import FrameObs


class Motion:
    def __init__(self, p: DetectParams):
        self.p = p
        self.raw: dict[str, list] = {}  # id -> last m_win positions
        self.sm: dict[str, list] = {}  # id -> last k_lag + 1 (t, smoothed position)
        self.st: dict[str, dict] = {}  # id -> {moving, run, off, cand, start}

    @property
    def moving(self) -> dict[str, bool]:
        return {k: s["moving"] for k, s in self.st.items()}

    def step(self, frame: FrameObs) -> list[AddEvidence]:
        ops = []
        for oid, o in frame.objects.items():
            if o.cls != "container":
                continue
            if not o.visible:
                self.raw[oid], self.sm[oid] = [], []
                self.st[oid] = {**self._state(oid), "moving": False, "run": 0, "off": 0}
                continue
            raw = (self.raw.setdefault(oid, []) + [np.asarray(o.xyz, float)])[-self.p.m_win:]
            self.raw[oid] = raw
            sm = (self.sm.setdefault(oid, []) + [(frame.t, np.median(raw, axis=0))])[-(self.p.k_lag + 1):]
            self.sm[oid] = sm
            if len(sm) == self.p.k_lag + 1:
                ops += self._update(oid, frame.t, sm)
        return ops

    def _state(self, oid: str) -> dict:
        return self.st.setdefault(oid, {"moving": False, "run": 0, "off": 0, "cand": None, "start": None,
                                        "peak": 0.0, "low": None})

    def _update(self, oid: str, t: int, sm: list) -> list[AddEvidence]:
        p, s = self.p, self._state(oid)
        v = sm[-1][1][:2] - sm[0][1][:2]
        speed = float(np.linalg.norm(v)) / p.k_lag
        if not s["moving"]:
            s["run"] = s["run"] + 1 if speed > p.v_start else 0
            s["cand"] = sm[0] if s["run"] == 1 else s["cand"]
            if s["run"] >= p.n_on:
                s.update(moving=True, off=0, start=s["cand"][1], peak=speed, low=None)
                return [self._ev(t, "move_start", oid, *s["cand"])]
            return []
        s["peak"] = max(s["peak"], speed)
        if p.valley and s["low"] is None and speed < p.valley * s["peak"]:
            s["low"] = sm[-1]  # a candidate split point
        elif p.valley and s["low"] is not None and speed > 2 * p.valley * s["peak"]:
            low, s["low"] = s["low"], None
            s.update(start=low[1], off=0, peak=speed)
            return [self._ev(t, "move_stop", oid, *low), self._ev(t, "move_start", oid, *low)]
        seg = sm[-1][1][:2] - s["start"][:2]
        if np.linalg.norm(seg) > p.rev_min and speed > p.v_start and \
                float(v @ seg) / (np.linalg.norm(v) * np.linalg.norm(seg)) < p.rev_cos:
            s.update(start=sm[0][1], off=0, peak=speed, low=None)
            return [self._ev(t, "move_stop", oid, *sm[0]), self._ev(t, "move_start", oid, *sm[0])]
        s["off"] = s["off"] + 1 if speed < p.v_stop else 0
        if s["off"] >= p.n_off:
            s.update(moving=False, run=0)
            return [self._ev(t, "move_stop", oid, t, sm[-1][1])]
        return []

    @staticmethod
    def _ev(t: int, kind: str, oid: str, t_at: int, xyz) -> AddEvidence:
        return AddEvidence(t, kind, (oid,), {"xyz": [round(float(v), 4) for v in xyz], "t_at": t_at,
                                             "obs": [obs_id(t_at, oid)]})
