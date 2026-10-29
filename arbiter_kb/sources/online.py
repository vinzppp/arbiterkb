"""Perception slot rows -> FrameObs, one frame at a time, for the closed loop (harness.py on the pod). The causal
version of PerceivedSource(relinked=True) with the occlusion step: the same Relinker (bit-identical to
relink_margin), and an occlusion step whose cube height z0 is the median of the first `first` sightings SO FAR (the
batch occluded() also uses later sightings until it has `first`; NOTES "Closed loop" gives the agreement on G).
Ids: container slot s -> bin_<s>, cube slot s -> cube_<colour>, from the perception classes.
"""
from __future__ import annotations

import numpy as np

from ..relink import Relinker
from ..sg_compat import F_HIDDEN, F_U, F_V, F_X, F_Z
from ..types import FrameObs, ObjObs
from .perceived import MARGIN_CAP


def _seen(row: np.ndarray, s: int) -> bool:
    return bool(row[s, F_HIDDEN] == 0 and np.all(np.isfinite(row[s, F_U:F_Z + 1])))


def _cube(cls: str, s: int, classes: list[str]) -> tuple[str, dict]:
    """(cube_<colour>, {"color": colour}) for a coloured phrase named once ("green cube."); (cube_s<slot>, {}) for a
    generic "cube." slot (the open word list has several, which must not share a name; smoke run 2026-10-29)."""
    words = cls.rstrip(".").split()
    if len(words) > 1 and sum(c == cls for c in classes) == 1:
        return f"cube_{words[0]}", {"color": words[0]}
    return f"cube_s{s}", {}


class Occluder:
    """A cube track over a visible container and more than dz above its own height counts as covered; visible
    again only back below z0 + dz / 2."""

    def __init__(self, n_bins: int, n_slots: int, r: float, dz: float, first: int = 10):
        self.n, self.S, self.r, self.dz, self.first = n_bins, n_slots, r, dz, first
        self.zs: dict[int, list[float]] = {s: [] for s in range(n_bins, n_slots)}
        self.hid = {s: False for s in range(n_bins, n_slots)}

    def step(self, row: np.ndarray) -> np.ndarray:
        out = np.zeros(self.S, bool)
        for s in range(self.n, self.S):
            if not _seen(row, s):
                continue
            p = row[s, F_X:F_Z + 1]
            if len(self.zs[s]) < self.first:
                self.zs[s].append(float(p[2]))
            z0 = float(np.median(self.zs[s]))
            over = any(_seen(row, c) and np.linalg.norm(row[c, F_X:F_X + 2] - p[:2]) < self.r for c in range(self.n))
            if not self.hid[s] and over and p[2] > z0 + self.dz:
                self.hid[s] = True
            elif self.hid[s] and p[2] < z0 + self.dz / 2:
                self.hid[s] = False
            out[s] = self.hid[s]
        return out


class OnlineObs:
    def __init__(self, classes: list[str], n_bins: int, r: float = 0.08, dz: float = 0.01, relinked: bool = True):
        self.names = [(f"bin_{s}", "container", {}) if s < n_bins else (_cube(c, s, classes)[0], "cube",
                      _cube(c, s, classes)[1]) for s, c in enumerate(classes)]
        self.n, self.t = n_bins, 0
        self.relinker = Relinker(n_bins) if relinked else None
        self.occ = Occluder(n_bins, len(classes), r, dz)

    def objects(self) -> list[tuple[str, str, dict]]:
        return list(self.names)

    def step(self, row: np.ndarray, ee_xyz, gripper_width: float, is_video: bool = False) -> FrameObs:
        margin = np.full(self.n, np.inf)
        if self.relinker is not None:
            row, margin = self.relinker.step(row)
        occ = self.occ.step(row)
        objs = {}
        for s, (oid, cls, _) in enumerate(self.names):
            vis = _seen(row, s) and not occ[s]
            q = min(1.0, margin[s] / MARGIN_CAP) if cls == "container" and self.relinker is not None else 1.0
            objs[oid] = ObjObs(oid, cls, vis, row[s, F_X:F_Z + 1].astype(float) if vis else None,
                               row[s, F_U:F_V + 1].astype(float) if vis else None, q)
        f = FrameObs(self.t, objs, np.asarray(ee_xyz, float), float(gripper_width), is_video)
        self.t += 1
        return f
