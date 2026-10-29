"""PerceivedSource: G's perceived VideoUnmaskSwap episodes (stress/perceive.py npz files) as FrameObs (M6 item 2).

Ids: container slot s -> bin_<s>, cube slot s -> cube_<colour>. The tracker's ids are kept as they are, so a tracker
swap reaches the memory as it would in the closed loop.

Occlusion step (NOTES "Does the KB-in-words gain survive real perception?"): when a container covers a cube, the
tracker's cube mask moves onto the container top instead of vanishing. A cube track therefore counts as not
visible while its centre lies within `r` (xy) of a visible container track and more than `dz` above the cube's own
height (its median z over its first visible frames). It is visible again only back at table height.
With `relinked=True` the container tracks go through relink.py first. With `oracle_cubes=True` the cubes' visibility comes from the simulator instead (the privileged reference row).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..relink import relink
from ..sg_compat import F_HIDDEN, F_U, F_V, F_X, F_Z
from ..types import EpisodeMeta, FrameObs, ObjObs

G_DIR = Path(__file__).resolve().parents[2] / "data" / "g_perceived"


def _seen(rows: np.ndarray, t: int, s: int) -> bool:
    return bool(rows[t, s, F_HIDDEN] == 0 and np.all(np.isfinite(rows[t, s, F_U:F_Z + 1])))


def occluded(per: np.ndarray, n_bins: int, r: float, dz: float, first: int = 10) -> np.ndarray:
    """[T, S] True where a seen cube track counts as covered (see the module docstring)."""
    T, S = per.shape[:2]
    out = np.zeros((T, S), bool)
    for s in range(n_bins, S):
        zs = [per[t, s, F_Z] for t in range(T) if _seen(per, t, s)][:first]
        if not zs:
            continue
        z0, hid = float(np.median(zs)), False
        for t in range(T):
            if not _seen(per, t, s):
                continue
            p = per[t, s, F_X:F_Z + 1]
            over = any(_seen(per, t, c) and np.linalg.norm(per[t, c, F_X:F_X + 2] - p[:2]) < r for c in range(n_bins))
            if not hid and over and p[2] > z0 + dz:
                hid = True
            elif hid and p[2] < z0 + dz / 2:
                hid = False
            out[t, s] = hid
    return out


class PerceivedSource:
    def __init__(self, r: float = 0.05, dz: float = 0.02, oracle_cubes: bool = False, relinked: bool = False,
                 root: Path = G_DIR):
        self.r, self.dz, self.oracle_cubes, self.relinked, self.root = r, dz, oracle_cubes, relinked, root
        self.files = sorted(root.glob("*.npz"))

    def episodes(self, split: str) -> list[int]:
        """train = episodes 0-9 of each setting, test = 10-19 (as G and motion v2)."""
        return [i for i, f in enumerate(self.files) if (int(f.stem.split("_ep")[1]) < 10) == (split == "train")]

    def load(self, ep: int) -> dict:
        z = np.load(self.files[ep], allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        per = relink(z["perceived"], meta["n_bins"]) if self.relinked else z["perceived"]
        return {"name": self.files[ep].stem, "meta": meta, "perceived": per, "oracle": z["oracle"]}

    def ids(self, d: dict) -> list[tuple[str, str, dict]]:
        nb, cls = d["meta"]["n_bins"], d["meta"]["classes"]
        return [(f"bin_{s}", "container", {}) if s < nb else (f"cube_{cls[s].split()[0]}", "cube",
                                                                {"color": cls[s].split()[0]})
                for s in range(len(cls))]

    def meta(self, ep: int) -> EpisodeMeta:
        d = self.load(ep)
        goal = f"pick up the container that hides the {d['meta']['colors'][0]} cube"
        return EpisodeMeta("g_perceived", ep, "train" if ep in self.episodes("train") else "test", goal, self.ids(d),
                           None, None)

    def frames(self, ep: int):
        d = self.load(ep)
        per, ora, nb = d["perceived"], d["oracle"], d["meta"]["n_bins"]
        occ = occluded(per, nb, self.r, self.dz)
        names = self.ids(d)
        for t in range(len(per)):
            objs = {}
            for s, (oid, cls, _) in enumerate(names):
                covered = not _seen(ora, t, s) if self.oracle_cubes and cls == "cube" else occ[t, s]
                vis = _seen(per, t, s) and not covered
                row = per[t, s]
                objs[oid] = ObjObs(oid, cls, vis, row[F_X:F_Z + 1].astype(float) if vis else None,
                                   row[F_U:F_V + 1].astype(float) if vis else None, 1.0)
            yield FrameObs(t, objs, np.array([0.0, 0.0, 0.45]), 0.08, t < len(per) - 1)
