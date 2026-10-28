"""RoboMME-like VideoUnmaskSwap episodes generated directly as FrameObs, with exact ground truth.

Mirrors the real mechanics read from robomme_benchmark@53673b3 (evidence-gaps-2026-10-28.md §7):
- t 0-31: cubes visible, containers absent; t = 32: containers appear over the cubes in ONE frame
  (a teleport, no descending motion) and the cubes vanish in that same frame;
- swaps from t = 64, `swap_steps` each, back to back; each moves one container and its nearest
  neighbour along a straight line plus a sideways bulge lane_offset·sin(πα), smoothstep timing
  (statechange.swap_flat_two_lane); the other containers stay put;
- hold 10 frames; t* = the last frame.
Front camera as in VideoUnmaskSwap._default_sensor_configs: eye (0.3, 0, 0.4), target (0, 0, -0.2),
256 x 256, fov 90°.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import sg_compat as sg
from ..types import EpisodeMeta, FrameObs, GroundTruth, ObjObs

COVER_T, SWAP_T0, HOLD = 32, 64, 10
COLORS = ("red", "green", "blue")
LAYOUTS = {
    3: [np.array([[-0.05, -0.1], [-0.05, 0.1], [0.1, 0.0]]), np.array([[0.0, -0.15], [0.0, 0.15], [0.0, 0.0]])],
    4: [np.array([[-0.05, -0.1], [-0.05, 0.1], [0.1, 0.1], [0.1, -0.1]])],
}
BIN_Z, CUBE_Z = 0.03, 0.017
SPLIT_OFFSET = {"train": 0, "test": 1000}


def look_at(eye, target, size=256, fov=np.pi / 2):
    """Pinhole K [3,3] and world->camera E [3,4] (OpenCV: x right, y down, z forward)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    fx = (size / 2) / np.tan(fov / 2)
    K = np.array([[fx, 0, size / 2], [0, fx, size / 2], [0, 0, 1.0]])
    return K, np.hstack([R, (-R @ eye)[:, None]])


K_FRONT, E_FRONT = look_at([0.3, 0.0, 0.4], [0.0, 0.0, -0.2])


@dataclass
class Episode:
    meta: EpisodeMeta
    gt: GroundTruth
    xyz: dict  # id -> [T, 3] true positions
    visible: dict  # id -> [T] bool


def smoothstep(a):
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3 - 2 * a)


def swap_path(A, B, alpha, lane):
    """Positions of the two swapping containers at progress alpha (statechange.swap_flat_two_lane)."""
    d = B - A
    n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
    s, off = smoothstep(alpha), lane * np.sin(np.pi * smoothstep(alpha))
    return A + d * s + n * off, B - d * s - n * off


LANES = (0.0, 0.01, 0.02, 0.04, 0.07)  # "mix": overlapping (<= 0.03) and clearly separated passes
STEPS = (20, 35, 50)


class SyntheticSource:
    """lane_offset / swap_steps: a number, or "mix" to draw one per episode from LANES / STEPS
    (with a separate generator, so episodes with fixed settings stay identical to M0's)."""

    name = "synthetic"

    def __init__(self, n_containers=None, n_swaps=None, swap_steps=50, lane_offset=0.07):
        self.n_containers, self.n_swaps = n_containers, n_swaps
        self.swap_steps, self.lane_offset = swap_steps, lane_offset
        self._cache: dict[int, Episode] = {}

    def settings(self, ep: int) -> tuple[float, int]:
        r = np.random.default_rng([ep, 1])
        lane = float(r.choice(LANES)) if self.lane_offset == "mix" else float(self.lane_offset)
        steps = int(r.choice(STEPS)) if self.swap_steps == "mix" else int(self.swap_steps)
        return lane, steps

    def episodes(self, split: str, n: int = 30) -> list[int]:
        return [SPLIT_OFFSET[split] + i for i in range(n)]

    def meta(self, ep: int) -> EpisodeMeta:
        return self._episode(ep).meta

    def ground_truth(self, ep: int) -> GroundTruth:
        return self._episode(ep).gt

    def frames(self, ep: int):
        e = self._episode(ep)
        T = len(next(iter(e.xyz.values())))
        for t in range(T):
            objs = {}
            for oid, cls, _ in e.meta.objects:
                vis = bool(e.visible[oid][t])
                xyz = e.xyz[oid][t].copy() if vis else None
                uv = sg.project_points(xyz, K_FRONT, E_FRONT)[0] if vis else None
                objs[oid] = ObjObs(oid, cls, vis, xyz, uv, 1.0)
            yield FrameObs(t, objs, np.array([0.0, 0.0, 0.45]), 0.08, t < T - 1)

    # ---- generation ---------------------------------------------------------------------
    def _episode(self, ep: int) -> Episode:
        if ep not in self._cache:
            self._cache[ep] = self._generate(ep)
        return self._cache[ep]

    def _generate(self, ep: int) -> Episode:
        rng = np.random.default_rng(ep)
        lane, steps = self.settings(ep)
        nb = self.n_containers or int(rng.choice([3, 4]))
        ns = self.n_swaps or int(rng.integers(1, 4))
        bins_xy = self._layout(rng, nb)
        colors = list(rng.permutation(COLORS))
        T = SWAP_T0 + ns * steps + HOLD
        track = np.repeat(bins_xy[None], T, axis=0)  # [T, nb, 2]
        for k in range(ns):
            self._swap(track, rng, k, [0, 1, 2], lane, steps)
        bins = [f"bin_{i}" for i in range(nb)]
        cubes = [f"cube_{c}" for c in colors]
        xyz, visible = {}, {}
        for i, b in enumerate(bins):
            xyz[b] = np.concatenate([track[:, i], np.full((T, 1), BIN_Z)], 1)
            visible[b] = np.arange(T) >= COVER_T
        for i, c in enumerate(cubes):  # cube i starts under bin i (RoboMME: the first 3 bins)
            xyz[c] = np.concatenate([np.repeat(bins_xy[i][None], T, 0), np.full((T, 1), CUBE_Z)], 1)
            visible[c] = np.arange(T) < COVER_T
        target = cubes[int(rng.integers(3))]
        tb = bins[cubes.index(target)]
        objects = [(b, "container", {}) for b in bins] + [(c, "cube", {"color": c[5:]}) for c in cubes]
        meta = EpisodeMeta("synthetic", ep, "test" if ep >= 1000 else "train",
                           f"pick up the container that hides the {target[5:]} cube", objects,
                           K_FRONT, np.repeat(E_FRONT[None], T, 0))
        moved = bool(np.linalg.norm(xyz[tb][-1, :2] - xyz[tb][COVER_T, :2]) > 0.03)
        gt = GroundTruth([target], {c: bins[i] for i, c in enumerate(cubes)}, {target: T - 1},
                         {target: xyz[tb][-1].copy()}, {target: moved})
        return Episode(meta, gt, xyz, visible)

    def _layout(self, rng, nb):
        base = LAYOUTS[nb][int(rng.integers(len(LAYOUTS[nb])))]
        a = np.deg2rad(rng.uniform(0, 180))
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return base @ R.T + rng.uniform(-0.01, 0.01, size=base.shape)

    def _swap(self, track, rng, k, cube_bins, lane, steps):
        """Swap k: a random cube-carrying container and its nearest neighbour, steps [t0, t1)."""
        t0, t1 = SWAP_T0 + k * steps, SWAP_T0 + (k + 1) * steps
        pos = track[t0].copy()
        a = int(rng.choice(cube_bins))
        dist = [np.linalg.norm(pos[a] - pos[j]) if j != a else np.inf for j in range(len(pos))]
        b = int(np.argmin(dist))
        for t in range(t0, len(track)):
            alpha = (t - t0) / (t1 - t0)
            track[t, a], track[t, b] = swap_path(pos[a], pos[b], alpha, lane)
