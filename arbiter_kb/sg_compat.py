"""Functions copied verbatim from scene-graph-robot/notebooks/pilot/sg_pilot.py @ e2e06f9.

ArbiterKB copies instead of importing so its results cannot change when the other repo changes
(DEVELOPMENT.md §0.4). tests/test_sg_compat.py checks equality with the original when it is present.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# copied from sg_pilot.py:47 @ e2e06f9
F_ID, F_LABEL, F_U, F_V, F_X, F_Y, F_Z, F_HIDDEN = range(8)

# copied from sg_pilot.py:579 @ e2e06f9
PICK_Z = 0.15  # metres: RoboMME counts a container as picked up when its pose z exceeds this


# copied from sg_pilot.py:364 @ e2e06f9
@dataclass
class Replay:
    task: str
    episode: int
    front_rgb: np.ndarray  # [T, H, W, 3] uint8
    wrist_rgb: np.ndarray  # [T, H, W, 3] uint8
    front_depth_m: np.ndarray  # [T, H, W] float32 metres
    wrist_depth_m: np.ndarray  # [T, H, W] float32 metres
    front_seg: np.ndarray  # [T, H, W] int32 segmentation ids
    wrist_seg: np.ndarray  # [T, H, W] int32
    K_front: np.ndarray  # [3, 3]
    K_wrist: np.ndarray  # [3, 3]
    E_front: np.ndarray  # [T, 3, 4] world -> camera (OpenCV), extrinsic_cv
    E_wrist: np.ndarray  # [T, 3, 4]
    joint_state: np.ndarray  # [T, 7]
    gripper_state: np.ndarray  # [T, 2] finger widths
    ee_xyz: np.ndarray  # [T, 3]
    is_video: np.ndarray  # [T] bool, True for the conditioning video shown before acting
    policy_frame: np.ndarray  # [K] frame index observed before policy action k
    actions: np.ndarray  # [K, 8] joint_action used for the replay
    task_goal: str
    scene: dict  # from scene_objects(), minus live actors
    status: str
    # [T, N, 3] actor pose positions (world, m) per node; NaN for the video frames before the last
    # reset frame (reset plays the video internally, so only its final state is observable)
    obj_xyz: np.ndarray | None = None


# copied from sg_pilot.py:539 @ e2e06f9
def load_replay(path: Path, keys: list[str] | None = None) -> Replay:
    """A cached replay. keys: load only these arrays (others become empty), e.g. to keep RAM low;
    a full replay is ~0.5 GB in memory (depth and segmentation of every frame, both cameras)."""
    import dataclasses

    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    meta["scene"]["cube_to_bin"] = {
        int(k): int(v) for k, v in meta["scene"]["cube_to_bin"].items()
    }
    meta["scene"]["seg_ids"] = {int(k): v for k, v in meta["scene"]["seg_ids"].items()}
    arrays = [k for k in z.files if k != "meta"]
    want = arrays if keys is None else [k for k in arrays if k in keys]
    data = {k: z[k] for k in want}
    for f in dataclasses.fields(Replay):  # skipped or absent arrays: empty, so a wrong use fails loudly
        if f.name not in data and f.name not in meta and f.default is dataclasses.MISSING:
            data[f.name] = np.zeros(0)
    return Replay(**data, **meta)


# copied from sg_pilot.py:583 @ e2e06f9
def target_cube_nodes(scene: dict, task_goal: str) -> list[int]:
    """The cubes named in the instruction, in the order they must be picked.

    Some episodes ask for two: '... hiding the blue cube, finally pick up another container hiding
    the red cube'. The order of mention is the env's subgoal order (VideoUnmaskSwap._load_scene).
    """
    found = []
    for o in scene["objects"]:
        if o["label_id"] == 1 and o.get("color"):
            i = task_goal.find(f"{o['color']} cube")
            if i >= 0:
                found.append((i, o["node"]))
    if not found:
        raise ValueError(
            f"no cube colour of {[o.get('color') for o in scene['objects']]} found in {task_goal!r}"
        )
    return [n for _, n in sorted(found)]


# copied from sg_pilot.py:602 @ e2e06f9
class ActiveTarget:
    """The target currently asked for: the first until its container is picked up, then the next.

    A pick counts once the target's container (the cube rides with it) has pose z > PICK_Z, the
    test RoboMME itself uses to advance the subgoal. Latched: a container put back down stays picked.
    """

    def __init__(self, targets: list[int], cube_to_bin: dict):
        self.targets, self.bins = list(targets), [cube_to_bin[c] for c in targets]
        self.k = 0

    def update(self, obj_xyz: np.ndarray) -> int:
        """obj_xyz [N, 3] pose positions this frame (NaN rows ignored) -> active target node."""
        while self.k < len(self.targets) - 1 and obj_xyz[self.bins[self.k], 2] > PICK_Z:
            self.k += 1
        return self.targets[self.k]


# copied from sg_pilot.py:620 @ e2e06f9
def active_targets(rep: Replay, targets: list[int]) -> np.ndarray:
    """[T] active target node per replayed frame (ActiveTarget over the recorded poses)."""
    at = ActiveTarget(targets, rep.scene["cube_to_bin"])
    return np.array([at.update(rep.obj_xyz[t]) for t in range(len(rep.obj_xyz))])


# copied from sg_pilot.py:626 @ e2e06f9
def decision_frame(rep: Replay) -> int:
    """First frame the policy acts on: the remembered fact (which container) is needed here."""
    return int(rep.policy_frame[0])


# copied from sg_pilot.py:634 @ e2e06f9
def cam_to_world(E_world_to_cam: np.ndarray) -> np.ndarray:
    """Invert a 3x4 OpenCV extrinsic (world -> camera) into a 4x4 camera -> world matrix."""
    R, t = E_world_to_cam[:3, :3], E_world_to_cam[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


# copied from sg_pilot.py:643 @ e2e06f9
def project_points(xyz: np.ndarray, K: np.ndarray, E_world_to_cam: np.ndarray) -> np.ndarray:
    """World points [N, 3] -> pixel (u, v) [N, 2] with the pinhole model (u = column, v = row)."""
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    cam = (E_world_to_cam[:3, :3] @ xyz.T + E_world_to_cam[:3, 3:4]).T
    z = np.where(np.abs(cam[:, 2:3]) < 1e-9, 1e-9, cam[:, 2:3])
    uv = (K @ (cam / z).T).T[:, :2]
    return uv


# copied from sg_pilot.py:652 @ e2e06f9
def backproject_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    E_world_to_cam: np.ndarray,
    zmin: float = 0.05,
    zmax: float = 3.0,
) -> np.ndarray | None:
    """Mean world point of the masked pixels with valid depth; None if no valid pixel."""
    v, u = np.nonzero(mask & (depth_m > zmin) & (depth_m < zmax))
    if len(u) == 0:
        return None
    z = depth_m[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z, np.ones_like(z)], 1)
    world = (cam_to_world(E_world_to_cam) @ pts.T).T[:, :3]
    return world.mean(0)


# copied from sg_pilot.py:672 @ e2e06f9
def oracle_graph(rep: Replay, min_px: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """oracle_graph [T, N, 8] from simulator segmentation + depth.

    Visible in the front camera: (u, v) = mask pixel centroid, xyz = back-projected mask mean.
    Visible only in the wrist camera: xyz from the wrist mask, (u, v) = its front projection.
    Hidden in both: hidden = 1 and xyz/u/v = NaN (the update rules in Step 6 fill these).
    Also returns full_xyz [T, N, 3]: hidden cubes take their paired container's xyz, because the
    simulator moves each cube with its container during the swap (VideoUnmaskSwap.step,
    lift_and_drop_objectA_onto_objectB).
    """
    objs = rep.scene["objects"]
    T, N = len(rep.front_rgb), len(objs)
    g = np.full((T, N, 8), np.nan, np.float32)
    for o in objs:
        g[:, o["node"], F_ID] = o["node"]
        g[:, o["node"], F_LABEL] = o["label_id"]
    for t in range(T):
        for o in objs:
            n, ids = o["node"], rep.scene["seg_ids"][o["node"]]
            mf = np.isin(rep.front_seg[t], ids)
            mw = np.isin(rep.wrist_seg[t], ids)
            if mf.sum() >= min_px:
                v, u = np.nonzero(mf)
                g[t, n, F_U], g[t, n, F_V] = u.mean(), v.mean()
                xyz = backproject_mask(rep.front_depth_m[t], mf, rep.K_front, rep.E_front[t])
                if xyz is not None:
                    g[t, n, F_X : F_Z + 1] = xyz
                g[t, n, F_HIDDEN] = 0
            elif mw.sum() >= min_px:
                xyz = backproject_mask(rep.wrist_depth_m[t], mw, rep.K_wrist, rep.E_wrist[t])
                if xyz is not None:
                    g[t, n, F_X : F_Z + 1] = xyz
                    g[t, n, F_U : F_V + 1] = project_points(xyz, rep.K_front, rep.E_front[t])[
                        0
                    ]
                g[t, n, F_HIDDEN] = 0
            else:
                g[t, n, F_HIDDEN] = 1
    full_xyz = g[..., F_X : F_Z + 1].copy()
    for cube, bin_ in rep.scene["cube_to_bin"].items():
        hid = g[:, cube, F_HIDDEN] == 1
        full_xyz[hid, cube] = g[hid, bin_, F_X : F_Z + 1]
    return g, full_xyz


# copied from sg_pilot.py:720 @ e2e06f9
@dataclass
class RuleParams:
    r_bind: float = 0.06  # metres: hidden object's last xy must lie within r of the cover's xy
    g_closed: float = 0.03  # gripper total width (m) below which the gripper counts as closed
    d_grip: float = 0.05  # metres: end-effector within d of the object for the gripper rule


# copied from sg_pilot.py:727 @ e2e06f9
class GraphTracker:
    """Fills hidden nodes frame by frame, using only frames <= t.

    rule = 'frozen': keep the last seen position.
    rule = 'bound' : at the frame an object disappears, bind it to the nearest other node visible at
                     that frame whose xy lies within r_bind (tie-break: nearest), or to the end effector
                     if the gripper is closed within d_grip; afterwards follow the partner with the offset
                     measured at binding time. No partner -> frozen.
    """

    def __init__(self, n_nodes: int, rule: str, params: RuleParams | None = None):
        assert rule in ("frozen", "bound")
        self.rule, self.p, self.n = rule, params or RuleParams(), n_nodes
        self.last = np.full((n_nodes, 3), np.nan)
        self.partner: dict[int, int | str] = {}
        self.offset: dict[int, np.ndarray] = {}
        self.prev_hidden = np.ones(n_nodes, bool)
        self.events: list[tuple[int, int, Any]] = []
        self.t = -1

    def update(
        self, frame: np.ndarray, gripper_width: float, ee_xyz: np.ndarray
    ) -> np.ndarray:
        """frame [N, 8] with NaN for hidden nodes -> filled [N, 8] (u, v left NaN for the caller to project)."""
        self.t += 1
        out = frame.copy()
        hidden = frame[:, F_HIDDEN] == 1
        vis_xyz = frame[:, F_X : F_Z + 1]
        # nodes that just disappeared: choose a partner (bound) or nothing (frozen)
        for n in np.nonzero(hidden & ~self.prev_hidden)[0]:
            self.partner.pop(n, None)
            if self.rule == "bound" and np.all(np.isfinite(self.last[n])):
                closed = gripper_width < self.p.g_closed
                if closed and np.linalg.norm(ee_xyz - self.last[n]) < self.p.d_grip:
                    self.partner[n], self.offset[n] = "ee", self.last[n] - ee_xyz
                else:
                    cands = [
                        (np.linalg.norm(vis_xyz[m, :2] - self.last[n, :2]), m)
                        for m in range(self.n)
                        if m != n and not hidden[m] and np.all(np.isfinite(vis_xyz[m]))
                    ]
                    cands = [c for c in cands if c[0] < self.p.r_bind]
                    if cands:
                        m = min(cands)[1]
                        self.partner[n], self.offset[n] = m, self.last[n] - vis_xyz[m]
                self.events.append((self.t, int(n), self.partner.get(n)))
        # visible nodes: record; release any binding
        for n in np.nonzero(~hidden)[0]:
            if np.all(np.isfinite(vis_xyz[n])):
                self.last[n] = vis_xyz[n]
            self.partner.pop(n, None)
        # hidden nodes: fill from partner (partners filled first if they are hidden themselves)
        for n in np.nonzero(hidden)[0]:
            p = self.partner.get(n)
            if p == "ee":
                self.last[n] = ee_xyz + self.offset[n]
            elif p is not None and np.all(np.isfinite(self.last[p])):
                self.last[n] = self.last[p] + self.offset[n]
            out[n, F_X : F_Z + 1] = self.last[n]
            out[n, F_U : F_V + 1] = np.nan
        self.prev_hidden = hidden
        return out


# copied from sg_pilot.py:791 @ e2e06f9
def apply_rule(
    graph: np.ndarray,
    rule: str,
    gripper_width: np.ndarray,
    ee_xyz: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    params: RuleParams | None = None,
) -> tuple[np.ndarray, list]:
    """Run GraphTracker over [T, N, 8]; hidden nodes get xyz from the rule and (u, v) by projection."""
    T, N, _ = graph.shape
    tr = GraphTracker(N, rule, params)
    out = np.empty_like(graph)
    for t in range(T):
        f = tr.update(graph[t], float(gripper_width[t]), ee_xyz[t])
        hid = f[:, F_HIDDEN] == 1
        ok = hid & np.all(np.isfinite(f[:, F_X : F_Z + 1]), 1)
        if ok.any():
            f[ok, F_U : F_V + 1] = project_points(f[ok, F_X : F_Z + 1], K, E[t])
        out[t] = f
    return out, tr.events


# copied from sg_pilot.py:1997 @ e2e06f9
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


# copied from sg_pilot.py:1332 @ e2e06f9 (G: match perceived slots to oracle nodes)
def assign_slots(perceived: np.ndarray, oracle: np.ndarray, t: int, radius_px: float = 25.0):
    """Hungarian assignment slot -> oracle node at frame t on (u, v) distance, same label only."""
    from scipy.optimize import linear_sum_assignment

    S, N = perceived.shape[1], oracle.shape[1]
    cost = np.full((S, N), 1e6)
    for s in range(S):
        for n in range(N):
            if (
                perceived[t, s, F_LABEL] == oracle[t, n, F_LABEL]
                and np.all(np.isfinite(perceived[t, s, F_U : F_V + 1]))
                and np.all(np.isfinite(oracle[t, n, F_U : F_V + 1]))
            ):
                cost[s, n] = np.linalg.norm(
                    perceived[t, s, F_U : F_V + 1] - oracle[t, n, F_U : F_V + 1]
                )
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < radius_px}


def assign_slots_seeded(perceived: np.ndarray, oracle: np.ndarray, radius_px: float = 25.0):
    """Slot -> oracle node, each slot matched at the first frame it has a mask (its seed frame).

    Slots seed at different frames (cubes at frame 0, containers only once they enter the view), so a
    single-frame assignment would leave some slots unmatched. Slots sharing a seed frame are matched
    jointly (assign_slots, same label, Hungarian on (u, v)); a node is used at most once.
    Returns (assignment, {slot: seed frame}).
    """
    S = perceived.shape[1]
    seed = {}
    for s in range(S):
        vis = np.nonzero(perceived[:, s, F_HIDDEN] == 0)[0]
        if len(vis):
            seed[s] = int(vis[0])
    out: dict[int, int] = {}
    for t in sorted(set(seed.values())):
        group = [s for s in seed if seed[s] == t]
        sub = perceived[:, group].copy()
        orc = oracle.copy()
        orc[:, list(out.values()), F_U : F_V + 1] = np.nan  # already taken
        for i, n in assign_slots(sub, orc, t, radius_px).items():
            out[group[i]] = n
    return out, seed
