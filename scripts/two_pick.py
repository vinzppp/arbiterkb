"""Scripted robot continuations after an opening video, for requests rebuilt without a simulator
(tools/cache_sim.py, scripts/k2_offline.py; NOTES "K2: initial state, change log, current state").

Each motion is 30 frames, every third (and the last) fed to the KB, as the closed loop feeds stride-3 frames. A held
bin follows the hand; the cube under a lifted bin becomes visible at that bin's spot. Observation noise, when given,
is a residual drawn per frame and object from real perception on still containers (scripts/still_noise.py); none
otherwise. The robot follows the script whatever a model answers (open loop), so every arm reads memories of the same
events.
- `plan`: the plain two-pick (above A, close, lift, put A aside, above B, close, lift), for the cost simulation.
- `probe_plan`: the offline test's script, asked only where the answer needs memory (NOTES "K vs K2 offline,
  redesigned").
"""
from __future__ import annotations

import numpy as np

from arbiter_kb.closed_loop import gripper_closed
from arbiter_kb.self_memory import record
from arbiter_kb.types import FrameObs, ObjObs

OPEN, CLOSED = 0.08, 0.01  # gripper width (m)
HOME = [0.0, 0.0, 0.45]
DOWN = [-3.1416, 0.0, 0.0]  # hand pointing down
# The camera calibration as the state carries it (fixed for an episode); stand-ins of the real shape.
CALIB = {"wrist_camera_K": [[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]],
         "front_camera_K": [[221.7, 0.0, 128.0], [0.0, 221.7, 128.0], [0.0, 0.0, 1.0]],
         "front_camera_extrinsic_world_to_camera": [[0.0, 1.0, 0.0, 0.0], [0.5547, 0.0, -0.8321, 0.2219],
                                                    [-0.8321, 0.0, -0.5547, 1.2481]]}


def noisy(frame: FrameObs, rng, resid: np.ndarray | None) -> FrameObs:
    """The frame with one residual (drawn from `resid`) added to each visible object; unchanged without a pool."""
    if resid is None:
        return frame
    objs = {k: ObjObs(o.id, o.cls, o.visible, o.xyz + resid[rng.integers(len(resid))] if o.visible else None, o.uv,
                      o.track_q) for k, o in frame.objects.items()}
    return FrameObs(frame.t, objs, frame.ee_xyz, frame.gripper_width, frame.is_video, frame.rgb)


class Robot:
    def __init__(self, last: FrameObs, rng, reveal: dict, resid: np.ndarray | None = None):
        """`reveal`: {bin: (cube id, xyz)}, the cube that becomes visible when that bin is lifted."""
        self.objs = dict(last.objects)
        self.pos = {k: (o.xyz.copy() if o.xyz is not None else None) for k, o in self.objs.items()}  # the truth
        self.ee, self.width, self.held, self.t = np.array(HOME), OPEN, None, last.t
        self.rng, self.reveal, self.resid, self.closed = rng, reveal, resid, False

    def frames(self, to, gripper: int, n: int = 30) -> list[FrameObs]:
        """Move the hand to `to` in n frames, then apply the gripper command (it acts at the end of the motion)."""
        start, out = self.ee.copy(), []
        for k in range(1, n + 1):
            self.ee = start + (np.asarray(to, float) - start) * k / n
            if k == n and gripper:
                self.width = CLOSED if gripper < 0 else OPEN
                self.held = min((b for b in self.pos if b.startswith("bin_") and self.pos[b] is not None),
                                key=lambda b: np.linalg.norm(self.pos[b][:2] - self.ee[:2])) if gripper < 0 else None
            if self.held:
                self.pos[self.held] = self.ee.copy()
                if self.ee[2] > 0.08 and self.held in self.reveal:
                    c, p = self.reveal.pop(self.held)
                    self.pos[c] = p
            self.t += 1
            if k % 3 == 0 or k == n:
                out.append(self._frame())
        for _ in range(2 if gripper else 0):  # the fingers take time to act: the hand waits 2 observed frames (6 steps;
            self.t += 3  # RoboMME's gripper settles in ~8 dataset frames), so a grasp is seen settled (actions.py)
            out.append(self._frame())
        return out

    def _frame(self) -> FrameObs:
        objs = {oid: ObjObs(oid, o.cls, self.pos[oid] is not None, None if self.pos[oid] is None else
                            self.pos[oid].copy(), None, 1.0) for oid, o in self.objs.items()}
        return noisy(FrameObs(self.t, objs, self.ee.copy(), self.width, False), self.rng, self.resid)

    def state(self) -> dict:
        """The state text the closed loop sends (closed_loop.state), from the scripted robot."""
        return {"tcp_pose_xyz_rpy": [round(float(v), 4) for v in self.ee] + DOWN,
                "finger_positions_m": [self.width / 2] * 2, "elapsed_control_steps": int(self.t * 3),
                "wrist_camera_K": CALIB["wrist_camera_K"],
                "wrist_camera_extrinsic_world_to_camera": [[0.0, -1.0, 0.0, round(float(self.ee[1]), 4)],
                                                           [-1.0, 0.0, 0.0, round(float(self.ee[0]), 4)],
                                                           [0.0, 0.0, -1.0, round(float(self.ee[2]) + 0.1, 4)]],
                "front_camera_K": CALIB["front_camera_K"],
                "front_camera_extrinsic_world_to_camera": CALIB["front_camera_extrinsic_world_to_camera"]}

    def act(self, kb, t: int, target: str | None, wp: list, g: int) -> list[FrameObs]:
        """Record the decision as Episode.execute does (gripper 0 keeps the state), then move."""
        record(kb, t, np.array(list(wp) + DOWN), gripper_closed(g, self.closed), target, ())
        self.closed = gripper_closed(g, self.closed)
        return self.frames(wp, g)


def plan(A: str, B: str, pa, pb) -> list[tuple[str, list, int]]:
    """The seven scripted actions: (container acted on, waypoint xyz, gripper command)."""
    side = [pa[0] - 0.05, pa[1] + (0.15 if pa[1] < 0.1 else -0.15), 0.035]
    return [(A, [*pa[:2], 0.17], 1), (A, [*pa[:2], 0.035], -1), (A, [*pa[:2], 0.22], 0), (A, side, 1),
            (B, [*pb[:2], 0.17], 0), (B, [*pb[:2], 0.035], -1), (B, [*pb[:2], 0.22], 0)]


def nearest_other(pos: dict, x: str, skip: set) -> str:
    """The container nearest container x in xy, among those not in `skip`."""
    return min((b for b in pos if b.startswith("bin_") and b not in skip and pos[b] is not None),
               key=lambda b: float(np.linalg.norm(pos[b][:2] - pos[x][:2])))


def probe_plan(A: str, B: str, pos: dict) -> list[tuple[str | None, str | None, list, int]]:
    """(right container if the model is asked BEFORE this action else None, recorded target, waypoint, gripper).
    The five asks: at home; above D1; at home after A was put aside; above A's new spot; above D2."""
    pa, pb = pos[A], pos[B]
    d1, d2 = nearest_other(pos, A, {A, B}), nearest_other(pos, B, {A, B})
    side = [pa[0] - 0.05, pa[1] + (0.15 if pa[1] < 0.1 else -0.15), 0.035]
    return [(A, None, [*pos[d1][:2], 0.17], 0), (A, A, [*pa[:2], 0.17], 0), (None, A, [*pa[:2], 0.035], -1),
            (None, A, [*pa[:2], 0.22], 0), (None, A, side, 1), (None, None, HOME, 0),
            (B, None, [*side[:2], 0.17], 0), (B, None, [*pos[d2][:2], 0.17], 0), (B, None, None, 0)]
