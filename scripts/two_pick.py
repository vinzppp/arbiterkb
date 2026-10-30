"""A scripted two-pick continuation after an opening video, for requests rebuilt without a simulator
(tools/cache_sim.py, scripts/k2_offline.py; NOTES "K2: initial state, change log, current state").

After the video the hand goes: above bin A, down and close, lift, put A down to one side and open, above bin B, down
and close, lift. Each motion is 30 frames, every third (and the last) fed to the KB, as the closed loop feeds
stride-3 frames. A held bin follows the hand; the cube under a lifted bin becomes visible at that bin's spot; every
visible object gets N(0, 3 mm) noise per frame (perception is not still). The robot follows the script whatever a
model answers (open loop), so every arm reads memories of the same events.
"""
from __future__ import annotations

import numpy as np

from arbiter_kb.closed_loop import gripper_closed
from arbiter_kb.self_memory import record
from arbiter_kb.types import FrameObs, ObjObs

OPEN, CLOSED, NOISE = 0.08, 0.01, 0.003  # gripper width (m), observation noise (m)
DOWN = [-3.1416, 0.0, 0.0]  # hand pointing down
# The camera calibration as the state carries it (fixed for an episode); stand-ins of the real shape.
CALIB = {"wrist_camera_K": [[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]],
         "front_camera_K": [[221.7, 0.0, 128.0], [0.0, 221.7, 128.0], [0.0, 0.0, 1.0]],
         "front_camera_extrinsic_world_to_camera": [[0.0, 1.0, 0.0, 0.0], [0.5547, 0.0, -0.8321, 0.2219],
                                                    [-0.8321, 0.0, -0.5547, 1.2481]]}


class Robot:
    def __init__(self, last: FrameObs, rng, reveal: dict):
        """`reveal`: {bin: (cube id, xyz)}, the cube that becomes visible when that bin is lifted."""
        self.objs = dict(last.objects)
        self.pos = {k: (o.xyz.copy() if o.xyz is not None else None) for k, o in self.objs.items()}  # the truth
        self.ee, self.width, self.held, self.t = np.array([0.0, 0.0, 0.45]), OPEN, None, last.t
        self.rng, self.reveal, self.closed = rng, reveal, False

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
                out.append(FrameObs(self.t, self._observe(), self.ee.copy(), self.width, False))
        return out

    def _observe(self) -> dict:
        objs = {}
        for oid, o in self.objs.items():
            p = self.pos[oid]
            vis = p is not None  # a covered cube has no position until its bin is lifted
            objs[oid] = ObjObs(oid, o.cls, vis, p + self.rng.normal(0, NOISE, 3) if vis else None, None, 1.0)
        return objs

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

    def act(self, kb, t: int, target: str, wp: list, g: int) -> list[FrameObs]:
        """Record the decision as Episode.execute does (gripper 0 keeps the state), then move."""
        record(kb, t, np.array(wp + DOWN), gripper_closed(g, self.closed), target, ())
        self.closed = gripper_closed(g, self.closed)
        return self.frames(wp, g)


def plan(A: str, B: str, pa, pb) -> list[tuple[str, list, int]]:
    """The seven scripted actions: (container acted on, waypoint xyz, gripper command)."""
    side = [pa[0] - 0.05, pa[1] + (0.15 if pa[1] < 0.1 else -0.15), 0.035]
    return [(A, [*pa[:2], 0.17], 1), (A, [*pa[:2], 0.035], -1), (A, [*pa[:2], 0.22], 0), (A, side, 1),
            (B, [*pb[:2], 0.17], 0), (B, [*pb[:2], 0.035], -1), (B, [*pb[:2], 0.22], 0)]
