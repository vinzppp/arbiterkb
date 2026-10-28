"""Franka Panda forward kinematics (numpy): 7 joint angles -> TCP position in the world frame.

Modified (Craig) DH parameters from Franka's documentation; the flange is 0.107 m past joint 7 and the TCP
(between the fingertips) a further 0.1034 m. RoboMME's robot base sits at (-0.615, 0, 0) in the world frame
(doc/env_format.md; the same value the controller prompt uses). Checked against the dataset's grounded
subgoals in NOTES (table-level TCP positions vs the subgoal pixel coordinates).
"""
from __future__ import annotations

import numpy as np

BASE = np.array([-0.615, 0.0, 0.0])
# (a, d, alpha) per joint, modified DH; theta = joint angle
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
FLANGE_D, TCP_D = 0.107, 0.1034


def _link(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Modified DH: RotX(alpha) TransX(a) RotZ(theta) TransZ(d)."""
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def tcp(q: np.ndarray) -> np.ndarray:
    """World xyz of the TCP for joint angles q [7]."""
    T = np.eye(4)
    for (a, d, alpha), theta in zip(DH, q[:7]):
        T = T @ _link(a, d, alpha, float(theta))
    T = T @ _link(0.0, FLANGE_D + TCP_D, 0.0, 0.0)
    return T[:3, 3] + BASE


def tcp_path(states: np.ndarray) -> np.ndarray:
    """[T, 3] TCP positions for states [T, >=7]."""
    return np.stack([tcp(s) for s in states])
