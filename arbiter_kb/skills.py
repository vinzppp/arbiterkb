"""Skills turn a target pose into RoboMME `waypoint` actions: [x, y, z, roll, pitch, yaw, gripper]
in the world frame, gripper -1 close / 1 open / 0 keep (robomme doc/env_format.md)."""
from __future__ import annotations

import numpy as np

SAFE_Z = 0.20  # m: pre-grasp and lift height (RoboMME's own solve_pickup_bin reaches at z = 0.2)


def pick(xyz: np.ndarray, yaw: float = 0.0) -> np.ndarray:
    """Four waypoints: pre-grasp, descend, close, lift. Orientation top-down (π, 0, yaw).

    TODO M5: yaw from the container's orientation; reach check before sending (an unreachable
    waypoint ends a RoboMME episode: FailAwareWrapper returns status 'error')."""
    x, y, z = (float(v) for v in xyz)
    rpy = [np.pi, 0.0, float(yaw)]
    return np.array([
        [x, y, SAFE_Z, *rpy, 1.0],
        [x, y, z, *rpy, 0.0],
        [x, y, z, *rpy, -1.0],
        [x, y, SAFE_Z, *rpy, 0.0],
    ])
