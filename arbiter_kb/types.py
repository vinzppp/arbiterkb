"""Data contracts shared by sources, systems and evaluation (DEVELOPMENT.md §4). Fixed since M0.

Units: metres, frames, world frame (+x forward, +y left, +z up, table at z = 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjObs:
    """One object in one frame, as a Source reports it. `id` is the track id and may be WRONG
    under corruption; positions are always the truth of whatever the track is following."""

    id: str  # e.g. "bin_0", "cube_green"
    cls: str  # "container" | "cube"
    visible: bool
    xyz: np.ndarray | None  # world metres; None if not visible
    uv: np.ndarray | None  # front-image pixel (u = column, v = row); None if not visible / no camera
    track_q: float  # 0..1 tracker quality (oracle and synthetic: 1.0)


@dataclass
class FrameObs:
    t: int
    objects: dict[str, ObjObs]
    ee_xyz: np.ndarray  # [3] end-effector position
    gripper_width: float  # metres, sum of finger widths
    is_video: bool
    rgb: np.ndarray | None = None  # front image, only when a later phase needs it


@dataclass
class EpisodeMeta:
    source: str
    episode: int
    split: str  # "train" | "test"
    task_goal: str
    objects: list[tuple[str, str, dict]]  # (id, cls, attributes e.g. {"color": "green"})
    K: np.ndarray | None  # [3, 3] front intrinsics
    E: np.ndarray | None  # [T, 3, 4] world -> camera (OpenCV)


@dataclass
class GroundTruth:
    """Evaluation only. Systems never see it (E5b)."""

    targets: list[str]  # cube ids in pick order
    cube_to_bin: dict[str, str]  # which container hides each cube (the truth for Inside)
    t_star: dict[str, int]  # decision frame per target
    bin_xyz_at_t_star: dict[str, np.ndarray]  # true container position at that target's t*
    moved: dict[str, bool]  # target container ended > 3 cm from where it covered the cube


@dataclass
class PlanStep:
    skill: str  # "pick" | "verify" | "unreachable"
    target: str | None  # container id
    conf: float  # probability of the chosen target
    decision: str  # "act" | "act_on_argmax" | "verify"
    support: list[str] = field(default_factory=list)  # evidence / observation ids behind the choice
    waypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))  # [4, 7] x y z r p y g
