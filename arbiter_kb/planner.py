"""Planner: belief about a cube -> (skill, target container, confidence, decision, waypoints).

Positions come only from the system's own geometry (the KB's L0 for ArbiterKB)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import skills
from .beliefs import Distribution
from .monitor import decide
from .params import MonitorParams
from .types import PlanStep


def plan_pick(dist: Distribution, pose_of: Callable[[str], np.ndarray | None], mp: MonitorParams,
              support: list[str]) -> PlanStep:
    target, p = dist.argmax()
    xyz = pose_of(target) if target is not None else None
    if xyz is None:
        return PlanStep("verify", target, p, "verify", list(support))
    return PlanStep("pick", target, p, decide(p, mp), list(support), skills.pick(xyz))
