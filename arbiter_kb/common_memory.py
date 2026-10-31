"""Deterministic memory arms used by the common RoboMME Plan harness."""
from __future__ import annotations

from .params import Params
from .systems.arbiter_kb import ArbiterKBSystem
from .systems.soft_haircut import SoftHaircut
from .types import EpisodeMeta, FrameObs
from .views import controller_view


def make_system(arm: str, params: Params):
    if arm == "soft":
        return SoftHaircut(params)
    if arm == "arbiter":
        return ArbiterKBSystem(params)
    raise ValueError(f"no deterministic system for {arm!r}")


def _points(frame: FrameObs | None) -> list[str]:
    if frame is None:
        return []
    return [f"  [{name}] {o.cls} at pixel [{o.uv[1]:.0f}, {o.uv[0]:.0f}]"
            for name, o in frame.objects.items() if o.visible and o.uv is not None]


def _soft_view(system: SoftHaircut, goal: str) -> str:
    lines = [f"Goal: {goal}", "Beliefs about hidden cubes:"]
    for cube in system.cubes:
        d = system.believe_inside(cube)
        values = " | ".join(f"{k or 'elsewhere'} {p:.2f}" for k, p in sorted(d.probs.items(), key=lambda x: -x[1]))
        lines.append(f"  [{cube}] inside {values}")
    return "\n".join(lines)


class DeterministicMemory:
    def __init__(self, arm: str, params: Params):
        self.arm, self.system, self.goal = arm, make_system(arm, params), ""
        self.frame: FrameObs | None = None

    def reset(self, meta: EpisodeMeta) -> None:
        self.goal, self.frame = meta.task_goal, None
        self.system.reset(meta)

    def feed(self, frames: list[FrameObs]) -> None:
        for frame in frames:
            self.system.step(frame)
            self.frame = frame

    def text(self) -> str:
        body = _soft_view(self.system, self.goal) if self.arm == "soft" else controller_view(self.system.kb)
        return "\n".join([body, "Visible object-to-pixel lookup:", *_points(self.frame)])

    def beliefs(self) -> list[dict]:
        cubes = list(self.system.cubes) if self.arm == "soft" else [
            k for k, o in self.system.kb.objects.items() if o.cls == "cube"]
        out = []
        for cube in cubes:
            d = self.system.believe_inside(cube)
            answer, p = d.argmax()
            out.append({"cube": cube, "answer": answer, "p_max": p, "probs": d.probs})
        return out
