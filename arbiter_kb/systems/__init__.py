"""Memory systems behind one interface, so baselines and ArbiterKB share the evaluation path (E5d)."""
from __future__ import annotations

from typing import Protocol

from ..beliefs import Distribution
from ..types import EpisodeMeta, FrameObs, PlanStep


class MemorySystem(Protocol):
    name: str

    def reset(self, meta: EpisodeMeta) -> None: ...

    def step(self, frame: FrameObs) -> None: ...

    def believe_inside(self, cube: str) -> Distribution: ...

    def pose(self, obj: str): ...  # best world xyz [3] or None

    def plan(self, cube: str) -> PlanStep: ...

    def trace_rows(self) -> list[dict]: ...
