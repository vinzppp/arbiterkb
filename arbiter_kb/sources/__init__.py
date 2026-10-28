"""Sources produce FrameObs one frame at a time. Only sources know where observations come from."""
from __future__ import annotations

from typing import Iterator, Protocol

from ..types import EpisodeMeta, FrameObs, GroundTruth


class Source(Protocol):
    name: str

    def episodes(self, split: str) -> list[int]: ...

    def meta(self, ep: int) -> EpisodeMeta: ...

    def frames(self, ep: int) -> Iterator[FrameObs]: ...

    def ground_truth(self, ep: int) -> GroundTruth: ...  # evaluation only
