"""Operations: the only way rules, detectors and the arbiter change the KB (applied by KB.apply)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .beliefs import Distribution


@dataclass
class SetPose:
    obj: str; xyz: Any; conf: float; t: int; source: str; support: list  # noqa: E702


@dataclass
class SetVis:
    obj: str; state: str; conf: float; t: int; support: list  # noqa: E702


@dataclass
class SetDist:
    pred: str; subj: str; dist: Distribution  # noqa: E702


@dataclass
class MixDist:
    pred: str; subj: str; a: str; b: str; w: float; t: int; ev_id: str  # noqa: E702


@dataclass
class FlattenDist:
    pred: str; subj: str; f: float; plausible: list; t: int  # noqa: E702


@dataclass
class AddEvidence:
    t: int; kind: str; args: tuple; data: dict  # noqa: E702


@dataclass
class AddEvent:
    t: int; type: str; args: tuple; conf: float; support: list; source: str = "rule"  # noqa: E702


@dataclass
class Retract:
    ident: str
