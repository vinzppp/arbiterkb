"""Minimal canonical projection for the zero-cost memory identifiability audit.

This is deliberately not the full RoboMME controller.  It projects a memory's
identity distribution into one arm-neutral schema and asks whether that state
supports a legal ``multi_choice`` pick.  Truth is never an input here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .beliefs import Distribution


@dataclass(frozen=True)
class CanonicalIdentityState:
    target: str
    candidates: tuple[str | None, ...]
    probabilities: tuple[tuple[str | None, float], ...]
    visible_points: tuple[tuple[str, tuple[float, float]], ...]
    evidence_t: int
    support: tuple[str, ...]
    provenance: tuple[tuple[str, str], ...] = ()

    def probs(self) -> dict[str | None, float]:
        return dict(self.probabilities)

    def points(self) -> dict[str, tuple[float, float]]:
        return dict(self.visible_points)

    def top(self) -> tuple[str | None, float]:
        probs = self.probs()
        key = max(self.candidates, key=lambda k: probs[k])
        return key, probs[key]


def project_identity(target: str, distribution: Distribution, candidates: list[str],
                     visible_points: Mapping[str, tuple[float, float]],
                     provenance: Mapping[str, str] | None = None) -> CanonicalIdentityState:
    """Project a complete candidate space: every container plus unknown/elsewhere.

    Missing hypotheses receive zero probability.  The projection never adds
    probability mass, object identity, or a point not present in the memory's
    authorized current observation.
    """
    keys: tuple[str | None, ...] = tuple(dict.fromkeys([*candidates, None]))
    raw = {k: float(distribution.probs.get(k, 0.0)) for k in keys}
    total = sum(raw.values())
    if total <= 0:
        raw = {k: float(k is None) for k in keys}
    else:
        raw = {k: v / total for k, v in raw.items()}
    points = tuple(sorted((k, (float(v[0]), float(v[1]))) for k, v in visible_points.items()
                          if k in candidates))
    return CanonicalIdentityState(
        target=target,
        candidates=keys,
        probabilities=tuple((k, raw[k]) for k in keys),
        visible_points=points,
        evidence_t=int(distribution.t),
        support=tuple(distribution.support),
        provenance=tuple(sorted((provenance or {}).items())),
    )


def hard_pick_probe(state: CanonicalIdentityState) -> dict:
    """Select the top-1 visible target, or report that no justified pick exists."""
    target, confidence = state.top()
    point = state.points().get(target) if target is not None else None
    if target is None or point is None:
        return {"legal": False, "choice": None, "target": None, "point": None,
                "reason": "unknown_or_not_visible", "confidence": confidence}
    return {"legal": True, "choice": "a", "target": target, "point": list(point),
            "reason": "top1_visible", "confidence": confidence}


def threshold_pick_probe(state: CanonicalIdentityState, threshold: float) -> dict:
    """Confidence-gated probe; blocking is not presented as a legal observation action."""
    action = hard_pick_probe(state)
    if action["legal"] and action["confidence"] < threshold:
        return {**action, "legal": False, "choice": None, "target": None, "point": None,
                "reason": "ambiguous_no_legal_information_action"}
    return action
