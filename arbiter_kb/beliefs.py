"""Belief (one value with confidence, time, source, support) and Distribution (exclusive hypotheses).

Distribution math is DEVELOPMENT.md §3.3. Every constructor and operation returns a normalised
distribution (invariant I1); operations never mutate, they return a new object.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SOURCES = ("observed", "inferred", "prior", "verified", "arbiter")


@dataclass
class Belief:
    value: Any
    conf: float  # 0..1
    t: int  # frame of last confirmation
    source: str  # one of SOURCES
    support: list[str] = field(default_factory=list)  # evidence / observation / event ids


@dataclass
class Distribution:
    """Mutually exclusive hypotheses over container ids; None means 'unknown / elsewhere'.

    hyp_support[k] lists the ids that put mass on hypothesis k (used by retract);
    support lists ids that changed the whole distribution (e.g. a near-pass that mixed it).
    """

    probs: dict[str | None, float]
    t: int
    support: list[str] = field(default_factory=list)
    hyp_support: dict[str | None, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.probs = _normalise(self.probs)

    # ---- constructors -------------------------------------------------------------------
    @staticmethod
    def point(key: str | None, t: int, support: list[str]) -> "Distribution":
        return Distribution({key: 1.0}, t, list(support), {key: list(support)})

    @staticmethod
    def from_distances(d: dict[str, float], sigma: float, t: int, support: dict[str, list[str]],
                       eps_none: float = 0.0) -> "Distribution":
        """p(k) ∝ exp(-d_k² / 2σ²); no candidate -> {None: 1}. support[k]: ids behind candidate k."""
        if not d:
            return Distribution.point(None, t, [])
        probs: dict[str | None, float] = {k: math.exp(-(v * v) / (2 * sigma * sigma)) for k, v in d.items()}
        if eps_none > 0:
            total = sum(probs.values())
            probs = {k: (1 - eps_none) * v / total for k, v in probs.items()}
            probs[None] = eps_none
        all_ids = sorted({i for ids in support.values() for i in ids})
        return Distribution(probs, t, all_ids, {k: list(support.get(k, [])) for k in probs})

    # ---- operations ---------------------------------------------------------------------
    def swap_mix(self, a: str, b: str, w: float, t: int, ev_id: str) -> "Distribution":
        """With probability w the tracks a and b exchanged identity: move mass w between them."""
        pa, pb = self.probs.get(a, 0.0), self.probs.get(b, 0.0)
        probs = dict(self.probs)
        probs[a], probs[b] = (1 - w) * pa + w * pb, (1 - w) * pb + w * pa
        hs = {k: list(v) for k, v in self.hyp_support.items()}
        both = sorted(set(hs.get(a, [])) | set(hs.get(b, [])))  # after a possible exchange, either
        hs[a], hs[b] = list(both), list(both)  # container's mass rests on both containers' evidence
        return Distribution(probs, t, sorted(set(self.support) | {ev_id}), hs)

    def flatten(self, f: float, plausible: list[str | None], t: int) -> "Distribution":
        """p <- f·p + (1-f)·uniform(plausible). f = λ^Δt for decay (R7) or 1 - w_risk for R8."""
        u = 1.0 / len(plausible)
        probs = {k: f * self.probs.get(k, 0.0) for k in set(self.probs) | set(plausible)}
        for k in plausible:
            probs[k] += (1 - f) * u
        return Distribution(probs, t, list(self.support), {k: list(v) for k, v in self.hyp_support.items()})

    def drop(self, ident: str, t: int) -> "Distribution":
        """Retract `ident`: remove the mass it supports; if it only supported the whole
        distribution, move half of the mass to 'unknown' (None)."""
        backed = [k for k, ids in self.hyp_support.items() if ident in ids]
        if backed:
            probs = {k: p for k, p in self.probs.items() if k not in backed}
            hs = {k: [i for i in v if i != ident] for k, v in self.hyp_support.items() if k not in backed}
            sup = [i for i in self.support if i != ident]
            if sum(probs.values()) <= 0:
                return Distribution.point(None, t, sup)
            return Distribution(probs, t, sup, hs)
        if ident in self.support:
            flat = self.flatten(0.5, [None], t)
            flat.support = [i for i in self.support if i != ident]
            return flat
        return self

    # ---- reads --------------------------------------------------------------------------
    def argmax(self) -> tuple[str | None, float]:
        k = max(self.probs, key=lambda key: (self.probs[key], str(key)))
        return k, self.probs[k]

    def max_prob(self) -> float:
        return self.argmax()[1]

    def entropy(self) -> float:
        return -sum(p * math.log(p) for p in self.probs.values() if p > 0)

    def as_dict(self) -> dict[str, float]:
        return {str(k): round(p, 6) for k, p in sorted(self.probs.items(), key=lambda kv: str(kv[0]))}


def _normalise(probs: dict) -> dict:
    probs = {k: float(v) for k, v in probs.items() if v > 1e-12}
    total = sum(probs.values())
    if total <= 0:
        return {None: 1.0}
    return {k: v / total for k, v in probs.items()}
