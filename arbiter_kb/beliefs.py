"""Belief (one value with confidence, time, source, support) and Distribution (exclusive hypotheses).

Distribution math is DEVELOPMENT.md §3.3. Every constructor and operation returns a normalised
distribution (invariant I1); operations never mutate, they return a new object.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

SOURCES = ("observed", "inferred", "prior", "verified", "arbiter", "vlm")  # enforced by KB.check(); vlm: the keyframe call (v1.9; the annotator from v1.6)


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
                       eps_none: float = 0.0, none_support: list[str] | None = None) -> "Distribution":
        """p(k) ∝ exp(-d_k² / 2σ²); no candidate -> {None: 1} backed by none_support.
        The exponent is shifted by the smallest d² so no valid candidate underflows to 0."""
        if not d:
            return Distribution.point(None, t, list(none_support or []))
        d2min = min(v * v for v in d.values())
        probs: dict[str | None, float] = {k: math.exp(-(v * v - d2min) / (2 * sigma * sigma)) for k, v in d.items()}
        if eps_none > 0:
            total = sum(probs.values())
            probs = {k: (1 - eps_none) * v / total for k, v in probs.items()}
            probs[None] = eps_none
        all_ids = sorted({i for ids in support.values() for i in ids} | set(none_support or []))
        hs = {k: list(support.get(k, [])) for k in probs}
        if None in probs:
            hs[None] = list(none_support or [])  # "elsewhere" rests on the evidence that triggered R1
        return Distribution(probs, t, all_ids, hs)

    # ---- operations ---------------------------------------------------------------------
    def swap_mix(self, a: str, b: str, w: float, t: int, ev_id: str) -> "Distribution":
        """With probability w the tracks a and b exchanged identity: move mass w between them.

        Each hypothesis keeps its own evidence (a key that had none inherits the other's), so a
        later retract removes the CURRENT mass on the hypotheses that evidence backs. This is an
        approximation after mixing; exact per-origin mass tracking is deferred to M5 (R4 reveal)."""
        assert 0.0 <= w <= 1.0, f"swap weight {w} outside [0, 1]"
        pa, pb = self.probs.get(a, 0.0), self.probs.get(b, 0.0)
        probs = dict(self.probs)
        probs[a], probs[b] = (1 - w) * pa + w * pb, (1 - w) * pb + w * pa
        hs = {k: list(v) for k, v in self.hyp_support.items()}
        sa, sb = hs.get(a, []), hs.get(b, [])
        hs[a], hs[b] = (sa or list(sb)), (sb or list(sa))
        return Distribution(probs, t, sorted(set(self.support) | {ev_id}), hs)

    def flatten(self, f: float, plausible: list[str | None], t: int) -> "Distribution":
        """p <- f·p + (1-f)·uniform(plausible). f = 1 - w_risk for R8 (R7 decay was removed in v1.6)."""
        if not plausible:
            return self
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
        k = max(self.probs, key=self.probs.__getitem__)  # ties -> the first key (callers order keys)
        return k, self.probs[k]

    def max_prob(self) -> float:
        return self.argmax()[1]

    def entropy(self) -> float:
        return -sum(p * math.log(p) for p in self.probs.values() if p > 0)

    def as_dict(self) -> dict[str, float]:
        return {str(k): round(p, 6) for k, p in sorted(self.probs.items(), key=lambda kv: str(kv[0]))}


def joint_exclusive(dists: dict[str, Distribution], t: int, max_states: int = 5000) -> dict[str, Distribution] | None:
    """R5: at most one cube per container. Treats the cube marginals as INDEPENDENT evidence, forms the
    joint over assignments with no container used twice (None may repeat), and returns the marginals.
    Only valid when the marginals really are independent (e.g. merging a new cover); after swap_mix the
    cubes are correlated and this would be overconfident, so R5 skips mixed cubes. Returns None when no
    assignment is consistent (e.g. two cubes certain of the same container); too many states -> input."""
    cubes = list(dists)
    keys = [list(dists[c].probs) for c in cubes]
    if not cubes or math.prod(len(k) for k in keys) > max_states:
        return dict(dists)
    marg = {c: {} for c in cubes}
    total = 0.0
    for combo in itertools.product(*keys):
        used = [k for k in combo if k is not None]
        if len(used) != len(set(used)):
            continue
        w = math.prod(dists[c].probs[k] for c, k in zip(cubes, combo))
        total += w
        for c, k in zip(cubes, combo):
            marg[c][k] = marg[c].get(k, 0.0) + w
    if total <= 0:
        return None
    sup = sorted({i for d in dists.values() for i in d.support})
    return {c: Distribution(marg[c], t, sup, {k: list(v) for k, v in dists[c].hyp_support.items()}) for c in cubes}


def _normalise(probs: dict) -> dict:
    probs = {k: float(v) for k, v in probs.items() if v > 1e-12}
    total = sum(probs.values())
    if total <= 0:
        return {None: 1.0}
    return {k: v / total for k, v in probs.items()}
