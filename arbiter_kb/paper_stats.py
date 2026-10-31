"""Small, deterministic statistics used by the paper comparison audit.

The resampling unit is an episode.  Rows inside an episode are correlated because
they share the same scene and tracker history.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .metrics import aurc, ece
from .sg_compat import wilson


def rate_interval(values: list[bool]) -> list[float | int]:
    """Rate, Wilson 95% interval, numerator and denominator."""
    k, n = sum(values), len(values)
    if not n:
        return [math.nan, math.nan, math.nan, 0, 0]
    lo, hi = wilson(k, n)
    return [k / n, max(0.0, lo), min(1.0, hi), k, n]


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Two-sided exact McNemar test; ``a_only`` means a right and b wrong."""
    if len(a) != len(b):
        raise ValueError("paired outcomes must have the same length")
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    n, tail = a_only + b_only, min(a_only, b_only)
    p = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(tail + 1)) / 2**n)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_exact": p}


def belief_summary(rows: list[dict], tau_hi: float) -> dict:
    """Accuracy, selective-risk and calibration summaries over belief rows."""
    acc = rate_interval([bool(r["correct"]) for r in rows])
    silent = rate_interval([not r["correct"] and r["p_max"] >= tau_hi for r in rows])
    coverage = rate_interval([r["p_max"] >= tau_hi for r in rows])
    return {
        "episodes": len({r["ep"] for r in rows}),
        "n": len(rows),
        "accuracy": acc,
        "silent_failure": silent,
        "coverage": coverage,
        "aurc": aurc(rows),
        "brier": float(np.mean([r["brier"] for r in rows])),
        "ece": ece(rows),
    }


def _metric(rows: list[dict], name: str) -> float:
    if name == "accuracy":
        return float(np.mean([r["correct"] for r in rows]))
    if name == "aurc":
        return aurc(rows)
    if name == "brier":
        return float(np.mean([r["brier"] for r in rows]))
    raise ValueError(name)


def paired_bootstrap(a: list[dict], b: list[dict], metric: str, n_boot: int = 5000,
                     seed: int = 0) -> list[float]:
    """Metric(a)-metric(b), percentile CI, paired by row and resampled by episode."""
    key: Callable[[dict], tuple] = lambda r: (r["ep"], r["cube"])
    aa, bb = {key(r): r for r in a}, {key(r): r for r in b}
    if set(aa) != set(bb):
        raise ValueError("paired rows differ")
    eps = sorted({k[0] for k in aa})
    by_ep = {ep: [k for k in aa if k[0] == ep] for ep in eps}
    rng, draws = np.random.default_rng(seed), []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(eps), len(eps))
        keys = [k for i in chosen for k in by_ep[eps[i]]]
        draws.append(_metric([aa[k] for k in keys], metric) - _metric([bb[k] for k in keys], metric))
    point = _metric(list(aa.values()), metric) - _metric(list(bb.values()), metric)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(point), float(lo), float(hi)]


def paired_correct(a: list[dict], b: list[dict]) -> tuple[list[bool], list[bool]]:
    """Aligned correctness vectors for a paired test."""
    key = lambda r: (r["ep"], r["cube"])  # noqa: E731
    aa, bb = {key(r): bool(r["correct"]) for r in a}, {key(r): bool(r["correct"]) for r in b}
    if set(aa) != set(bb):
        raise ValueError("paired rows differ")
    keys = sorted(aa)
    return [aa[k] for k in keys], [bb[k] for k in keys]
