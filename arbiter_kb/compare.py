"""M2 comparisons (DEVELOPMENT §6 M2 (c)): bootstrap CIs for AURC, a paired bootstrap for the AURC
difference between two systems, and silent failure at matched clean coverage.

Bootstraps resample EPISODES (all rows of an episode together: its cubes share one scene), with a fixed
seed. The paired bootstrap draws the same episodes for both systems. Matched coverage: on calibration
rows (train seeds), pick the baseline threshold θ whose clean coverage P(p_max ≥ θ) is nearest the
reference system's clean coverage at τ_hi; then report the baseline's silent rate P(wrong ∧ p_max ≥ θ)
on evaluation rows, next to the reference's at τ_hi. θ comes from calibration rows only.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .sg_compat import wilson

N_BOOT, SEED = 2000, 0


def load_rows(run_dir: Path) -> list[dict]:
    """rows.csv of a run, with p_max, correct and episode typed back."""
    with open(Path(run_dir) / "rows.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["p_max"], r["correct"], r["episode"] = float(r["p_max"]), r["correct"] == "True", int(r["episode"])
    return rows


def select(rows: list[dict], condition: str, system: str) -> list[dict]:
    return [r for r in rows if r["condition"] == condition and r["system"] == system]


def aurc_np(p: np.ndarray, err: np.ndarray) -> float:
    """metrics.aurc on arrays: ties at one confidence enter together."""
    vals, inv = np.unique(-p, return_inverse=True)  # ascending -p = descending confidence
    cnt, nerr = np.bincount(inv, minlength=len(vals)), np.bincount(inv, weights=err, minlength=len(vals))
    cum_n, cum_err = np.cumsum(cnt), np.cumsum(nerr)
    return float(np.sum(cnt * cum_err / cum_n) / len(p))


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(p, err, per-episode row indices) in a fixed episode order."""
    p = np.array([r["p_max"] for r in rows], float)
    err = np.array([0.0 if r["correct"] else 1.0 for r in rows])
    eps = sorted({r["episode"] for r in rows})
    pos = {e: i for i, e in enumerate(eps)}
    idx: list[list[int]] = [[] for _ in eps]
    for i, r in enumerate(rows):
        idx[pos[r["episode"]]].append(i)
    return p, err, [np.array(x) for x in idx]


def _draws(n_eps: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_eps, size=(n_boot, n_eps))


def bootstrap_aurc(rows: list[dict], n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """(AURC, 2.5%, 97.5%) over episode resamples."""
    p, err, idx = _arrays(rows)
    stats = [aurc_np(p[ii], err[ii]) for d in _draws(len(idx), n_boot, seed)
             for ii in [np.concatenate([idx[k] for k in d])]]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return aurc_np(p, err), float(lo), float(hi)


def paired_aurc_diff(ref: list[dict], base: list[dict], n_boot: int = N_BOOT,
                     seed: int = SEED) -> tuple[float, float, float]:
    """(AURC_ref - AURC_base, 2.5%, 97.5%), the same episodes drawn for both. Negative = ref better."""
    key = lambda r: (r["episode"], r["cube"])  # noqa: E731
    ref, base = sorted(ref, key=key), sorted(base, key=key)
    assert [key(r) for r in ref] == [key(r) for r in base], "paired bootstrap needs the same rows"
    (pa, ea, idx), (pb, eb, _) = _arrays(ref), _arrays(base)
    stats = []
    for d in _draws(len(idx), n_boot, seed):
        ii = np.concatenate([idx[k] for k in d])
        stats.append(aurc_np(pa[ii], ea[ii]) - aurc_np(pb[ii], eb[ii]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return aurc_np(pa, ea) - aurc_np(pb, eb), float(lo), float(hi)


def coverage(rows: list[dict], theta: float) -> float:
    return sum(r["p_max"] >= theta for r in rows) / len(rows)


def threshold_for_coverage(rows: list[dict], target: float) -> float:
    """The θ among the rows' confidences whose coverage is nearest `target`; a tie goes to the higher
    coverage (the baseline answers at least as much as asked). A binary haircut can only hit a few
    coverages, so the achieved one is always reported next to the target."""
    cands = sorted({r["p_max"] for r in rows}, reverse=True)
    return min(cands, key=lambda th: (round(abs(coverage(rows, th) - target), 9), -coverage(rows, th)))


def silent_at(rows: list[dict], theta: float) -> tuple[float, float, float]:
    """(silent rate, Wilson 95% lo, hi) with the answer threshold θ."""
    k, n = sum((not r["correct"]) and r["p_max"] >= theta for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return k / n, max(0.0, lo), min(1.0, hi)
