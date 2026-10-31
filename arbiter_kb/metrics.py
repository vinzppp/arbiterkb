"""Metrics over evaluation rows (DEVELOPMENT.md §3.10). Rates carry 95% Wilson intervals."""
from __future__ import annotations

import math
from collections import defaultdict

from .sg_compat import wilson

KEYS = ("condition", "system")


def brier(probs: dict, truth: str, keys: list[str]) -> float:
    """Σ_c (p_c - 1[c = truth])² over every container, 'elsewhere' (None) and, when the truth is not
    among them (memory lost), the truth itself with p = 0."""
    ks = list(dict.fromkeys(list(keys) + ["None", truth]))
    return sum((probs.get(k, 0.0) - (1.0 if k == truth else 0.0)) ** 2 for k in ks)


def ece(rows: list[dict], bins: int = 10) -> float:
    """Expected calibration error on p_max with equal-width bins."""
    if not rows:
        return math.nan
    groups = defaultdict(list)
    for r in rows:
        groups[min(int(r["p_max"] * bins), bins - 1)].append(r)
    return sum(len(g) / len(rows) * abs(sum(x["correct"] for x in g) / len(g) - sum(x["p_max"] for x in g) / len(g))
               for g in groups.values())


def aurc(rows: list[dict]) -> float:
    """Area under the risk-coverage curve: answer rows from most to least confident; risk at a row =
    error rate among all rows at least that confident (ties enter together). Lower is better; it does
    not depend on a threshold, so a system cannot win by uniformly lowering its confidence."""
    if not rows:
        return math.nan
    groups = defaultdict(list)
    for row in rows:
        groups[row["p_max"]].append(row)
    risk, n_seen, err_seen, total = 0.0, 0, 0, len(rows)
    for p in sorted(groups, reverse=True):
        g = groups[p]
        n_seen, err_seen = n_seen + len(g), err_seen + sum(1 for r in g if not r["correct"])
        risk += len(g) * err_seen / n_seen
    return risk / total


def risk_coverage(rows: list[dict]) -> list[tuple[float, float]]:
    """[(coverage, risk)] at each distinct confidence, most confident first (for the figure)."""
    out, n, err, groups = [], 0, 0, defaultdict(list)
    for row in rows:
        groups[row["p_max"]].append(row)
    for p in sorted(groups, reverse=True):
        g = groups[p]
        n, err = n + len(g), err + sum(1 for r in g if not r["correct"])
        out.append((n / len(rows), err / n))
    return out


def _rate(rows, pred) -> tuple[float, float, float, int, int]:
    k, n = sum(1 for r in rows if pred(r)), len(rows)
    if n == 0:
        return math.nan, math.nan, math.nan, 0, 0
    lo, hi = wilson(k, n)
    lo, hi = max(0.0, lo), min(1.0, hi)
    return k / n, lo, hi, k, n


def summarize(rows: list[dict], tau_hi: float) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in KEYS)].append(r)
    out = []
    for key, g in sorted(groups.items()):
        wrong = [r for r in g if not r["correct"]]
        right = [r for r in g if r["correct"]]
        acc = _rate(g, lambda r: r["correct"])
        silent = _rate(g, lambda r: (not r["correct"]) and r["p_max"] >= tau_hi)
        cov = _rate(g, lambda r: r["p_max"] >= tau_hi)
        corr = [r for r in g if r["corrupted"]]
        row = dict(zip(KEYS, key))
        row.update({
            "n": len(g), "acc": acc[0], "acc_lo": acc[1], "acc_hi": acc[2],
            "silent": silent[0], "silent_lo": silent[1], "silent_hi": silent[2],
            "coverage": cov[0], "cov_lo": cov[1], "cov_hi": cov[2], "aurc": aurc(g),
            "detect_among_wrong": _rate(wrong, lambda r: r["p_max"] < tau_hi)[0],
            "brier": sum(r["brier"] for r in g) / len(g), "ece": ece(g),
            "conf_right": _mean([r["p_max"] for r in right]), "conf_wrong": _mean([r["p_max"] for r in wrong]),
            "acc_moved": _rate([r for r in g if r["moved"]], lambda r: r["correct"])[0],
            "acc_back": _rate([r for r in g if not r["moved"]], lambda r: r["correct"])[0],
            "n_corrupted": len(corr),
            "silent_on_corrupted": _rate(corr, lambda r: (not r["correct"]) and r["p_max"] >= tau_hi)[0],
            "grasp_err_m": _mean([r["grasp_err_m"] for r in g if not math.isnan(r["grasp_err_m"])]),
        })
        out.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan
