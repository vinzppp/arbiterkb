"""Statistics for perceived_qa.py: Wilson CIs per row / arm / effort, AUROC of the stated Q1 confidence, paired
bootstrap over episodes (accuracy, top-2, AUROC differences). "Swapped" cubes exclude unknowns (swapped = None)."""
from __future__ import annotations

import re

import numpy as np

from video_windows import wilson

EFFORT = ("default", "off")


def auroc(conf: list[float], ok: list[bool]) -> float | None:
    pos, neg = [c for c, y in zip(conf, ok) if y], [c for c, y in zip(conf, ok) if not y]
    if not pos or not neg:
        return None
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def report(rows: list[dict]) -> dict:
    out = {}
    for row in sorted({r["row"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows}):
            for eff in EFFORT:
                rs = [r for r in rows if (r["row"], r["arm"], r["effort"]) == (row, arm, eff)]
                q1 = [v for r in rs for v in r["q1"].values()]
                stock = [v["right"] for r in rs if r["ep"].startswith("stock") for v in r["q1"].values()]
                stress = [v["right"] for r in rs if not r["ep"].startswith("stock") for v in r["q1"].values()]
                sw = [v for v in q1 if v["swapped"] and v["conf"] is not None]
                other = [k for k in rs[0] if re.fullmatch(r"q[2-9]", k)] if rs else []
                out[f"{row}/{arm}/{eff}"] = {
                    "q1": wilson(sum(v["right"] for v in q1), len(q1)), "q1_stock": wilson(sum(stock), len(stock)),
                    "q1_stress": wilson(sum(stress), len(stress)),
                    "q1_swapped": wilson(sum(v["right"] for v in q1 if v["swapped"]), sum(bool(v["swapped"]) for v in q1)),
                    "auroc_swapped": auroc([v["conf"] for v in sw], [v["right"] for v in sw]),
                    "auroc_all": auroc([v["conf"] for v in q1 if v["conf"] is not None],
                                       [v["right"] for v in q1 if v["conf"] is not None]),
                    "top2": round(float(np.mean([v["top2"] for v in q1])), 3) if q1 else None,
                    "prompt_tokens_median": int(np.median([r.get("prompt_tokens", 0) for r in rs])) if rs else None,
                    "turns_mean": round(float(np.mean([r.get("turns", 1) for r in rs])), 2) if rs else None,
                    **{k: round(float(np.mean([r[k] for r in rs if k in r])), 3) for k in other}}
        for eff in EFFORT:
            for a in ("KT", "KT2"):
                if {"K", a} <= {r["arm"] for r in rows if r["row"] == row}:
                    out[f"{row}/{a}-K/{eff}"] = {"q1": paired(rows, row, a, "K", eff, "all")}
            for a in ("KA", "KN"):
                if {"K", a} <= {r["arm"] for r in rows if r["row"] == row}:
                    out[f"{row}/{a}-K/{eff}"] = {"auroc_all": auroc_pair(rows, row, a, "K", eff),
                                                 "top2": paired(rows, row, a, "K", eff, "all", "top2"),
                                                 "q1": paired(rows, row, a, "K", eff, "all")}
            for a, b in [p for p in (("KR", "R"), ("K", "R"), ("KR", "K")) if {p[0], p[1]} <= {r["arm"] for r in rows}]:
                out[f"{row}/{a}-{b}/{eff}"] = {part: paired(rows, row, a, b, eff, part)
                                               for part in ("all", "stock", "stress")}
    return out


def paired(rows, row, a, b, eff, part, field: str = "right") -> list[float] | None:
    keep = lambda ep: part == "all" or ep.startswith("stock") == (part == "stock")  # noqa: E731
    acc = {(r["arm"], r["ep"]): np.mean([v[field] for v in r["q1"].values()]) for r in rows
           if r["row"] == row and r["effort"] == eff and r["q1"] and keep(r["ep"])}
    d = np.array([acc[(a, ep)] - acc[(b, ep)] for (arm, ep) in acc if arm == a and (b, ep) in acc])
    if not len(d):
        return None
    boot = np.random.default_rng(0).choice(d, (2000, len(d))).mean(1)
    return [round(float(d.mean()), 3), round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]


def auroc_pair(rows, row, a, b, eff) -> list[float] | None:
    """AUROC(a) - AUROC(b) of the stated Q1 confidence, bootstrap over the cubes both answered with a confidence."""
    cube = lambda arm: {(r["ep"], c): v for r in rows if (r["row"], r["arm"], r["effort"]) == (row, arm, eff)  # noqa: E731
                        for c, v in r["q1"].items() if v["conf"] is not None}
    x, y = cube(a), cube(b)
    keys = sorted(k for k in x if k in y)
    if not keys:
        return None
    eps = sorted({k[0] for k in keys})  # resample episodes, not cubes: an episode's cubes are correlated (code review)
    by_ep = {e: [k for k in keys if k[0] == e] for e in eps}
    rng, diffs = np.random.default_rng(0), []
    for _ in range(1000):
        ks = [k for i in rng.integers(0, len(eps), len(eps)) for k in by_ep[eps[i]]]
        da, db = auroc([x[k]["conf"] for k in ks], [x[k]["right"] for k in ks]), \
            auroc([y[k]["conf"] for k in ks], [y[k]["right"] for k in ks])
        if da is not None and db is not None:
            diffs.append(da - db)
    pa, pb = auroc([x[k]["conf"] for k in keys], [x[k]["right"] for k in keys]), \
        auroc([y[k]["conf"] for k in keys], [y[k]["right"] for k in keys])
    return [round(pa - pb, 3), round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]
