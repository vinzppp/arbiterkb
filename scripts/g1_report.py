"""G1 report for the "memory or policy?" study: discordance per P2 contrast, suggested n, termination
paths, fairness on the real requests, cost and time (main NOTES 7327cf1, "Power and n" and G1).

Not a hypothesis test: G1 measures d, the share of (task, episode) pairs where two arms disagree on
success, and from it n_10 = (2.81 sqrt d + 0.84 sqrt(d - 0.01))^2 / 0.01 and n_TOST = 2.925^2 d / 0.0025.
Free; reads result.json files only.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

P2 = {"P2a MM-m vs LD-m": ("mm_m", "ld_m"), "P2b KB-m vs LD-m": ("kb_m", "ld_m"),
      "P2c FH-m vs LD-m": ("fh_m", "ld_m")}
OTHER = {"split fidelity MM-m vs MEMI": ("mm_m", "memi"), "FH-m vs FH-g": ("fh_m", "fh_g"),
         "MEMI vs FH-g": ("memi", "fh_g")}
PRICES = {"fresh_input": 0.1, "cached_input": 0.01, "cache_write": 0.125, "output": 0.5}


def cost(row: dict) -> float:
    u = (row.get("policy") or {}).get("usage") or {}
    total, cached, written = int(u.get("input_tokens", 0)), int(u.get("cached_tokens", 0)), int(u.get("cache_write_tokens", 0))
    return ((total - cached - written) * PRICES["fresh_input"] + cached * PRICES["cached_input"]
            + written * PRICES["cache_write"] + int(u.get("output_tokens", 0)) * PRICES["output"]) / 1e6


def n_10(d: float) -> float:
    return (2.81 * math.sqrt(d) + 0.84 * math.sqrt(max(d - 0.01, 0.0))) ** 2 / 0.01 if d > 0 else 0.0


def detectable(d: float, n: int) -> float:
    """Smallest difference detectable at 80% power, Holm worst case, with discordance d and n pairs."""
    lo, hi = 1e-4, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        need = (2.81 * math.sqrt(d) + 0.84 * math.sqrt(max(d - mid ** 2, 0.0))) ** 2 / mid ** 2
        lo, hi = (mid, hi) if need > n else (lo, mid)
    return hi


def discordance(cells: dict, a: str, b: str) -> dict:
    pairs = [k for k in cells if a in cells[k] and b in cells[k]]
    a_only = sum(cells[k][a] and not cells[k][b] for k in pairs)
    b_only = sum(cells[k][b] and not cells[k][a] for k in pairs)
    by_task = defaultdict(lambda: [0, 0])
    for k in pairs:
        by_task[k[0]][0] += cells[k][a] != cells[k][b]
        by_task[k[0]][1] += 1
    return {"pairs": len(pairs), "a_only": a_only, "b_only": b_only,
            "d": round((a_only + b_only) / len(pairs), 3) if pairs else None,
            "by_task": {t: f"{x}/{n}" for t, (x, n) in sorted(by_task.items())}}


def fairness(rows: list[dict]) -> dict:
    """Per (task, policy): one policy sha across memories; per memory: one memory-note sha across policies."""
    pol, note, models, statuses = defaultdict(set), defaultdict(set), Counter(), Counter()
    for r in rows:
        for c in (r.get("policy") or {}).get("log") or []:
            statuses[c.get("status")] += 1
            models[c.get("model_returned") or "(none)"] += 1
            if r["arm"] != "memi":
                mem, p = r["arm"].split("_")
                pol[(r["task"], p)].add(c.get("policy_sha256"))
                note[mem].add(c.get("memory_note_sha256"))
    return {"policy_text_identical_across_memories": all(len(v) == 1 for v in pol.values()),
            "memory_note_identical_across_policies": all(len(v) == 1 for v in note.values()),
            "returned_models": dict(models), "call_statuses": dict(statuses)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="run directory (…/memory_or_policy_g1/<run id>)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted(a.run.rglob("result.json"))]
    cells = defaultdict(dict)
    for r in rows:
        if (r.get("terminal") or {}).get("primary_eligible", True):
            cells[(r["task"], r["episode"])][r["arm"]] = bool(r.get("success"))
    p2 = {name: discordance(cells, *arms) for name, arms in P2.items()}
    d_max = max((v["d"] or 0) for v in p2.values())
    arms = sorted({r["arm"] for r in rows})
    report = {
        "cells": len(rows), "pairs": len(cells),
        "success_by_arm": {arm: f"{sum(c.get(arm, False) for c in cells.values())}/"
                                f"{sum(arm in c for c in cells.values())}" for arm in arms},
        "success_by_task_arm": {f"{t}/{arm}": sum(cells[k].get(arm, False) for k in cells if k[0] == t)
                                for t in sorted({k[0] for k in cells}) for arm in arms},
        "p2_discordance": p2, "other_discordance": {n: discordance(cells, *x) for n, x in OTHER.items()},
        "d_max": d_max, "n_10": round(n_10(d_max)), "n_tost": round(2.925 ** 2 * d_max / 0.0025),
        "detectable_at_250": round(detectable(d_max, 250), 3) if d_max else None,
        "terminal_status": dict(Counter((r.get("terminal") or {}).get("status") for r in rows)),
        "outcomes": dict(Counter(r.get("outcome") for r in rows)),
        "max_calls": max(len((r.get("policy") or {}).get("log") or []) for r in rows) if rows else 0,
        "fairness": fairness(rows),
        "cost_usd": round(sum(cost(r) for r in rows), 4),
        "cost_by_arm": {arm: round(sum(cost(r) for r in rows if r["arm"] == arm), 4) for arm in arms},
        "episode_seconds_sum": round(sum(float(r.get("seconds") or 0) for r in rows), 1),
        "mean_seconds_by_task": {t: round(sum(float(r.get("seconds") or 0) for r in rows if r["task"] == t)
                                          / max(1, sum(r["task"] == t for r in rows)), 1)
                                 for t in sorted({r["task"] for r in rows})},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("cells", "d_max", "n_10", "detectable_at_250", "cost_usd",
                                             "terminal_status")}, indent=1))


if __name__ == "__main__":
    main()
