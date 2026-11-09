"""Report for count_test.py runs (rows.csv + meta.json): every number in NOTES' counting tables comes from here.

  python scripts/count_report.py RUN_DIR [RUN_DIR ...] [--ref K3] [--pair M/ARM/EFF:M/ARM/EFF] [--out FILE]

Scoring (fixed in v1, NOTES "Counting test results"): an answer is right when it equals the truth; for SwingXtimes a
wrap-up truth (put the cube on the table / press the button) also accepts the other wrap-up action, for every arm.
Per (model, arm, effort): accuracy with a Wilson 95% interval, per task, median input tokens and seconds. Paired
against --ref at the same model and effort: mean difference over decision points with a bootstrap 95% interval
(2000 draws, seed 0), decisions won / lost, and the most common errors. With goal specs in meta.json (arm K), the
parsed target is checked against the dataset's execution subgoals. Also split by the episode's true target count
(the same subgoals): accuracy, n and median input tokens per (model, arm, effort) and target.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.sources import lerobot as L  # noqa: E402

WRAP = {"put the cube on the table", "press the button"}
TASKS = ("PickXtimes", "SwingXtimes", "BinFill")
COUNTED = {"PickXtimes": ("onto the target", {"pick_up", "release"}), "BinFill": ("into the bin", {"pick_up", "release"}),
           "SwingXtimes": ("left-side target", {"hold_left", "hold_right"})}


def right(r: dict) -> bool:
    return r["answer"] == r["truth"] or (r["task"] == "SwingXtimes" and r["truth"] in WRAP and r["answer"] in WRAP)


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0, 0.0]
    p, z = k / n, 1.96
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 3), round(c - h, 3), round(c + h, 3)]


def cells(rows: list[dict]) -> dict:
    out = {}
    for key in sorted({(r["model"], r["arm"], r["effort"]) for r in rows}):
        rs = [r for r in rows if (r["model"], r["arm"], r["effort"]) == key]
        out["/".join(key)] = {"right": wilson(sum(map(right, rs)), len(rs)), "n": len(rs),
                              **{t: round(float(np.mean([right(r) for r in rs if r["task"] == t])), 3)
                                 for t in TASKS if any(r["task"] == t for r in rs)},
                              "tokens_in": int(np.median([float(r["prompt_tokens"]) for r in rs])),
                              "median_s": round(float(np.median([float(r["seconds"]) for r in rs])), 1)}
    return out


def pair(rows: list[dict], x: tuple, y: tuple) -> dict | None:
    """x - y over the decision points both answered; x, y = (model, arm, effort)."""
    key = lambda r: (r["task"], r["episode"], r["k"])  # noqa: E731
    a = {key(r): right(r) for r in rows if (r["model"], r["arm"], r["effort"]) == x}
    b = {key(r): right(r) for r in rows if (r["model"], r["arm"], r["effort"]) == y}
    common = [k for k in a if k in b]
    if not common:
        return None
    d = np.array([a[k] - b[k] for k in common], float)
    boot = np.random.default_rng(0).choice(d, (2000, len(d))).mean(1)
    errs = collections.Counter(f"{r['task']} | {r['truth']} -> {r['answer']}" for r in rows
                               if (r["model"], r["arm"], r["effort"]) == x and not right(r))
    return {"diff": [round(float(d.mean()), 3), round(float(np.percentile(boot, 2.5)), 3),
                     round(float(np.percentile(boot, 97.5)), 3)],
            "won_lost": [sum(a[k] and not b[k] for k in common), sum(b[k] and not a[k] for k in common)],
            "errors": dict(errs.most_common(4))}


def paired(rows: list[dict], ref: str, extra: list[str]) -> dict:
    """Every arm against `ref` at the same model and effort, plus the --pair requests 'm/arm/eff:m/arm/eff'."""
    todo = [((m, arm, e), (m, ref, e)) for m, arm, e in sorted({(r["model"], r["arm"], r["effort"]) for r in rows})
            if arm != ref]
    todo += [tuple(tuple(side.rsplit("/", 2)) for side in p.split(":")) for p in extra]
    out = {f"{'/'.join(x)} - {'/'.join(y)}": pair(rows, x, y) for x, y in todo}
    return {k: v for k, v in out.items() if v}


def targets(tasks: set) -> dict:
    """(task, episode) -> the true target count, from the execution subgoals."""
    out = {}
    for task in tasks:
        key = COUNTED[task][0]
        for p in L.episode_files(task):
            ep = L.load(p)
            out[(task, str(ep["meta"]["episode"]))] = sum(key in s[2] for s in L.segments(ep["simple"])
                                                          if not ep["demo"][s[0]])
    return out


def by_target(rows: list[dict]) -> dict:
    tgt, out = targets({r["task"] for r in rows}), {}
    for key in sorted({(r["model"], r["arm"], r["effort"], tgt[(r["task"], str(r["episode"]))]) for r in rows}):
        rs = [r for r in rows if (r["model"], r["arm"], r["effort"], tgt[(r["task"], str(r["episode"]))]) == key]
        out["/".join(map(str, key))] = {"right": round(float(np.mean([right(r) for r in rs])), 3), "n": len(rs),
                                        "tokens_in": int(np.median([float(r["prompt_tokens"]) for r in rs]))}
    return out


def parse_accuracy(specs: dict) -> dict:
    out = {}
    for task, (key, kinds) in COUNTED.items():
        ok = n = 0
        for p in L.episode_files(task):
            ep = L.load(p)
            if str(ep["meta"]["episode"]) not in specs:
                continue
            tgt = sum(key in s[2] for s in L.segments(ep["simple"]) if not ep["demo"][s[0]])
            sp, n = specs[str(ep["meta"]["episode"])], n + 1
            ok += bool(sp) and sp["count"] in kinds and sp["target"] == tgt
        if n:
            out[task] = [ok, n]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default="K3")
    ap.add_argument("--out", default="")
    ap.add_argument("--pair", action="append", default=[], help="cross-model pair, e.g. "
                    "anthropic/claude-sonnet-5.5/P/default:qwen/qwen3.6-35b-a3b/K/default")
    a = ap.parse_args()
    rows, specs = [], {}
    for run in a.runs:
        for f in sorted(Path(run).glob("rows*.csv")):  # rows_<x>.csv pairs with meta_<x>.json (v3 kept two runs)
            meta = json.loads((f.parent / f.name.replace("rows", "meta").replace(".csv", ".json")).read_text())
            specs.update(meta.get("goal_specs") or {})
            rows += [{**r, "model": r.get("model") or meta["args"]["model"]} for r in csv.DictReader(open(f))]
    rep = {"runs": a.runs, "ref": a.ref, "cells": cells(rows), "paired": paired(rows, a.ref, a.pair),
           "by_target": by_target(rows)}
    if specs:
        rep["goal_parse"] = parse_accuracy({str(k): v for k, v in specs.items()})
    text = json.dumps(rep, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
