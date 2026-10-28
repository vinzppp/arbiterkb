"""Summarise dry runs (scripts/dry_run.py) per (arm, effort) and compare each against a reference, paired by episode.

  python scripts/dry_report.py results/runs/dry_synthetic_<stamp> [--ref K:default] [--margin 0.10] [--keep-as NAME]

Per group: right-container rate and within-3-cm rate with Wilson 95% CIs, median xy error, mean tokens /
seconds per call, total $. Each other group vs the reference: the paired difference of the right-container
rate with a 95% episode bootstrap CI (2000 draws, seed 0), labelled by the rule fixed BEFORE the run (NOTES):
  better / worse  the CI excludes 0;
  same            the CI lies inside [-margin, +margin];
  undecided       otherwise (n too small to say).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.sg_compat import wilson  # noqa: E402


def load(run: Path) -> dict:
    groups: dict = defaultdict(dict)
    with open(run / "dry.csv") as f:
        for r in csv.DictReader(f):
            groups[(r["arm"], r.get("effort", "default"))][int(r["episode"])] = r
    return groups


def rate(rows: list[dict], key) -> tuple[float, float, float]:
    k, n = sum(map(key, rows)), len(rows)
    lo, hi = wilson(k, n)
    return round(k / n, 3), round(max(lo, 0.0), 3), round(min(hi, 1.0), 3)


def right(r) -> bool:
    return r["target_is_true_bin"] == "True"


def near(r) -> bool:
    return bool(r["xy_err_m"]) and float(r["xy_err_m"]) < 0.03


def summary(key, g: dict) -> dict:
    rows = list(g.values())
    errs = [float(r["xy_err_m"]) for r in rows if r["xy_err_m"]]
    mean = lambda k: round(float(np.mean([float(r.get(k) or 0) for r in rows])), 1)  # noqa: E731
    return {"arm": key[0], "effort": key[1], "n": len(rows), "right": rate(rows, right), "within_3cm": rate(rows, near),
            "median_err_m": round(float(np.median(errs)), 3) if errs else None, "in_tok": mean("prompt_tokens"),
            "out_tok": mean("completion_tokens"), "reasoning_tok": mean("reasoning_tokens"), "s_per_call": mean("seconds"),
            "usd": round(sum(float(r.get("usd") or 0) for r in rows), 4)}


def paired(a: dict, b: dict, margin: float) -> dict:
    eps = sorted(set(a) & set(b))
    d = np.array([float(right(a[e])) - float(right(b[e])) for e in eps])
    boot = [d[i].mean() for i in np.random.default_rng(0).integers(0, len(d), (2000, len(d)))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    label = "better" if lo > 0 else "worse" if hi < 0 else "same" if -margin <= lo and hi <= margin else "undecided"
    return {"n_paired": len(eps), "diff": round(float(d.mean()), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
            "label": label}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ref", default="K:default")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    groups, ref = load(a.run), tuple(a.ref.split(":"))
    out = {"run": str(a.run), "ref": a.ref, "margin": a.margin, "groups": [summary(k, g) for k, g in sorted(groups.items())],
           "vs_ref": {f"{k[0]}:{k[1]}": paired(g, groups[ref], a.margin) for k, g in sorted(groups.items()) if k != ref}}
    print(json.dumps(out, indent=1))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(json.dumps(out, indent=1))
        (d / "dry.csv").write_bytes((a.run / "dry.csv").read_bytes())


if __name__ == "__main__":
    main()
