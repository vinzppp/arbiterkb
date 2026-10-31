"""Score a completed frozen RoboMME common-harness stage."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from arbiter_kb.paper_stats import belief_summary, holm_adjust, mcnemar_exact, rate_interval

ROOT = Path(__file__).resolve().parents[1]


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("result.json")):
        row = json.loads(path.read_text())
        row["_path"], row["cell"] = str(path), path.parents[1].name
        rows.append(row)
    return rows


def usage(row: dict) -> dict:
    return ((row.get("policy") or {}).get("usage") or {})


def dollars(row: dict, prices: dict) -> float:
    u = usage(row)
    total, cached = int(u.get("input_tokens", 0)), int(u.get("cached_tokens", 0))
    written, output = int(u.get("cache_write_tokens", 0)), int(u.get("output_tokens", 0))
    value = ((total - cached - written) * prices["fresh_input"] + cached * prices["cached_input"]
             + written * prices["cache_write"] + output * prices["output"])
    return value / 1e6


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def group_summary(rows: list[dict], prices: dict) -> dict:
    scored = [r for r in rows if r.get("outcome") != "harness_error"]
    return {"success": rate_interval([bool(r.get("success")) for r in scored]),
            "episodes": len(scored), "harness_errors": len(rows) - len(scored),
            "model_calls_mean": _mean(len((r.get("policy") or {}).get("log") or []) for r in scored),
            "input_tokens_mean": _mean(usage(r).get("input_tokens", 0) for r in scored),
            "output_tokens_mean": _mean(usage(r).get("output_tokens", 0) for r in scored),
            "cost_usd_mean": _mean(dollars(r, prices) for r in scored),
            "model_latency_s_mean": _mean(usage(r).get("latency", 0) for r in scored),
            "wall_s_mean": _mean(r.get("seconds", 0) for r in scored),
            "invalid_decisions": sum((r.get("policy") or {}).get("statuses", {}).get("valid", 0)
                                     < len((r.get("policy") or {}).get("log") or []) for r in scored)}


def summaries(rows: list[dict], prices: dict) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cell"], row["task"], row["arm"])].append(row)
        groups[(row["cell"], "ALL", row["arm"])].append(row)
    return {"/".join(key): group_summary(value, prices) for key, value in sorted(groups.items())}


def _paired(rows: list[dict], a: str, b: str) -> tuple[list[bool], list[bool], list[tuple]]:
    valid = [r for r in rows if r.get("outcome") != "harness_error"]
    aa = {(r["task"], r["episode"]): bool(r.get("success")) for r in valid if r["arm"] == a}
    bb = {(r["task"], r["episode"]): bool(r.get("success")) for r in valid if r["arm"] == b}
    keys = sorted(set(aa) & set(bb))
    return [aa[k] for k in keys], [bb[k] for k in keys], keys


def paired_tests(rows: list[dict], contrasts: list[list[str]]) -> dict:
    groups = {(cell, task) for cell in {r["cell"] for r in rows}
              for task in {"ALL"} | {r["task"] for r in rows}}
    out = {}
    for cell, task in sorted(groups):
        sample = [r for r in rows if r["cell"] == cell and (task == "ALL" or r["task"] == task)]
        tests, raw = {}, {}
        for a, b in contrasts:
            av, bv, keys = _paired(sample, a, b)
            name = f"{a}_vs_{b}"
            tests[name] = {**mcnemar_exact(av, bv), "n_paired": len(keys)}
            raw[name] = tests[name]["p_exact"]
        adjusted = holm_adjust(raw)
        for name in tests:
            tests[name]["p_holm"] = adjusted[name]
        out[f"{cell}/{task}"] = tests
    return out


def paired_mean_ci(rows: list[dict], a: str, b: str, field, seed: int = 0) -> list[float]:
    aa = {(r["task"], r["episode"]): field(r) for r in rows if r["arm"] == a and r.get("outcome") != "harness_error"}
    bb = {(r["task"], r["episode"]): field(r) for r in rows if r["arm"] == b and r.get("outcome") != "harness_error"}
    values = np.asarray([aa[k] - bb[k] for k in sorted(set(aa) & set(bb))], float)
    if not len(values):
        return [math.nan, math.nan, math.nan]
    rng = np.random.default_rng(seed)
    draws = np.mean(values[rng.integers(0, len(values), (10000, len(values)))], axis=1)
    return [float(np.mean(values)), *[float(x) for x in np.percentile(draws, [2.5, 97.5])]]


def resource_differences(rows: list[dict], contrasts: list[list[str]], prices: dict) -> dict:
    out = {}
    for cell in sorted({r["cell"] for r in rows}):
        sample = [r for r in rows if r["cell"] == cell]
        for a, b in contrasts:
            out[f"{cell}/{a}_vs_{b}"] = {
                "cost_usd": paired_mean_ci(sample, a, b, lambda r: dollars(r, prices)),
                "model_latency_s": paired_mean_ci(sample, a, b, lambda r: usage(r).get("latency", 0)),
                "model_calls": paired_mean_ci(sample, a, b, lambda r: len((r.get("policy") or {}).get("log") or [])),
            }
    return out


def beliefs(rows: list[dict], tau: float = .8) -> dict:
    groups = defaultdict(list)
    for row in rows:
        for call in (row.get("policy") or {}).get("log") or []:
            for belief in call.get("scored_beliefs") or []:
                groups[(row["cell"], row["task"], row["arm"])].append(
                    {**belief, "ep": row["episode"]})
    return {"/".join(key): belief_summary(value, tau) for key, value in sorted(groups.items())}


def audit(rows: list[dict], arms: list[str]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cell"], row["task"], row["episode"])].append(row)
    issues = []
    for key, group in groups.items():
        if {r["arm"] for r in group} != set(arms):
            issues.append(f"{key}: incomplete arms")
        if len({r.get("reset_digest") for r in group}) != 1:
            issues.append(f"{key}: reset digest differs")
        if len({json.dumps((r.get("provenance") or {}).get("episode"), sort_keys=True) for r in group}) != 1:
            issues.append(f"{key}: episode seed differs")
    dirty = [r["_path"] for r in rows if (r.get("provenance") or {}).get("arbiter_dirty")]
    return {"passed": not issues and not dirty, "issues": issues, "dirty_rows": dirty,
            "harness_errors": [r["_path"] for r in rows if r.get("outcome") == "harness_error"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--manifest", type=Path, default=ROOT / "experiments/robomme_common_v1.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    spec, rows = json.loads(a.manifest.read_text()), load_rows(a.root)
    report = {"manifest_sha": __import__("hashlib").sha256(a.manifest.read_bytes()).hexdigest()[:16],
              "rows": len(rows), "summary": summaries(rows, spec["prices_per_million"]),
              "paired_tests": paired_tests(rows, spec["pairwise_contrasts"]),
              "paired_resources": resource_differences(rows, spec["pairwise_contrasts"], spec["prices_per_million"]),
              "beliefs": beliefs(rows), "audit": audit(rows, spec["arms"])}
    out = a.out or a.root / "summary.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(json.dumps({"rows": len(rows), "audit": report["audit"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
