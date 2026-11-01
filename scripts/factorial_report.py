"""Frozen analysis and provenance artifacts for the memory x policy pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.experiment_manifest import load_manifest  # noqa: E402
from arbiter_kb.paper_stats import holm_adjust, mcnemar_exact, rate_interval  # noqa: E402
from scripts.common_harness_report import (dollars, load_rows, primary_eligible,  # noqa: E402
                                           primary_success, reset_digest, terminal_status, usage)

ARMS = ["ledger_generic", "ledger_matched", "arbiter_generic", "arbiter_matched"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values, seed=20261031) -> list[float]:
    values = np.asarray(list(values), float)
    if not len(values):
        return [math.nan, math.nan, math.nan]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return [float(values.mean()), *[float(x) for x in np.percentile(draws, [2.5, 97.5])]]


def paired(rows, left, right) -> tuple[list[float], list[tuple], list[bool], list[bool]]:
    valid = [row for row in rows if primary_eligible(row)]
    a = {(r["task"], r["episode"]): float(primary_success(r)) for r in valid if r["arm"] == left}
    b = {(r["task"], r["episode"]): float(primary_success(r)) for r in valid if r["arm"] == right}
    keys = sorted(set(a) & set(b))
    left_values, right_values = [bool(a[key]) for key in keys], [bool(b[key]) for key in keys]
    return [a[key] - b[key] for key in keys], keys, left_values, right_values


def pairwise(rows, contrasts) -> dict:
    out, raw = {}, {}
    for left, right in contrasts:
        diffs, keys, av, bv = paired(rows, left, right)
        test = mcnemar_exact(av, bv)
        name = f"{left}_minus_{right}"
        outcomes = [{"task": key[0], "episode": key[1], "left": av[i], "right": bv[i]}
                    for i, key in enumerate(keys)]
        out[name] = {"n_paired": len(keys), "outcomes": outcomes,
                     "risk_difference": bootstrap(diffs), **test}
        raw[name] = test["p_exact"]
    adjusted = holm_adjust(raw)
    for name, value in adjusted.items():
        out[name]["p_holm"] = value
    return out


def factorial_effects(rows) -> dict:
    valid = {(r["task"], r["episode"], r["arm"]): float(primary_success(r))
             for r in rows if primary_eligible(r)}
    blocks = sorted({(r["task"], r["episode"]) for r in rows})
    values = defaultdict(list)
    for task, episode in blocks:
        try:
            a, b, c, d = (valid[(task, episode, arm)] for arm in ARMS)
        except KeyError:
            continue
        values["memory_main"].append(((c - a) + (d - b)) / 2)
        values["policy_main"].append(((b - a) + (d - c)) / 2)
        values["interaction"].append((d - c) - (b - a))
    return {name: {"n_blocks": len(vals), "estimate_ci": bootstrap(vals)} for name, vals in values.items()}


def group_table(rows, prices) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["arm"])].append(row)
        groups[("ALL", row["arm"])].append(row)
    out = {}
    for key, group in sorted(groups.items()):
        valid = [r for r in group if primary_eligible(r)]
        values = [primary_success(r) for r in valid]
        out["/".join(key)] = {"success": rate_interval(values), "eligible": len(valid), "total": len(group),
                              "terminal": dict(Counter(terminal_status(r) for r in group)),
                              "actions_mean": float(np.mean([len(r.get("decisions") or []) for r in valid])) if valid else None,
                              "cost_usd": sum(dollars(r, prices) for r in group)}
    return out


def read_events(row) -> list[dict]:
    path = Path(row["_path"]).with_name("events.jsonl")
    if not path.exists():
        return row.get("partial_events") or []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def phase_times(row) -> dict:
    starts, totals = {}, defaultdict(float)
    for event in read_events(row):
        key = (event.get("phase"), event.get("decision"))
        if event.get("event") == "phase_start":
            starts[key] = float(event.get("elapsed_s", 0))
        elif event.get("event") == "phase_end" and key in starts:
            totals[str(key[0])] += float(event.get("elapsed_s", 0)) - starts.pop(key)
    return {**{name: round(value, 6) for name, value in totals.items()},
            "total": float(row.get("seconds") or 0)}


def compact_trace(row) -> dict:
    calls = (row.get("policy") or {}).get("log") or []
    final_audit = next((c.get("memory_audit") for c in reversed(calls) if c.get("memory_audit")), None)
    actions = [((d.get("command") or {}).get("choice"), d.get("result"), d.get("status"))
               for d in row.get("decisions") or []]
    repeats = sum(actions[i][0] == actions[i - 1][0] for i in range(1, len(actions)))
    return {"task": row.get("task"), "episode": row.get("episode"), "arm": row.get("arm"),
            "outcome": row.get("outcome"), "terminal": terminal_status(row), "success": primary_success(row),
            "actions": actions, "consecutive_repeats": repeats, "invalid_calls": sum(c.get("status") != "valid" for c in calls),
            "final_memory_audit": final_audit, "usage": usage(row), "cost_usd": dollars(row, PRICE),
            "timing_s": phase_times(row), "result_path": row["_path"]}


def fairness(rows, spec) -> dict:
    issues, groups = [], defaultdict(list)
    for row in rows:
        groups[(row["task"], row["episode"])].append(row)
    for key, group in groups.items():
        if {r["arm"] for r in group} != set(ARMS):
            issues.append(f"{key}: incomplete arms")
        if len({reset_digest(r) for r in group}) != 1:
            issues.append(f"{key}: reset digest mismatch")
        seeds = {json.dumps((r.get("provenance") or {}).get("episode"), sort_keys=True) for r in group}
        if len(seeds) != 1:
            issues.append(f"{key}: episode resolution mismatch")
    for row in rows:
        policy = row.get("policy") or {}
        expected = spec["factorial"]["policy_sha256"]
        want = expected["generic"] if policy.get("policy_kind") == "generic" else expected.get(f"{row['task']}/matched")
        if policy.get("policy_sha256") != want:
            issues.append(f"{row['_path']}: policy hash mismatch")
        returned = {c.get("model_returned") for c in policy.get("log") or [] if c.get("model_returned")}
        if returned != {spec["model"]}:
            issues.append(f"{row['_path']}: returned model mismatch")
    return {"passed": len(rows) == 24 and not issues, "row_count": len(rows), "issues": issues,
            "terminal_statuses": dict(Counter(terminal_status(r) for r in rows)),
            "infrastructure_or_incomplete": [r["_path"] for r in rows if terminal_status(r) in {"infrastructure_error", "incomplete"}]}


def write_repro(out, manifest, manifest_sha, run_root, probe, rows) -> None:
    files = [manifest, ROOT / "arbiter_kb/factorial_memory.py", ROOT / "stress/common_policy.py",
             ROOT / "stress/common_harness_pod.py", ROOT / "stress/common_episode.py",
             ROOT / "paper/ROBOMME_MEMORY_POLICY_FACTORIAL_PROTOCOL_ZH.md", ROOT / "params.json"]
    repo = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    data = {"manifest": {"path": str(manifest.relative_to(ROOT)), "sha16": manifest_sha, "sha256": sha(manifest)},
            "arbiter_commit": repo, "run_root": str(run_root), "probe": probe,
            "sources": (rows[0].get("provenance") or {}).get("sources") if rows else {},
            "artifacts": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "credential": "authorized macOS Keychain item; secret never persisted", "heldout_accessed": False,
            "observation_degradation_run": False, "selective_reruns": False}
    (out / "reproducibility_manifest.json").write_text(json.dumps(data, indent=2))


def main() -> int:
    global PRICE
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec, manifest_sha = load_manifest(a.manifest)
    PRICE = spec["prices_per_million"]
    rows, probe = load_rows(a.run_root / "pilot"), json.loads(a.probe.read_text())
    audit = fairness(rows, spec)
    scored_cost = sum(dollars(row, PRICE) for row in rows)
    summary = {"manifest_sha": manifest_sha, "rows": len(rows), "fairness_audit": audit,
               "success_table": group_table(rows, PRICE), "paired": pairwise(rows, spec["pairwise_contrasts"]),
               "factorial_effects": factorial_effects(rows),
               "cost": {"probe_usd": probe.get("cost_usd", 0), "scored_usd": scored_cost,
                        "total_usd": scored_cost + float(probe.get("cost_usd", 0)), "cap_usd": .10},
               "timing_s": {"sum": {name: sum(phase_times(r).get(name, 0) for r in rows)
                                      for name in ("reset", "llm", "executor", "total")}},
               "decision": "GO_ANALYZE" if audit["passed"] and scored_cost + float(probe.get("cost_usd", 0)) < .10
               else "NO_GO_INCOMPLETE_OR_INTEGRITY"}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (a.out / "per_cell_traces.json").write_text(json.dumps([compact_trace(r) for r in rows], indent=2))
    (a.out / "fairness_leakage_audit.json").write_text(json.dumps(audit, indent=2))
    write_repro(a.out, a.manifest, manifest_sha, a.run_root, probe, rows)
    print(json.dumps({"rows": len(rows), "audit": audit["passed"], "cost_usd": summary["cost"]["total_usd"]}))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
