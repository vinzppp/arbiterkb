"""Score a completed frozen RoboMME common-harness stage."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.paper_stats import belief_summary, holm_adjust, mcnemar_exact, rate_interval
from arbiter_kb.experiment_manifest import load_manifest


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("result.json")):
        row = json.loads(path.read_text())
        row["arm_recorded"] = row.get("arm")
        if row.get("arm") in (None, "", "unknown"):
            suffix = path.parent.name.rsplit("_", 1)[-1]
            if suffix in {"fullhist", "memi", "soft", "arbiter"}:
                row["arm"] = suffix
        row["_path"], row["cell"] = str(path), path.parents[1].name
        rows.append(row)
    return rows
def usage(row: dict) -> dict:
    return ((row.get("policy") or {}).get("usage") or {})
def terminal_status(row: dict) -> str:
    status = (row.get("terminal") or {}).get("status")
    if status:
        return str(status)
    if row.get("outcome") == "harness_error":
        return "infrastructure_error"
    return "success" if row.get("success") else "task_failure"
def reset_digest(row: dict):
    if row.get("reset_digest") is not None:
        return row["reset_digest"]
    resets = (e for e in reversed(row.get("partial_events") or [])
              if e.get("event") == "phase_end" and e.get("phase") == "reset")
    return next(resets, {}).get("reset_digest")
def primary_eligible(row: dict) -> bool:
    terminal = row.get("terminal") or {}
    return bool(terminal.get("primary_eligible")) if terminal else row.get("outcome") != "harness_error"
def primary_success(row: dict) -> bool:
    terminal = row.get("terminal") or {}
    return bool(terminal.get("primary_success")) if terminal else bool(row.get("success"))
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
    scored = [r for r in rows if primary_eligible(r)]
    statuses = Counter(terminal_status(row) for row in rows)
    diagnostic = bool(scored) and all((r.get("policy") or {}).get("model") == "none" for r in scored)
    if diagnostic:
        return {"diagnostic_only": True, "episodes": len(scored),
                "outcomes": dict(Counter(r.get("outcome", "") for r in scored)),
                "terminal_statuses": dict(statuses),
                "wall_s_mean": _mean(r.get("seconds", 0) for r in scored)}
    return {"success": rate_interval([primary_success(r) for r in scored]),
            "episodes": len(scored), "total_cells": len(rows),
            "terminal_statuses": dict(statuses),
            "infrastructure_errors": statuses["infrastructure_error"],
            "incomplete": statuses["incomplete"], "timeouts": statuses["timeout"],
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
    valid = [r for r in rows if primary_eligible(r)]
    aa = {(r["task"], r["episode"]): primary_success(r) for r in valid if r["arm"] == a}
    bb = {(r["task"], r["episode"]): primary_success(r) for r in valid if r["arm"] == b}
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
            tests[name]["risk_difference"] = paired_mean_ci(sample, a, b, primary_success)
            raw[name] = tests[name]["p_exact"]
        adjusted = holm_adjust(raw)
        for name in tests:
            tests[name]["p_holm"] = adjusted[name]
        out[f"{cell}/{task}"] = tests
    return out


def paired_mean_ci(rows: list[dict], a: str, b: str, field, seed: int = 0) -> list[float]:
    aa = {(r["task"], r["episode"]): field(r) for r in rows
          if r["arm"] == a and primary_eligible(r)}
    bb = {(r["task"], r["episode"]): field(r) for r in rows
          if r["arm"] == b and primary_eligible(r)}
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


def _cell_name(cfg: dict) -> str:
    return cfg["mode"] if cfg["mode"] == "clean" else (
        f"dropout{cfg['dropout']:.2f}" if cfg["mode"] == "dropout"
        else f"jitter{cfg['jitter_px']:.1f}px")


def expected_stage_keys(spec: dict, stage_name: str, arms: list[str]) -> set[tuple] | None:
    stage = (spec.get("stages") or {}).get(stage_name)
    if not stage:
        return None
    return {(_cell_name(cfg), task, episode) for cfg in stage["observations"]
            for task in spec["tasks"] for episode in (stage.get("episodes_by_task") or {}).get(task, stage["episodes"])}


def audit(rows: list[dict], arms: list[str], expected_model: str | None = None,
          expected_keys: set[tuple] | None = None, require_terminal: bool = False) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cell"], row["task"], row["episode"])].append(row)
    issues = []
    if expected_keys is not None:
        missing = sorted(expected_keys - set(groups))
        issues.extend(f"{key}: missing cell" for key in missing)
    for key, group in groups.items():
        if {r["arm"] for r in group} != set(arms):
            issues.append(f"{key}: incomplete arms")
        if len({reset_digest(r) for r in group}) != 1:
            issues.append(f"{key}: reset digest differs")
        if len({json.dumps((r.get("provenance") or {}).get("episode"), sort_keys=True) for r in group}) != 1:
            issues.append(f"{key}: episode seed differs")
    if require_terminal:
        issues.extend(f"{row['_path']}: missing terminal status" for row in rows
                      if not (row.get("terminal") or {}).get("status"))
    wrong_model = []
    if expected_model:
        for row in rows:
            returned = {call.get("model_returned") for call in (row.get("policy") or {}).get("log") or []
                        if call.get("model_returned")}
            if returned and returned != {expected_model}:
                wrong_model.append(row["_path"])
        issues.extend(f"{path}: wrong returned model" for path in wrong_model)
    dirty = [r["_path"] for r in rows if (r.get("provenance") or {}).get("arbiter_dirty")]
    statuses = Counter(terminal_status(row) for row in rows)
    illegal = [r["_path"] for r in rows if any(
        decision.get("result") == "api_error" for decision in r.get("decisions") or [])]
    return {"passed": not issues and not dirty, "issues": issues, "dirty_rows": dirty,
            "terminal_statuses": dict(statuses),
            "infrastructure_errors": [r["_path"] for r in rows
                                      if terminal_status(r) == "infrastructure_error"],
            "incomplete": [r["_path"] for r in rows if terminal_status(r) == "incomplete"],
            "wrong_model": wrong_model, "illegal_or_rejected_actions": illegal,
            "harness_errors": [r["_path"] for r in rows if r.get("outcome") == "harness_error"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--manifest", type=Path, default=ROOT / "experiments/robomme_common_v1.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    (spec, manifest_sha), rows = load_manifest(a.manifest), load_rows(a.root)
    diagnostic = a.root.name == "executor_preflight"
    arms = spec["arms"][:1] if diagnostic else spec["arms"]
    audit_result = audit(
        rows,
        arms,
        expected_model=(spec.get("transport") or {}).get("wire_model") or spec.get("model"),
        expected_keys=expected_stage_keys(spec, a.root.name, arms),
        require_terminal=int(spec.get("schema_version", 1)) >= 2,
    )
    terminal_blockers = audit_result["infrastructure_errors"] + audit_result["incomplete"]
    gate_pass = audit_result["passed"] and not terminal_blockers and not audit_result["harness_errors"]
    gate = {
        "decision": "GO" if gate_pass else "NO_GO",
        "approved_stages": ["validation"] if gate_pass and a.root.name == "smoke" else [],
        "reason": "complete integrity audit" if gate_pass else "audit or terminal blockers remain",
    }
    report = {"manifest_sha": manifest_sha,
              "diagnostic_only": diagnostic,
              "rows": len(rows), "summary": summaries(rows, spec["prices_per_million"]),
              "paired_tests": {} if diagnostic else paired_tests(rows, spec["pairwise_contrasts"]),
              "paired_resources": {} if diagnostic else resource_differences(
                  rows, spec["pairwise_contrasts"], spec["prices_per_million"]),
              "beliefs": beliefs(rows),
              "audit": audit_result, "gate": gate}
    out = a.out or a.root / "summary.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(json.dumps({"rows": len(rows), "audit": report["audit"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
