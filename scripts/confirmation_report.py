"""Preregistered analysis for the memory-or-policy held-out confirmation run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.experiment_manifest import load_manifest
from arbiter_kb.paper_stats import holm_adjust, rate_interval
from arbiter_kb.confirmation_audit import ARBITER_COMMIT, audit, find_pair_drops
from scripts.common_harness_report import dollars, primary_eligible, primary_success, terminal_status

MEMORIES = ("fh", "ld", "mm", "kb")
P2 = {"P2a_mm_m_minus_ld_m": ("mm_m", "ld_m"),
      "P2b_kb_m_minus_ld_m": ("kb_m", "ld_m"),
      "P2c_fh_m_minus_ld_m": ("fh_m", "ld_m")}


def sign_flip(values: list[float], n_perm: int = 10_000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value with a finite-sample correction."""
    x = np.asarray(values, float)
    if not len(x):
        return math.nan
    observed = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    draws = (rng.integers(0, 2, (n_perm, len(x))) * 2 - 1) * x
    extreme = int(np.sum(np.abs(draws.mean(axis=1)) >= observed - 1e-15))
    return (extreme + 1) / (n_perm + 1)


def stratified_ci(values: dict[tuple[str, int], float], n_boot: int = 10_000,
                  seed: int = 0) -> list[float]:
    """Mean and percentile CI, resampling pairs independently within each task."""
    groups = [np.asarray([value for (task, _), value in sorted(values.items()) if task == name], float)
              for name in sorted({task for task, _ in values})]
    if not groups or any(not len(group) for group in groups):
        return [math.nan, math.nan, math.nan]
    rng = np.random.default_rng(seed)
    draws = np.mean(np.stack([
        group[rng.integers(0, len(group), (n_boot, len(group)))].mean(axis=1)
        for group in groups
    ]), axis=0)
    return [float(np.mean([group.mean() for group in groups])),
            float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _tree_sha(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def load_rows(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.rglob("result.json")):
        row = json.loads(path.read_text())
        row["_path"] = str(path)
        rows.append(row)
    return rows


def paired_scores(rows: list[dict]) -> tuple[dict[tuple[str, int], dict[str, bool]], list[str]]:
    cells: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    issues = []
    for row in rows:
        if not primary_eligible(row):
            issues.append(f"ineligible row: {row['_path']}")
            continue
        cells[(row["task"], int(row["episode"]))][row["arm"]] = primary_success(row)
    return cells, issues


def score_map(cells, fn) -> dict[tuple[str, int], float]:
    return {key: float(fn(arms)) for key, arms in cells.items()}


def contrast(cells, left: str, right: str) -> dict:
    values = score_map(cells, lambda arms: int(arms[left]) - int(arms[right]))
    return {"estimate_ci95": stratified_ci(values),
            "left_only": sum(value > 0 for value in values.values()),
            "right_only": sum(value < 0 for value in values.values()),
            "discordant": sum(value != 0 for value in values.values()),
            "n_pairs": len(values)}


def classify_h1(ci: list[float], p_holm: float) -> str:
    if p_holm < .05 and ci[0] >= .10:
        return "supported"
    if ci[2] < .10:
        return "falsified"
    return "inconclusive"


def classify_h3(ci: list[float], p_holm: float) -> str:
    if p_holm < .05 and ci[0] > 0:
        return "supported"
    if ci[2] < 0:
        return "falsified"
    return "inconclusive"


def primary_analysis(cells: dict, spec: dict) -> dict:
    p1_values = score_map(cells, lambda a: np.mean([int(a[f"{m}_m"]) - int(a[f"{m}_g"])
                                                    for m in MEMORIES]))
    p3_values = score_map(cells, lambda a: (int(a["fh_m"]) - int(a["fh_g"]))
                                           - .5 * (int(a["memi"]) - int(a["fh_g"])))
    p1_ci, p3_ci = stratified_ci(p1_values), stratified_ci(p3_values)
    p_raw = {"P1": sign_flip(list(p1_values.values())),
             "P3": sign_flip(list(p3_values.values()))}
    p_holm = holm_adjust(p_raw)
    h3_by_task = {}
    for task in spec["tasks"]:
        sample = {key: value for key, value in p3_values.items() if key[0] == task}
        h3_by_task[task] = {"estimate_ci95": stratified_ci(sample), "n_pairs": len(sample)}

    numerator = score_map(cells, lambda a: int(a["fh_m"]) - int(a["fh_g"]))
    denominator = score_map(cells, lambda a: int(a["memi"]) - int(a["fh_g"]))
    numerator_ci, denominator_ci = stratified_ci(numerator), stratified_ci(denominator)
    ratio = (numerator_ci[0] / denominator_ci[0]
             if denominator_ci[1] > 0 or denominator_ci[2] < 0 else None)
    return {
        "P1_policy_main_effect": {"estimate_ci95": p1_ci, "p_raw": p_raw["P1"],
                                  "p_holm": p_holm["P1"],
                                  "reading": classify_h1(p1_ci, p_holm["P1"])},
        "P3_portability": {"estimate_ci95": p3_ci, "p_raw": p_raw["P3"],
                           "p_holm": p_holm["P3"],
                           "reading": classify_h3(p3_ci, p_holm["P3"]),
                           "by_task": h3_by_task,
                           "fh_matched_minus_generic_ci95": numerator_ci,
                           "memi_minus_fh_generic_ci95": denominator_ci,
                           "recovery_ratio_if_denominator_ci_excludes_zero": ratio}}


def secondary_analysis(cells: dict) -> dict:
    p2 = {name: contrast(cells, *arms) for name, arms in P2.items()}
    fidelity = contrast(cells, "mm_m", "memi")
    fidelity["beyond_preregistered_10pp_margin"] = abs(fidelity["estimate_ci95"][0]) > .10
    return {"P2_intervals_only": p2, "H2_equivalence_claimed": False,
            "detectable_P2_difference_at_n250": .165,
            "split_fidelity_mm_m_minus_stock_memi": fidelity}


def descriptive_analysis(rows: list[dict], cells: dict, spec: dict, run: Path) -> dict:
    arms = spec["arms"]
    success_by_arm = {arm: rate_interval([cells[key][arm] for key in sorted(cells)]) for arm in arms}
    success_by_task_arm = {
        f"{task}/{arm}": rate_interval([cells[key][arm] for key in sorted(cells) if key[0] == task])
        for task in spec["tasks"] for arm in arms
    }
    prices = spec["prices_per_million"]
    terminal = Counter(terminal_status(row) for row in rows)
    reasons = Counter((row.get("terminal") or {}).get("reason") for row in rows)
    cost_by_arm = {arm: sum(dollars(row, prices) for row in rows if row["arm"] == arm) for arm in arms}
    calls_by_arm = {arm: sum(sum((row.get("policy") or {}).get("calls", {}).values())
                              for row in rows if row["arm"] == arm) for arm in arms}
    result_paths = [Path(row["_path"]) for row in rows]
    all_raw = result_paths + [path.with_name("events.jsonl") for path in result_paths]
    created = json.loads((run / "run.json").read_text()).get("created_utc")
    finished = max(path.stat().st_mtime for path in result_paths)
    return {
        "success_by_arm": success_by_arm, "success_by_task_arm": success_by_task_arm,
        "terminal_status": dict(terminal), "terminal_reason": dict(reasons),
        "resources": {"cost_usd": sum(cost_by_arm.values()), "cost_by_arm": cost_by_arm,
                      "calls_by_arm": calls_by_arm,
                      "episode_seconds_sum": sum(float(row.get("seconds") or 0) for row in rows),
                      "run_created_utc": created, "last_result_mtime_epoch": finished},
        "provenance": {"arbiter_commit": ARBITER_COMMIT, "sources": spec["sources"],
                       "params_sha": spec["params_sha"],
                       "prompt_sha": (rows[0].get("provenance") or {}).get("prompt_sha"),
                       "model": spec["model"], "effort": spec["effort"],
                       "execution_host": spec.get("execution_host"),
                       "execution_image": spec.get("execution_image"),
                       "raw_result_and_event_count": len(all_raw),
                       "raw_result_and_event_tree_sha256": _tree_sha(all_raw, run)}}


def build_report(run: Path, rows: list[dict], spec: dict, manifest_sha: str) -> dict:
    cells, eligibility_issues = paired_scores(rows)
    integrity = audit(rows, spec, manifest_sha)
    integrity["issues"].extend(eligibility_issues)
    integrity["passed"] = integrity["passed"] and not eligibility_issues
    report = {
        "schema": 1, "run_id": run.name, "manifest_sha": manifest_sha,
        "integrity": integrity, "primary": primary_analysis(cells, spec),
        "secondary": secondary_analysis(cells)}
    report.update(descriptive_analysis(rows, cells, spec, run))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec, manifest_sha = load_manifest(args.manifest)
    report = build_report(args.run, load_rows(args.run), spec, manifest_sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"integrity": report["integrity"]["passed"],
                      "P1": report["primary"]["P1_policy_main_effect"],
                      "P3": {k: v for k, v in report["primary"]["P3_portability"].items()
                             if k != "by_task"}, "cost": report["resources"]["cost_usd"]}, indent=2))
    return 0 if report["integrity"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
