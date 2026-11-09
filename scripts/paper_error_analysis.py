"""Paired error analysis and targeted ablations for the perception/belief paper.

This is an append-only analysis of the already-read G test split. Test results are
post-hoc diagnostics, never model selection. Resampling and randomization use the
episode as the independent unit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT)]

from paper_compare import brier_set, data_manifest  # noqa: E402
from perceived_qa import OCCLUSION, truth  # noqa: E402
from run import git_state, load_params  # noqa: E402
from arbiter_kb import ledger  # noqa: E402
from arbiter_kb.paper_stats import (belief_summary, holm_adjust, mcnemar_exact, paired_bootstrap,  # noqa: E402
                                    paired_correct, paired_randomization)
from arbiter_kb.sources.perceived import PerceivedSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.systems.baselines import TrackerSystem  # noqa: E402
from arbiter_kb.systems.soft_haircut import SoftHaircut  # noqa: E402


def _probs(dist) -> dict[str, float]:
    return {"None" if k is None else k: float(v) for k, v in dist.probs.items()}


def _audit(system, cube: str) -> dict:
    if isinstance(system, SoftHaircut):
        passes = system.passes.get(cube, [])
        return {"pass_count": len(passes), "passes": [list(x) for x in passes]}
    if not isinstance(system, ArbiterKBSystem):
        return {}
    target = f"Inside({cube})"
    ops = [op for row in system.trace_rows() for op in row["ops"] if op["target"] == target]
    rules = Counter(op["rule"] for op in ops)
    return {"dist_updates": len(ops), "rule_updates": dict(sorted(rules.items())),
            "near_pass_count": sum(e.kind == "near_pass" for e in system.kb.evidence),
            "last_seen": system.kb.objects[cube].t_last_seen}


def detailed_rows(src, factory, split: str) -> list[dict]:
    rows = []
    for ep in src.episodes(split):
        data, meta, system = src.load(ep), src.meta(ep), factory()
        system.reset(meta)
        for frame in src.frames(ep):
            system.step(frame)
        bins = [oid for oid, cls, _ in meta.objects if cls == "container"]
        for question, want in truth(data).items():
            if question.startswith("q1_"):
                cube = "cube_" + question[3:]
                dist, pred = system.believe_inside(cube), system.believe_inside(cube).argmax()[0]
                rows.append({"ep": data["name"], "cube": question[3:], "correct": pred in want,
                             "pred": pred, "truth": want, "p_max": float(dist.max_prob()),
                             "brier": brier_set(dist.probs, want, bins), "probs": _probs(dist),
                             "audit": _audit(system, cube)})
    return rows


def configurations(params) -> dict:
    relinked = PerceivedSource(**OCCLUSION, relinked=True)
    no_r6 = replace(params, rule=replace(params.rule, w_max=0.0))
    point = replace(params, rule=replace(params.rule, sigma_cover=0.001, eps_none=0.0, eps_far=0.0))
    return {
        "relink_bound": (relinked, lambda: TrackerSystem("bound", params)),
        "relink_soft": (relinked, lambda: SoftHaircut(params, "min_3d")),
        "r1_only": (relinked, lambda: ArbiterKBSystem(no_r6, skip=("R4", "R5"))),
        "arbiterkb": (relinked, lambda: ArbiterKBSystem(params)),
        "kb_no_R4": (relinked, lambda: ArbiterKBSystem(params, skip=("R4",))),
        "kb_no_R5": (relinked, lambda: ArbiterKBSystem(params, skip=("R5",))),
        "kb_no_R6": (relinked, lambda: ArbiterKBSystem(no_r6)),
        "kb_point_R1": (relinked, lambda: ArbiterKBSystem(point)),
    }


def compare(a: list[dict], b: list[dict], n_resamples: int) -> dict:
    av, bv = paired_correct(a, b)
    out = {"mcnemar": mcnemar_exact(av, bv)}
    for seed, metric in enumerate(("accuracy", "aurc", "brier")):
        out[metric] = {"delta_a_minus_b_ci95": paired_bootstrap(a, b, metric, n_resamples, seed),
                       "p_episode_randomization": paired_randomization(a, b, metric, n_resamples, seed + 20)}
    out["accuracy"]["p_exact_mcnemar"] = out["mcnemar"]["p_exact"]
    return out


def apply_holm(pairs: dict) -> None:
    raw = {}
    for pair, stats in pairs.items():
        raw[f"{pair}/accuracy"] = stats["accuracy"]["p_exact_mcnemar"]
        for metric in ("aurc", "brier"):
            raw[f"{pair}/{metric}"] = stats[metric]["p_episode_randomization"]
    adjusted = holm_adjust(raw)
    for pair, stats in pairs.items():
        for metric in ("accuracy", "aurc", "brier"):
            stats[metric]["p_holm_family"] = adjusted[f"{pair}/{metric}"]


def _by_key(rows: list[dict]) -> dict:
    return {(r["ep"], r["cube"]): r for r in rows}


def paired_diagnostics(full: list[dict], soft: list[dict]) -> dict:
    aa, bb = _by_key(full), _by_key(soft)
    keys = sorted(aa)
    counts, details, brier_delta = Counter(), [], []
    for key in keys:
        a, b = aa[key], bb[key]
        label = ("full_only_correct" if a["correct"] and not b["correct"] else
                 "soft_only_correct" if b["correct"] and not a["correct"] else
                 "both_correct" if a["correct"] else "both_wrong")
        counts[label] += 1
        counts["prediction_disagrees" if a["pred"] != b["pred"] else "prediction_agrees"] += 1
        delta = a["brier"] - b["brier"]
        brier_delta.append(delta)
        row = {"ep": key[0], "cube": key[1], "class": label, "brier_delta_full_minus_soft": delta,
               "full": a, "soft": b}
        if a["pred"] != b["pred"]:
            details.append(row)
    worst = sorted(({"ep": k[0], "cube": k[1], "delta": aa[k]["brier"] - bb[k]["brier"],
                     "full": aa[k], "soft": bb[k]} for k in keys), key=lambda x: -x["delta"])
    return {"counts": dict(sorted(counts.items())), "mean_brier_delta": sum(brier_delta) / len(brier_delta),
            "full_brier_worse_n": sum(x > 0 for x in brier_delta), "discordant_predictions": details,
            "largest_brier_degradations": worst[:20]}


def activity_groups(full: list[dict], soft: list[dict]) -> dict:
    sb, groups = _by_key(soft), defaultdict(list)
    for row in full:
        rules = row["audit"].get("rule_updates", {})
        signature = "+".join(k for k in ("R1", "R5", "R6") if rules.get(k)) or "none"
        groups[signature].append((row, sb[(row["ep"], row["cube"])]))
    return {name: {"n": len(items), "full_accuracy": sum(a["correct"] for a, _ in items) / len(items),
                   "soft_accuracy": sum(b["correct"] for _, b in items) / len(items),
                   "mean_brier_delta": sum(a["brier"] - b["brier"] for a, b in items) / len(items)}
            for name, items in sorted(groups.items())}


def run_split(params, split: str, n_resamples: int) -> dict:
    rows = {name: detailed_rows(src, factory, split)
            for name, (src, factory) in configurations(params).items()}
    plan = {"arbiterkb-relink_soft": ("arbiterkb", "relink_soft"),
            "r1_only-relink_bound": ("r1_only", "relink_bound"),
            "arbiterkb-r1_only": ("arbiterkb", "r1_only"),
            "relink_soft-relink_bound": ("relink_soft", "relink_bound"),
            "kb_no_R4-arbiterkb": ("kb_no_R4", "arbiterkb"),
            "kb_no_R5-arbiterkb": ("kb_no_R5", "arbiterkb"),
            "kb_no_R6-arbiterkb": ("kb_no_R6", "arbiterkb"),
            "kb_point_R1-arbiterkb": ("kb_point_R1", "arbiterkb")}
    pairs = {name: compare(rows[a], rows[b], n_resamples) for name, (a, b) in plan.items()}
    apply_holm(pairs)
    return {"table": {name: belief_summary(rs, params.monitor.tau_hi) for name, rs in rows.items()},
            "pairs": pairs, "paired_full_vs_soft": paired_diagnostics(rows["arbiterkb"], rows["relink_soft"]),
            "full_activity_groups": activity_groups(rows["arbiterkb"], rows["relink_soft"]), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/latest/paper_error_analysis/summary.json")
    ap.add_argument("--resamples", type=int, default=5000)
    args, params, state = ap.parse_args(), load_params(), git_state()
    ledger.record("READ", params.hash(), "G perceived test, paper_error_analysis.py post-hoc diagnosis; no tuning")
    result = {"schema": 1, "design": {"status": "post-hoc", "unit": "episode", "resamples": args.resamples,
                                      "multiplicity": "Holm across 8 contrasts x 3 metrics per split"},
              "train": run_split(params, "train", args.resamples),
              "test": run_split(params, "test", args.resamples),
              "provenance": {"git_before_read": state, "params_hash": params.hash(),
                             "data_train": data_manifest("train"), "data_test": data_manifest("test"),
                             "script_commit_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({"out": str(args.out), "test_table": result["test"]["table"],
                      "test_counts": result["test"]["paired_full_vs_soft"]["counts"]}, indent=1))


if __name__ == "__main__":
    main()
