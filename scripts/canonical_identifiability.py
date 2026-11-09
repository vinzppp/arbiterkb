"""Zero-cost canonical-memory identifiability audit on G-train and recorded runs.

The script is fail-closed to G-train episode suffixes 0--9.  It never opens a
G-test npz, starts RoboMME, or calls an LLM.  Oracle arrays are used only after
memory decisions are produced, for evaluation and tracker-error measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from run import git_state, load_params  # noqa: E402
from perceived_qa import OCCLUSION, truth  # noqa: E402
from arbiter_kb.canonical_probe import (hard_pick_probe, project_identity,  # noqa: E402
                                        threshold_pick_probe)
from arbiter_kb.canonical_audit import (assert_train_only, episode_cluster_bootstrap,  # noqa: E402
                                        progress_conflicts, sha256, tracker_error_audit,
                                        train_suffix, write_outputs)
from arbiter_kb.factorial_memory import FactorialMemory  # noqa: E402
from arbiter_kb.sources.perceived import PerceivedSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.systems.baselines import TrackerSystem  # noqa: E402


def visible_points(frame) -> dict[str, tuple[float, float]]:
    return {oid: (float(obj.uv[1]), float(obj.uv[0])) for oid, obj in frame.objects.items()
            if obj.cls == "container" and obj.visible and obj.uv is not None}


def truth_score(probs: dict, wanted: list[str], candidates: list[str]) -> tuple[bool, float, float]:
    top = max([*candidates, None], key=lambda k: probs.get(k, 0.0))
    correct = top in wanted
    target = 1.0 / len(wanted)
    brier = sum((probs.get(k, 0.0) - (target if k in wanted else 0.0)) ** 2
                for k in [*candidates, None])
    p_true = sum(probs.get(k, 0.0) for k in wanted)
    return bool(correct), float(brier), float(-math.log(max(p_true, 1e-12)))


def run_memories(src: PerceivedSource, ep: int, params) -> tuple[dict, list]:
    frames = list(src.frames(ep))
    systems = {
        "strong_ledger": TrackerSystem("bound", params),
        "arbiterkb": ArbiterKBSystem(params),
        "factorial_ledger": FactorialMemory("ledger", params),
    }
    for name, system in systems.items():
        system.reset(src.meta(ep))
        if name == "factorial_ledger":
            system.feed(frames)
        else:
            for frame in frames:
                system.step(frame)
    return systems, frames


def memory_distribution(system, arm: str, cube: str):
    if arm == "factorial_ledger":
        row = next(r for r in system.beliefs() if r["cube"] == cube)
        from arbiter_kb.beliefs import Distribution
        return Distribution(row["probs"], system.frame.t if system.frame else -1, [])
    return system.believe_inside(cube)


def memory_rows(params) -> tuple[list[dict], dict]:
    src = PerceivedSource(**OCCLUSION, relinked=True)
    eps = src.episodes("train")
    assert_train_only(src, eps)
    rows = []
    for ep in eps:
        data = src.load(ep)
        targets = truth(data)
        systems, frames = run_memories(src, ep, params)
        frame = frames[-1]
        candidates = [oid for oid, cls, _ in src.meta(ep).objects if cls == "container"]
        points = visible_points(frame)
        for question, wanted in targets.items():
            if not question.startswith("q1_"):
                continue
            colour, cube = question[3:], "cube_" + question[3:]
            states, actions = {}, {}
            for arm, system in systems.items():
                dist = memory_distribution(system, arm, cube)
                state = project_identity(cube, dist, candidates, points,
                                         {"source": arm, "episode": data["name"]})
                hard = hard_pick_probe(state)
                gated = threshold_pick_probe(state, params.monitor.tau_hi)
                correct, brier, logloss = truth_score(state.probs(), wanted, candidates)
                states[arm] = {
                    "top1": state.top()[0], "p_max": state.top()[1],
                    "probabilities": {"unknown" if k is None else k: v for k, v in state.probs().items()},
                    "evidence_t": state.evidence_t, "support": list(state.support),
                    "correct": correct, "brier": brier, "log_loss": logloss,
                }
                actions[arm] = {"hard": hard, "threshold_0_8": gated}
            rows.append({
                "episode": data["name"], "setting": data["name"].rsplit("_ep", 1)[0],
                "episode_suffix": train_suffix(src.files[ep]), "cube": cube, "colour": colour,
                "truth": wanted, "tracker_swapped": (targets.get("swapped") or {}).get(colour),
                "states": states, "actions": actions,
            })
    manifest = {src.files[ep].name: sha256(src.files[ep]) for ep in eps}
    return rows, {"episodes": len(eps), "files": manifest, "occlusion": OCCLUSION, "relinked": True}


def calibration(rows: list[dict], arm: str) -> dict:
    vals = [r["states"][arm] for r in rows]
    correct = np.asarray([v["correct"] for v in vals], float)
    conf = np.asarray([v["p_max"] for v in vals], float)
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        hi = lo + .1
        take = (conf >= lo) & (conf <= hi if hi == 1 else conf < hi)
        if take.any():
            ece += take.mean() * abs(conf[take].mean() - correct[take].mean())
    return {
        "n": len(vals), "top1_accuracy": float(correct.mean()),
        "mean_brier": float(np.mean([v["brier"] for v in vals])),
        "mean_log_loss": float(np.mean([v["log_loss"] for v in vals])),
        "ece_10_bin": float(ece), "mean_p_max": float(conf.mean()),
        "unknown_top1": int(sum(v["top1"] is None for v in vals)),
    }


def pair_summary(rows: list[dict], left: str, right: str) -> dict:
    top = lambda r, a: r["states"][a]["top1"]  # noqa: E731
    hard = lambda r, a: r["actions"][a]["hard"]  # noqa: E731
    gated = lambda r, a: r["actions"][a]["threshold_0_8"]  # noqa: E731
    disagreements = [r for r in rows if top(r, left) != top(r, right)]
    both_legal = [r for r in rows if hard(r, left)["legal"] and hard(r, right)["legal"]
                  and hard(r, left)["target"] != hard(r, right)["target"]]
    consequence = [r for r in disagreements if r["states"][left]["correct"] != r["states"][right]["correct"]]
    return {
        "n": len(rows), "top1_disagreement": len(disagreements),
        "top1_disagreement_rate": len(disagreements) / len(rows),
        "unknown_disagreement": sum((top(r, left) is None) != (top(r, right) is None) for r in rows),
        "both_legal_different_pick": len(both_legal),
        "hard_actionability_disagreement": sum(hard(r, left)["legal"] != hard(r, right)["legal"] for r in rows),
        "threshold_actionability_disagreement": sum(gated(r, left)["legal"] != gated(r, right)["legal"] for r in rows),
        "different_correctness": len(consequence),
        "left_only_correct": sum(r["states"][left]["correct"] and not r["states"][right]["correct"] for r in rows),
        "right_only_correct": sum(r["states"][right]["correct"] and not r["states"][left]["correct"] for r in rows),
        "mean_abs_pmax_difference": float(np.mean([abs(r["states"][left]["p_max"] -
                                                        r["states"][right]["p_max"]) for r in rows])),
        "disagreement_on_measured_swap": sum(r["tracker_swapped"] is True for r in disagreements),
        "disagreement_on_no_measured_swap": sum(r["tracker_swapped"] is False for r in disagreements),
        "disagreement_swap_unknown": sum(r["tracker_swapped"] is None for r in disagreements),
        "disagreement_rows": [{"episode": r["episode"], "cube": r["cube"], "truth": r["truth"],
                                "tracker_swapped": r["tracker_swapped"],
                                left: r["states"][left], right: r["states"][right],
                                "actions": {left: r["actions"][left], right: r["actions"][right]}}
                               for r in disagreements],
    }


def build_summary(rows: list[dict], data: dict, params) -> dict:
    return {
        "schema": 1, "scope": "G-train only plus previously recorded factorial traces",
        "zero_cost": True, "new_simulator_rollouts": 0, "new_llm_calls": 0,
        "heldout_records_opened": False, "params_hash": params.hash(),
        "git": {"commit": git_state()[0], "dirty": git_state()[1]},
        "data": data,
        "state_metrics": {arm: calibration(rows, arm) for arm in
                          ("strong_ledger", "arbiterkb", "factorial_ledger")},
        "pairs": {
            "arbiterkb_vs_strong_ledger": pair_summary(rows, "arbiterkb", "strong_ledger"),
            "arbiterkb_vs_factorial_ledger": pair_summary(rows, "arbiterkb", "factorial_ledger"),
        },
        "paired_episode_cluster_bootstrap": episode_cluster_bootstrap(
            rows, "arbiterkb", "strong_ledger"),
        "tracker_errors": tracker_error_audit(OCCLUSION),
        "recorded_factorial_progress_conflicts": progress_conflicts(),
        "identifiability": {
            "hard_state_action_identifiable": True,
            "calibration_closed_loop_identifiable": False,
            "reason": "G-train contains different legal pick targets, but RoboMME exposes no legal abstain or information-gathering action; confidence alone can only block, not choose a recovery action.",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    params = load_params()
    rows, data = memory_rows(params)
    summary = build_summary(rows, data, params)
    write_outputs(args.out, summary, rows, data)
    print(json.dumps({"rows": len(rows), "episodes": data["episodes"],
                      "top1_disagreement": summary["pairs"]["arbiterkb_vs_strong_ledger"]["top1_disagreement"],
                      "both_legal_different_pick": summary["pairs"]["arbiterkb_vs_strong_ledger"]["both_legal_different_pick"],
                      "heldout_records_opened": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
