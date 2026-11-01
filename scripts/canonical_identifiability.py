"""Zero-cost canonical-memory identifiability audit on G-train and recorded runs.

The script is fail-closed to G-train episode suffixes 0--9.  It never opens a
G-test npz, starts RoboMME, or calls an LLM.  Oracle arrays are used only after
memory decisions are produced, for evaluation and tracker-error measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from run import git_state, load_params  # noqa: E402
from perceived_qa import OCCLUSION, truth  # noqa: E402
from arbiter_kb.canonical_probe import (hard_pick_probe, project_identity,  # noqa: E402
                                        threshold_pick_probe)
from arbiter_kb.factorial_memory import FactorialMemory  # noqa: E402
from arbiter_kb.sg_compat import F_HIDDEN, F_LABEL, F_U, F_V, F_X, F_Z, assign_slots_seeded  # noqa: E402
from arbiter_kb.sources.perceived import PerceivedSource, _seen, occluded  # noqa: E402
from arbiter_kb.swapstats import episode_passes, identities  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.systems.baselines import TrackerSystem  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_suffix(path: Path) -> int:
    return int(path.stem.rsplit("_ep", 1)[1])


def assert_train_only(src: PerceivedSource, episodes: list[int]) -> None:
    bad = [src.files[i].name for i in episodes if train_suffix(src.files[i]) >= 10]
    if bad:
        raise RuntimeError(f"refusing non-train G records: {bad}")


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


def episode_cluster_bootstrap(rows: list[dict], left: str, right: str,
                              draws: int = 10_000, seed: int = 20261101) -> dict:
    """Paired uncertainty with episode, rather than query, as the resampling unit."""
    groups = {episode: [r for r in rows if r["episode"] == episode]
              for episode in sorted({r["episode"] for r in rows})}
    episodes = list(groups)
    rng = np.random.default_rng(seed)
    observed, samples = {}, {"top1_accuracy": [], "mean_brier": [], "mean_log_loss": []}

    def diffs(selected: list[dict]) -> dict[str, float]:
        return {
            "top1_accuracy": float(np.mean([r["states"][left]["correct"] for r in selected]) -
                                   np.mean([r["states"][right]["correct"] for r in selected])),
            "mean_brier": float(np.mean([r["states"][left]["brier"] for r in selected]) -
                                np.mean([r["states"][right]["brier"] for r in selected])),
            "mean_log_loss": float(np.mean([r["states"][left]["log_loss"] for r in selected]) -
                                  np.mean([r["states"][right]["log_loss"] for r in selected])),
        }

    observed = diffs(rows)
    for _ in range(draws):
        picked = rng.choice(episodes, size=len(episodes), replace=True)
        selected = [row for episode in picked for row in groups[str(episode)]]
        for metric, value in diffs(selected).items():
            samples[metric].append(value)
    return {
        "unit": "G-train episode", "episodes": len(episodes), "draws": draws, "seed": seed,
        "contrast": f"{left} minus {right}",
        "metrics": {metric: {"estimate": observed[metric],
                              "percentile_95_interval": [float(np.percentile(values, 2.5)),
                                                         float(np.percentile(values, 97.5))]}
                    for metric, values in samples.items()},
    }


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    out, start = [], None
    for i, value in enumerate([*flags, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    return out


def tracker_error_audit() -> dict:
    src = PerceivedSource(**OCCLUSION)
    eps = src.episodes("train")
    assert_train_only(src, eps)
    pass_rows, identity_acc = [], []
    observed, missed, associated, false_assoc, unmatched = 0, 0, 0, 0, 0
    wrong_runs = recovered_runs = 0
    occ_lengths = []
    for ep in eps:
        path = src.files[ep]
        with np.load(path, allow_pickle=True) as z:
            data = {"perceived": z["perceived"], "oracle": z["oracle"], "bbox": z["bbox"]}
        rows, summary = episode_passes(data)
        pass_rows.extend({**r, "episode": path.stem} for r in rows)
        identity_acc.append(summary["identity_accuracy"])
        per, ora = data["perceived"], data["oracle"]
        nodes = [int(n) for n in range(ora.shape[1]) if np.nanmax(ora[:, n, F_LABEL]) == 0]
        assignment, _ = assign_slots_seeded(per, ora)
        slots = [s for s, n in assignment.items() if n in nodes]
        ids = identities(per, ora, slots, nodes)
        for slot in slots:
            node = assignment[slot]
            wrong = []
            for t in range(len(per)):
                oracle_visible = bool(ora[t, node, F_HIDDEN] == 0 and
                                      np.all(np.isfinite(ora[t, node, F_U:F_Z + 1])))
                if oracle_visible:
                    observed += 1
                    if not _seen(per, t, slot):
                        missed += 1
                if _seen(per, t, slot):
                    associated += 1
                    unmatched += int(ids[t, slot] < 0)
                    false_assoc += int(ids[t, slot] >= 0 and ids[t, slot] != node)
                wrong.append(bool(ids[t, slot] >= 0 and ids[t, slot] != node))
            for start, stop in _runs(wrong):
                wrong_runs += 1
                recovered_runs += int(any(ids[t, slot] == node for t in range(stop, len(per))))
        nb = len(nodes)
        occ = occluded(per, nb, OCCLUSION["r"], OCCLUSION["dz"])
        for slot in range(nb, occ.shape[1]):
            occ_lengths.extend(stop - start for start, stop in _runs(occ[:, slot].tolist()))
    kinds = Counter(r["kind"] for r in pass_rows if r["swapped"])
    return {
        "episodes": len(eps), "pass_count": len(pass_rows),
        "swap_count": int(sum(r["swapped"] for r in pass_rows)),
        "swap_rate_per_pass": float(np.mean([r["swapped"] for r in pass_rows])) if pass_rows else None,
        "swap_kind_counts": dict(kinds),
        "mean_episode_identity_accuracy": float(np.nanmean(identity_acc)),
        "container_missed_detection_rate": missed / observed if observed else None,
        "container_missed_frames": missed, "container_oracle_visible_frames": observed,
        "false_association_rate_when_track_visible": false_assoc / associated if associated else None,
        "false_association_frames": false_assoc, "visible_association_frames": associated,
        "unmatched_visible_rate": unmatched / associated if associated else None,
        "wrong_identity_runs": wrong_runs, "recovered_wrong_identity_runs": recovered_runs,
        "recovery_rate": recovered_runs / wrong_runs if wrong_runs else None,
        "inferred_cube_occlusion_runs": len(occ_lengths),
        "inferred_cube_occlusion_duration_frames": {
            "median": float(np.median(occ_lengths)) if occ_lengths else None,
            "p90": float(np.percentile(occ_lengths, 90)) if occ_lengths else None,
            "max": max(occ_lengths) if occ_lengths else None,
        },
        "definitions": {
            "swap": "episode_passes: a tracker slot ends on the other oracle container after a simultaneous-motion pass",
            "missed_detection": "the initially assigned tracker slot is absent while its oracle container is visible",
            "false_association": "a visible slot is within 25 px of an oracle container other than its initial assignment",
            "recovery": "after a contiguous false-association run, the slot is later associated with its initial container",
        },
    }


NORMAL = re.compile(r"Execution progress: pick-ups (\d+), releases (\d+)")
LEGACY = re.compile(r"Progress:.*?pick-ups (\d+), releases (\d+)")


def progress_conflicts() -> dict:
    base = ROOT / "results/robomme_common/factorial_v1/pilot_20261101T0453Z/pilot/clean"
    rows = []
    for path in sorted(base.glob("*_arbiter_*/events.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            text = ((event.get("memory_before") or {}).get("text") or "")
            n, old = NORMAL.search(text), LEGACY.search(text)
            if not n or not old:
                continue
            normalized, legacy = tuple(map(int, n.groups())), tuple(map(int, old.groups()))
            rows.append({"cell": path.parent.name, "decision": event.get("decision"),
                         "normalized_pick_release": normalized, "legacy_pick_release": legacy,
                         "conflict": normalized != legacy})
    return {"decisions_with_both_views": len(rows), "conflicts": sum(r["conflict"] for r in rows),
            "rows": [r for r in rows if r["conflict"]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    params = load_params()
    rows, data = memory_rows(params)
    summary = {
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
        "tracker_errors": tracker_error_audit(),
        "recorded_factorial_progress_conflicts": progress_conflicts(),
        "identifiability": {
            "hard_state_action_identifiable": True,
            "calibration_closed_loop_identifiable": False,
            "reason": "G-train contains different legal pick targets, but RoboMME exposes no legal abstain or information-gathering action; confidence alone can only block, not choose a recovery action.",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "rows.json").write_text(json.dumps(rows, indent=2))
    tracked = [
        "arbiter_kb/canonical_probe.py", "arbiter_kb/factorial_memory.py",
        "scripts/canonical_identifiability.py", "scripts/canonical_timing.py",
        "tests/test_canonical_probe.py", "tests/test_factorial.py", "params.json",
    ]
    manifest = {
        "protocol": "canonical-memory-v2-zero-cost-identifiability",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "main_ref": subprocess.check_output(
            ["git", "rev-parse", "main"], cwd=ROOT, text=True).strip(),
        "source_dirty_before_output": bool(summary["git"]["dirty"]),
        "sources": {
            "memi": "bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0",
            "robomme": "fae9f7300347f1146c2ae4844faa4eef7e38098e",
            "maniskill": "07be6fbc66350ddca200abfb0a11b692f078f7fd",
        },
        "artifacts_sha256": {path: sha256(ROOT / path) for path in tracked},
        "g_train_files_sha256": data["files"],
        "execution": {
            "new_simulator_rollouts": 0, "new_llm_calls": 0,
            "new_api_cost_usd": 0.0, "heldout_records_opened": False,
            "observation_degradation_run": False, "selective_reruns": False,
        },
        "reproduce": (
            "PYTHONPATH=.:scripts .venv/bin/python scripts/canonical_identifiability.py "
            "--out results/robomme_common/canonical_memory_v2/identifiability"
        ),
    }
    (args.out / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"rows": len(rows), "episodes": data["episodes"],
                      "top1_disagreement": summary["pairs"]["arbiterkb_vs_strong_ledger"]["top1_disagreement"],
                      "both_legal_different_pick": summary["pairs"]["arbiterkb_vs_strong_ledger"]["both_legal_different_pick"],
                      "heldout_records_opened": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
