"""Support functions for the canonical memory audit."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from arbiter_kb.sg_compat import F_HIDDEN, F_LABEL, F_U, F_Z, assign_slots_seeded
from arbiter_kb.sources.perceived import PerceivedSource, _seen, occluded
from arbiter_kb.swapstats import episode_passes, identities

ROOT = Path(__file__).resolve().parents[1]
NORMAL = re.compile(r"Execution progress: pick-ups (\d+), releases (\d+)")
LEGACY = re.compile(r"Progress:.*?pick-ups (\d+), releases (\d+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_suffix(path: Path) -> int:
    return int(path.stem.rsplit("_ep", 1)[1])


def assert_train_only(src: PerceivedSource, episodes: list[int]) -> None:
    bad = [src.files[i].name for i in episodes if train_suffix(src.files[i]) >= 10]
    if bad:
        raise RuntimeError(f"refusing non-train G records: {bad}")


def episode_cluster_bootstrap(rows: list[dict], left: str, right: str,
                              draws: int = 10_000, seed: int = 20261101) -> dict:
    groups = {episode: [r for r in rows if r["episode"] == episode]
              for episode in sorted({r["episode"] for r in rows})}
    episodes = list(groups)
    rng = np.random.default_rng(seed)
    samples = {"top1_accuracy": [], "mean_brier": [], "mean_log_loss": []}
    observed = _metric_differences(rows, left, right)
    for _ in range(draws):
        picked = rng.choice(episodes, size=len(episodes), replace=True)
        selected = [row for episode in picked for row in groups[str(episode)]]
        for metric, value in _metric_differences(selected, left, right).items():
            samples[metric].append(value)
    metrics = {metric: {"estimate": observed[metric], "percentile_95_interval": [
        float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}
        for metric, values in samples.items()}
    return {"unit": "G-train episode", "episodes": len(episodes), "draws": draws,
            "seed": seed, "contrast": f"{left} minus {right}", "metrics": metrics}


def _metric_differences(rows: list[dict], left: str, right: str) -> dict[str, float]:
    def mean(arm: str, metric: str) -> float:
        return float(np.mean([r["states"][arm][metric] for r in rows]))
    return {
        "top1_accuracy": mean(left, "correct") - mean(right, "correct"),
        "mean_brier": mean(left, "brier") - mean(right, "brier"),
        "mean_log_loss": mean(left, "log_loss") - mean(right, "log_loss"),
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


def _slot_counts(per, ora, slot: int, node: int, ids) -> np.ndarray:
    counts = np.zeros(7, dtype=int)
    wrong = []
    for t in range(len(per)):
        oracle_visible = bool(ora[t, node, F_HIDDEN] == 0 and
                              np.all(np.isfinite(ora[t, node, F_U:F_Z + 1])))
        seen = _seen(per, t, slot)
        counts[0] += oracle_visible
        counts[1] += oracle_visible and not seen
        counts[2] += seen
        counts[3] += seen and ids[t, slot] >= 0 and ids[t, slot] != node
        counts[4] += seen and ids[t, slot] < 0
        wrong.append(bool(ids[t, slot] >= 0 and ids[t, slot] != node))
    for _, stop in _runs(wrong):
        counts[5] += 1
        counts[6] += any(ids[t, slot] == node for t in range(stop, len(per)))
    return counts


def _episode_tracker(src: PerceivedSource, ep: int, occlusion: dict) -> tuple[list, float, np.ndarray, list[int]]:
    path = src.files[ep]
    with np.load(path, allow_pickle=True) as z:
        data = {"perceived": z["perceived"], "oracle": z["oracle"], "bbox": z["bbox"]}
    rows, summary = episode_passes(data)
    per, ora = data["perceived"], data["oracle"]
    nodes = [int(n) for n in range(ora.shape[1]) if np.nanmax(ora[:, n, F_LABEL]) == 0]
    assignment, _ = assign_slots_seeded(per, ora)
    slots = [s for s, n in assignment.items() if n in nodes]
    ids = identities(per, ora, slots, nodes)
    counts = sum((_slot_counts(per, ora, slot, assignment[slot], ids) for slot in slots),
                 start=np.zeros(7, dtype=int))
    occ, nb = occluded(per, len(nodes), occlusion["r"], occlusion["dz"]), len(nodes)
    lengths = [stop - start for slot in range(nb, occ.shape[1])
               for start, stop in _runs(occ[:, slot].tolist())]
    return [{**r, "episode": path.stem} for r in rows], summary["identity_accuracy"], counts, lengths


def tracker_error_audit(occlusion: dict) -> dict:
    src = PerceivedSource(**occlusion)
    eps = src.episodes("train")
    assert_train_only(src, eps)
    pass_rows, accuracy, lengths = [], [], []
    counts = np.zeros(7, dtype=int)
    for ep in eps:
        rows, acc, add, occ = _episode_tracker(src, ep, occlusion)
        pass_rows.extend(rows)
        accuracy.append(acc)
        counts += add
        lengths.extend(occ)
    observed, missed, associated, false, unmatched, wrong, recovered = map(int, counts)
    kinds = Counter(r["kind"] for r in pass_rows if r["swapped"])
    return _tracker_summary(eps, pass_rows, accuracy, lengths, kinds,
                            observed, missed, associated, false, unmatched, wrong, recovered)


def _tracker_summary(eps, rows, acc, lengths, kinds, observed, missed, associated,
                     false, unmatched, wrong, recovered) -> dict:
    return {
        "episodes": len(eps), "pass_count": len(rows),
        "swap_count": int(sum(r["swapped"] for r in rows)),
        "swap_rate_per_pass": float(np.mean([r["swapped"] for r in rows])) if rows else None,
        "swap_kind_counts": dict(kinds), "mean_episode_identity_accuracy": float(np.nanmean(acc)),
        "container_missed_detection_rate": missed / observed if observed else None,
        "container_missed_frames": missed, "container_oracle_visible_frames": observed,
        "false_association_rate_when_track_visible": false / associated if associated else None,
        "false_association_frames": false, "visible_association_frames": associated,
        "unmatched_visible_rate": unmatched / associated if associated else None,
        "wrong_identity_runs": wrong, "recovered_wrong_identity_runs": recovered,
        "recovery_rate": recovered / wrong if wrong else None,
        "inferred_cube_occlusion_runs": len(lengths),
        "inferred_cube_occlusion_duration_frames": {
            "median": float(np.median(lengths)) if lengths else None,
            "p90": float(np.percentile(lengths, 90)) if lengths else None,
            "max": max(lengths) if lengths else None},
        "definitions": {
            "swap": "episode_passes: a tracker slot ends on the other oracle container after a simultaneous-motion pass",
            "missed_detection": "the initially assigned tracker slot is absent while its oracle container is visible",
            "false_association": "a visible slot is within 25 px of an oracle container other than its initial assignment",
            "recovery": "after a contiguous false-association run, the slot is later associated with its initial container"},
    }


def progress_conflicts() -> dict:
    base = ROOT / "results/robomme_common/factorial_v1/pilot_20261101T0453Z/pilot/clean"
    rows = []
    for path in sorted(base.glob("*_arbiter_*/events.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            text = ((event.get("memory_before") or {}).get("text") or "")
            new, old = NORMAL.search(text), LEGACY.search(text)
            if new and old:
                normalized, legacy = tuple(map(int, new.groups())), tuple(map(int, old.groups()))
                rows.append({"cell": path.parent.name, "decision": event.get("decision"),
                             "normalized_pick_release": normalized, "legacy_pick_release": legacy,
                             "conflict": normalized != legacy})
    return {"decisions_with_both_views": len(rows), "conflicts": sum(r["conflict"] for r in rows),
            "rows": [r for r in rows if r["conflict"]]}


def write_outputs(out: Path, summary: dict, rows: list[dict], data: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    tracked = ["arbiter_kb/canonical_probe.py", "arbiter_kb/factorial_memory.py",
               "scripts/canonical_identifiability.py", "arbiter_kb/canonical_audit.py",
               "scripts/canonical_timing.py", "tests/test_canonical_probe.py",
               "tests/test_factorial.py", "params.json"]
    manifest = {
        "protocol": "canonical-memory-v2-zero-cost-identifiability",
        "source_commit": _git("HEAD"), "main_ref": _git("main"),
        "source_dirty_before_output": bool(summary["git"]["dirty"]),
        "sources": {"memi": "bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0",
                    "robomme": "fae9f7300347f1146c2ae4844faa4eef7e38098e",
                    "maniskill": "07be6fbc66350ddca200abfb0a11b692f078f7fd"},
        "artifacts_sha256": {path: sha256(ROOT / path) for path in tracked},
        "g_train_files_sha256": data["files"],
        "execution": {"new_simulator_rollouts": 0, "new_llm_calls": 0,
                      "new_api_cost_usd": 0.0, "heldout_records_opened": False,
                      "observation_degradation_run": False, "selective_reruns": False},
        "reproduce": "PYTHONPATH=.:scripts .venv/bin/python scripts/canonical_identifiability.py --out results/robomme_common/canonical_memory_v2/identifiability"}
    (out / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2))


def _git(ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
