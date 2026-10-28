"""Evaluation harness: run systems frame by frame on a source and score them at each target's t*.

Systems never see GroundTruth and are never told t* (E5b); frames are consumed lazily, one at a time
(E5a); the test split needs a frozen parameter set (E5c). All systems share this path (E5d).
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from .metrics import brier, summarize
from .params import Params


def physical_truth(system, bins: list[str], true_xyz: np.ndarray, r: float = 0.05) -> str | None:
    """The label THIS system uses for the true container: the one whose believed position is nearest
    the true position. Under id corruption labels lie, so correctness is judged physically."""
    cands = [(float(np.linalg.norm(x[:2] - true_xyz[:2])), b) for b in bins if (x := system.pose(b)) is not None]
    if not cands or min(cands)[0] > r:
        return None
    return min(cands)[1]


def score(system, cube: str, gt, bins: list[str], ep: int, t: int, condition: str, corrupted: bool) -> dict:
    d = system.believe_inside(cube)
    pred, p = d.argmax()
    step = system.plan(cube)
    truth = physical_truth(system, bins, gt.bin_xyz_at_t_star[cube])
    probs = {str(k): v for k, v in d.probs.items()}
    err = math.nan
    if len(step.waypoints):
        err = float(np.linalg.norm(step.waypoints[1, :2] - gt.bin_xyz_at_t_star[cube][:2]))
    return {
        "condition": condition, "system": system.name, "episode": ep, "cube": cube, "t_star": t,
        "truth_id": gt.cube_to_bin[cube], "truth": str(truth), "pred": str(pred), "p_max": round(float(p), 6),
        "correct": truth is not None and pred == truth, "brier": round(brier(probs, str(truth), bins), 6),
        "entropy": round(d.entropy(), 6), "decision": step.decision, "n_support": len(step.support),
        "grasp_err_m": round(err, 5), "moved": gt.moved[cube], "corrupted": corrupted,
    }


def run(systems: list, source, episodes: list[int], params: Params, condition: str,
        corrupted_of=lambda ep: False) -> tuple[list[dict], dict, list[float]]:
    """-> (rows, traces {(system, ep): rows}, per-frame step seconds of the last system)."""
    if any(ep >= 1000 for ep in episodes) and not params.frozen:
        raise RuntimeError("E5c: the test split needs a frozen params.json")
    rows, traces, step_s = [], {}, []
    for ep in episodes:
        meta, gt = source.meta(ep), source.ground_truth(ep)
        for s in systems:
            s.reset(meta)
        at = {t: c for c, t in gt.t_star.items()}
        bins = [oid for oid, cls, _ in meta.objects if cls == "container"]
        for frame in source.frames(ep):
            for s in systems:
                t0 = time.perf_counter()
                s.step(frame)
                if s is systems[-1]:
                    step_s.append(time.perf_counter() - t0)
            if frame.t in at:
                rows += [score(s, at[frame.t], gt, bins, ep, frame.t, condition, corrupted_of(ep)) for s in systems]
        traces.update({(s.name, ep): s.trace_rows() for s in systems})
    return rows, traces, step_s


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_run(run_dir: Path, rows: list[dict], params: Params, meta: dict, traces: dict) -> list[dict]:
    """rows.csv, summary.csv, params.json, meta.json, traces/<system>_<ep>.jsonl. Returns the summary."""
    from .trace import write_trace

    summary = summarize(rows, params.monitor.tau_hi)
    write_csv(run_dir / "rows.csv", rows)
    write_csv(run_dir / "summary.csv", summary)
    params.save(run_dir / "params.json")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    for (sysname, ep), trows in traces.items():
        write_trace(run_dir / "traces" / f"{meta.get('trace_prefix', '')}{sysname}_{ep}.jsonl", trows)
    return summary
