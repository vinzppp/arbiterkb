"""Measurements of the synthetic generator that NOTES cites (E10), on TRAIN seeds only.
Writes results/m0_synthetic_stats.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.sources.corrupt import CorruptedSource  # noqa: E402
from arbiter_kb.sources.synthetic import COVER_T, SyntheticSource  # noqa: E402


def stats(n: int = 40) -> dict:
    src = SyntheticSource()
    peak, min_px, jumps, max_other_step = [], [], [], []
    for ep in src.episodes("train", n):
        fr = list(src.frames(ep))
        bins = [o for o, c, _ in src.meta(ep).objects if c == "container"]
        xy = {b: np.array([f.objects[b].xyz[:2] for f in fr[COVER_T:]]) for b in bins}
        uv = {b: np.array([f.objects[b].uv for f in fr[COVER_T:]]) for b in bins}
        sp = {b: np.linalg.norm(np.diff(xy[b], axis=0), axis=1) for b in bins}
        peak += [float(v.max()) for v in sp.values() if v.max() > 1e-6]
        max_other_step.append(float(max(v.max() for v in sp.values())))
        for i, a in enumerate(bins):
            for b in bins[i + 1:]:
                both = (sp[a] > 1e-3) & (sp[b] > 1e-3)
                if both.any():
                    min_px.append(float(np.linalg.norm(uv[a][1:] - uv[b][1:], axis=1)[both].min()))
        bad = CorruptedSource(src, p=1.0)
        cf = list(bad.frames(ep))
        log = bad.log[ep]
        if log["corrupted"]:
            a, t = log["ids"][0], log["t"]
            jumps.append(float(np.linalg.norm(cf[t].objects[a].xyz[:2] - cf[t - 1].objects[a].xyz[:2])))
    q = lambda v: [round(float(x), 4) for x in np.percentile(v, [0, 50, 100])]  # noqa: E731
    return {"seeds": f"train 0..{n - 1}", "peak_container_speed_m_per_frame_min_med_max": q(peak),
            "min_image_distance_of_passing_containers_px_min_med_max": q(min_px),
            "id_swap_label_jump_m_min_med_max": q(jumps),
            "largest_true_per_frame_step_m_max": round(max(max_other_step), 4), "n_corrupted": len(jumps)}


if __name__ == "__main__":
    out = ROOT / "results" / "m0_synthetic_stats.json"
    out.write_text(json.dumps(stats(), indent=2))
    print(out.read_text())
