"""Perception noise on still containers, G's perceived train episodes (NOTES "Move threshold from real perception
noise"; the rule is fixed there before this ran). Free: no LLM.

Still: the oracle container a perceived track follows (the nearest in xy at that frame) moved < 0.1 mm since the
previous frame. A still segment is a maximal run of still frames of one track on one oracle container, seen in every
frame. Output: the xy displacement between every pair of frames
in a segment (percentiles; T = the 99th), and the pool of residuals (perceived xyz - the segment's median) that the
offline test's noisy condition draws from.

  python scripts/still_noise.py            # prints the numbers, writes data/g_perceived_still_residuals.npy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.sg_compat import F_HIDDEN, F_X, F_Y, F_Z  # noqa: E402
from arbiter_kb.sources.perceived import PerceivedSource, _seen  # noqa: E402

STILL = 1e-4  # m: oracle xy change below this between frames = still
OUT = ROOT / "data" / "g_perceived_still_residuals.npy"


def _oracle_xy(ora: np.ndarray, t: int, n_bins: int) -> dict[int, np.ndarray]:
    return {c: ora[t, c, F_X:F_Y + 1] for c in range(n_bins)
            if ora[t, c, F_HIDDEN] == 0 and np.all(np.isfinite(ora[t, c, F_X:F_Y + 1]))}


def segments(per: np.ndarray, ora: np.ndarray, s: int, n_bins: int) -> list[np.ndarray]:
    """Perceived xyz of each still segment of track s. The track's object at frame t is the oracle container nearest
    to it in xy (perceived and oracle slots are numbered differently: the first run of this script paired them by
    index, NOTES); the frame is still when that object moved < STILL since t-1, and a segment keeps one object."""
    out, cur, obj = [], [], None
    for t in range(1, len(per)):
        now, before = _oracle_xy(ora, t, n_bins), _oracle_xy(ora, t - 1, n_bins)
        c = min(now, key=lambda k: np.linalg.norm(now[k] - per[t, s, F_X:F_Y + 1])) if now and _seen(per, t, s) \
            else None
        still = c is not None and c in before and np.linalg.norm(now[c] - before[c]) < STILL
        if still and (obj is None or c == obj):
            cur.append(per[t, s, F_X:F_Z + 1].astype(float))
            obj = c
        else:
            out += [np.array(cur)] if len(cur) > 1 else []
            cur, obj = ([per[t, s, F_X:F_Z + 1].astype(float)], c) if still else ([], None)
    return out + ([np.array(cur)] if len(cur) > 1 else [])


def main() -> None:
    src = PerceivedSource()
    pairs, resid, n_seg = [], [], 0
    for ep in src.episodes("train"):
        d = src.load(ep)
        for s in range(d["meta"]["n_bins"]):
            for seg in segments(d["perceived"], d["oracle"], s, d["meta"]["n_bins"]):
                n_seg += 1
                xy = seg[:, :2]
                i, j = np.triu_indices(len(xy), 1)
                pairs.append(np.linalg.norm(xy[j] - xy[i], axis=1))
                resid.append(seg - np.median(seg, axis=0))
    pairs, resid = np.concatenate(pairs), np.concatenate(resid)
    pct = {q: float(np.percentile(pairs, q)) for q in (50, 95, 99)}
    out = {"episodes": len(src.episodes("train")), "segments": n_seg, "pairs": int(len(pairs)),
           "pair_xy_m": {**{f"p{q}": round(v, 5) for q, v in pct.items()}, "max": round(float(pairs.max()), 5)},
           "T_m": round(pct[99], 4), "residuals": int(len(resid)),
           "residual_std_xyz_m": [round(float(v), 5) for v in resid.std(axis=0)]}
    np.save(OUT, resid.astype(np.float32))
    print(json.dumps(out, indent=1), "\n", OUT)


if __name__ == "__main__":
    main()
