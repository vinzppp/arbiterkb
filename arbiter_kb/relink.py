"""Re-link container tracks by motion (NOTES "Fixing the tracker's swaps"): the containers are identical, so the
tracker's ids can jump between two containers when they pass close. Each identity keeps a position and a smoothed
velocity; every frame the visible container slots are assigned to the identities' predicted positions (Hungarian on
xy distance), so a crossing is followed by its direction of motion rather than by the tracker's label. An identity
starts on its own slot the first time that slot is seen. Cube slots are left as they are.

`margin` (claim 3, NOTES "Claim 3 on G"): for each identity and frame, how much worse (m) the assignment would be if
it exchanged slots with its closest competitor. Near 0 means the motion cannot tell the two apart: the place where the
tracker, and this re-linking, can swap identities. inf when no other identity was matched in that frame.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .sg_compat import F_HIDDEN, F_ID, F_LABEL, F_U, F_X, F_Z


def _seen(rows: np.ndarray, t: int, s: int) -> bool:
    return bool(rows[t, s, F_HIDDEN] == 0 and np.all(np.isfinite(rows[t, s, F_U:F_Z + 1])))


def relink(per: np.ndarray, n_bins: int, alpha: float = 0.3, v_max: float = 0.05) -> np.ndarray:
    return relink_margin(per, n_bins, alpha, v_max)[0]


def relink_margin(per: np.ndarray, n_bins: int, alpha: float = 0.3, v_max: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """per [T, S, 8] -> the same array with the container rows re-assigned to motion-consistent identities.
    alpha: velocity smoothing; v_max (m/frame): larger steps are tracker jumps and do not update the velocity."""
    out = per.copy()
    margin = np.full((len(per), n_bins), np.inf)
    pos: list[np.ndarray | None] = [None] * n_bins
    vel = [np.zeros(2) for _ in range(n_bins)]
    for t in range(len(per)):
        obs = [s for s in range(n_bins) if _seen(per, t, s)]
        live = [j for j in range(n_bins) if pos[j] is not None]
        match: dict[int, int] = {}
        if live and obs:
            cost = np.array([[np.linalg.norm(pos[j] + vel[j] - per[t, s, F_X:F_X + 2]) for s in obs] for j in live])
            rows, cols = linear_sum_assignment(cost)
            match = {live[i]: obs[k] for i, k in zip(rows, cols)}
            col = dict(zip(rows, cols))
            for i in col:
                for i2 in col:
                    if i2 != i:
                        d = cost[i, col[i2]] + cost[i2, col[i]] - cost[i, col[i]] - cost[i2, col[i2]]
                        margin[t, live[i]] = min(margin[t, live[i]], float(d))
        for j in range(n_bins):
            if pos[j] is None and j in obs and j not in match.values():
                match[j] = j
        out[t, :n_bins, F_U:F_Z + 1] = np.nan
        out[t, :n_bins, F_HIDDEN] = 1
        for j, s in match.items():
            p = per[t, s, F_X:F_X + 2]
            if pos[j] is not None and np.linalg.norm(p - pos[j]) < v_max:
                vel[j] = alpha * (p - pos[j]) + (1 - alpha) * vel[j]
            pos[j] = p.copy()
            out[t, j] = per[t, s]
            out[t, j, F_ID], out[t, j, F_LABEL] = j, 0
    return out, margin
