"""Phase G: how often, and how, does the real tracker swap container ids? (DEVELOPMENT.md §6 G)

Input: one episode as arrays (written on the pod by stress/perceive.py):
  perceived [T, S, 8]  tracker slots in sg_compat row layout (u, v, x, y, z, hidden ...)
  bbox      [T, S, 4]  normalised [row_min, row_max, col_min, col_max]; zeros when the slot is absent
  oracle    [T, N, 8]  simulator ground truth (segmentation + depth), same layout
Every pass (two containers moving at the same time, swapped or not) becomes one row of features, so the
features can be judged by ROC: can a trivial rule tell swapped from clean passes?
"""
from __future__ import annotations

import numpy as np

from .sg_compat import F_HIDDEN, F_LABEL, F_U, F_V, F_X, F_Z, assign_slots_seeded

FEATURES = ("min_3d", "min_img", "max_iou", "area_ratio", "max_jump", "min_margin")
SWAP_DIRECTION = {"min_3d": -1, "min_img": -1, "max_iou": 1, "area_ratio": -1, "max_jump": 1, "min_margin": -1}


def identities(perceived: np.ndarray, oracle: np.ndarray, slots: list[int], nodes: list[int],
               radius_px: float = 25.0) -> np.ndarray:
    """[T, S] oracle container node each container slot sits on per frame (-1: absent or too far)."""
    T = len(perceived)
    out = np.full((T, perceived.shape[1]), -1, int)
    for t in range(T):
        for s in slots:
            if perceived[t, s, F_HIDDEN] != 0 or not np.all(np.isfinite(perceived[t, s, F_U:F_V + 1])):
                continue
            d = [np.linalg.norm(oracle[t, n, F_U:F_V + 1] - perceived[t, s, F_U:F_V + 1]) for n in nodes]
            d = np.where(np.isfinite(d), d, np.inf)
            if len(d) and d.min() < radius_px:
                out[t, s] = nodes[int(np.argmin(d))]
    return out


def oracle_passes(oracle: np.ndarray, nodes: list[int], v_move: float = 0.001) -> list[dict]:
    """Intervals where two containers move at once (oracle 3D), with their closest approach."""
    xyz = oracle[:, :, F_X:F_Z + 1]
    step = np.linalg.norm(np.diff(xyz[:, :, :2], axis=0), axis=-1)
    moving = np.vstack([np.zeros((1, xyz.shape[1]), bool), np.nan_to_num(step) > v_move])
    out = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            both = moving[:, a] & moving[:, b]
            t = 0
            while t < len(both):
                if not both[t]:
                    t += 1
                    continue
                t1 = t
                while t1 < len(both) and both[t1]:
                    t1 += 1
                d = np.linalg.norm(xyz[t:t1, a] - xyz[t:t1, b], axis=-1)
                k = int(np.nanargmin(d)) if np.isfinite(d).any() else 0
                out.append({"a": a, "b": b, "t_open": t, "t_close": t1, "t_min": t + k, "min_3d": float(np.nanmin(d))})
                t = t1
    return sorted(out, key=lambda p: p["t_min"])


def _slot_on(ids: np.ndarray, node: int, t: int, look: int = 5) -> int | None:
    """The slot sitting on `node` in the last frames before t (None if no slot was)."""
    for u in range(t - 1, max(-1, t - 1 - look), -1):
        hit = np.nonzero(ids[u] == node)[0]
        if len(hit):
            return int(hit[0])
    return None


def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    if not (b1.any() and b2.any()):
        return 0.0
    r = max(0.0, min(b1[1], b2[1]) - max(b1[0], b2[0]))
    c = max(0.0, min(b1[3], b2[3]) - max(b1[2], b2[2]))
    area = lambda b: (b[1] - b[0]) * (b[3] - b[2])  # noqa: E731
    inter = r * c
    return float(inter / (area(b1) + area(b2) - inter + 1e-12))


def _outcome(ids: np.ndarray, sa: int, sb: int, p: dict, t1: int, settle: int) -> tuple[int, int]:
    """(swapped, lost): the containers the two slots settle on after the pass."""
    after = {s: [n for n in ids[t1 - settle:t1, s] if n >= 0] for s in (sa, sb)}
    end = {s: (max(set(v), key=v.count) if v else -1) for s, v in after.items()}
    return int(end[sa] == p["b"] or end[sb] == p["a"]), int(end[sa] < 0 or end[sb] < 0)


def _image_features(ep: dict, p: dict, sa: int, sb: int, win: range) -> dict:
    """Closest image distance (oracle), max bbox IoU of the two slots, worst bbox-area ratio vs before."""
    box, orc, t0 = ep["bbox"], ep["oracle"], p["t_open"]
    img = [np.linalg.norm(orc[t, p["a"], F_U:F_V + 1] - orc[t, p["b"], F_U:F_V + 1]) for t in win]
    area = lambda s, t: (box[t, s, 1] - box[t, s, 0]) * (box[t, s, 3] - box[t, s, 2])  # noqa: E731
    pre = {s: np.median([area(s, t) for t in range(max(0, t0 - 10), t0)] or [np.nan]) for s in (sa, sb)}
    ratio = [min(area(s, t) / pre[s] for s in (sa, sb)) for t in win if pre[sa] > 0 and pre[sb] > 0]
    return {"min_img": _nanmin(img), "max_iou": max([_iou(box[t, sa], box[t, sb]) for t in win] or [np.nan]),
            "area_ratio": _nanmin(ratio)}


def _motion_features(per: np.ndarray, sa: int, sb: int, win: range) -> dict:
    """Largest one-frame jump of either slot (m) and the smallest keep-vs-exchange margin:
    (cost of exchanging the two tracks - cost of keeping them) / keep cost, frame to frame."""
    xyz = per[:, :, F_X:F_Z + 1]
    jumps, margins = [], []
    for t in win:
        if t == 0:
            continue
        a0, b0, a1, b1 = xyz[t - 1, sa], xyz[t - 1, sb], xyz[t, sa], xyz[t, sb]
        if not np.all(np.isfinite([a0, b0, a1, b1])):
            continue
        keep = np.linalg.norm(a1 - a0) + np.linalg.norm(b1 - b0)
        swap = np.linalg.norm(a1 - b0) + np.linalg.norm(b1 - a0)
        jumps.append(max(np.linalg.norm(a1 - a0), np.linalg.norm(b1 - b0)))
        margins.append((swap - keep) / max(keep, 1e-4))
    return {"max_jump": float(np.nanmax(jumps)) if jumps else np.nan, "min_margin": _nanmin(margins)}


def _nanmin(v) -> float:
    return float(np.nanmin(v)) if len(v) and np.isfinite(v).any() else np.nan


def pass_features(p: dict, ep: dict, ids: np.ndarray, settle: int = 5) -> dict | None:
    """One feature row for a pass, and whether the two tracker slots exchanged their containers."""
    per = ep["perceived"]
    sa, sb = _slot_on(ids, p["a"], p["t_open"]), _slot_on(ids, p["b"], p["t_open"])
    if sa is None or sb is None or sa == sb:
        return None
    t1 = min(len(per), p["t_close"] + settle)
    swapped, lost = _outcome(ids, sa, sb, p, t1, settle)
    win = range(p["t_open"], min(len(per), p["t_close"]))
    row = {"node_a": p["a"], "node_b": p["b"], "slot_a": sa, "slot_b": sb, "t_open": p["t_open"],
           "t_min": p["t_min"], "duration": p["t_close"] - p["t_open"], "swapped": swapped, "lost": lost,
           "min_3d": p["min_3d"]}
    return {**row, **_image_features(ep, p, sa, sb, win), **_motion_features(per, sa, sb, win)}


def episode_passes(ep: dict, v_move: float = 0.001) -> tuple[list[dict], dict]:
    """(one feature row per pass, identity summary) for one perceived episode."""
    per, orc = ep["perceived"], ep["oracle"]
    nodes = [int(n) for n in range(orc.shape[1]) if np.nanmax(orc[:, n, F_LABEL]) == 0]
    assignment, _ = assign_slots_seeded(per, orc)
    slots = [s for s, n in assignment.items() if n in nodes]
    ids = identities(per, orc, slots, nodes)
    rows = [r for p in oracle_passes(orc, nodes, v_move) if (r := pass_features(p, ep, ids)) is not None]
    seen = [(t, s) for s in slots for t in range(len(per)) if ids[t, s] >= 0]
    ok = sum(ids[t, s] == assignment[s] for t, s in seen)
    summary = {"identity_accuracy": ok / len(seen) if seen else np.nan, "n_slots": len(slots),
               "n_container_nodes": len(nodes), "n_passes": len(rows), "n_swaps": sum(r["swapped"] for r in rows)}
    return rows, summary


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC (Mann-Whitney) of `score` for label 1; ties count half; nan if a class is missing."""
    ok = np.isfinite(score)
    s, y = score[ok], label[ok].astype(bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def logistic_cv_auc(X: np.ndarray, y: np.ndarray, folds: int = 5, l2: float = 1e-2, seed: int = 0) -> float:
    """Cross-validated AUC of an L2 logistic regression (Newton steps) on standardised features.
    Rows with a missing feature are dropped. nan if a class has fewer than `folds` rows."""
    ok = np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok].astype(float)
    if min(y.sum(), len(y) - y.sum()) < folds:
        return np.nan
    order = np.random.default_rng(seed).permutation(len(y))
    score = np.zeros(len(y))
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        A = np.hstack([np.ones((len(train), 1)), (X[train] - mu) / sd])
        w = np.zeros(A.shape[1])
        for _ in range(25):
            p = 1 / (1 + np.exp(-A @ w))
            H = A.T @ (A * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
            w -= np.linalg.solve(H, A.T @ (p - y[train]) + l2 * w)
        score[test] = np.hstack([np.ones((len(test), 1)), (X[test] - mu) / sd]) @ w
    return auc(score, y)


def separability(rows: list[dict]) -> dict:
    """AUC of each feature (oriented so that higher means 'swapped'), and the best <= 2-feature rule."""
    y = np.array([r["swapped"] for r in rows])
    col = {f: np.array([r[f] for r in rows], float) for f in FEATURES}
    single = {f: auc(SWAP_DIRECTION[f] * col[f], y) for f in FEATURES}
    pairs = {f"{a}+{b}": logistic_cv_auc(np.stack([col[a], col[b]], 1), y)
             for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]}
    finite = {k: v for k, v in {**single, **pairs}.items() if np.isfinite(v)}
    best = max(finite, key=finite.get) if finite else None
    return {"single": single, "pairs": pairs, "best_rule": best, "best_auc": finite.get(best, np.nan)}
