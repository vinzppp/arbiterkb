"""G2 (descriptive): the stock tracker error signature through the executor's point snapping.

Main NOTES 7327cf1, gate G2.  RoboMME's ``multi_choice`` wrapper turns a point into the
available actor whose projected centre is nearest to it (``select_target_with_pixel``).
A memory that stores a perceived container's pixel therefore changes the executed target
only when the perceived pixel lands nearer another container than its own.

Data: G **train** perceived replays only (``data/g_perceived/<setting>_ep000-009.npz``);
the script refuses any suffix >= 10 (G test) before opening a file.  Each perceived slot
is bound to an oracle container at its seed frame (``sg_compat.assign_slots_seeded``,
the identity the memory would carry).  A decision is (episode, container) at the last
frame, after every exchange: the memory points at the slot's latest perceived pixel and
the snap goes to the nearest oracle container pixel.  "Changes target" = snapped
container != the slot's own container.  All frames are reported as a sensitivity row.

Approximation, stated: oracle (u, v) is the mask centroid from ``oracle_graph``, not the
projected actor centre the wrapper uses.  Free; no simulator, no model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from arbiter_kb.sg_compat import F_HIDDEN, F_LABEL, F_U, F_V, assign_slots_seeded, wilson

CONTAINER = 0.0
TRAIN = re.compile(r"^(?P<setting>[a-z0-9_]+)_ep00(?P<ep>[0-9])\.npz$")


def train_files(root: Path, settings: list[str]) -> list[Path]:
    out = []
    for name in sorted(p.name for p in root.iterdir()):
        m = TRAIN.match(name)
        if m and m["setting"] in settings:
            out.append(root / name)
    for path in out:
        if not TRAIN.match(path.name):
            raise SystemExit(f"refusing {path.name}: not a G-train file")
    return out


def last_seen(per: np.ndarray, slot: int, t: int) -> np.ndarray | None:
    vis = np.nonzero(per[: t + 1, slot, F_HIDDEN] == 0)[0]
    return per[vis[-1], slot, F_U : F_V + 1] if len(vis) else None


def snap(point: np.ndarray, oracle_t: np.ndarray, nodes: list[int]) -> tuple[int | None, float, float]:
    """Nearest container node to `point`, its distance, and the margin to the second nearest."""
    d = sorted((float(np.linalg.norm(point - oracle_t[n, F_U : F_V + 1])), n) for n in nodes
               if np.all(np.isfinite(oracle_t[n, F_U : F_V + 1])))
    if not d:
        return None, float("nan"), float("nan")
    margin = d[1][0] - d[0][0] if len(d) > 1 else float("inf")
    return d[0][1], d[0][0], margin


def episode_rows(path: Path) -> list[dict]:
    z = np.load(path, allow_pickle=False)
    per, orc = z["perceived"], z["oracle"]
    meta = json.loads(str(z["meta"]))
    assignment, _ = assign_slots_seeded(per, orc)
    nodes = [n for n in range(orc.shape[1]) if np.nanmax(orc[:, n, F_LABEL]) == CONTAINER
             and np.any(orc[:, n, F_HIDDEN] == 0)]
    rows = []
    for t in range(per.shape[0]):
        visible_nodes = [n for n in nodes if orc[t, n, F_HIDDEN] == 0]
        if len(visible_nodes) < 2:
            continue
        for slot, node in assignment.items():
            if node not in nodes:
                continue
            seen_now = per[t, slot, F_HIDDEN] == 0
            point = per[t, slot, F_U : F_V + 1] if seen_now else last_seen(per, slot, t)
            if point is None or not np.all(np.isfinite(point)):
                continue
            got, dist, margin = snap(point, orc[t], visible_nodes)
            own = orc[t, node, F_U : F_V + 1]
            rows.append({"episode": path.stem, "setting": meta["setting"], "t": t, "final": t == per.shape[0] - 1,
                         "slot": slot, "node": node, "snapped": got, "changed": got != node,
                         "slot_visible": bool(seen_now),
                         "pixel_error": float(np.linalg.norm(point - own)) if np.all(np.isfinite(own)) else None,
                         "snap_distance": dist, "margin": margin})
    return rows


def summarize(rows: list[dict]) -> dict:
    n, k = len(rows), sum(r["changed"] for r in rows)
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    err = np.array([r["pixel_error"] for r in rows if r["pixel_error"] is not None])
    margins = np.array([r["margin"] for r in rows if np.isfinite(r["margin"])])
    return {"decisions": n, "changed": k, "rate": k / n if n else None, "wilson95": [lo, hi],
            "slot_hidden_at_decision": sum(not r["slot_visible"] for r in rows),
            "pixel_error_px": {"median": float(np.median(err)), "p95": float(np.percentile(err, 95)),
                               "max": float(err.max())} if len(err) else None,
            "snap_margin_px": {"min": float(margins.min()), "p05": float(np.percentile(margins, 5)),
                               "median": float(np.median(margins))} if len(margins) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("data/g_perceived"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    groups = {"stock": ["stock"], "stock_camera_shift": ["cam_left", "cam_right"]}
    out = {"source": "G train perceived replays, suffix 0-9 only; G test not opened",
           "decision": "last frame of the replay (after every exchange); each container slot = one decision",
           "approximation": "oracle mask-centroid pixel stands in for the wrapper's projected actor centre",
           "results": {}}
    for name, settings in groups.items():
        files = train_files(a.root, settings)
        rows = [r for path in files for r in episode_rows(path)]
        out["results"][name] = {"episodes": len(files),
                                "final_frame": summarize([r for r in rows if r["final"]]),
                                "all_frames": summarize(rows)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["results"], indent=1))


if __name__ == "__main__":
    main()
