"""Layer 1 on real scenes: does the model's first waypoint point at the right object, and does its first close happen
at the object's height? (NOTES "Layer 1 on real scenes", rules fixed there before running.) Scoring only, no I/O: the
pod runner (stress/score_callout.py) reads the closed loop's result.json files and the simulator's poses.

`truth`: {actor name: [x, y, z, yaw]} from env.unwrapped (stress/pod_checks.truth), read after one reset.
"""
from __future__ import annotations

import re
from statistics import median

import numpy as np

from .controller import BASE
from .task_items import parse_task_items

RIGHT_XY, UNCERTAIN_GAP = 0.03, 0.02  # m: right = nearest and closer than this; uncertain = d2 - d1 below this
TABLE = ((-0.6, 0.6), (-0.6, 0.6), (-0.01, 0.30))
BASE_XY, SAME = 0.10, 0.001  # m: robot-base exclusion (as online.not_objects); actors this close are one object
NOT_OBJECT = re.compile(r"table|ground|floor|camera|wall|goal_site|robot", re.I)


def objects(truth: dict) -> dict:
    """The graspable scene: actors inside the table box, away from the robot base, not scene furniture; actors at the
    same position (one actor under two attribute names) kept once, under the first name."""
    out = {}
    for name, p in truth.items():
        x, y, z = (float(v) for v in p[:3])
        if NOT_OBJECT.search(name) or not all(lo <= v <= hi for v, (lo, hi) in zip((x, y, z), TABLE)):
            continue
        if np.hypot(x - BASE[0], y - BASE[1]) < BASE_XY:
            continue
        if any(np.linalg.norm(np.subtract(q[:3], (x, y, z))) <= SAME for q in out.values()):
            continue
        out[name] = [x, y, z]
    return out


COVER_XY, COVER_DZ = 0.03, 0.01  # m: an object this close in xy under a higher one is covered (VideoUnmaskSwap's cube)


def uncovered(objs: dict) -> dict:
    """The objects a waypoint can point at: a cube under its container shares the container's xy, so it would make
    every pick of that container "uncertain"; it still decides the right object (`right_object` gets all)."""
    return {n: p for n, p in objs.items()
            if not any(m != n and np.hypot(q[0] - p[0], q[1] - p[1]) < COVER_XY and q[2] - p[2] >= COVER_DZ
                       for m, q in objs.items())}


def right_object(goal: str, objs: dict) -> tuple[str | None, str]:
    """(name, how) from the goal and the simulator: the goal's first coloured cube; with a container named, the
    container nearest that cube in xy (the one hiding it), else the cube. None when the cube's actor is not unique."""
    items = parse_task_items(goal)
    cube = next((it for it in items if it["noun"] == "cube" and it["color"]), None)
    if cube is None:
        return None, "no coloured cube in the goal"
    hits = [n for n in objs if cube["color"] in n.lower()]
    if len(hits) != 1:
        return None, f"{len(hits)} actors named {cube['color']!r}"
    if not any(it["noun"] == "container" for it in items):
        return hits[0], "the goal's cube"
    bins = [n for n in objs if re.search(r"bin|container", n, re.I)]
    if not bins:
        return None, "no container actor"
    c = objs[hits[0]]
    return min(bins, key=lambda n: np.hypot(objs[n][0] - c[0], objs[n][1] - c[1])), f"container nearest {hits[0]}"


def actions(steps: list) -> list:
    """The steps that carry an action (a retry's rejected answer has none)."""
    return [np.asarray(s["action"], float) for s in steps if s.get("action") is not None]


def nearest(xy, objs: dict) -> list[tuple[float, str]]:
    return sorted((float(np.hypot(p[0] - xy[0], p[1] - xy[1])), n) for n, p in objs.items())


def score(steps: list, objs: dict, right: str | None) -> dict:
    """One episode's row (NOTES rules): the first waypoint's category and distances, the first close's dz."""
    acts, row = actions(steps), {"right_object": right}
    if not acts:
        return {**row, "category": "no_action"}
    d_right = float(np.hypot(*(np.subtract(objs[right][:2], acts[0][:2])))) if right else None
    objs = uncovered(objs)
    near = nearest(acts[0][:2], objs)
    d1, n1 = near[0] if near else (None, None)
    d2 = near[1][0] if len(near) > 1 else None
    row.update(first_xyz=np.round(acts[0][:3], 4).tolist(), nearest=n1, d1=d1, d2=d2, d_right=d_right)
    if d1 is None:
        cat = "no_objects"
    elif d2 is not None and d2 - d1 < UNCERTAIN_GAP:
        cat = "uncertain"
    elif right is None:
        cat = "unknown"
    else:
        cat = "right" if n1 == right and d1 < RIGHT_XY else "wrong"
    row["category"] = cat
    close = next((a for a in acts if a[6] < 0), None)
    if close is not None and objs:
        _, n = nearest(close[:2], objs)[0]
        row.update(close_xyz=np.round(close[:3], 4).tolist(), close_object=n, close_dz=float(close[2] - objs[n][2]))
    return row


CATEGORIES = ("right", "uncertain", "wrong", "no_action", "unknown", "no_objects")


def summarize(rows: list[dict]) -> list[dict]:
    """Per task x arm x model: counts per category, median / max d1, d_right and |dz| at the first close."""
    groups = {}
    for r in rows:
        groups.setdefault((r["task"], r["arm"], r["model"]), []).append(r)

    def mm(vals):
        vals = [v for v in vals if v is not None]
        return (round(median(vals), 4), round(max(vals), 4)) if vals else (None, None)
    out = []
    for (task, arm, model), rs in sorted(groups.items()):
        s = {"task": task, "arm": arm, "model": model, "n": len(rs)}
        s.update({c: sum(r["category"] == c for r in rs) for c in CATEGORIES})
        for key, vals in (("d1", [r.get("d1") for r in rs]), ("d_right", [r.get("d_right") for r in rs]),
                          ("abs_close_dz", [abs(r["close_dz"]) if r.get("close_dz") is not None else None
                                            for r in rs])):
            s[f"{key}_median"], s[f"{key}_max"] = mm(vals)
        s["closes"] = sum(r.get("close_dz") is not None for r in rs)
        out.append(s)
    return out
