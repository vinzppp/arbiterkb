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
BASE_XY, PART = 0.10, 0.01  # m: robot-base exclusion (as online.not_objects); actors this close are one object's parts
NOT_OBJECT = re.compile(r"table|ground|floor|camera|wall|goal_site|robot", re.I)


def objects(truth: dict) -> dict:
    """The scene's objects: actors inside the table box, away from the robot base, not scene furniture. Every actor is
    kept (keyed by its own name), so the right object can be found even where two coincide."""
    out = {}
    for name, p in truth.items():
        x, y, z = (float(v) for v in p[:3])
        if NOT_OBJECT.search(name) or not all(lo <= v <= hi for v, (lo, hi) in zip((x, y, z), TABLE)):
            continue
        if np.hypot(x - BASE[0], y - BASE[1]) < BASE_XY:
            continue
        out[name] = [x, y, z]
    return out


COVER_XY, COVER_DZ = 0.03, 0.01  # m: an object this close in xy under a higher one is covered (VideoUnmaskSwap's cube)


def pointable(objs: dict) -> dict:
    """The objects a waypoint can point at, for the nearest / second-nearest distances. Actors within 1 cm are one
    object's parts, kept under the first name (VideoPlaceOrder's button and its cap_link, 7 mm apart). A covered
    object (a cube under its container) shares the container's xy and would make every pick of it "uncertain"; it is
    left out here but still decides the right object (`right_object` gets all)."""
    one = {}
    for n, p in objs.items():
        if not any(np.linalg.norm(np.subtract(q, p)) < PART for q in one.values()):
            one[n] = p
    return {n: p for n, p in one.items()
            if not any(m != n and np.hypot(q[0] - p[0], q[1] - p[1]) < COVER_XY and q[2] - p[2] >= COVER_DZ
                       for m, q in one.items())}


def right_object(goal: str, objs: dict) -> tuple[str | None, str]:
    """(name, how) from the goal and the simulator: the goal's first coloured cube; with a container named, the
    container nearest that cube in xy (the one hiding it), else the cube. None when the cube's actor is not unique."""
    items = parse_task_items(goal)
    cube = next((it for it in items if it["noun"] == "cube" and it["color"]), None)
    if cube is None:
        return None, "no coloured cube in the goal"
    hits = [n for n in objs if cube["color"] in n.lower() and "cube" in n.lower()]
    if len(hits) != 1:
        return None, f"{len(hits)} cube actors named {cube['color']!r}"
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
    objs = pointable(objs)
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


# --- after the first move: what each hold did (NOTES "Astra's 7 VideoUnmaskSwap failures") ---
LIFTED, MAPPED = 0.05, 0.03  # m: a hold counts as lifted above this rise; a close farther than this from every object
# at its reset position is flagged (the object may have moved since)
LABELS = ("planning failure", "wrong object at close", "premature done", "lift too low", "grasp missed (inferred)",
          "other")


def right_sequence(goal: str, objs: dict) -> list[str]:
    """The objects the goal asks for, in order: the container hiding each coloured cube it names (VideoUnmaskSwap's
    "..., finally pick up another container hiding the blue cube"), or the cube itself when no container is named."""
    container = any(it["noun"] == "container" for it in parse_task_items(goal))
    return [right_object(f"pick up the container hiding the {it['color']} cube" if container else
                         f"pick up the {it['color']} cube", objs)[0]
            for it in parse_task_items(goal) if it["noun"] == "cube" and it["color"]]


def holds(steps: list, objs: dict) -> list[dict]:
    """Every close (gripper -1) with the object nearest it at its reset position, held until the next open (+1):
    {step, object, d, flagged, close_z, lift (highest waypoint z while held), rise (lift - close_z)}."""
    near_objs, out, cur = pointable(objs), [], None
    for i, a in enumerate(actions(steps)):
        if a[6] < 0 and cur is None:
            d, n = nearest(a[:2], near_objs)[0]
            cur = {"step": i, "object": n, "d": round(d, 4), "flagged": d > MAPPED, "close_z": round(float(a[2]), 4),
                   "lift": float(a[2])}
            out.append(cur)
        elif cur is not None:
            cur["lift"] = max(cur["lift"], float(a[2]))
            if a[6] > 0:
                cur = None
    for h in out:
        h["lift"], h["rise"] = round(h["lift"], 4), round(h["lift"] - h["close_z"], 4)
    return out


def right_picks(hs: list[dict], sequence: list[str]) -> tuple[list[tuple[int, dict]], dict | None]:
    """The lifted holds matched in order to the goal's sequence ([(k, hold)]), and the first lifted hold of an object
    that is neither the next right object nor the one just picked (None if there is none)."""
    picks = []
    for h in (h for h in hs if h["rise"] >= LIFTED):
        k = len(picks)
        if k < len(sequence) and h["object"] == sequence[k]:
            picks.append((k, h))
        elif not (picks and h["object"] == picks[-1][1]["object"]):
            return picks, h
    return picks, None


def label(row: dict, hs: list[dict], sequence: list[str], min_lift: dict) -> tuple[str, str]:
    """(label, why) for one failed episode, the first rule that applies (NOTES). `min_lift`: {k: lowest lift of the
    k-th right pick among successful episodes}."""
    picks, wrong = right_picks(hs, sequence)
    end = row.get("end_reason")
    if end == "env_error":
        return LABELS[0], f"env_error after {len(picks)} of {len(sequence)} right pick(s)"
    if wrong is not None:
        nxt = sequence[len(picks)] if len(picks) < len(sequence) else "none"
        return LABELS[1], f"lifted {wrong['object']} at step {wrong['step']} (next right: {nxt})"
    if end == "model_answer:declared_complete":
        return LABELS[2], f"declared complete after {len(picks)} of {len(sequence)} right pick(s)"
    for k, h in picks:
        if k in min_lift and h["lift"] < min_lift[k]:
            return LABELS[3], f"pick {k + 1} lifted to z {h['lift']} < successes' lowest {min_lift[k]}"
    if picks:
        return LABELS[4], f"{len(picks)} of {len(sequence)} right pick(s) lifted as high as the successes'; ended {end}"
    return LABELS[5], f"{len(hs)} close(s), none a lifted right pick; ended {end}"


def went_back(steps: list, objs: dict, sequence: list[str]) -> tuple[str, str]:
    """On a two-pick goal: after the first right pick is put down (the first open after its close, at P; the gripper
    command is taken to act at the end of the motion), does a later lifted hold go back to P rather than to the second
    right object (S, at its reset position)? P and the later closes are the robot's own positions, so no object pose
    after reset is needed. ("yes" | "no" | "ambiguous" | "n/a", why). A "yes" is a behaviour, not a failure: a model
    may move the first bin aside on purpose (Astra R ep 1 did, then succeeded)."""
    if len(sequence) < 2:
        return "n/a", "one-pick goal"
    acts = actions(steps)
    picks, _ = right_picks(holds(steps, objs), sequence)
    if not picks:
        return "n/a", "no lifted first pick"
    k = next((j for j in range(picks[0][1]["step"] + 1, len(acts)) if acts[j][6] > 0), None)
    if k is None:
        return "n/a", "first pick never put down"
    put, s = acts[k][:2], objs[sequence[1]][:2]
    if np.hypot(*np.subtract(put, s)) < MAPPED:
        return "ambiguous", f"put down within {MAPPED} m of {sequence[1]}"
    rest = acts[k + 1:]
    for h in holds([{"action": a.tolist()} for a in rest], objs):
        a = rest[h["step"]]
        if h["rise"] >= LIFTED and np.hypot(*np.subtract(a[:2], put)) < np.hypot(*np.subtract(a[:2], s)):
            return "yes", f"lifted at {np.round(a[:2], 3).tolist()} near the put-down {np.round(put, 3).tolist()}"
    return "no", f"no later lifted hold nearer the put-down than {sequence[1]}"
