"""Stage rules of the model screen (scripts/model_screen.py; NOTES "Model screen: rules fixed before any call").

Every number comes from RoboMME's code (github.com/RoboMME/robomme_benchmark @ 69a2fe2, src/robomme/robomme_env;
ManiSkill's pick_cube_cfgs.py for the cube size) or is marked "ours". The container ("bin") is an open box turned
upside down with a 4 cm handle cube at its origin; cube_half_size = 0.02 (PICK_CUBE_CONFIGS["panda"]).
"""
from __future__ import annotations

import json

import numpy as np

from arbiter_kb.controller import check, parse_answer, to_action  # check: workspace box and reach, as the loop does

BIN_Z = 0.052  # bin origin at rest: build_bin sets p.z = floor half 0.002 + wall height 0.05 (2.5 x cube half)
GRASP_Z = 0.062  # solve_pickup_bin: grasp pose at the bin origin, then 0.01 back along the approach (up)
HANDLE_TOP, PLATE_TOP = 0.072, 0.054  # handle box: origin +/- 0.02; floor plate (now on top): origin + 0.002
PICKED = 0.15  # is_bin_pickup: bin origin z > 0.15
SCRIPT_GRASP_Z = 0.045  # ours, not RoboMME's 0.062: the KB records a pick-up only below z_low = 0.05 (actions.py;
# NOTES "Model screen"), and the screen's memory must be right; inside GRASP_BAND
LIFT_MIN = PICKED + (SCRIPT_GRASP_Z - BIN_Z)  # 0.143: TCP height that lifts the origin above 0.15 after that grasp
ABOVE_MIN = 0.10  # handle top 0.072 + FINGER_LENGTH 0.025 (solve_pickup_bin) = 0.097, rounded up
Z_MAX = 0.40  # ours: below the script's home height 0.45 (RoboMME's own reach / lift height is 0.2)
GRASP_BAND = (0.02, HANDLE_TOP + 0.005)  # ours on both ends: fingertips off the table; at most 5 mm above the handle
RELEASE_BAND = (0.06, 0.12)  # bin bottom at the table when TCP = 0.062 (minus 2 mm, ours); a drop <= 5.8 cm (ours;
# solve_putdown_whenhold releases at 0.07; is_bin_putdown: bin origin <= 0.07, not grasped, TCP z > 0.05)
XY_ABOVE = 0.03  # bin walls' outer half width: inner half 0.025 + wall 0.005 (build_bin)
XY_HANDLE, XY_BODY = 0.02, 0.01  # grasp above the plate: handle half 0.02; below it the fingers (0.08 open, prompt:
# 0.04 m each) must straddle the 0.06 m walls: (0.08 - 0.06) / 2
FREE = 0.075  # spawn_random_bin: centres >= 2 x bin half 0.0275 + min_gap 0.02 (VideoUnmaskSwap: cube_half x 1)
CARRY_MOVE = 0.03  # ours: a "carry" answer at the put-down stage must move the hand at least this far in xy
STAGES = {1: "above_A", 2: "grasp_A", 3: "lift_A", 4: "put_down_A", 5: "above_B", 6: "grasp_B", 7: "done"}
OPEN_G, HOLD_G = (0, 1), (-1, 0)


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a[:2], float) - np.asarray(b[:2], float)))


def _grasp_tol(z: float) -> float:
    return XY_HANDLE if z >= PLATE_TOP else XY_BODY


def _above(a, g, x) -> str | None:
    if g not in OPEN_G:
        return f"gripper {g} while moving above (must be 0 or 1)"
    if _dist(a, x) > XY_ABOVE:
        return f"xy {_dist(a, x):.3f} m from the container > {XY_ABOVE}"
    return None if ABOVE_MIN <= a[2] <= Z_MAX else f"z {a[2]:.3f} outside [{ABOVE_MIN}, {Z_MAX}]"


def _grasp(a, g, x, gripper_ok) -> str | None:
    if g not in gripper_ok:
        return f"gripper {g} not in {gripper_ok}"
    if not GRASP_BAND[0] <= a[2] <= GRASP_BAND[1]:
        return f"z {a[2]:.3f} outside the grasp band {GRASP_BAND}"
    tol = _grasp_tol(a[2])
    return None if _dist(a, x) <= tol else f"xy {_dist(a, x):.3f} m from the container > {tol}"


def _free(a, others) -> str | None:
    near = [k for k, p in others.items() if _dist(a, p) < FREE]
    return f"xy within {FREE} m of {near}" if near else None


def judge(stage: int, a: np.ndarray, truth: dict) -> tuple[str | None, str]:
    """(reason it fails or None, variant) for one executable action at one stage. `truth`: the true xy of A and B
    (`A`, `B`), the hand's xy (`hand`) and the other containers (`others`: every bin except the one acted on)."""
    g, x = int(a[6]), truth["A" if stage <= 4 else "B"]
    if stage in (1, 5):
        if g == -1:
            return _grasp(a, g, x, (-1,)), "direct_grasp"
        return _above(a, g, x), "above"
    if stage in (2, 6):
        return _grasp(a, g, x, (-1,) if g == -1 else OPEN_G), "close" if g == -1 else "descend_first"
    if stage == 3:
        if g not in HOLD_G:
            return f"gripper {g} while lifting (must be -1 or 0)", "lift"
        return (None if LIFT_MIN <= a[2] <= Z_MAX else f"z {a[2]:.3f} outside [{LIFT_MIN:.2f}, {Z_MAX}]"), "lift"
    if g == 1:
        lo, hi = RELEASE_BAND
        return (f"z {a[2]:.3f} outside the release band {RELEASE_BAND}" if not lo <= a[2] <= hi
                else _free(a, truth["others"])), "release"
    if not LIFT_MIN <= a[2] <= Z_MAX:
        return f"carry at z {a[2]:.3f} outside [{LIFT_MIN:.2f}, {Z_MAX}]", "carry"
    if _dist(a, truth["hand"]) < CARRY_MOVE:
        return f"carry moves the hand {_dist(a, truth['hand']):.3f} m < {CARRY_MOVE}", "carry"
    return _free(a, truth["others"]), "carry"


def score(stage: int, truth: dict, text: str | None) -> dict:
    """One answer at one stage: valid JSON, pass, variant, reason. Never raises."""
    try:
        ans = parse_answer(text or "")
        action, kind = to_action(ans)
    except Exception as e:  # noqa: BLE001 (a malformed answer must not stop the screen)
        ans, action, kind = None, None, f"parse_error:{e!r}"[:80]
    valid = action is not None or kind == "declared_complete"
    out = {"valid_json": int(valid), "passed": 0, "variant": kind or "", "reason": "", "d_xy": None,
           "z": None if action is None else round(float(action[2]), 4),
           "gripper": None if action is None else int(action[6]),
           "target": ans.get("target") if isinstance(ans, dict) and isinstance(ans.get("target"), str) else None}
    if not valid:
        return {**out, "reason": f"not a valid answer ({kind})"}
    if stage == 7:
        return {**out, "passed": int(action is None), "reason": "" if action is None else "acts after the goal is done"}
    if action is None:
        return {**out, "reason": "premature done (waypoint null before both containers were picked up)"}
    bad = check(action, {}, set())  # workspace and reach only: target and cites are not scored
    why, variant = judge(stage, action, truth)
    why = "; ".join(bad) if bad else why
    out["d_xy"] = round(_dist(action, truth["A" if stage <= 4 else "B"]), 4)
    return {**out, "passed": int(why is None), "variant": variant, "reason": why or ""}


def canonical(item: dict) -> str:
    """The answer a model that follows RoboMME's own script would give (used by the mock model only)."""
    t, s = item["truth"], item["stage"]
    if s == 7:
        return json.dumps({"description": "both containers picked up", "waypoint": None, "gripper": 0})
    if s == 4:
        hand = np.asarray(t["hand"])
        spots = [hand + d for d in np.array([(0, 0.15), (0, -0.15), (0.15, 0), (-0.15, 0), (0.15, 0.15), (-0.15, -0.15)])]
        xy = next((p for p in spots if all(_dist(p, o) >= FREE + 0.01 for o in t["others"].values())), spots[0])
        wp, g = [*xy, 0.07], 1
    else:
        x = t["hand"] if s == 3 else t["A" if s <= 4 else "B"]
        wp, g = [*x, {1: 0.2, 2: SCRIPT_GRASP_Z, 3: 0.2, 5: 0.2, 6: SCRIPT_GRASP_Z}[s]], {1: 1, 2: -1, 3: 0, 5: 0, 6: -1}[s]
    return json.dumps({"description": "mock", "waypoint": [float(v) for v in wp] + [-3.1416, 0.0, 0.0], "gripper": g,
                       "target": item["A"] if s <= 4 else item["B"], "cites": []})


def mock_transport(items: list[dict]):
    """A fake OpenRouter for the scoring path: the canonical answer to every request, except prose at the first
    episode's stage 2 and a premature "done" at the second episode's stage 3 (both must score 0)."""
    eps = sorted({x["episode"] for x in items})
    bad = {(eps[0], 2): "I will now close the gripper.", (eps[min(1, len(eps) - 1)], 3): '{"waypoint": null}'}
    table = {json.dumps(x["messages"], sort_keys=True): bad.get((x["episode"], x["stage"])) or canonical(x)
             for x in items}

    def transport(body: dict, key: str) -> dict:
        return {"choices": [{"message": {"content": table[json.dumps(body["messages"], sort_keys=True)]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001}, "provider": "mock"}
    return transport
