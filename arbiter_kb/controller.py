"""LLM controller (DEVELOPMENT §3.8): one world-frame waypoint per call, as in arXiv 2610.17350.
v1.7: every arm's prompt states the containment rule (ENDING), so no arm gets it for free from its memory.

The system prompt, answer parsing and action checks follow that paper's harness
(github.com/jackbjed/embodied-astra-robomme, harness/astra_waypoint_eval_v3.py, MIT licence): the robot and
interface text is theirs, verbatim where possible. Ours: the memory section per arm, and the "target" /
"cites" fields (E6; they explain an action and never constrain it). Arms:
- H: the image history (their memory); the caller chains all turns, as they do.
- K: the KB view (views.controller_view) + the current front / wrist frames; a fresh request each step.
- K-L3: K without the event log.  R: the raw tracker log as text + the current frames.
- K0 (v1.6, M3a goal review): the current frame's object table only (3D centres now; no relations, events
  or history). It separates "memory" from "being given coordinates".
- S (v1.6): the current frames + one sentence per hidden cube naming its most likely container (Bound-like).
  If S ties K, the KB's structure adds nothing on that task.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np

ARMS = ("H", "K", "K-L3", "R", "K0", "S")
BASE = np.array([-0.615, 0.0, 0.0])
SHOULDER = BASE + [0, 0, 0.333]  # Panda joint 2 height above the base
REACH = 0.855  # m: Panda reach from the shoulder (datasheet)
WORKSPACE = ((-0.6, 0.6), (-0.6, 0.6), (0.0, 0.7))  # world box, m

ROBOT = """You control a simulated Franka Panda robot arm on a table, by choosing end-effector waypoints.
World frame, metres: the robot's base is at (-0.615, 0, 0) and faces +x; +y is the robot's left, +z is up.

Each turn, answer with ONE JSON object:
{"description": "what happened in the latest frames and what you will do next",
 "waypoint": [x, y, z, roll, pitch, yaw],
 "gripper": -1, 0 or 1,
 "target": "id of the object this action is about, or null",
 "cites": ["ids of the memory lines this action relies on"]}
The robot moves its TCP (the point between the fingertips) to (x, y, z) with orientation (roll, pitch, yaw) in
radians, using the same convention as the TCP pose reported to you, along a planned path; then it applies the
gripper command: -1 = close, 1 = open, 0 = keep. Pointing straight down is roll = -3.1416, pitch = 0; yaw turns the
hand about the vertical axis. If a pose cannot be reached the episode fails. A waypoint identical to the current
TCP pose also fails: to only open or close the gripper, move the TCP by at least 2 mm (for example 5 mm up or down).
"""

MEMORY = {
    "H": """Observations: frames from the fixed front camera (256 x 256) in time order, of the video so far and of each
motion, then the current front and wrist images; the TCP pose; the finger positions (0.04 m each = fully open,
0 = closed); the elapsed control steps. You get no depth and no object positions. The task may require
remembering what happened earlier in the video.""",
    "K": """Memory: instead of the image history you get the robot's memory, rebuilt every turn from everything seen so
far: the goal and progress, each object with its 3D centre (world frame, m; from depth), relations with
probabilities (Inside(cube): which container hides it), and an event log with times and where-from / where-to.
Each line starts with an id in brackets; cite the ids you rely on. A probability below 1 means the memory is
unsure: decide whether to act on the most likely answer or to look first. Then the current front and wrist
images, the TCP pose, the finger positions (0.04 m each = fully open, 0 = closed) and the elapsed control steps.""",
    "R": """Memory: instead of the image history you get the perception system's raw log: for sampled frames, each
tracked object's 3D centre (world frame, m; from depth) or "hidden". Nothing else is inferred for you. Then the
current front and wrist images, the TCP pose, the finger positions and the elapsed control steps.""",
}
MEMORY["K-L3"] = MEMORY["K"].replace(", and an event log with times and where-from / where-to", "")
MEMORY["K0"] = """Memory: none. You get a table of the objects visible in the current frame with their 3D centres (world
frame, m; from depth), then the current front and wrist images, the TCP pose, the finger positions (0.04 m each
= fully open, 0 = closed) and the elapsed control steps. Nothing about earlier frames is given."""
MEMORY["S"] = """Memory: one sentence per hidden object saying where the robot's memory thinks it is. Then the current front
and wrist images, the TCP pose, the finger positions (0.04 m each = fully open, 0 = closed) and the elapsed
control steps. You get no depth and no object positions."""

ENDING = """Objects can be hidden: an object covered by a container moves with that container until it is seen again.
Positions recorded earlier are where things were at that time, not necessarily where they are now.
Notes: time passes only while the robot executes a motion; everything else in the scene keeps moving during a
motion. The gripper opens or closes only when you command it (-1 or 1). The episode ends automatically when the
environment judges success or failure, or after 1300 control steps. If you believe the task is complete,
answer with "waypoint": null."""


def system_prompt(arm: str) -> str:
    assert arm in ARMS, arm
    return "\n".join([ROBOT, MEMORY[arm], ENDING])


def image_part(rgb: np.ndarray) -> dict:
    """PNG data URL (matplotlib, so no extra dependency)."""
    import matplotlib.image as mpimg

    buf = io.BytesIO()
    mpimg.imsave(buf, np.asarray(rgb, np.uint8), format="png")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def raw_tracker_log(frames: list, stride: int = 10) -> str:
    """Arm R: per sampled frame, each object's observed centre or 'hidden'; no events, no beliefs."""
    lines = []
    for f in frames[::stride] + ([frames[-1]] if (len(frames) - 1) % stride else []):
        objs = [f"{k} " + (", ".join(f"{v:.3f}" for v in o.xyz) if o.visible else "hidden") for k, o in f.objects.items()]
        lines.append(f"t={f.t}: " + "; ".join(objs))
    return "\n".join(lines)


def object_table(frame) -> str:
    """Arm K0: the objects visible now, with their 3D centres; nothing from earlier frames."""
    rows = [f"{k} ({o.cls}) at " + ", ".join(f"{v:.3f}" for v in o.xyz) for k, o in frame.objects.items() if o.visible]
    return "Visible objects now:\n" + "\n".join(rows)


def one_sentence(inside: dict) -> str:
    """Arm S: {cube: most likely container} -> 'cube_red is under bin_0.' per hidden cube."""
    return " ".join(f"{c} is under {b}." for c, b in inside.items() if b) or "No object is hidden."


def user_content(arm: str, goal: str, memory_text: str, current: list, state: dict) -> list:
    """One turn's content for K / K-L3 / R (H builds its frame list in the harness, as the paper does)."""
    head = [] if memory_text.startswith("Goal:") else [text_part(f"Task goal: {goal}")]  # the KB view has it
    head += [text_part(memory_text)] if memory_text else []
    imgs = [p for rgb, label in current for p in (text_part(label + ":"), image_part(rgb))]
    return head + imgs + [text_part("Robot state: " + json.dumps(state))]


def parse_answer(text: str) -> dict | None:
    """The JSON object asked for, with or without ```json fences; None if there is none (theirs)."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def to_action(answer) -> tuple[np.ndarray | None, str | None]:
    """(7-vector, None) or (None, kind): declared_complete / unparseable / invalid_action (theirs)."""
    if not isinstance(answer, dict):
        return None, "unparseable"
    if "waypoint" in answer and answer["waypoint"] is None:
        return None, "declared_complete"
    wp, g = answer.get("waypoint"), answer.get("gripper", 0)
    if not (isinstance(wp, list) and len(wp) == 6 and all(isinstance(v, (int, float)) for v in wp)) \
            or g not in (-1, 0, 1) or not np.all(np.isfinite(wp)):
        return None, "invalid_action"
    return np.array([*wp, g], dtype=np.float64), None


def check(action: np.ndarray, answer: dict, known: set[str]) -> list[str]:
    """Our checks before execution: workspace box, coarse reach (shoulder sphere; the authoritative IK /
    mplib check runs on the pod), and that target / cites name real KB ids. Problems are returned, not raised:
    the harness gives the model one retry with them, as an executor error would."""
    errs = [f"{'xyz'[i]}={action[i]:.3f} outside [{lo}, {hi}]" for i, (lo, hi) in enumerate(WORKSPACE)
            if not lo <= action[i] <= hi]
    if (d := float(np.linalg.norm(action[:3] - SHOULDER))) > REACH:
        errs.append(f"waypoint {d:.3f} m from the shoulder > reach {REACH} m")
    tgt, cites = answer.get("target"), answer.get("cites") or []
    if tgt is not None and tgt not in known:
        errs.append(f"unknown target {tgt!r}")
    errs += [f"unknown cite {c!r}" for c in cites if not _citable(c, known)]
    return errs


def _citable(c, known: set[str]) -> bool:
    """An evidence / event / object id the KB holds, or an observation id o<t>:<object | ee | image>."""
    if not isinstance(c, str):
        return False
    if c in known:
        return True
    head, _, obj = c.partition(":")
    return head[:1] == "o" and head[1:].isdigit() and (obj in known or obj in ("ee", "image"))


def target_distance(action: np.ndarray, answer: dict, pose_of) -> float | None:
    """Logged, never enforced: xy distance from the waypoint to the cited target's KB centre."""
    p = pose_of(answer.get("target")) if answer.get("target") else None
    return None if p is None else round(float(np.linalg.norm(action[:2] - np.asarray(p)[:2])), 4)
