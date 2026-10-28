"""How the controller view words the robot's semantic actions (actions.py), v3 of the counting test (NOTES).

- Ordinals per kind: "picked up an object (pick-up #2)", so the controller does not have to count.
- A hold on one side directly followed by a release at the same spot (within merge_xy) is ONE line,
  "held it up on the robot's right side, then placed it down there": a carry over a target or bin is not
  shown as a separate step, and the hold still counts (SwingXtimes's last hold is a task step).
- Neutral wording for releases (no guess about what was under the hand: there is no vision here).
- Coordinates only when the KB has objects to relate them to.
"""
from __future__ import annotations

import numpy as np

from .kb import KB
from .params import DetectParams

KINDS = ("pick_up", "release", "press", "hold_above")
RELEASE = {"placed": "placed it down at table level", "set_down": "set it down from a small height",
           "dropped": "released it from above (as into a container)"}
PLURAL = {"pick_up": "pick-ups", "release": "releases", "press": "button presses"}


def _text(e, n: int) -> str:
    d = e.data
    if e.kind == "pick_up":
        return f"picked up an object (pick-up #{n})"
    if e.kind == "release":
        return f"{RELEASE[d['how']]} (release #{n})"
    if e.kind == "press":
        return f"pressed down with the closed gripper, a button press (press #{n})"
    return f"held the object up on the robot's {d['side']} side ({d['side']} hold #{n})"


def action_rows(kb: KB, coords: bool, p: DetectParams = DetectParams()) -> list[tuple[int, int, str]]:
    """[(t, evidence number, line)] for the semantic actions, holds merged into the release that follows."""
    acts = [e for e in kb.evidence if e.kind in KINDS]
    count: dict[str, int] = {}
    rows, pending = [], None
    for i, e in enumerate(acts):
        key = e.kind if e.kind != "hold_above" else f"hold_{e.data['side']}"
        count[key] = count.get(key, 0) + 1
        text = _text(e, count[key]) + (f" at {_fmt(e.data['ee'])}" if coords else "")
        nxt = acts[i + 1] if i + 1 < len(acts) else None
        if e.kind == "hold_above" and nxt is not None and nxt.kind == "release" and \
                np.linalg.norm(np.subtract(nxt.data["ee"][:2], e.data["ee"][:2])) < p.merge_xy:
            pending = (e, text.removeprefix("held the object up"))
            continue
        if pending is not None:
            text = f"held it up{pending[1]}, then {text}"
            e, pending = pending[0], None
        rows.append((e.t, int(e.id[1:]), f"[{e.id}] t={e.t} {text}"))
    return rows


def progress(kb: KB) -> list[str]:
    """Counts of each completed action kind, holds by side; nothing for kinds that never happened."""
    acts = [e for e in kb.evidence if e.kind in KINDS]
    out = [f"{PLURAL[k]} {n}" for k in PLURAL if (n := sum(e.kind == k for e in acts))]
    sides = [e.data["side"] for e in acts if e.kind == "hold_above"]
    return out + ([f"holds left {sides.count('left')}, right {sides.count('right')}"] if sides else [])


def _fmt(v) -> str:
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"
