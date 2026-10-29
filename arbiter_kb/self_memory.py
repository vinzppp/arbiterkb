"""The controller's own record in the memory (NOTES "B — the controller's actions and their outcomes"): what it
decided at each call, whether the robot did it, and what changed since its last decision. Kept in kb.task
["decisions"] (L4 bookkeeping, like the goal; beliefs still change only through KB.apply). Outcomes come from the
robot-state actions (actions.py), so they need no vision. It reports; it suggests no action.

Intent of a decision: the controller's own "intent" field when it gives one (like "target" and "cites", it explains
an action and never constrains it); otherwise read off the waypoint: "pick_up" (gripper closes, nothing held),
"release" (gripper opens while holding), "press" (gripper already closed, nothing held, hand low: RoboMME presses
with a closed gripper), else "move".
"""
from __future__ import annotations

from .kb import KB
from .narrative import inside, name

ACTION_KINDS = ("pick_up", "release", "press", "hold_above")
Z_LOW = 0.05  # m: a waypoint this low with a closed gripper and nothing held reads as a press


def holding(kb: KB) -> bool:
    acts = [e.kind for e in kb.evidence if e.kind in ("pick_up", "release")]
    return bool(acts) and acts[-1] == "pick_up"


def intent(kb: KB, waypoint, gripper_closed: bool) -> str:
    held = holding(kb)
    if gripper_closed and not held:
        return "press" if waypoint is not None and float(waypoint[2]) < Z_LOW and _was_closed(kb) else "pick_up"
    if not gripper_closed and held:
        return "release"
    return "move"


def _was_closed(kb: KB) -> bool:
    d = kb.task.get("decisions", [])
    return bool(d) and d[-1]["gripper_closed"]


def record(kb: KB, t: int, waypoint, gripper_closed: bool, target: str | None = None, cites=(),
           declared: str | None = None, detail: str | None = None) -> dict:
    """`detail`: what the action's evidence must also say, e.g. "left" for hold_above or "dropped" for a release
    (goal review: a hold on the wrong side read as done)."""
    snap = {c: (h[0][0], round(h[0][1], 2)) for c, h in inside(kb).items() if h}
    said = declared if declared in ACTION_KINDS + ("move",) else None
    d = {"t": t, "intent": said or intent(kb, waypoint, gripper_closed), "gripper_closed": bool(gripper_closed),
         "waypoint": None if waypoint is None else [round(float(v), 3) for v in waypoint[:3]], "target": target,
         "cites": list(cites), "beliefs": snap, "n_evidence": len(kb.evidence), "detail": detail}
    kb.task.setdefault("decisions", []).append(d)
    return d


def outcome(kb: KB, d: dict, until: int | None = None) -> str:
    """'done', 'not done yet', or what happened instead, from the actions since the decision."""
    acts = [e for e in kb.evidence[d["n_evidence"]:] if e.kind in ACTION_KINDS and (until is None or e.t <= until)]
    if d["intent"] == "move":
        return "moved" if not acts else "moved; also " + ", ".join(f"{e.kind} at t={e.t}" for e in acts)
    hit = next((e for e in acts if e.kind == d["intent"] and (not d.get("detail") or d["detail"] in
                                                               (e.data.get("side"), e.data.get("how")))), None)
    if hit:
        return f"done ({hit.kind} at t={hit.t} [{hit.id}])"
    if acts:
        return "not done; instead " + ", ".join(f"{e.kind} at t={e.t}" for e in acts)
    return "not done yet"


def changes(kb: KB, d: dict) -> list[str]:
    """Events and belief changes since decision d."""
    out = [f"t={v.t}: {v.type} " + " ".join(name(kb, a) for a in v.args if isinstance(a, str))
           for v in kb.events if v.t > d["t"]]
    now = {c: (h[0][0], round(h[0][1], 2)) for c, h in inside(kb).items() if h}
    for c, (k, p) in now.items():
        old = d["beliefs"].get(c)
        if old is None:
            out.append(f"{name(kb, c)}: now believed under {k} ({p:.2f})")
        elif old[0] != k:
            out.append(f"{name(kb, c)}: most likely container changed from {old[0]} to {k} ({p:.2f})")
        elif old[1] - p >= 0.1:
            out.append(f"{name(kb, c)}: less sure it is under {k} ({old[1]:.2f} -> {p:.2f})")
    return out


def self_view(kb: KB, last: int = 3) -> str:
    """The section that goes FIRST in the controller's prompt: changes since the last decision, then the last
    decisions with their outcomes. Empty before the first decision."""
    ds = kb.task.get("decisions", [])
    if not ds:
        return ""
    ch = changes(kb, ds[-1]) or ["nothing"]
    rows = [f"  t={d['t']}: {d['intent']}" + (f" {d['target']}" if d["target"] else "") + f" -> {outcome(kb, d, nxt)}"
            for d, nxt in zip(ds[-last:], [x["t"] for x in ds[-last:]][1:] + [None])]
    return "\n".join([f"Since your last decision (t={ds[-1]['t']}):", *("  " + x for x in ch),
                      "Your last decisions and what happened:", *rows])
