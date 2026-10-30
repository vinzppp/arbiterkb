"""Arm K2's memory in three parts (NOTES "K2: initial state, change log, current state"; user design 2026-10-30).

A. Initial: the goal, the camera calibration and every object as first seen (the first frame's observations in the
   trace). The same at every call of an episode.
B. Change log: at each call, one new message with what changed since the previous call (moves with their displacement
   and end point, objects first seen or seen again or no longer seen, covers, close passes, the robot's pick-ups and
   put-downs, new beliefs and corrections that supersede earlier ones). Appended, never edited.
C. Current state, rewritten every call and authoritative: the hand and what it holds, every object (visible with its
   position and when it was seen, held, or hidden with its Inside probabilities), relative positions, the task's items
   with what the robot did to the container believed to hide each, and a summary of what has happened.
A and the B messages are a prefix that grows only at its end, so the provider can cache it (NOTES "K's request layout
and the provider's prompt cache"); C, the current frames and the changing state come last. Nothing here is new
information: every line is computed from the KB and its trace, with the KB's own frozen parameters (no new threshold).
"""
from __future__ import annotations

import json
import re

import numpy as np

from .action_view import KINDS, progress
from .kb import KB
from .narrative import _timeline, inside, joined_facts, seen
from .rules import held
from .task_items import is_task_item
from .views import _event_text, geometric, rel_id

FIXED_STATE = ("wrist_camera_K", "front_camera_K", "front_camera_extrinsic_world_to_camera")
RELEASE = {"placed": "put it down at table level", "set_down": "set it down from a small height",
           "dropped": "dropped it from above"}


def _f(v) -> str:
    return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


class K2View:
    """Per episode: the fixed initial text, the log messages so far, and where the last call's log ended."""

    def __init__(self):
        self.initial, self.logs, self.n_ev, self.n_events, self.start, self.beliefs = None, [], 0, 0, {}, {}
        self.t_last = 0

    def update(self, system) -> tuple[str, list[str], str]:
        kb, rows = system.kb, system.rows
        if self.initial is None:
            self.initial = initial(kb, rows)
        if lines := self._changes(kb, rows):
            t0 = rows[0]["t"] if not self.logs else self.t_last + 1
            self.logs.append(f"Changes, t={t0}-{rows[-1]['t']} (appended; never edited):\n"
                             + "\n".join("  " + x for x in lines))
        self.t_last = rows[-1]["t"] if rows else 0
        return self.initial, list(self.logs), current(kb, rows)

    def _changes(self, kb: KB, rows: list) -> list[str]:
        covered = {s for v in kb.events if v.type == "cover" for s in v.support}
        out = []
        for e in kb.evidence[self.n_ev:]:
            if (line := self._line(e, rows, covered)) is not None:
                out.append((e.t, f"[{e.id}] t={e.t} {line}"))
        out += [(v.t, f"[{v.id}] t={v.t} " + _event_text(v, kb)) for v in kb.events[self.n_events:]]
        self.n_ev, self.n_events = len(kb.evidence), len(kb.events)
        t = rows[-1]["t"] if rows else 0
        for c, hyps in inside(kb).items():
            now, was = _hyps(hyps), self.beliefs.get(c)
            if was is None:
                out.append((t, f"[Inside({c})] t={t} new belief: {c} inside {now}"))
            elif hyps[0][0] != was[1] or was[2] - hyps[0][1] >= 0.1:  # self_memory.changes' rule
                out.append((t, f"[Inside({c})] t={t} correction: {c} now inside {now}; supersedes {was[0]} "
                               f"stated at t={was[3]}"))
            if was is None or hyps[0][0] != was[1] or abs(was[2] - hyps[0][1]) >= 0.1:
                self.beliefs[c] = (now, hyps[0][0], hyps[0][1], t)
        return [x for _, x in sorted(out, key=lambda r: r[0])]

    def _line(self, e, rows: list, covered: set) -> str | None:
        d, a = e.data, ", ".join(e.args)
        if e.kind == "move_start":
            self.start[e.args[0]] = (d.get("t_at", e.t - 1), d.get("xyz"))
        elif e.kind == "move_stop" and e.args[0] in self.start and self.start[e.args[0]][1] is not None:
            t0, xyz0 = self.start.pop(e.args[0])
            return f"{a} moved by {_f(np.subtract(d['xyz'], xyz0))} to {_f(d['xyz'])} (t={t0}-{e.t})"
        elif e.kind == "appear" and e.t != rows[0]["t"]:  # the first frame's objects are the initial state
            xyz = _seen_at(rows, e.t, e.args[0])
            return f"{a} seen at {_f(xyz)}" if xyz is not None else f"{a} seen"
        elif e.kind == "disappear" and e.id not in covered:
            return f"{a} no longer seen; last at {_f(d['last_xyz'])}"
        elif e.kind == "near_pass":
            return f"{a} passed close to each other, closest {100 * d['min_3d']:.1f} cm (detected)"
        elif e.kind == "pick_up":
            return f"the robot picked up an object at {_f(d['ee'])}"
        elif e.kind == "release":
            return f"the robot {RELEASE[d['how']]} at {_f(d['ee'])}"
        elif e.kind in KINDS:
            return f"the robot {e.kind.replace('_', ' ')} at {_f(d['ee'])}"
        return None


def _hyps(hyps) -> str:
    return " | ".join(f"{k or 'elsewhere'} {p:.2f}" for k, p in hyps)


def _seen_at(rows: list, t: int, oid: str):
    row = next((r for r in rows if r["t"] == t), None)
    o = row["obs"].get(oid) if row else None
    return o[1] if o and o[0] else None


def initial(kb: KB, rows: list) -> str:
    """Every object as observed in the first frame; the rest 'not seen yet'."""
    first = rows[0]["obs"] if rows else {}
    lines = [f"Initial state (t={rows[0]['t'] if rows else 0}, the first frame; later lines may supersede it):"]
    for o in kb.objects.values():
        ob = first.get(o.id)
        where = f"seen at {_f(ob[1])}" if ob and ob[0] else "not seen yet"
        lines.append(f"  [{o.id}] {o.description}, {where}")
    return "\n".join(lines)


def current(kb: KB, rows: list) -> str:
    t = rows[-1]["t"] if rows else 0
    ee = rows[-1]["obs"]["ee"] if rows else [True, [0, 0, 0], 0.08]
    h = held(kb)
    items = kb.task.get("items", [])
    lines = [f"Current state at t={t} (authoritative: where it differs from anything above, this holds):",
             f"Robot: hand at {_f(ee[1])}; holding {', '.join(f'[{x}]' for x in h) if h else 'nothing'}.",
             "Objects (world frame, m):"]
    ins = inside(kb)
    for o in kb.objects.values():
        if not seen(o):
            continue
        tag = " (task item)" if is_task_item(o.attributes, o.cls, items) else ""
        if o.id in h:
            state = f"held, at {_f(o.pose.value)}" if o.pose.value is not None else "held"
        elif o.visibility.value == "hidden" and o.id in ins:
            state = f"hidden since t={o.t_last_seen + 1}, inside {_hyps(ins[o.id])}"
        elif o.visibility.value == "visible":
            state = f"visible at {_f(o.pose.value)}, seen t={o.t_last_seen}"
        else:
            state = f"not seen since t={o.t_last_seen}" + (f", last at {_f(o.pose.value)}" if o.pose.value is not None
                                                            else "")
        lines.append(f"  [{o.id}] {o.description}{tag}: {state}")
    lines.append("Relative positions:")
    lines += ["  " + x for x in joined_facts(kb)[:2]]
    lines += [f"  [{rel_id(f)}]" for f, _, _ in geometric(kb) if f[0] != "LeftOf"]
    lines += ["Task (the goal's items in order):"] + _task(kb, items, ins)
    lines += ["Summary of what has happened (oldest first):"] + ["  " + x for x in _summary(kb)]
    return "\n".join(lines)


def _task(kb: KB, items: list, ins: dict) -> list[str]:
    """Per coloured item: where the memory puts it, and what the robot did to the container believed to hide it."""
    done = {}
    for v in kb.events:
        if v.type in ("grasp", "place"):
            done.setdefault(v.args[0], []).append(f"{'picked up' if v.type == 'grasp' else 'put down'} at t={v.t}")
    out = []
    for it in items:
        objs = [o for o in kb.objects.values() if o.attributes.get("color") == it["color"] and it["color"]]
        if not objs:
            continue
        o = objs[0]
        where = f"inside {_hyps(ins[o.id])}" if o.id in ins and o.visibility.value == "hidden" else o.visibility.value
        box = ins[o.id][0][0] if o.id in ins else None
        acts = f"; {box}: " + ", ".join(done[box]) if box in done else f"; {box}: not picked up yet" if box else ""
        out.append(f"  {len(out) + 1}. {it['phrase']} [{o.id}]: {where}{acts}")
    acts = progress(kb)
    return out + [f"  robot actions so far: {', '.join(acts) if acts else 'none'}"]


def _summary(kb: KB) -> list[str]:
    rows = [(int(re.match(r"t=(\d+)", x).group(1)), x) for x in _timeline(kb)]
    rows += [(v.t, f"t={v.t}: {'picked up' if v.type == 'grasp' else 'put down'} {v.args[0]}")
             for v in kb.events if v.type in ("grasp", "place")]
    return [x for _, x in sorted(rows, key=lambda r: r[0])]


def request(view: K2View, system, goal: str, current: list, state: dict, breakpoint: bool, note: list) -> tuple:
    """(user messages, the memory text joined, for check()). `current`: [(rgb, label)] as controller.user_content."""
    from .controller import image_part, text_part
    parts = [p for rgb, label in current for p in (text_part(label + ":"), image_part(rgb))]
    a, logs, c = view.update(system)
    fixed = [text_part(f"Task goal: {goal}"),
             text_part("Robot state, fixed part (camera calibration): "
                       + json.dumps({k: state[k] for k in FIXED_STATE if k in state})), text_part(a)]
    msgs = [fixed] + [[text_part(x)] for x in logs]
    if breakpoint:
        for m in msgs:
            m[-1] = {**m[-1], "prompt_cache_breakpoint": {"mode": "explicit"}}
    rest = {k: v for k, v in state.items() if k not in FIXED_STATE}
    msgs.append([text_part(c)] + parts + [text_part("Robot state: " + json.dumps(rest))] + note)
    return [{"role": "user", "content": m} for m in msgs], "\n\n".join([a, *logs, c])
