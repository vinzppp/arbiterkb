"""Arm K2's memory in three parts (NOTES "K2: initial state, change log, current state"; user design 2026-10-30;
revised after the review in NOTES "K2 fixes").

A. Initial: the goal, the camera calibration and every object as first seen (the first frame's observations in the
   trace). The same at every call of an episode.
B. Change log: at each call, one new message with what changed since the previous call: moves (displacement and end
   point), objects seen or no longer seen, covers, close passes, grasps and releases with the hand's numbers (where it
   closed, the highest it rose while holding, where it opened), reveals (a cube seen under a lifted container), the
   controller's own decisions, and each Inside belief with its own id, a change naming the line it replaces.
   Appended, never edited.
C. Now, rewritten every call and authoritative: time, hand, finger opening and what is held; the goal's items with
   what was done to each; every object (a table); relative positions; a summary of the semantic events only.
A and the B messages are a prefix that grows only at its end, so the provider can cache it; C, the current frames and
the changing state come last. Every line is computed from the KB and its trace (numbers, not labels such as "at table
level"), with the KB's own frozen parameters: no new threshold.
"""
from __future__ import annotations

import json

import numpy as np

from .action_view import KINDS
from .kb import KB
from .narrative import inside, seen
from .rules import held
from .task_items import is_task_item
from .views import _event_text, geometric, rel_id

FIXED_STATE = ("wrist_camera_K", "front_camera_K", "front_camera_extrinsic_world_to_camera")
BASE = np.array([-0.615, 0.0])  # the robot's base (world frame, m)


def _f(v) -> str:
    return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def _hyps(hyps) -> str:
    return " | ".join(f"{k or 'elsewhere'} {p:.2f}" for k, p in hyps)


def _ee(rows: list, t: int):
    row = next((r for r in rows if r["t"] == t), None)
    return row["obs"]["ee"][1] if row else None


def holds(kb: KB, rows: list) -> dict[str, list[dict]]:
    """{object: [{grasp t, hand at grasp, highest hand z while held, release t, hand at release, on}]}."""
    out: dict[str, list[dict]] = {}
    for v in kb.events:
        if v.type == "grasp":
            out.setdefault(v.args[0], []).append({"t": v.t, "at": _ee(rows, v.t), "top": None, "t_off": None})
        elif v.type == "place" and out.get(v.args[0]):
            out[v.args[0]][-1].update(t_off=v.t, off=_ee(rows, v.t), on=v.args[1])
    for h in (h for hs in out.values() for h in hs):
        zs = [r["obs"]["ee"][1][2] for r in rows if h["t"] <= r["t"] <= (h["t_off"] or rows[-1]["t"])]
        h["top"] = max(zs) if zs else None
    return out


def _hold_text(h: dict) -> str:
    s = f"grasped t={h['t']} (hand at {_f(h['at'])})" if h["at"] is not None else f"grasped t={h['t']}"
    s += f", highest hand z while held {h['top']:.3f}" if h["top"] is not None else ""
    if h["t_off"] is None:
        return s + ", still held"
    on = "" if h["on"] == "table" else f" onto [{h['on']}]"
    return s + f", released t={h['t_off']}{on} (hand at {_f(h['off'])})" if h["off"] is not None else s


class K2View:
    """Per episode: the fixed initial text, the log messages so far, and how far the last call's log reached."""

    def __init__(self):
        self.initial, self.logs, self.start, self.beliefs, self.n_b = None, [], {}, {}, 0
        self.n_ev = self.n_events = self.n_dec = 0

    def update(self, system) -> tuple[str, list[str], str]:
        kb, rows = system.kb, system.rows
        if self.initial is None:
            self.initial = initial(kb, rows)
        if lines := self._changes(kb, rows):
            head = "new since the previous turn" if self.logs else "from the start"
            self.logs.append(f"Log, {head}, up to t={rows[-1]['t']} (appended; never edited):\n"
                             + "\n".join("  " + x for x in lines))
        return self.initial, list(self.logs), current(kb, rows, self.beliefs)

    def _changes(self, kb: KB, rows: list) -> list[str]:
        covered = {s for v in kb.events if v.type == "cover" for s in v.support}
        out = [(e.t, f"[{e.id}] t={e.t} {x}") for e in kb.evidence[self.n_ev:]
               if (x := self._line(e, rows, covered)) is not None]
        hs = holds(kb, rows)
        out += [(v.t, f"[{v.id}] t={v.t} " + _event(v, kb, hs)) for v in kb.events[self.n_events:]]
        ds = kb.task.get("decisions", [])
        out += [(d["t"], f"[d{i}] t={d['t']} you: waypoint {_f(d['waypoint'])}, gripper "
                         f"{'closed' if d['gripper_closed'] else 'open'} after it, target {d['target']}")
                for i, d in enumerate(ds[self.n_dec:], self.n_dec + 1) if d["waypoint"] is not None]
        self.n_ev, self.n_events, self.n_dec = len(kb.evidence), len(kb.events), len(ds)
        out += self._beliefs(kb)
        return [x for _, x in sorted(out, key=lambda r: r[0])]

    def _beliefs(self, kb: KB) -> list:
        """A belief line when it first appears, and again (naming the line it replaces) when its most likely
        container changes, its top probability falls by >= 0.1 (self_memory.changes' rule), or its set of
        candidate containers changes."""
        out = []
        for c, hyps in inside(kb).items():
            was = self.beliefs.get(c)
            keys = {k for k, _ in hyps}
            if was and hyps[0][0] == was[1] and was[2] - hyps[0][1] < 0.1 and keys == was[3]:
                continue
            self.n_b += 1
            t = kb.relations[("Inside", c)].t  # when the belief last changed
            rep = f"; replaces [{was[0]}]" if was else ""
            out.append((t, f"[b{self.n_b}] t={t} Inside({c}): {_hyps(hyps)}{rep}"))
            self.beliefs[c] = (f"b{self.n_b}", hyps[0][0], hyps[0][1], keys)
        return out

    def _line(self, e, rows: list, covered: set) -> str | None:
        d, a = e.data, ", ".join(e.args)
        if e.kind == "move_start":
            self.start[e.args[0]] = (d.get("t_at", e.t - 1), d.get("xyz"))
        elif e.kind == "move_stop" and e.args[0] in self.start and self.start[e.args[0]][1] is not None:
            t0, xyz0 = self.start.pop(e.args[0])
            return f"{a} moved by {_f(np.subtract(d['xyz'], xyz0))} to {_f(d['xyz'])} (t={t0}-{e.t})"
        elif e.kind == "appear" and e.t != rows[0]["t"]:  # the first frame's objects are the initial state
            row = next((r for r in rows if r["t"] == e.t), None)
            o = row["obs"].get(e.args[0]) if row else None
            return f"{a} seen at {_f(o[1])}" if o and o[0] else f"{a} seen"
        elif e.kind == "disappear" and e.id not in covered:
            return f"{a} no longer seen; last at {_f(d['last_xyz'])}"
        elif e.kind == "near_pass":
            return f"{a} passed within {100 * d['min_3d']:.1f} cm of each other (detected)"
        elif e.kind in KINDS and e.kind not in ("pick_up", "release"):  # grasps / releases: the events below
            return f"the robot: {e.kind.replace('_', ' ')}, hand at {_f(d['ee'])}"
        return None


def _event(v, kb: KB, hs: dict) -> str:
    """A grasp line says where the hand closed; a release line where it opened and the highest it rose while holding."""
    h = next((h for h in hs.get(v.args[0], []) if v.t in (h["t"], h["t_off"])), None) if v.args else None
    if v.type == "grasp":
        return f"[{v.args[0]}] grasped" + (f", hand at {_f(h['at'])}" if h and h["at"] is not None else "")
    if v.type == "place":
        on = "" if v.args[1] == "table" else f" onto [{v.args[1]}]"
        if h is None or h.get("off") is None:
            return f"[{v.args[0]}] released{on}"
        return f"[{v.args[0]}] released{on}, hand at {_f(h['off'])}; highest hand z while held {h['top']:.3f}"
    if v.type == "reveal":
        return f"[{v.args[1]}] seen where [{v.args[0]}] was grasped: [{v.args[0]}] hid it"
    return _event_text(v, kb)


def initial(kb: KB, rows: list) -> str:
    """Every object as observed in the first frame; the rest 'not seen yet'."""
    first = rows[0]["obs"] if rows else {}
    lines = [f"Initial state (t={rows[0]['t'] if rows else 0}, the first frame; later lines may supersede it):"]
    for o in kb.objects.values():
        ob = first.get(o.id)
        lines.append(f"  [{o.id}] {o.description}, " + (f"seen at {_f(ob[1])}" if ob and ob[0] else "not seen yet"))
    return "\n".join(lines)


def current(kb: KB, rows: list, beliefs: dict) -> str:
    t, ee = (rows[-1]["t"], rows[-1]["obs"]["ee"]) if rows else (0, [True, [0, 0, 0], 0.08])
    h, ins, hs = held(kb), inside(kb), holds(kb, rows)
    lines = [f"Now, t={t} (authoritative: where it differs from anything above, this holds):",
             f"  hand at {_f(ee[1])}, finger opening {ee[2]:.3f} m, holding "
             + (", ".join(f"[{x}]" for x in h) if h else "nothing"),
             "Goal items, in order:"] + _items(kb, ins, hs, beliefs)
    lines.append("Objects (world frame, m): id | state | x, y, z | last seen t | inside (hidden objects)")
    for o in (o for o in kb.objects.values() if seen(o)):
        vis = "held" if o.id in h else o.visibility.value
        xyz = _f(o.pose.value) if o.pose.value is not None and vis != "hidden" else "-"
        hid = f"{_hyps(ins[o.id])} [{beliefs[o.id][0]}]" if vis == "hidden" and o.id in ins and o.id in beliefs \
            else "-"
        lines.append(f"  [{o.id}] | {vis} | {xyz} | {o.t_last_seen} | {hid}")
    bins = [o.id for o in kb.objects.values() if o.cls == "container" and o.pose.value is not None and seen(o)]
    lines += ["Relative positions:",
              "  containers from the robot's left (+y) to its right: "
              + ", ".join(sorted(bins, key=lambda b: -kb.objects[b].pose.value[1])),
              "  containers from nearest to farthest from the robot's base (-0.615, 0): " + ", ".join(sorted(
                  bins, key=lambda b: float(np.linalg.norm(np.asarray(kb.objects[b].pose.value[:2]) - BASE))))]
    lines += [f"  [{rel_id(f)}]" for f, _, _ in geometric(kb) if f[0] != "LeftOf"]
    return "\n".join(lines + ["What happened (semantic events only, oldest first):"] + _summary(kb, hs))


def _items(kb: KB, ins: dict, hs: dict, beliefs: dict) -> list[str]:
    """Per coloured item of the goal: the container that hid it (seen, or the belief) and what was done to it."""
    items, out = kb.task.get("items", []), []
    for it in (it for it in items if it["color"]):
        o = next((o for o in kb.objects.values() if o.attributes.get("color") == it["color"]), None)
        if o is None:
            continue
        rev = next((v.args[0] for v in reversed(kb.events) if v.type == "reveal" and v.args[1] == o.id), None)
        if rev:
            box, where = rev, f"seen under [{rev}]"
        else:
            box = ins[o.id][0][0] if o.id in ins else None
            where = f"inside {_hyps(ins[o.id])}" + (f" [{beliefs[o.id][0]}]" if o.id in beliefs else "") \
                if o.id in ins and o.visibility.value == "hidden" else o.visibility.value
        did = "; ".join(_hold_text(h) for h in hs.get(box, [])) or "not grasped yet"
        out.append(f"  {len(out) + 1}. {it['phrase']} [{o.id}]: {where}; [{box}]: {did}" if box else
                   f"  {len(out) + 1}. {it['phrase']} [{o.id}]: {where}")
    return out


def _summary(kb: KB, hs: dict) -> list[str]:
    """Covers, close passes, grasps, releases and reveals; no moves (the log has them)."""
    out = []
    for v in kb.events:
        if v.type == "cover":
            out.append(f"  t={v.t}: [{v.args[1]}] covered by [{v.args[0]}]")
        elif v.type == "swap":
            out.append(f"  t={v.t}: [{v.args[0]}] and [{v.args[1]}] passed close (identity may be confused)")
        elif v.type in ("grasp", "place", "reveal"):
            out.append(f"  t={v.t}: " + _event(v, kb, hs))
    return out


def request(view: K2View, system, goal: str, current: list, state: dict, breakpoint: bool, note: list) -> tuple:
    """(user messages, the memory text joined, for check()). `current`: [(rgb, label)] as controller.user_content."""
    from .controller import image_part, text_part
    parts = [p for rgb, label in current for p in (text_part(label + ":"), image_part(rgb))]
    a, logs, c = view.update(system)
    fixed = [text_part(f"Task goal: {goal}"),
             text_part("Robot state, fixed part (camera calibration): "
                       + json.dumps({k: state[k] for k in FIXED_STATE if k in state})), text_part(a)]
    msgs = [fixed] + [[text_part(x)] for x in logs]
    if breakpoint:  # OpenAI looks up the first 2 and the latest 50 explicit breakpoints and writes at most 4
        for m in msgs:
            m[-1] = {**m[-1], "prompt_cache_breakpoint": {"mode": "explicit"}}
    rest = {k: v for k, v in state.items() if k not in FIXED_STATE}
    msgs.append([text_part(c)] + parts + [text_part("Robot state: " + json.dumps(rest))] + note)
    return [{"role": "user", "content": m} for m in msgs], "\n\n".join([a, *logs, c])
