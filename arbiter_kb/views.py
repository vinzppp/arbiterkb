"""Views of the KB: predicate facts with course-style unify / ask, the text view the arbiter reads,
and (M3) the event log and the controller view (DEVELOPMENT §3.8).

Predicates are generated on demand from the typed tables (DEVELOPMENT.md §3.2); they are never stored.
Geometric relations (Near, LeftOf, On) come from the pose beliefs, with probability = the poses' confidence.
ask() returns answers WITH probabilities instead of True / False.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from .kb import KB
from .params import RuleParams
from .rules import held


def is_var(x) -> bool:
    return isinstance(x, str) and x.startswith("?")


def unify(pattern: tuple, fact: tuple, bindings: dict | None = None) -> dict | None:
    """Course Module 7 unify for flat tuples: variables are strings starting with '?'."""
    b = dict(bindings or {})
    if len(pattern) != len(fact):
        return None
    for p, f in zip(pattern, fact):
        if is_var(p):
            if p in b and b[p] != f:
                return None
            b[p] = f
        elif p != f:
            return None
    return b


def facts(kb: KB) -> list[tuple[tuple, float]]:
    """All facts the KB currently supports, each with its probability."""
    out = []
    for (pred, subj), d in kb.relations.items():
        out += [((pred, subj, k), p) for k, p in d.probs.items() if k is not None]
    for o in kb.objects.values():
        vis = o.visibility.value
        if vis == "hidden":
            out.append((("Hidden", o.id), o.visibility.conf))
        elif vis == "visible":
            out.append((("Visible", o.id), o.visibility.conf))
    out += [(("Held", k), 1.0) for k in held(kb)]
    return out + geometric(kb)


def geometric(kb: KB, p: RuleParams = RuleParams()) -> list[tuple[tuple, float]]:
    """Near(a, b): centres closer than r_near. LeftOf(a, b): a is on the robot's left of b (+y) by more
    than 2 cm. On(a, b): a above b with xy within r_on and a height gap of 0.5-8 cm. Visible or held
    objects only (a hidden object's pose is a guess; its place is the Inside distribution)."""
    known = [o for o in kb.objects.values() if o.pose.value is not None and o.visibility.value in ("visible", "held")]
    out = []
    for a, b in permutations(known, 2):
        pa, pb, c = a.pose.value, b.pose.value, round(a.pose.conf * b.pose.conf, 4)
        if a.id < b.id and np.linalg.norm(pa - pb) < p.r_near:
            out.append((("Near", a.id, b.id), c))
        if pa[1] - pb[1] > 0.02:
            out.append((("LeftOf", a.id, b.id), c))
        if np.linalg.norm(pa[:2] - pb[:2]) < p.r_on and 0.005 < pa[2] - pb[2] < 0.08:
            out.append((("On", a.id, b.id), c))
    return out


def ask(kb: KB, pattern: tuple) -> list[tuple[dict, float]]:
    """All bindings that satisfy `pattern`, most probable first."""
    answers = [(b, p) for f, p in facts(kb) if (b := unify(pattern, f)) is not None]
    return sorted(answers, key=lambda bp: (-bp[1], str(bp[0])))


def text_view(kb: KB) -> str:
    """One line per object, e.g.
    [cube_green] cube, hidden since t=32; inside bin_1 (.55) | bin_2 (.45); support e12,e31"""
    lines = []
    for o in kb.objects.values():
        vis = o.visibility.value
        line = f"[{o.id}] {o.description}, {vis}"
        if vis == "hidden":
            line += f" since t={o.t_last_seen + 1}"
        d = kb.inside(o.id)
        if d is not None:
            hyps = sorted(d.probs.items(), key=lambda kv: -kv[1])
            line += "; inside " + " | ".join(f"{k or 'elsewhere'} ({p:.2f})" for k, p in hyps)
            line += "; support " + ",".join(d.support[-4:])
        elif o.pose.value is not None and vis == "visible":
            x, y, _ = o.pose.value
            line += f"; at x={x:.2f} y={y:.2f}"
        lines.append(line)
    return "\n".join(lines)


def _xy(v) -> str:
    return f"({v[0]:.2f}, {v[1]:.2f})"


def _ev_line(e) -> str | None:
    """One line per evidence the controller cares about; None for the rest (appear of a visible scene)."""
    d, a = e.data, ", ".join(e.args)
    if e.kind == "disappear":
        return f"t={e.t} {a} disappeared at {_xy(d['last_xyz'])}"
    if e.kind == "move_stop" and "from" in d:
        return f"t={e.t} {a} moved {_xy(d['from'])} -> {_xy(d['xyz'])} (t={d['t_from']}-{e.t})"
    if e.kind == "near_pass":
        return f"t={e.t} {a} passed each other, closest {100 * d['min_3d']:.1f} cm"
    if e.kind in ("grip_close", "grip_open"):
        return f"t={e.t} gripper {'closed' if e.kind == 'grip_close' else 'opened'} at {_xy(d['ee'])}"
    return None


def event_log(kb: KB, last: int = 30) -> list[str]:
    """Evidence and events in time order, with ids so the controller can cite them. A move_stop is joined
    with its move_start, so each motion reads 'from -> to'."""
    start: dict[str, tuple] = {}
    rows = []
    for e in kb.evidence:
        if e.kind == "move_start":
            start[e.args[0]] = (e.t, e.data.get("xyz"))
            continue
        if e.kind == "move_stop" and e.args[0] in start and start[e.args[0]][1] is not None:
            t0, xyz0 = start.pop(e.args[0])
            e = type(e)(e.id, e.t, e.kind, e.args, {**e.data, "from": xyz0, "t_from": t0})
        if (line := _ev_line(e)) is not None:
            rows.append((e.t, 0, int(e.id[1:]), f"[{e.id}] {line}"))
    for v in kb.events:
        rows.append((v.t, 1, int(v.id[1:]), f"[{v.id}] t={v.t} " + _event_text(v)))
    return [r[3] for r in sorted(rows)][-last:]


def _event_text(v) -> str:
    """R6's `swap` event records that two containers exchanged PLACES (certain); whether their identities
    were confused is the Inside mix, so the log must not read it as a certain identity swap."""
    if v.type == "swap":
        return f"{v.args[0]} and {v.args[1]} exchanged places (identity may be confused: see Inside)"
    return f"event {v.type}({', '.join(v.args)}) conf {v.conf:.2f}"


def controller_view(kb: KB) -> str:
    """What the LLM controller reads (DEVELOPMENT §3.8): goal and progress, objects with 3D centres,
    relations with probabilities, then the event log. Every line names the ids it can cite."""
    lines = [f"Goal: {kb.task.get('goal', '?')}"]
    names = {"grasp": "grasps", "place": "places", "cover": "covers", "swap": "container exchanges"}
    done = [f"{names[t]} {n}" for t in names if (n := sum(e.type == t for e in kb.events))]
    lines.append("Progress: " + (", ".join(done) or "nothing yet"))
    lines.append("Objects (world frame, m):")
    for o in kb.objects.values():
        vis = "held" if o.id in held(kb) else o.visibility.value
        if vis == "hidden":  # its pose is a probability-weighted mean, possibly between two containers
            pos = ": see Inside below"
        else:
            pos = "" if o.pose.value is None else " at " + ", ".join(f"{v:.3f}" for v in o.pose.value)
            pos += " (inferred)" if o.pose.source == "inferred" else ""
        lines.append(f"  [{o.id}] {o.description}, {vis}{pos}, last seen t={o.t_last_seen}")
    lines.append("Relations:")
    for (pred, subj), d in kb.relations.items():
        hyps = " | ".join(f"{k or 'elsewhere'} {p:.2f}" for k, p in sorted(d.probs.items(), key=lambda kv: -kv[1]))
        lines.append(f"  {pred}({subj}): {hyps}  [{','.join(d.support[-3:])}]")
    lines += [f"  {f[0]}({', '.join(f[1:])}) {pr:.2f}" for f, pr in geometric(kb) if f[0] != "LeftOf"]
    lines.append("Event log (oldest first):")
    return "\n".join(lines + ["  " + x for x in event_log(kb)])
