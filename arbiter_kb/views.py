"""Views of the KB: predicate facts with course-style unify / ask, the text view the arbiter reads,
and (M3) the event log and the controller view (DEVELOPMENT §3.8).

Predicates are generated on demand from the typed tables (DEVELOPMENT.md §3.2); they are never stored.
Geometric relations (Near, LeftOf, On) come from the pose beliefs, with probability = the poses' confidence.
ask() returns answers WITH probabilities instead of True / False.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from .action_view import KINDS as ACTION_KINDS, action_rows, progress
from .kb import KB
from .params import RuleParams
from .rules import held
from .task_items import is_task_item


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


def facts(kb: KB, p: RuleParams = RuleParams()) -> list[tuple[tuple, float]]:
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
    out += [(k, b.conf) for k, b in kb.facts.items() if b.value is True]
    return out + [(f, c) for f, c, _ in geometric(kb, p)]


def geometric(kb: KB, p: RuleParams = RuleParams()) -> list[tuple[tuple, float, list[str]]]:
    """[(fact, c, support)] for Near(a, b): centres closer than r_near; LeftOf(a, b): a is on the robot's
    left of b (+y) by more than left_margin; On(a, b): a above b, xy within r_on, height gap in
    [on_dz_min, on_dz_max]. Visible or held objects only (a hidden object's pose is a guess; its place is
    the Inside distribution). c = the product of the two pose confidences: how sure the POSITIONS are,
    not a calibrated probability of the relation. support = the two poses' observation ids."""
    known = [o for o in kb.objects.values() if o.pose.value is not None and o.visibility.value in ("visible", "held")]
    out = []
    for a, b in permutations(known, 2):
        pa, pb, c = a.pose.value, b.pose.value, round(a.pose.conf * b.pose.conf, 4)
        sup = list(a.pose.support[-1:]) + list(b.pose.support[-1:])
        if a.id < b.id and np.linalg.norm(pa - pb) < p.r_near:
            out.append((("Near", a.id, b.id), c, sup))
        if pa[1] - pb[1] > p.left_margin:
            out.append((("LeftOf", a.id, b.id), c, sup))
        if np.linalg.norm(pa[:2] - pb[:2]) < p.r_on and p.on_dz_min < pa[2] - pb[2] < p.on_dz_max:
            out.append((("On", a.id, b.id), c, sup))
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


def _xyz(v) -> str:
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


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
        return f"t={e.t} {a} passed close to each other, closest {100 * d['min_3d']:.1f} cm (detected)"
    if e.kind == "hand_pause":
        return f"t={e.t} hand stopped at {_xyz(d['ee'])}, gripper {'closed' if d['closed'] else 'open'}"
    if e.kind in ("grip_close", "grip_open"):
        return f"t={e.t} gripper {'closed' if e.kind == 'grip_close' else 'opened'} at {_xy(d['ee'])}"
    return _vlm_line(e)


def _vlm_line(e) -> str | None:
    """The review call's evidence (review.py); a confirmation adds nothing to read, so it is not shown."""
    d, a = e.data, ", ".join(e.args)
    if e.kind == "vlm_summary":
        return f"t={e.t} video summary, t={d['t0']}-{e.t} (vlm): {d['text']}"
    if e.kind == "vlm_check" and d["verdict"] == "contradict":
        return f"t={e.t} the video contradicts [{a}] (vlm)"
    if e.kind == "vlm_missing":
        return f"t={e.t} seen in the video, not recorded: {d['what']} (vlm)"
    if e.kind == "vlm_identity" and d["now"] not in (a, "cannot_tell"):
        return f"t={e.t} {a} now looks like {d['now']} (vlm)"
    return None


KEEP = ("cover", "grasp", "place")  # events never dropped from the log: they say where things are


LOW_LEVEL = ("grip_close", "grip_open", "hand_pause")  # explained by the semantic actions when those exist


def event_log(kb: KB, last: int = 30, low_level: bool = True, coords: bool = True) -> list[str]:
    """Evidence and events in time order, with ids so the controller can cite them. A move_stop is joined
    with its move_start, so each motion reads 'from -> to'. Over `last` lines, the oldest others are dropped
    (cover / grasp / place events are always kept) and a first line says how many were omitted."""
    start: dict[str, tuple] = {}
    rows = []
    covered = {s for v in kb.events if v.type == "cover" for s in v.support}  # shown in the cover line instead
    for e in kb.evidence:
        if (e.kind == "disappear" and e.id in covered) or (not low_level and e.kind in LOW_LEVEL) \
                or e.kind in ACTION_KINDS:  # actions: action_view.action_rows (ordinals, merged holds)
            continue
        if e.kind == "move_start":
            start[e.args[0]] = (e.data.get("t_at", e.t - 1), e.data.get("xyz"))
            continue
        if e.kind == "move_stop" and e.args[0] in start and start[e.args[0]][1] is not None:
            t0, xyz0 = start.pop(e.args[0])
            e = type(e)(e.id, e.t, e.kind, e.args, {**e.data, "from": xyz0, "t_from": t0})
        if (line := _ev_line(e)) is not None:
            rows.append((e.t, 0, int(e.id[1:]), f"[{e.id}] {line}"))
    rows += [(t, 0, n, line, True) for t, n, line in action_rows(kb, coords)]
    for v in kb.events:
        rows.append((v.t, 1, int(v.id[1:]), f"[{v.id}] t={v.t} " + _event_text(v, kb), v.type in KEEP))
    rows = sorted(r if len(r) == 5 else (*r, False) for r in rows)
    drop = max(0, len(rows) - last)
    kept = [r for i, r in enumerate(rows) if r[4] or i >= drop]
    return ([f"({len(rows) - len(kept)} earlier lines omitted)"] if len(kept) < len(rows) else []) + [r[3] for r in kept]


def _event_text(v, kb: KB) -> str:
    """R6's `swap` event marks a DETECTED close pass (the detector can be wrong: M3 truth check), and
    whether identities were confused is the Inside mix; so the log reads it as neither certain nor a swap.
    A cover is one line with where it happened and the containment rule spelled out (v1.7: in ep 9 a separate
    'disappeared at' line made the model look for the cube at its old spot)."""
    if v.type == "cover":
        ev = next((e for e in kb.evidence if e.id in v.support and "last_xyz" in e.data), None)
        where = f" at {_xy(ev.data['last_xyz'])}" if ev else ""
        return (f"{v.args[1]} covered by {v.args[0]}{where} (conf {v.conf:.2f}) [{','.join(v.support)}]; "
                f"from then on it moves with {v.args[0]}")
    if v.type == "swap":
        return f"{v.args[0]} and {v.args[1]} passed close (identity may be confused: see Inside)"
    return f"event {v.type}({', '.join(v.args)}) conf {v.conf:.2f}"


def controller_view(kb: KB, p: RuleParams = RuleParams()) -> str:
    """What the LLM controller reads (DEVELOPMENT §3.8): goal and progress, objects with 3D centres,
    relations with probabilities, then the event log. Every line names the ids it can cite."""
    lines = [f"Goal: {kb.task.get('goal', '?')}"]
    names = {"grasp": "grasps", "place": "places", "cover": "covers", "swap": "close passes"}
    done = [f"{names[t]} {n}" for t in names if (n := sum(e.type == t for e in kb.events))]
    done += progress(kb)
    lines.append("Progress: " + (", ".join(done) or "nothing yet"))
    items = kb.task.get("items", [])
    lines.append("Task items (named by the goal): " + (", ".join(i["phrase"] for i in items) or "none"))
    if not kb.objects:  # robot state only: no objects or relations to show, positions mean nothing alone
        return "\n".join(lines + ["What the robot has done (oldest first):"]
                         + ["  " + x for x in event_log(kb, low_level=False, coords=False)])
    lines.append("Objects (world frame, m):")
    for o in kb.objects.values():
        vis = "held" if o.id in held(kb) else o.visibility.value
        if vis == "hidden":  # its pose is a probability-weighted mean, possibly between two containers
            pos = ": see Inside below"
        else:
            pos = "" if o.pose.value is None else " at " + ", ".join(f"{v:.3f}" for v in o.pose.value)
            pos += " (inferred)" if o.pose.source == "inferred" else ""
        sup = f" [{o.pose.support[-1]}]" if o.pose.support and vis != "hidden" else ""
        seen = kb.facts.get(("description", o.id))
        look = f'; looks like "{seen.value}" (vlm, t={seen.t})' if seen else ""
        tag = " (task item)" if is_task_item(o.attributes, o.cls, items) else ""
        lines.append(f"  [{o.id}] {o.description}{tag}, {vis}{pos}, last seen t={o.t_last_seen}{sup}{look}")
    lines.append("Relations:")
    for (pred, subj), d in kb.relations.items():
        hyps = " | ".join(f"{k or 'elsewhere'} {p:.2f}" for k, p in sorted(d.probs.items(), key=lambda kv: -kv[1]))
        lines.append(f"  [{pred}({subj})] {hyps}  [{','.join(d.support[-3:])}]")
    lines += [f"  [{rel_id(f)}] from positions [{','.join(sup)}]" for f, _, sup in geometric(kb, p) if f[0] != "LeftOf"]
    lines += [f"  [{rel_id(k)}] {b.conf:.2f} ({b.source}) [{','.join(b.support[-3:])}]"
              for k, b in kb.facts.items() if k[0] != "description"]
    lines.append("Event log (oldest first):")
    return "\n".join(lines + ["  " + x for x in event_log(kb, low_level=False)])


def rel_id(fact: tuple) -> str:
    """The id a relation line carries in the controller view, e.g. 'Inside(cube_red)', 'On(cube_r, bin_a)'."""
    return f"{fact[0]}({', '.join(fact[1:])})"


def citable_ids(kb: KB, p: RuleParams = RuleParams()) -> set[str]:
    """Every id the controller view prints: evidence, events, objects, relation lines (observation ids
    o<t>:<object> are accepted separately by controller.check)."""
    rels = {rel_id((pred, subj)) for pred, subj in kb.relations} | {rel_id(k) for k in kb.facts}
    return kb.known_ids() | set(kb.objects) | rels | {rel_id(f) for f, _, _ in geometric(kb, p)}
