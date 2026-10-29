"""The KB in words (arm KR, NOTES "Does a richer KB text make the facts easier to extract?"): the same beliefs and
events as views.controller_view, written for a reader rather than as a table. Positions become places relative to the
robot (its left is +y; near / far from its base), containers say what they probably hold, cubes say where they are and
whether their container moved since, and the timeline uses names and joins moves made at the same time into one
exchange. Nothing here is new information: every phrase is computed from the KB (no truth, no VLM).
"""
from __future__ import annotations

import numpy as np

from .kb import KB
from .task_items import is_task_item

BASE = np.array([-0.615, 0.0])  # the robot's base in the world frame (RoboMME)
SIDE = 0.03  # |y| below this is "in the middle"
HOLD_P = 0.2  # a container "may hold" a cube with at least this probability


def name(kb: KB, oid: str) -> str:
    o = kb.objects[oid]
    return f"the {o.attributes['color']} cube" if o.attributes.get("color") else oid


def moves(kb: KB) -> list[tuple[str, int, int]]:
    """[(object, t_start, t_end)] from move_start / move_stop evidence, in time order."""
    start, out = {}, []
    for e in kb.evidence:
        if e.kind == "move_start":
            start[e.args[0]] = e.data.get("t_at", e.t - 1)
        elif e.kind == "move_stop" and e.args[0] in start:
            out.append((e.args[0], start.pop(e.args[0]), e.t))
    return out


def exchanges(kb: KB) -> list[tuple[int, int, list[str]]]:
    """Moves that overlap by at least half of the shorter one, grouped: [(t_start, t_end, [objects])]. Back-to-back
    exchanges share a frame or two at their ends, so mere touching does not join them."""
    groups: list = []
    for obj, t0, t1 in moves(kb):
        g = next((g for g in groups if min(t1, g[1]) - max(t0, g[0]) >= 0.5 * min(t1 - t0, g[1] - g[0])), None)
        if g:
            g[0], g[1], g[2] = min(g[0], t0), max(g[1], t1), g[2] + [obj]
        else:
            groups.append([t0, t1, [obj]])
    return [(a, b, sorted(set(c))) for a, b, c in groups]


def inside(kb: KB) -> dict[str, list[tuple[str | None, float]]]:
    """{cube: [(container or None, p)] most likely first}."""
    return {s: sorted(d.probs.items(), key=lambda kv: -kv[1]) for (pred, s), d in kb.relations.items() if pred == "Inside"}


def _place(kb: KB, oid: str, near: list[str]) -> str:
    xy = np.asarray(kb.objects[oid].pose.value[:2])
    side = "on the robot's left" if xy[1] > SIDE else "on the robot's right" if xy[1] < -SIDE else "in the middle"
    rank = near.index(oid) + 1
    return f"{side}, {'nearest to the robot' if rank == 1 else f'number {rank} from the robot'}"


def _container(kb: KB, b: str, near: list[str], ins: dict, mv: list) -> str:
    holds = [f"{name(kb, c)} ({p:.2f})" for c, hyps in ins.items() for k, p in hyps if k == b and p >= HOLD_P]
    n = sum(m[0] == b for m in mv)
    last = next((g for g in reversed(exchanges(kb)) if b in g[2]), None)
    with_ = f", last at t={last[0]}-{last[1]} together with {', '.join(x for x in last[2] if x != b)}" if last and \
        len(last[2]) > 1 else ""
    return (f"  {b}: {_place(kb, b, near)}; " + (f"may hold {' or '.join(holds)}" if holds else
            "holds no cube as far as the memory knows") + f"; moved {n} time{'s' if n != 1 else ''}{with_}.")


def _cube(kb: KB, c: str, ins: dict, mv: list, items: list) -> str:
    o, tag = kb.objects[c], " (task item)" if is_task_item(kb.objects[c].attributes, kb.objects[c].cls, items) else ""
    if o.visibility.value != "hidden" or c not in ins:
        return f"  {name(kb, c)}{tag}: visible."
    hyps = [(k, p) for k, p in ins[c] if k and p >= HOLD_P]
    covered = o.t_last_seen
    after = {k: sum(m[0] == k and m[1] > covered for m in mv) for k, _ in hyps}
    where = " or ".join(f"under {k} ({p:.2f}; {k} has moved {after[k]} time{'s' if after[k] != 1 else ''} since, and "
                        f"the cube moved with it)" for k, p in hyps)
    doubt = "; which container is which became uncertain when containers passed close" if len(hyps) > 1 else ""
    return f"  {name(kb, c)}{tag}: hidden since t={covered + 1}, {where}{doubt}."


def _timeline(kb: KB) -> list[str]:
    rows = [(v.t, f"t={v.t}: {name(kb, v.args[1])} was covered by {v.args[0]}") for v in kb.events if v.type == "cover"]
    rows += [(a, f"t={a}-{b}: {' and '.join(objs)} {'exchanged places' if len(objs) > 1 else 'moved'}")
             for a, b, objs in exchanges(kb)]
    rows += [(v.t, f"t={v.t}: {v.args[0]} and {v.args[1]} passed close (identity may be confused)")
             for v in kb.events if v.type == "swap"]
    return [r for _, r in sorted(rows)]


def narrative(kb: KB) -> str:
    items = kb.task.get("items", [])
    bins = [o.id for o in kb.objects.values() if o.cls == "container" and o.pose.value is not None]
    near = sorted(bins, key=lambda b: float(np.linalg.norm(np.asarray(kb.objects[b].pose.value[:2]) - BASE)))
    ltr = sorted(bins, key=lambda b: -kb.objects[b].pose.value[1])
    ins, mv = inside(kb), moves(kb)
    lines = [f"Goal: {kb.task.get('goal', '?')}",
             f"Containers from the robot's left to its right: {', '.join(ltr)}.",
             f"Containers from nearest to farthest from the robot: {', '.join(near)}.", "Containers:"]
    lines += [_container(kb, b, near, ins, mv) for b in bins]
    lines.append("Cubes:")
    lines += [_cube(kb, c, ins, mv, items) for c, o in kb.objects.items() if o.cls == "cube"]
    return "\n".join(lines + ["What happened (oldest first):"] + ["  " + r for r in _timeline(kb)])
