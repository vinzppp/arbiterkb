"""The KB in words (arm KR, NOTES "Does a richer KB text make the facts easier to extract?"): the same beliefs and
events as views.controller_view, written for a reader rather than as a table. Positions become places relative to the
robot (its left is +y; near / far from its base), containers say what they probably hold, cubes say where they are and
whether their container moved since, and the timeline uses names and joins moves made at the same time into one
exchange. Nothing here is new information: every phrase is computed from the KB (no truth, no VLM).
"""
from __future__ import annotations

import re

import numpy as np

from .action_view import KINDS, action_rows
from .kb import KB
from .motion import reported
from .task_items import is_task_item

BASE = np.array([-0.615, 0.0])  # the robot's base in the world frame (RoboMME)
SIDE = 0.03  # |y| below this is "in the middle"
HOLD_P = 0.2  # a container "may hold" a cube with at least this probability


def name(kb: KB, oid: str) -> str:
    """A cube by its colour; anything else (a container, or "table": a put-down away from every container, rules.py)
    by its id (closed-loop smoke run: "table" raised a KeyError)."""
    o = kb.objects.get(oid)
    return f"the {o.attributes['color']} cube" if o is not None and o.attributes.get("color") else oid


def moves(kb: KB) -> list[tuple[str, int, int]]:
    """[(object, t_start, t_end)] from move_start / move_stop evidence, in time order; moves below the reporting
    threshold (motion.reported) are left out."""
    start, out = {}, []
    for e in kb.evidence:
        if e.kind == "move_start":
            start[e.args[0]] = (e.data.get("t_at", e.t - 1), e.data.get("xyz"))
        elif e.kind == "move_stop" and e.args[0] in start:
            t0, xyz0 = start.pop(e.args[0])
            if xyz0 is None or reported(xyz0, e.data.get("xyz")):
                out.append((e.args[0], t0, e.t))
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


def revealed(kb: KB) -> dict[str, tuple[str, int, str]]:
    """{cube: (container, t, event id)} from R4's reveal events (the last per cube)."""
    return {v.args[1]: (v.args[0], v.t, v.id) for v in kb.events if v.type == "reveal"}


def seen_again(kb: KB, cube: str, d, rev) -> str:
    """The Inside line of a cube that is not hidden now (NOTES "K's view after a cube is seen again")."""
    if rev:
        return f"  [Inside({cube})] was under {rev[0]}, revealed t={rev[1]} [{rev[2]}]"
    t = max((e.t for e in kb.evidence if e.kind == "appear" and e.args[0] == cube), default=kb.objects[cube].t_last_seen)
    top, p = d.argmax()
    now = kb.objects[cube].pose.value
    at = f", now at {', '.join(f'{v:.3f}' for v in now)}" if now is not None else ""
    return f"  [Inside({cube})] seen again at t={t}{at}; the earlier belief ({top or 'elsewhere'} {p:.2f}) no longer applies"


def _container(kb: KB, b: str, near: list[str], ins: dict, mv: list) -> str:
    """A container 'may hold' only cubes still hidden; after an R4 reveal it 'hid' that cube (NOTES "K's view after a
    cube is seen again")."""
    holds = [f"{name(kb, c)} ({p:.2f})" for c, hyps in ins.items() for k, p in hyps if k == b and p >= HOLD_P
             and kb.objects[c].visibility.value == "hidden"]
    hid = [f"hid {name(kb, c)} (revealed t={t})" for c, (box, t, _) in revealed(kb).items() if box == b]
    n = sum(m[0] == b for m in mv)
    last = next((g for g in reversed(exchanges(kb)) if b in g[2]), None)
    with_ = f", last at t={last[0]}-{last[1]} together with {', '.join(x for x in last[2] if x != b)}" if last and \
        len(last[2]) > 1 else ""
    return (f"  {b}: {_place(kb, b, near)}; " + "; ".join(hid + [f"may hold {' or '.join(holds)}" if holds else
            "holds no cube as far as the memory knows"]) + f"; moved {n} time{'s' if n != 1 else ''}{with_}.")


def seen(o) -> bool:
    """False for a tracker slot that never found its object (the open word list has spare slots; closed-loop smoke run:
    such a cube was listed as "visible" in the joined facts while the view's object list left it out)."""
    return not (o.visibility.value == "unknown" and o.t_last_seen < 0)


def _cube(kb: KB, c: str, ins: dict, mv: list, items: list) -> str:
    o, tag = kb.objects[c], " (task item)" if is_task_item(kb.objects[c].attributes, kb.objects[c].cls, items) else ""
    if o.visibility.value != "hidden" or c not in ins:
        rev = revealed(kb).get(c)
        return f"  {name(kb, c)}{tag}: visible" + (f"; it was under {rev[0]} (revealed t={rev[1]})." if rev else ".")
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


NUMBERS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10}
HOW = {"placed": "at table level", "set_down": "from a small height", "dropped": "from above (as into a container)"}


def goal_numbers(goal: str) -> list[str]:
    """The numbers the goal mentions, as written: 'two times' -> 'two (2)'. No reading of what they count."""
    words = re.findall(r"[a-z]+|\d+", goal.lower())
    return [w if w.isdigit() else f"{w} ({NUMBERS[w]})" for w in words if w.isdigit() or w in NUMBERS]


def robot_narrative(kb: KB, trim: bool = False) -> str:
    """The robot-state KB (no objects) in words: what the gripper holds, the last action, counts per kind, the
    numbers in the goal, then the same action timeline as the controller view (action_view.action_rows).
    trim (arm KT): no goal-numbers line, and only the kinds that happened, without the put-down breakdown; KR lost
    to K at wrap-ups with both in (NOTES "Counting with the KB in words")."""
    acts = [e for e in kb.evidence if e.kind in KINDS]
    held = bool(acts) and next((e.kind for e in reversed(acts) if e.kind in ("pick_up", "release")), "") == "pick_up"
    last = action_rows(kb, coords=False)[-1][2].split("] ", 1)[1] if acts else "nothing yet"
    n = {k: sum(e.kind == k for e in acts) for k in KINDS}
    rel = [f"{sum(e.data.get('how') == h for e in acts if e.kind == 'release')} {w}" for h, w in HOW.items()]
    sides = [e.data["side"] for e in acts if e.kind == "hold_above"]
    nums = goal_numbers(kb.task.get("goal", ""))
    so_far = (f"picked up an object {n['pick_up']} time(s); put an object down {n['release']} time(s) "
              f"({', '.join(rel)}); pressed the button {n['press']} time(s); held an object up on the left "
              f"{sides.count('left')} time(s) and on the right {sides.count('right')} time(s)")
    if trim:
        parts = [(n["pick_up"], "picked up an object"), (n["release"], "put an object down"),
                 (n["press"], "pressed the button"), (sides.count("left"), "held an object up on the left"),
                 (sides.count("right"), "held an object up on the right")]
        so_far = "; ".join(f"{w} {c} time{'s' if c != 1 else ''}" for c, w in parts if c) or "nothing yet"
    lines = [f"Goal: {kb.task.get('goal', '?')}"]
    lines += [] if trim else [f"Numbers the goal mentions: {', '.join(nums) if nums else 'none'}."]
    lines += [f"Right now the gripper is {'holding the object it last picked up' if held else 'empty'}.",
              f"Last thing the robot did: {last}.", f"So far: {so_far}.",
              "Everything the robot has done (oldest first):"]
    return "\n".join(lines + ["  " + r[2] for r in action_rows(kb, coords=False)])


def joined_facts(kb: KB) -> list[str]:
    """Facts the reader would otherwise join from several lines of the controller view (arm KH): orders of the
    containers, what each may hold and its moves, and each hidden cube's containers. No "now" sentences (they cost
    the counting decisions; NOTES "Trimmed results"). Empty when the KB has no containers."""
    bins = [o.id for o in kb.objects.values() if o.cls == "container" and o.pose.value is not None]
    if not bins:
        return []
    near = sorted(bins, key=lambda b: float(np.linalg.norm(np.asarray(kb.objects[b].pose.value[:2]) - BASE)))
    ltr = sorted(bins, key=lambda b: -kb.objects[b].pose.value[1])
    ins, mv, items = inside(kb), moves(kb), kb.task.get("items", [])
    return ([f"Containers from the robot's left to its right: {', '.join(ltr)}.",
             f"Containers from nearest to farthest from the robot: {', '.join(near)}."]
            + [_container(kb, b, near, ins, mv).strip() for b in bins]
            + [_cube(kb, c, ins, mv, items).strip() for c, o in kb.objects.items() if o.cls == "cube" and seen(o)])


def merged_view(view: str, kb: KB) -> str:
    """The controller view plus the joined facts (arm KH); the view alone when there is nothing to join."""
    facts = joined_facts(kb)
    return view + ("\nJoined facts (computed from the memory above):\n" + "\n".join("  " + f for f in facts)
                   if facts else "")
