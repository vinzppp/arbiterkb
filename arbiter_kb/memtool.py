"""Memory as a tool (NOTES "C — memory as a tool"): the controller reads a short working memory whose size does not
grow with the episode, and asks the memory for more with queries. The working memory holds the goal, the container
orders, and each hidden cube's most likely container; history, close passes, the event log and how a belief is known
are behind queries. A query is one JSON object {"query": {"tool": <name>, "arg": <object id or "">}}.
"""
from __future__ import annotations

import numpy as np

from .epistemic import epistemic_lines
from .kb import KB
from .narrative import BASE, _timeline, exchanges, inside, moves, name

TOOLS = {
    "history": "arg = an object id: when it moved and with which containers, and when it was last seen",
    "passes": "arg = a container id: every close pass it took part in, with the distance",
    "how_known": "arg = a cube colour: how the memory knows where that cube is, and what else could be true",
    "events": "arg = \"\": the whole event log, oldest first",
}


def working_memory(kb: KB) -> str:
    bins = [o.id for o in kb.objects.values() if o.cls == "container" and o.pose.value is not None]
    near = sorted(bins, key=lambda b: float(np.linalg.norm(np.asarray(kb.objects[b].pose.value[:2]) - BASE)))
    ltr = sorted(bins, key=lambda b: -kb.objects[b].pose.value[1])
    ins = inside(kb)
    cubes = [f"  {name(kb, c)}: " + (f"hidden, most likely under {ins[c][0][0]}" if o.visibility.value == "hidden"
                                     and ins.get(c) and ins[c][0][0] else o.visibility.value)
             for c, o in kb.objects.items() if o.cls == "cube"]
    tools = "\n".join(f"  {k}: {v}" for k, v in TOOLS.items())
    return "\n".join([f"Goal: {kb.task.get('goal', '?')}", "Working memory (now):",
                      f"  containers from the robot's left to its right: {', '.join(ltr)}",
                      f"  containers from nearest to farthest from the robot: {', '.join(near)}", *cubes,
                      "The memory holds more. To ask it, answer with only {\"query\": {\"tool\": <name>, \"arg\": <...>}} "
                      "(one query per turn, at most 4); answer the questions when you have what you need. Tools:",
                      tools])


def run_tool(kb: KB, tool: str, arg: str) -> str:
    arg = str(arg or "").strip()
    if tool == "history":
        if arg not in kb.objects:
            return f"unknown object {arg!r}"
        mv = [(a, b, objs) for a, b, objs in exchanges(kb) if arg in objs]
        rows = [f"t={a}-{b}: moved" + (f" together with {', '.join(x for x in objs if x != arg)}" if len(objs) > 1 else "")
                for a, b, objs in mv]
        n = sum(m[0] == arg for m in moves(kb))
        return f"{arg}: moved {n} times; last seen t={kb.objects[arg].t_last_seen}\n" + "\n".join(rows)
    if tool == "passes":
        rows = [f"t={e.t}: with {e.args[1] if e.args[0] == arg else e.args[0]}, {100 * e.data['min_3d']:.0f} cm apart"
                for e in kb.evidence if e.kind == "near_pass" and arg in e.args]
        return "\n".join(rows) or f"no close passes for {arg!r}"
    if tool == "how_known":
        hit = [x for x in epistemic_lines(kb) if x.startswith(f"the {arg.replace('cube_', '')} cube")]
        return hit[0] if hit else f"no hidden cube {arg!r}"
    if tool == "events":
        return "\n".join(_timeline(kb))
    return f"unknown tool {tool!r}; tools: {', '.join(TOOLS)}"
