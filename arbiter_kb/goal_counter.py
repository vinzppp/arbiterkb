"""A goal counter kept by the KB (L4 progress; NOTES "A goal counter kept by the KB"): the controller's model reads
the instruction once and says what to count and how many; from then on the KB counts its own robot-state actions
(actions.py evidence) every frame, and the view carries one line, so the controller reads a ready number instead of
counting. No "now" sentences: those cost the counting decisions (NOTES "Trimmed results").
"""
from __future__ import annotations

import json
import re

from .kb import KB

KINDS = ("pick_up", "release", "press", "hold_left", "hold_right")
WORDS = {"pick_up": "pick-ups", "release": "put-downs", "press": "button presses", "hold_left": "holds on the left",
         "hold_right": "holds on the right"}
PROMPT = """A robot gets this instruction: "{goal}".
The robot's memory counts these actions of its own: pick_up (it grasps an object), release (it lets an object go:
on a target, on the table or into a container), press (it presses a button), hold_left / hold_right (it holds an
object up over the left / right target). It cannot tell objects apart.
Which ONE of these actions shows how far the repetitions in the instruction have got, and how many of it does the
instruction ask for in total? If the instruction gives no number, the target is 1. A motion repeated back and forth
N times means N holds on each side. When the instruction moves an object back and forth between two places, count
the holds, whether or not it gives a number. Answer with JSON only:
{{"count": "pick_up|release|press|hold_left|hold_right", "target": <int>, "then": "<what to do after the repetitions>"}}"""


def parse_goal(llm, goal: str, reasoning: dict | None = None) -> dict | None:
    """One call; the checked spec, or None (then the view carries no counter line)."""
    text = llm.call([{"role": "user", "content": PROMPT.format(goal=goal)}], max_tokens=4000, reasoning=reasoning)[0]
    spec = last_json(text)
    ok = isinstance(spec, dict) and spec.get("count") in KINDS and isinstance(spec.get("target"), int) \
        and not isinstance(spec.get("target"), bool) and 1 <= spec["target"] <= 20
    return {"count": spec["count"], "target": spec["target"], "then": str(spec.get("then", ""))[:80]} if ok else None


def last_json(text: str) -> dict | None:
    """The last flat JSON object in the text: reasoning models sometimes write a draft before the answer (v1 lost
    16 of 40 SwingXtimes parses to a first-brace-to-last-brace read; NOTES "Goal parse v2")."""
    for m in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return None


def done(kb: KB, kind: str) -> int:
    """How many actions of `kind` the KB holds (holds by side)."""
    if kind.startswith("hold_"):
        return sum(e.kind == "hold_above" and e.data.get("side") == kind[5:] for e in kb.evidence)
    return sum(e.kind == kind for e in kb.evidence)


def progress_line(kb: KB, spec: dict | None) -> str:
    if not spec:
        return ""
    n = done(kb, spec["count"])
    tail = " — the repetitions are done" if n >= spec["target"] else ""
    return f"Goal progress: {WORDS[spec['count']]} {n} of {spec['target']}{tail}."
