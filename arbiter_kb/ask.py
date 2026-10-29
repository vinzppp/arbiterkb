"""The VLM's one runtime job besides listing (spec v1.11, §2.2 step 6): when the rules cannot decide, ask ONE closed
question about ONE marked frame, with no memory text, and write the checked answer into the KB.

Why so narrow (NOTES, 2026-10-28): with the KB's text in its prompt a small VLM copied the text (it confirmed 68% of
planted false events, and listed objects the text named); from the front video neither qwen3.5-9b nor Sonnet 5.5
could tell a pick-up from a put-down (<= 0.52); robot actions come from robot state, identity from the tracker, and
relations from geometry (as in ECoMEM and Ledger). What a VLM can add is what one frame shows about one object:
- occluder: an object just disappeared; is it hidden by the robot arm or by a container (Ledger's occlusion query)?
- appearance: what an object looks like, for the view's "looks like" (descriptions are not geometry).
Each answer is one of fixed options or "cannot_tell"; it is evidence or a belief with source "vlm", supported by the
frame, never overwriting the rules.
"""
from __future__ import annotations

import re

from .controller import image_part, parse_answer, text_part
from .kb import KB, AddEvidence, SetFact

OCCLUDERS = ("robot_arm", "container", "out_of_view", "cannot_tell")
QUESTIONS = {
    "occluder": "Object {obj} (marked in the image) has just disappeared. What hides it? Answer with JSON only: "
                '{{"answer": "robot_arm|container|out_of_view|cannot_tell"}}',
    "appearance": "Describe object {obj} (marked in the image): what it looks like, in at most 12 words, and its main "
                  'colour in one word. If you cannot see it clearly, use "cannot_tell". Answer with JSON only: '
                  '{{"description": "<words or cannot_tell>", "color": "<one word or cannot_tell>"}}',
}


def messages(kind: str, obj: str, rgb) -> list:
    return [{"role": "user", "content": [text_part(QUESTIONS[kind].format(obj=obj)), image_part(rgb)]}]


def check(kind: str, answer) -> tuple[dict | None, str]:
    """(the checked answer or None, the problem)."""
    if not isinstance(answer, dict):
        return None, "unparseable answer"
    if kind == "occluder":
        return (answer, "") if answer.get("answer") in OCCLUDERS else (None, f"bad option {answer.get('answer')!r}")
    d, c = str(answer.get("description", "")).strip(), str(answer.get("color", "")).strip().lower()
    ok = 0 < len(d.split()) <= 12 and not re.search(r"\d", d) and re.fullmatch(r"[a-z_]+", c or "-")
    return ({"description": d, "color": c}, "") if ok else (None, "description too long, has numbers, or bad colour")


def to_ops(kind: str, obj: str, ans: dict, t: int) -> list:
    sup = [f"o{t}:image"]
    if kind == "occluder":
        return [AddEvidence(t, "vlm_occluder", (obj,), {"answer": ans["answer"], "obs": sup})]
    if ans["description"] == "cannot_tell":
        return []
    return [SetFact("description", (obj,), ans["description"] + (f" ({ans['color']})" if ans["color"] not in
                                                                   ("", "cannot_tell") else ""), 1.0, t, "vlm", sup)]


def ask(kb: KB, llm, kind: str, obj: str, t: int, rgb, max_tokens: int = 300) -> dict:
    """One question about one object in one marked frame; check; write. A report for the trace."""
    assert kind in QUESTIONS and obj in kb.objects, (kind, obj)
    text = llm.chat(messages(kind, obj, rgb), max_tokens=max_tokens)
    ans, problem = check(kind, parse_answer(text))
    ops = to_ops(kind, obj, ans, t) if ans else []
    kb.apply(ops, "ask")
    return {"t": t, "kind": kind, "obj": obj, "answer": ans, "written": len(ops), "problem": problem,
            "raw": text[-300:]}
