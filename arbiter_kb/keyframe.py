"""VLM keyframe call (DEVELOPMENT §2.2 step 6, §2.5; v1.10): at a keyframe a small VLM, thinking off, sees ONLY the
front frame and the instruction, and lists the movable objects on the table, which become the detector's phrases
(one tracker slot per object). Not from the simulator's scene (privileged). Every movable object is listed, not only
the task's, so the memory stays task-agnostic; the ones the instruction names are marked `task_item`.

v1.9 put the annotator into this call; on 4 real frames the merged prompt listed 2 / 4 right with no memory and 0 / 4
with a memory in the prompt (it copied objects from the text), against 4 / 4 for this image-only prompt (NOTES
"Keyframe call check"). So listing reads the image alone, and relations and descriptions are the review call's job
(review.py), which sees a window of frames. Model: ArbiterParams.vlm_model via llm.vlm_client.
"""
from __future__ import annotations

import re

from .controller import image_part, parse_answer, text_part
from .task_items import NOUNS, parse_task_items

MAX_OBJECTS = 10
PROMPT = """This is a robot's front camera looking down at a table. The robot's instruction is: "{goal}".
List every movable object ON the table (not the table, not the robot itself; include an object the robot is
holding), at most {cap} objects in total. Mark task_item true for objects the instruction refers to.
Answer with JSON only:
{{"objects": [{{"name": "<1-2 words, e.g. cube, container>", "color": "<one word>", "count": <int>,
               "task_item": <true|false>}}]}}"""


def messages(rgb, goal: str) -> list:
    return [{"role": "user", "content": [text_part(PROMPT.format(goal=goal, cap=MAX_OBJECTS)), image_part(rgb)]}]


def check_clues(items) -> tuple[list[dict], list[str]]:
    """(valid entries, problems): names of 1-2 lowercase words, a one-word colour, counts 1..MAX, total <= MAX."""
    if not isinstance(items, list):
        return [], ["unparseable answer"]
    out, errs, total = [], [], 0
    for o in items:
        name = str(o.get("name", "")).strip().lower() if isinstance(o, dict) else ""
        color = str(o.get("color", "")).strip().lower() if isinstance(o, dict) else ""
        count = o.get("count") if isinstance(o, dict) else None
        ok = (re.fullmatch(r"[a-z]+( [a-z]+)?", name) and re.fullmatch(r"[a-z]*", color)
              and isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= MAX_OBJECTS
              and total + count <= MAX_OBJECTS)
        if not ok:
            errs.append(f"dropped clue object {o}")
            continue
        total += count
        out.append({"name": name, "color": color, "count": count, "task_item": o.get("task_item") is True})
    return out, errs


def phrase(o: dict) -> str:
    """'red cube.'; the colour is not repeated when the model already put it in the name ('red cube')."""
    color = o["color"] if o["color"] and not o["name"].startswith(o["color"] + " ") else ""
    return f"{(color + ' ') if color else ''}{o['name']}."


def phrases(objects: list[dict]) -> list[str]:
    """Detector phrases, one per tracker slot ("white cube." x3); task items first, as the cap would cut last."""
    ordered = sorted(objects, key=lambda o: not o["task_item"])
    return [phrase(o) for o in ordered for _ in range(o["count"])]


def mark_task_items(objects: list[dict], goal: str) -> list[dict]:
    """task_item = the instruction names it (noun and, if given, colour; task_items.parse_task_items) OR the VLM
    said so. The text match is the reliable part: the VLM's own flags were inconsistent on real frames."""
    items = parse_task_items(goal)
    for o in objects:
        words = o["name"].split() + ([o["color"]] if o["color"] else [])
        nouns = {NOUNS[w] for w in words if w in NOUNS}
        o["task_item"] = o["task_item"] or any(it["noun"] in nouns and (it["color"] is None or it["color"] in words)
                                               for it in items)
    return objects


def list_objects(llm, rgb, goal: str, max_tokens: int = 600) -> dict:
    """Ask, check. {"objects", "phrases", "problems", "raw"}; the caller decides when (first frame, keyframes)."""
    text = llm.chat(messages(rgb, goal), max_tokens=max_tokens)
    answer = parse_answer(text)
    objs, errs = check_clues(answer.get("objects") if isinstance(answer, dict) else None)
    objs = mark_task_items(objs, goal)
    return {"objects": objs, "phrases": phrases(objs), "problems": errs, "raw": text[-400:]}
