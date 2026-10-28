"""Clue-object lister (DEVELOPMENT §2.5): a small VLM, thinking off, lists the movable objects in a frame; the list
becomes the detector's phrases (one tracker slot per object). Not from the simulator's scene (privileged).

The VLM lists every movable object on the table, not only the task's, so the memory stays task-agnostic; it
flags the ones the instruction names (`task_item`). Checks drop bad entries one by one and cap the total.
Benchmark of candidate models: results/latest/vlm_list_bench (qwen3-vl-8b / qwen3.5-9b: 4/4 at about 1 s).
The model is ArbiterParams.vlm_model (default qwen/qwen3.5-9b, DeepInfra, thinking off): build the client with
llm.vlm_client(params.arbiter).
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


def check(answer) -> tuple[list[dict], list[str]]:
    """(valid entries, problems): names of 1-2 lowercase words, a one-word colour, counts 1..MAX, total <= MAX."""
    if not isinstance(answer, dict) or not isinstance(answer.get("objects"), list):
        return [], ["unparseable answer"]
    out, errs, total = [], [], 0
    for o in answer["objects"]:
        name = str(o.get("name", "")).strip().lower() if isinstance(o, dict) else ""
        color = str(o.get("color", "")).strip().lower() if isinstance(o, dict) else ""
        count = o.get("count") if isinstance(o, dict) else None
        ok = (re.fullmatch(r"[a-z]+( [a-z]+)?", name) and re.fullmatch(r"[a-z]*", color)
              and isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= MAX_OBJECTS
              and total + count <= MAX_OBJECTS)
        if not ok:
            errs.append(f"dropped {o}")
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
    objs, errs = check(parse_answer(text))
    objs = mark_task_items(objs, goal)
    return {"objects": objs, "phrases": phrases(objs), "problems": errs, "raw": text[-400:]}
