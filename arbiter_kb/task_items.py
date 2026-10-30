"""Task items and the open perception vocabulary (DEVELOPMENT §2.3: record task-agnostically, mark what the
task names).

- `parse_task_items(goal)`: the object phrases a task instruction names ("the red cube", "the container"),
  read from the instruction text only, never from the simulator's scene. A deterministic parser for
  RoboMME-style instructions; an LLM extractor (as scene-graph-robot's llm_entities.py does) is the
  general version when instructions are free-form.
- `open_vocabulary(items, ...)`: the detector's word list without privileged counts: generic words with a
  fixed number of slots each (unused slots stay empty), plus one slot per specific task item. Containers
  come first, as sg_pilot's detector requires (a container box enclosing an earlier cube mask is rejected).
- `is_task_item(attrs, cls, items)`: whether a KB object matches one of the task's phrases.
"""
from __future__ import annotations

import re

COLORS = ("red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black", "gray", "grey", "brown")
NOUNS = {"cube": "cube", "cubes": "cube", "block": "cube", "blocks": "cube", "container": "container",
         "containers": "container", "bin": "container", "bins": "container", "bowl": "container",
         "cup": "container", "button": "button", "buttons": "button", "peg": "peg", "stick": "stick",
         "hole": "hole", "target": "target", "targets": "target", "plate": "plate", "box": "box"}
GENERIC_SLOTS = {"container": 5, "cube": 4}  # upper bounds per generic word; the scene's true counts are not used


def parse_task_items(goal: str) -> list[dict]:
    """[{"phrase": "red cube", "noun": "cube", "color": "red"}] in order of mention, without duplicates."""
    words = re.findall(r"[a-z]+", goal.lower())
    out, seen = [], set()
    for i, w in enumerate(words):
        if w not in NOUNS:
            continue
        color = words[i - 1] if i > 0 and words[i - 1] in COLORS else None
        noun = NOUNS[w]
        phrase = f"{color} {noun}" if color else noun
        if phrase not in seen:
            seen.add(phrase)
            out.append({"phrase": phrase, "noun": noun, "color": color})
    return out


def place_word(items: list[dict]) -> str:
    """What the first generic slots look for: "target" when the goal names targets and no container (VideoPlaceOrder,
    PickXtimes, SwingXtimes: the cubes go onto flat targets, which hide nothing; with "container." the KB read a cube
    on a target as hidden inside a bin, val pilot), else "container"."""
    nouns = {it["noun"] for it in items}
    return "target" if "target" in nouns and "container" not in nouns else "container"


def open_vocabulary(items: list[dict], slots: dict = GENERIC_SLOTS) -> list[str]:
    """GroundingDINO phrases, one per tracker slot, each ending in '.' (GroundingDINO splits at '.')."""
    specific = [f"{it['phrase']}." for it in items if it["color"]]
    generic_rest = [f"{w}." for w, n in slots.items() if w != "container" for _ in range(n)]
    return [f"{place_word(items)}."] * slots.get("container", 0) + specific + generic_rest


def is_task_item(attrs: dict, cls: str, items: list[dict]) -> bool:
    """A KB object matches a task phrase when the nouns agree and, if the phrase has a colour, so does it."""
    return any(it["noun"] == cls and (it["color"] is None or it["color"] == attrs.get("color")) for it in items)
