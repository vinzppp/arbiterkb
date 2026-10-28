"""VLM keyframe call (DEVELOPMENT §2.2 step 6, §2.5, §3.7; v1.9 merges the clue-object lister and the annotator):
at a keyframe a small VLM, thinking off, reads the KB view (goal, objects with ids, relations, event log) and the
front frame with the tracked ids drawn on it, and returns in ONE answer:
  - clue_objects: every movable object on the table (not only the task's, so the memory stays task-agnostic),
    which become the detector's phrases (one tracker slot per object); not from the simulator's scene (privileged);
  - relations and descriptions for objects that already have ids, only what geometry cannot decide.
A new object has no id yet, so it is described at a later keyframe (or by the review call).

Every item is checked on its own and dropped alone. Relations and descriptions are written as beliefs with support,
never as fact: source "vlm", p from a fixed strength table, support = the keyframe id + the evidence it cites. The
answer must carry no numbers beyond clue counts: positions stay with perception, probabilities come only from the
table, so they can be calibrated against truth. The model is ArbiterParams.vlm_model (llm.vlm_client); the
controller's model is separate. Lister benchmark: results/latest/vlm_list_bench (qwen3.5-9b 4/4 frames).
"""
from __future__ import annotations

import json
import re

from .controller import image_part, parse_answer, text_part
from .kb import KB, SetFact
from .params import ArbiterParams
from .task_items import NOUNS, parse_task_items
from .views import controller_view

MAX_OBJECTS = 10
STRENGTHS = ("clear", "ambiguous", "unlikely")
PREDICATES = ("inside", "on", "under", "near", "left_of", "touching", "held", "stacked_on", "blocked_by", "same_as")

SYSTEM = """You look at a robot's front camera at one moment and keep its memory of a table-top scene. You get the
memory (the goal, objects with ids and 3D centres, the relations already known with probabilities, and an event log
with ids) and the current camera image, with the ids of the objects already tracked drawn on them.

Answer with ONE JSON object:
{"clue_objects": [{"name": "<1-2 words, e.g. cube, container>", "color": "<one word>", "count": <int>,
                   "task_item": <true|false>}],
 "relations": [{"subject": "<object id>", "predicate": "<word>", "object": "<object id>",
                "strength": "clear|ambiguous|unlikely", "cites": ["<event or evidence ids you rely on>"]}],
 "objects": [{"id": "<object id>", "description": "<= 12 words: what it looks like, its state"}]}
clue_objects: every movable object ON the table (not the table, not the robot; include an object the robot holds),
at most CAP in total; task_item true for objects the goal refers to.
relations and objects: only ids that appear in the memory; add what it is missing or correct what looks wrong.
Prefer these predicates: PREDS; another short word is allowed when none fits. "unlikely" marks a relation the
memory states but you believe is false. Never write numbers except the counts: no coordinates, no probabilities.
Use empty lists when you have nothing to add."""


def system_prompt() -> str:
    return SYSTEM.replace("PREDS", ", ".join(PREDICATES)).replace("CAP", str(MAX_OBJECTS))


def messages(kb: KB, rgb, t: int) -> list:
    content = [text_part(controller_view(kb)),
               text_part(f"Current front camera, frame {t}, ids drawn on tracked objects:"), image_part(rgb)]
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": content}]


def strength_p(s: str, ap: ArbiterParams) -> float:
    """The fixed table (ArbiterParams.p_*): the VLM never writes a probability itself."""
    return {"clear": ap.p_clear, "ambiguous": ap.p_ambiguous, "unlikely": ap.p_unlikely}[s]


def check_clues(items) -> tuple[list[dict], list[str]]:
    """(valid entries, problems): names of 1-2 lowercase words, a one-word colour, counts 1..MAX, total <= MAX."""
    if not isinstance(items, list):
        return [], ["clue_objects missing"]
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


def _has_number(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, dict):
        return any(_has_number(v) for v in x.values())
    if isinstance(x, list):
        return any(_has_number(v) for v in x)
    return False


def check_annotations(answer: dict, kb: KB) -> tuple[list[dict], list[dict], list[str]]:
    """(valid relations, valid descriptions, problems). Invalid items are dropped one by one, not the answer."""
    objs, known, errs, rels, descs = set(kb.objects), kb.known_ids(), [], [], []
    for r in answer.get("relations") or []:
        bad = [f"relation {r}: {why}" for why, ok in (
            ("unknown subject / object", isinstance(r, dict) and r.get("subject") in objs and r.get("object") in objs),
            ("bad strength", isinstance(r, dict) and r.get("strength") in STRENGTHS),
            ("predicate not a short word", isinstance(r, dict) and isinstance(r.get("predicate"), str)
             and 0 < len(r["predicate"]) <= 20 and " " not in r["predicate"].strip()),
            ("cites unknown ids", isinstance(r, dict) and all(c in known for c in r.get("cites") or [])),
            ("contains numbers", not _has_number(r))) if not ok]
        (errs.extend(bad[:1]) if bad else rels.append(r))
    for d in answer.get("objects") or []:
        ok = isinstance(d, dict) and d.get("id") in objs and isinstance(d.get("description"), str) \
            and 0 < len(d["description"].split()) <= 12
        (descs.append(d) if ok else errs.append(f"description {d}: unknown id or bad text"))
    return rels, descs, errs


def to_ops(rels: list[dict], descs: list[dict], t: int, frame_id: str, ap: ArbiterParams = ArbiterParams()) -> list:
    """Beliefs with support: the keyframe observation id + the cited ids."""
    ops = [SetFact(r["predicate"].strip().lower(), (r["subject"], r["object"]), True, strength_p(r["strength"], ap), t,
                   "vlm", [frame_id] + list(r.get("cites") or [])) for r in rels]
    return ops + [SetFact("description", (d["id"],), d["description"].strip(), 1.0, t, "vlm", [frame_id])
                  for d in descs]


def keyframe_call(kb: KB, llm, t: int, rgb, max_tokens: int = 1500, ap: ArbiterParams = ArbiterParams()) -> dict:
    """Ask once, check, write the beliefs. Returns the clue objects and detector phrases for perception, and a
    report for the trace; the caller decides when (first frame, keyframes: §2.5 item 4)."""
    text = llm.chat(messages(kb, rgb, t), max_tokens=max_tokens)
    answer = parse_answer(text)
    if not isinstance(answer, dict):
        return {"t": t, "clue_objects": [], "phrases": [], "written": 0, "problems": ["unparseable answer"],
                "raw": text[-400:]}
    clues, errs = check_clues(answer.get("clue_objects"))
    clues = mark_task_items(clues, kb.task.get("goal", ""))
    rels, descs, more = check_annotations(answer, kb)
    ops = to_ops(rels, descs, t, f"o{t}:image", ap)
    kb.apply(ops, "keyframe")
    return {"t": t, "clue_objects": clues, "phrases": phrases(clues), "written": len(ops), "problems": errs + more,
            "raw": text[-400:], "answer": json.dumps({"relations": rels, "objects": descs})[:800]}
