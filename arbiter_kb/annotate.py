"""VLM annotator (DEVELOPMENT §2.2 step 6, §3.7): at event moments a small VLM reads the KB (object table, the
recent event log, the relations the rules already hold) and optionally one keyframe, and returns relations and
object descriptions that geometry cannot decide. Its answer is checked and written as beliefs with support,
never as fact: source "vlm", p from a fixed strength table, support = the keyframe id + the evidence it cites.

The answer must not carry numbers (no coordinates, no probabilities): positions stay with perception, and
probabilities come only from the strength table, so they can be calibrated against truth.
"""
from __future__ import annotations

import json

from .controller import image_part, parse_answer, text_part
from .kb import KB, SetFact
from .params import ArbiterParams
from .views import controller_view

STRENGTHS = ("clear", "ambiguous", "unlikely")


def strength_p(s: str, ap: ArbiterParams) -> float:
    """The fixed table (ArbiterParams.p_*): the VLM never writes a probability itself."""
    return {"clear": ap.p_clear, "ambiguous": ap.p_ambiguous, "unlikely": ap.p_unlikely}[s]
PREDICATES = ("inside", "on", "under", "near", "left_of", "touching", "held", "stacked_on", "blocked_by", "same_as")

SYSTEM = """You annotate a robot's memory of a table-top scene. You get the memory (objects with ids and 3D
centres, the relations already known with probabilities, and an event log with ids), and possibly the current
camera image with the object ids drawn on it. Add what the memory is missing or correct what looks wrong.

Answer with ONE JSON object:
{"relations": [{"subject": "<object id>", "predicate": "<word>", "object": "<object id>",
                "strength": "clear|ambiguous|unlikely", "cites": ["<event or evidence ids you rely on>"]}],
 "objects": [{"id": "<object id>", "description": "<= 12 words: what it looks like, its state"}]}
Use only object ids that appear in the memory. Prefer these predicates: PREDS; another short word is allowed
when none fits. "unlikely" marks a relation the memory states but you believe is false. Never write numbers:
no coordinates, no probabilities. Return empty lists if you have nothing to add."""


def system_prompt() -> str:
    return SYSTEM.replace("PREDS", ", ".join(PREDICATES))


def messages(kb: KB, rgb=None, frame_t: int | None = None) -> list:
    content = [text_part(controller_view(kb))]
    if rgb is not None:
        content += [text_part(f"Current front camera, frame {frame_t}, ids drawn on objects:"), image_part(rgb)]
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": content}]


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


def check(answer, kb: KB) -> tuple[list[dict], list[dict], list[str]]:
    """(valid relations, valid descriptions, problems). Invalid items are dropped one by one, not the answer."""
    if not isinstance(answer, dict):
        return [], [], ["unparseable answer"]
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


def to_ops(rels: list[dict], descs: list[dict], t: int, frame_id: str | None, ap: ArbiterParams = ArbiterParams()) -> list:
    """Beliefs with support: the keyframe observation id (if an image was shown) + the cited ids."""
    base = [frame_id] if frame_id else []
    ops = [SetFact(r["predicate"].strip().lower(), (r["subject"], r["object"]), True, strength_p(r["strength"], ap), t,
                   "vlm", base + list(r.get("cites") or [])) for r in rels]
    ops += [SetFact("description", (d["id"],), d["description"].strip(), 1.0, t, "vlm", base)
            for d in descs if base]  # a description needs the image it describes
    return ops


def annotate(kb: KB, llm, t: int, rgb=None, max_tokens: int = 4000, ap: ArbiterParams = ArbiterParams()) -> dict:
    """Ask, check, write. Returns a report for the trace: problems, how many beliefs were written, usage."""
    text = llm.chat(messages(kb, rgb, t), max_tokens=max_tokens)
    rels, descs, errs = check(parse_answer(text), kb)
    frame_id = f"o{t}:image" if rgb is not None else None
    ops = to_ops(rels, descs, t, frame_id, ap)
    if frame_id is None:  # text only: the beliefs rest on the cited ids; drop those with none
        ops = [o for o in ops if o.support]
    kb.apply(ops, "annotator")
    return {"t": t, "written": len(ops), "problems": errs, "raw": text[-400:], "image": rgb is not None,
            "answer": json.dumps({"relations": rels, "objects": descs})[:800]}
