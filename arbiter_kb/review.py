"""VLM review call (DEVELOPMENT §2.2 step 6b, §3.7; v1.10 adds the annotator's work): every ~3 s, and once before the
controller acts on a target, a small VLM watches the last window of frames (tracker ids drawn on each) next to the
KB's objects and the events in that window, and returns:
  - checks: confirm / contradict / cannot_tell per KB event or evidence in the window;
  - missing: what happened in the video that the KB did not record;
  - relations and descriptions for objects with ids (the annotator's work, v1.5-v1.9);
  - identity: for marks that passed close, which mark each one is now;
  - summary: one line for the window.
Every item names the frames it rests on, so each answer is checkable against truth, and every item is checked alone.
The answers are written as evidence or beliefs with source "vlm", never overwriting the rules; relations take p from
the fixed strength table. Not yet: a contradiction does not retract anything (support-chain retraction is M4 item 10).

The prompt tells the model to judge from the frames only: the merged keyframe prompt (v1.9) copied objects from the
memory text into its answer (NOTES "Keyframe call check"). Model: ArbiterParams.vlm_model via llm.vlm_client.
"""
from __future__ import annotations

import json
import re

from .controller import image_part, parse_answer, text_part
from .kb import KB, AddEvidence, SetFact
from .params import ArbiterParams
from .views import event_log

STRENGTHS = ("clear", "ambiguous", "unlikely")
VERDICTS = ("confirm", "contradict", "cannot_tell")
PREDICATES = ("inside", "on", "under", "near", "left_of", "touching", "held", "stacked_on", "blocked_by", "same_as")

SYSTEM = """You check a robot's memory of a table-top scene against video. You get frames from the last few seconds
(each titled with its frame number; the tracked objects' ids are drawn on them), the objects in memory, and the
events the memory recorded in this window, with ids. Judge ONLY from what the frames show: the memory may be wrong,
and an object or event the frames do not show is "cannot_tell", never assumed.

Answer with ONE JSON object:
{"checks": [{"id": "<event or evidence id>", "verdict": "confirm|contradict|cannot_tell", "frames": [<frame numbers>]}],
 "missing": [{"what": "<= 12 words: what happened", "objects": ["<object id>"], "frames": [<frame numbers>]}],
 "relations": [{"subject": "<object id>", "predicate": "<word>", "object": "<object id>",
                "strength": "clear|ambiguous|unlikely", "frames": [<frame numbers>]}],
 "objects": [{"id": "<object id>", "description": "<= 12 words: what it looks like, its state", "frames": [<n>]}],
 "identity": [{"mark": "<object id asked about>", "now": "<object id it is in the last frame, or cannot_tell>",
               "frames": [<frame numbers>]}],
 "summary": "<= 25 words: what happened in this window"}
Check every listed event. Relations and descriptions only for ids drawn on the frames. Prefer these predicates:
PREDS; another short word is allowed when none fits. "unlikely" marks a relation the memory states but the frames
contradict. The only numbers allowed are frame numbers. Use empty lists when you have nothing to add."""


def system_prompt() -> str:
    return SYSTEM.replace("PREDS", ", ".join(PREDICATES))


def window_lines(kb: KB, t0: int, t1: int) -> list[str]:
    """The event log's lines whose time lies in [t0, t1] (evidence, events and semantic actions, with ids)."""
    out = []
    for line in event_log(kb, last=10_000):
        m = re.search(r"\bt=(\d+)", line)
        if m and t0 <= int(m.group(1)) <= t1:
            out.append(line)
    return out


def messages(kb: KB, frames: list[tuple[int, object]], ask_identity: list[str] = ()) -> list:
    """frames: [(t, rgb with ids drawn)], oldest first."""
    t0, t1 = frames[0][0], frames[-1][0]
    objs = ", ".join(f"{o.id} ({o.cls})" for o in kb.objects.values()) or "none"
    lines = window_lines(kb, t0, t1)
    text = (f"Goal: {kb.task.get('goal', '?')}\nObjects in memory: {objs}\n"
            f"Events recorded in frames {t0}-{t1}:\n" + ("\n".join("  " + x for x in lines) or "  none") +
            (f"\nIdentity question: these objects passed close: {', '.join(ask_identity)}. For each, which id is it "
             f"in the last frame?" if ask_identity else ""))
    content = [text_part(text)]
    for t, rgb in frames:
        content += [text_part(f"Frame {t}:"), image_part(rgb)]
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": content}]


def strength_p(s: str, ap: ArbiterParams) -> float:
    """The fixed table (ArbiterParams.p_*): the VLM never writes a probability itself."""
    return {"clear": ap.p_clear, "ambiguous": ap.p_ambiguous, "unlikely": ap.p_unlikely}[s]


def _frames_ok(x: dict, shown: set[int]) -> bool:
    f = x.get("frames")
    return isinstance(f, list) and len(f) > 0 and all(isinstance(n, int) and n in shown for n in f)


def _short(s, n: int) -> bool:
    return isinstance(s, str) and 0 < len(s.split()) <= n and not re.search(r"\d", s)


def _item_ok(kind: str, x: dict, kb: KB, window: set[str], marks: set[str]) -> bool:
    objs = set(kb.objects)
    if kind == "checks":
        return x.get("id") in window and x.get("verdict") in VERDICTS
    if kind == "missing":
        return _short(x.get("what"), 12) and all(o in objs for o in x.get("objects") or [])
    if kind == "relations":
        p = x.get("predicate")
        return (x.get("subject") in objs and x.get("object") in objs and x.get("strength") in STRENGTHS
                and isinstance(p, str) and 0 < len(p) <= 20 and " " not in p.strip())
    if kind == "objects":
        return x.get("id") in objs and _short(x.get("description"), 12)
    return x.get("mark") in marks and (x.get("now") in objs or x.get("now") == "cannot_tell")


def check(answer, kb: KB, shown: set[int], window: set[str], marks: set[str]) -> tuple[dict, list[str]]:
    """({kind: valid items, "summary": text or None}, problems); each item is dropped alone, with the reason."""
    if not isinstance(answer, dict):
        return {}, ["unparseable answer"]
    good, errs = {}, []
    for kind in ("checks", "missing", "relations", "objects", "identity"):
        good[kind] = []
        for x in answer.get(kind) or []:
            ok = isinstance(x, dict) and _frames_ok(x, shown) and _item_ok(kind, x, kb, window, marks)
            (good[kind].append(x) if ok else errs.append(f"{kind} {x}: bad ids, frames or text"))
    s = answer.get("summary")
    good["summary"] = s.strip() if _short(s, 25) else None
    if s and good["summary"] is None:
        errs.append(f"summary {s!r}: too long or has numbers")
    return good, errs


def to_ops(good: dict, t0: int, t1: int, ap: ArbiterParams = ArbiterParams()) -> list:
    """Evidence (checks, missing, identity, summary) and beliefs (relations, descriptions), each supported by the
    frames it names (observation ids o<t>:image)."""
    sup = lambda x: [f"o{n}:image" for n in x["frames"]]  # noqa: E731
    ops = [AddEvidence(t1, "vlm_check", (c["id"],), {"verdict": c["verdict"], "obs": sup(c)}) for c in good["checks"]]
    ops += [AddEvidence(t1, "vlm_missing", tuple(m.get("objects") or []), {"what": m["what"], "obs": sup(m)})
            for m in good["missing"]]
    ops += [AddEvidence(t1, "vlm_identity", (i["mark"],), {"now": i["now"], "obs": sup(i)}) for i in good["identity"]]
    ops += [SetFact(r["predicate"].strip().lower(), (r["subject"], r["object"]), True, strength_p(r["strength"], ap),
                    t1, "vlm", sup(r)) for r in good["relations"]]
    ops += [SetFact("description", (d["id"],), d["description"].strip(), 1.0, t1, "vlm", sup(d))
            for d in good["objects"]]
    if good["summary"]:
        ops.append(AddEvidence(t1, "vlm_summary", (), {"text": good["summary"], "t0": t0, "obs": []}))
    return ops


def review(kb: KB, llm, frames: list[tuple[int, object]], ask_identity: list[str] = (), max_tokens: int = 1500,
           ap: ArbiterParams = ArbiterParams()) -> dict:
    """Ask once over the window, check, write. Returns a report for the trace."""
    t0, t1 = frames[0][0], frames[-1][0]
    window = {re.match(r"\[([ev]\d+)\]", x).group(1) for x in window_lines(kb, t0, t1) if re.match(r"\[[ev]\d+\]", x)}
    text = llm.chat(messages(kb, frames, ask_identity), max_tokens=max_tokens)
    good, errs = check(parse_answer(text), kb, {t for t, _ in frames}, window, set(ask_identity))
    ops = to_ops(good, t0, t1, ap) if good else []
    kb.apply(ops, "reviewer")
    return {"t0": t0, "t1": t1, "written": len(ops), "problems": errs, "raw": text[-400:],
            "answer": json.dumps({k: v for k, v in good.items()})[:800]}
