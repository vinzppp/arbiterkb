"""Arbiter (M0 stub): trigger policy, the RuleOnly backend (never changes anything) and the patch
validator that M4's LLM backend must pass (DEVELOPMENT.md §3.7)."""
from __future__ import annotations

from .kb import KB
from .params import ArbiterParams

STRENGTHS = {"clear", "ambiguous", "unlikely"}
EVENT_TYPES = {"swap", "cover", "none"}
COORD_KEYS = {"x", "y", "z", "xyz", "pose", "position", "u", "v", "uv"}


class Trigger:
    """Fires on a closed near-pass, or every `every_k` frames while some object is hidden."""

    def __init__(self, p: ArbiterParams):
        self.p, self.calls, self.last = p, 0, -10**9

    def fires(self, kb: KB, new_ev: list, t: int) -> str | None:
        if self.calls >= self.p.max_calls:
            return None
        reason = None
        if any(e.kind == "near_pass" for e in new_ev):
            reason = "near_pass"
        elif any(o.visibility.value == "hidden" for o in kb.objects.values()) and t - self.last >= self.p.every_k:
            reason = "periodic"
        if reason:
            self.calls, self.last = self.calls + 1, t
        return reason


class RuleOnlyBackend:
    name = "rule_only"

    def review(self, kb_text: str, new_evidence: list[str]) -> dict | None:
        return None


def validate(patch: dict, kb: KB) -> list[str]:
    """Errors of an arbiter patch; [] means valid. No numbers are allowed anywhere (strength is
    categorical), so the arbiter can never write a coordinate or a probability."""
    errs = _numbers(patch, "patch")
    if set(patch) - {"events", "hypotheses", "risks"}:
        errs.append(f"unknown keys {sorted(set(patch) - {'events', 'hypotheses', 'risks'})}")
    known, objs = kb.known_ids(), kb.objects
    bins = {o for o, f in objs.items() if f.cls == "container"}

    def cites_ok(item, where):
        cites = item.get("cites") or []
        if not isinstance(cites, list):
            errs.append(f"{where}: cites must be a list")
            return
        if not cites:
            errs.append(f"{where}: cites is empty")
        errs.extend(f"{where}: unknown evidence id {c}" for c in cites if c not in known)

    for i, ev in enumerate(patch.get("events", [])):
        if ev.get("type") not in EVENT_TYPES or ev.get("strength") not in STRENGTHS:
            errs.append(f"events[{i}]: bad type or strength")
        errs.extend(f"events[{i}]: unknown object {a}" for a in ev.get("args", []) if a not in objs)
        cites_ok(ev, f"events[{i}]")
    for i, h in enumerate(patch.get("hypotheses", [])):
        if objs.get(h.get("cube")) is None or objs[h["cube"]].cls != "cube":
            errs.append(f"hypotheses[{i}]: unknown cube {h.get('cube')}")
        errs.extend(f"hypotheses[{i}]: {k} is not a container" for k in h.get("keep", []) if k not in bins)
        d = kb.inside(h.get("cube", ""))
        cited = {a for e in kb.evidence if e.id in (h.get("cites") or []) for a in e.args}
        held = {k for k, p in d.probs.items() if p > 0} if d else set()
        errs.extend(f"hypotheses[{i}]: {k} holds no mass and is not cited" for k in h.get("keep", [])
                    if k in bins and k not in held | cited)
        cites_ok(h, f"hypotheses[{i}]")
    for i, r in enumerate(patch.get("risks", [])):
        if r.get("obj") not in objs or len(str(r.get("reason", ""))) > 80:
            errs.append(f"risks[{i}]: unknown object or reason too long")
        cites_ok(r, f"risks[{i}]")
    return errs


def _numbers(x, path: str) -> list[str]:
    if isinstance(x, bool):
        return []
    if isinstance(x, (int, float)):
        return [f"{path}: numeric value {x} (no numbers allowed)"]
    if isinstance(x, dict):
        out = [f"{path}.{k}: coordinate-like key" for k in x if str(k).lower() in COORD_KEYS]
        return out + [e for k, v in x.items() for e in _numbers(v, f"{path}.{k}")]
    if isinstance(x, list):
        return [e for i, v in enumerate(x) for e in _numbers(v, f"{path}[{i}]")]
    return []
