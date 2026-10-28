"""The knowledge base: L0/L1 object table, L2 relation distributions, evidence log, L3 events, L4 task.

Only `KB.apply(ops)` mutates state, and it logs every change (DEVELOPMENT.md §1, §3.2).
Invariants are checked by `check()` after every apply when ARBITER_DEBUG=1 (the default in tests).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from .beliefs import Belief, Distribution
from .ops import (AddEvent, AddEvidence, FlattenDist, MixDist, Retract, SetDist,  # noqa: F401 (re-exported)
                  SetPose, SetVis)

_ID = re.compile(r"^(e\d+|v\d+|o\d+:[\w.]+)$")  # evidence, event, observation ids


@dataclass
class ObjectFrame:
    id: str
    cls: str
    description: str
    attributes: dict
    pose: Belief  # value: np.ndarray [3] world metres, or None before first sighting
    visibility: Belief  # value: visible | hidden | held | occluded_by_arm | out_of_view | unknown
    track_quality: float = 1.0
    t_last_seen: int = -1
    extent: np.ndarray | None = None
    stationary: list = field(default_factory=list)


@dataclass
class Evidence:
    id: str
    t: int
    kind: str  # appear | disappear | move_start | move_stop | near_pass | ...
    args: tuple
    data: dict  # data["obs"]: observation ids this evidence was computed from


@dataclass
class Event:
    id: str
    t: int
    type: str  # cover | swap | pick | place | reveal | press
    args: tuple
    conf: float
    support: list[str]
    source: str  # rule | arbiter


def _r(x):
    return None if x is None else [round(float(v), 4) for v in np.asarray(x).ravel()]


class KB:
    def __init__(self, objects: list[tuple[str, str, dict]]):
        self.objects: dict[str, ObjectFrame] = {}
        for oid, cls, attrs in objects:
            desc = f"{attrs.get('color', '')} {cls}".strip()
            self.objects[oid] = ObjectFrame(oid, cls, desc, dict(attrs), Belief(None, 0.0, -1, "prior"),
                                            Belief("unknown", 1.0, -1, "prior"))
        self.relations: dict[tuple[str, str], Distribution] = {}
        self.evidence: list[Evidence] = []
        self.events: list[Event] = []
        self.task: dict = {}
        self.log: list[dict] = []  # changes since the caller last called pop_log()
        self.debug = os.environ.get("ARBITER_DEBUG", "0") == "1"

    # ---- mutation ------------------------------------------------------------------------
    def apply(self, ops: list, rule: str = "") -> list[str]:
        """Apply ops in order; return the ids of created evidence / events."""
        new_ids = []
        for op in ops:
            new_ids += self._apply_one(op, rule)
        if self.debug:
            self.check()
        return new_ids

    def _apply_one(self, op, rule: str) -> list[str]:
        name = type(op).__name__
        if isinstance(op, SetPose):
            o = self.objects[op.obj]
            before = (_r(o.pose.value), o.pose.conf)
            o.pose = Belief(np.asarray(op.xyz, float), op.conf, op.t, op.source, list(op.support))
            if op.source == "observed":
                o.t_last_seen = op.t  # observations are in the trace's obs field; not logged twice
            elif before[0] is None or np.abs(np.subtract(before[0], _r(op.xyz))).max() > 1e-3 \
                    or abs(before[1] - op.conf) > 1e-3:
                self._log(name, op.obj, before, (_r(op.xyz), op.conf), rule)
        elif isinstance(op, SetVis):
            o = self.objects[op.obj]
            before = o.visibility.value
            o.visibility = Belief(op.state, op.conf, op.t, "observed" if op.state == "visible" else "inferred",
                                  list(op.support))
            if before != op.state:
                self._log(name, op.obj, before, op.state, rule)
        elif isinstance(op, (SetDist, MixDist, FlattenDist)):
            self._apply_dist(op, rule)
        elif isinstance(op, AddEvidence):
            ev = Evidence(f"e{len(self.evidence) + 1}", op.t, op.kind, tuple(op.args), dict(op.data))
            self.evidence.append(ev)
            self._log(name, ev.id, None, {"kind": ev.kind, "args": list(ev.args), "data": ev.data}, rule)
            return [ev.id]
        elif isinstance(op, AddEvent):
            ev = Event(f"v{len(self.events) + 1}", op.t, op.type, tuple(op.args), op.conf, list(op.support),
                       op.source)
            self.events.append(ev)
            self._log(name, ev.id, None, {"type": ev.type, "args": list(ev.args), "conf": ev.conf,
                                          "support": ev.support}, rule)
            return [ev.id]
        elif isinstance(op, Retract):
            self.retract(op.ident)
        else:
            raise TypeError(f"unknown op {op!r}")
        return []

    def _apply_dist(self, op, rule: str) -> None:
        key = (op.pred, op.subj)
        old = self.relations.get(key)
        if isinstance(op, SetDist):
            new = op.dist
        elif old is None:
            return
        elif isinstance(op, MixDist):
            new = old.swap_mix(op.a, op.b, op.w, op.t, op.ev_id)
        else:
            new = old.flatten(op.f, op.plausible, op.t)
        self.relations[key] = new
        self._log(type(op).__name__, f"{op.pred}({op.subj})", old.as_dict() if old else None,
                  {"probs": new.as_dict(), "support": new.support}, rule)

    def retract(self, ident: str, _seen: set | None = None) -> None:
        """Downgrade everything whose support contains `ident`, transitively through events."""
        seen = _seen if _seen is not None else set()
        if ident in seen:
            return
        seen.add(ident)
        for o in self.objects.values():
            for attr in ("pose", "visibility"):
                b = getattr(o, attr)
                if ident in b.support:  # with no support left it is only a prior guess (keeps I4)
                    rest = [i for i in b.support if i != ident]
                    setattr(o, attr, Belief(b.value, b.conf * 0.5, b.t, "inferred" if rest else "prior", rest))
                    self._log("Retract", f"{o.id}.{attr}", b.conf, b.conf * 0.5, ident)
        for key, d in list(self.relations.items()):
            nd = d.drop(ident, d.t)
            if nd is not d:
                self.relations[key] = nd
                self._log("Retract", f"{key[0]}({key[1]})", d.as_dict(), nd.as_dict(), ident)
        for ev in self.events:
            if ident in ev.support:
                ev.support = [i for i in ev.support if i != ident]
                ev.conf *= 0.5
                self._log("Retract", ev.id, None, {"conf": ev.conf}, ident)
                self.retract(ev.id, seen)

    def _log(self, op, target, before, after, rule) -> None:
        self.log.append({"op": op, "target": target, "before": before, "after": after, "rule": rule})

    def pop_log(self) -> list[dict]:
        out, self.log = self.log, []
        return out

    # ---- reads ---------------------------------------------------------------------------
    def inside(self, cube: str) -> Distribution | None:
        return self.relations.get(("Inside", cube))

    def count(self, event_type: str, obj: str) -> int:
        """Derived from the event log, never stored (invariant I3)."""
        return sum(1 for e in self.events if e.type == event_type and obj in e.args and e.conf >= 0.5)

    def known_ids(self) -> set[str]:
        return {e.id for e in self.evidence} | {e.id for e in self.events}

    def check(self) -> None:
        """Invariants I1, I2, I4, I5 (DEVELOPMENT.md §3.2). Raises AssertionError."""
        known = self.known_ids()
        for key, d in self.relations.items():
            assert abs(sum(d.probs.values()) - 1) < 1e-9, f"I1 {key} sums to {sum(d.probs.values())}"
            assert all(isinstance(k, str) for k in key), f"I2 non-string key {key}"
        items = [(o.id, o.pose) for o in self.objects.values()] + [(o.id, o.visibility) for o in self.objects.values()]
        for oid, b in items:
            if b.source in ("inferred", "arbiter") and not (b.value is None or isinstance(b.value, str)
                                                           and b.value == "unknown"):
                assert b.support, f"I4 {oid} {b.source} belief without support"
            assert b.source != "arbiter" or not isinstance(b.value, np.ndarray), f"I5 arbiter wrote a pose for {oid}"
        for e in self.events:
            assert e.support or e.conf < 0.5, f"I4 event {e.id} without support"
        for ident in self._all_support_ids():
            assert _ID.match(ident), f"I4 malformed support id {ident!r}"
            assert not ident[0] in "ev" or ident in known, f"I4 unknown support id {ident}"

    def _all_support_ids(self) -> set[str]:
        ids = set()
        for o in self.objects.values():
            ids |= set(o.pose.support) | set(o.visibility.support)
        for d in self.relations.values():
            ids |= set(d.support) | {i for v in d.hyp_support.values() for i in v}
        for e in self.events:
            ids |= set(e.support)
        return ids

    def snapshot(self) -> dict:
        return {
            "objects": {o.id: {"pose": _r(o.pose.value), "conf": o.pose.conf, "vis": o.visibility.value,
                               "t_last_seen": o.t_last_seen} for o in self.objects.values()},
            "relations": {f"{p}({s})": d.as_dict() for (p, s), d in self.relations.items()},
            "n_evidence": len(self.evidence), "n_events": len(self.events),
        }
