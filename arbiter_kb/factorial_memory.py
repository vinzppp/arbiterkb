"""Matched memory adapters for the RoboMME memory × policy factorial.

Both arms receive the same leakage-safe frames and the same normalized task-state
schema.  The ledger is deterministic and point-valued; the KB keeps the frozen
probabilistic rules.  Demonstration and execution events are detected separately.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .detectors import Detectors
from .params import Params
from .systems.arbiter_kb import ArbiterKBSystem
from .types import EpisodeMeta, FrameObs
from .views import controller_view


def _point(frame: FrameObs | None, oid: str) -> str:
    obj = frame.objects.get(oid) if frame is not None else None
    if obj is None or not obj.visible or obj.uv is None:
        return "not currently visible"
    return f"pixel [{obj.uv[1]:.0f}, {obj.uv[0]:.0f}]"


class NormalizedTaskState:
    """Task-neutral state exposed with an identical schema to both memory arms."""

    def __init__(self, params: Params):
        self.params = params

    def reset(self, meta: EpisodeMeta) -> None:
        self.meta, self.frame = meta, None
        self.demo_detector = Detectors(self.params.detect)
        self.exec_detector = Detectors(self.params.detect)
        self.demo_target: str | None = None
        self.exec_events = []
        self.last_xyz: dict[str, np.ndarray] = {}

    def step(self, frame: FrameObs) -> list:
        self.frame = frame
        for oid, obj in frame.objects.items():
            if obj.visible and obj.xyz is not None:
                self.last_xyz[oid] = np.asarray(obj.xyz, float).copy()
        detector = self.demo_detector if frame.is_video else self.exec_detector
        events = detector.step(frame)
        if frame.is_video:
            self._remember_demo_target(events, frame)
        else:
            self.exec_events.extend(e for e in events if e.kind in {"pick_up", "release", "press", "hold_above"})
        return events

    def _remember_demo_target(self, events: list, frame: FrameObs) -> None:
        for event in events:
            if event.kind != "grip_close":
                continue
            at = np.asarray(event.data.get("ee", frame.ee_xyz), float)
            candidates = []
            for oid, obj in frame.objects.items():
                xyz = obj.xyz if obj.visible and obj.xyz is not None else self.last_xyz.get(oid)
                if obj.cls == "cube" and xyz is not None:
                    candidates.append((float(np.linalg.norm(np.asarray(xyz) - at)), oid))
            if candidates and min(candidates)[0] < self.params.rule.r_grasp:
                self.demo_target = min(candidates)[1]

    def counts(self) -> dict[str, int]:
        c = Counter(e.kind for e in self.exec_events)
        return {name: c[name] for name in ("pick_up", "release", "press", "hold_above")}

    def text(self) -> str:
        c = self.counts()
        target = self.demo_target or "unknown"
        return "\n".join([
            "NORMALIZED TASK STATE (identical schema in both memory arms)",
            "Phase separation: demonstration events are excluded from execution progress.",
            f"Demonstration-grasped cube: {target} ({_point(self.frame, target) if self.demo_target else 'unknown'}).",
            ("Execution progress: "
             f"pick-ups {c['pick_up']}, releases {c['release']}, button presses {c['press']}, "
             f"holds {c['hold_above']}."),
        ])

    def audit(self) -> dict:
        return {"demo_target": self.demo_target, "execution_counts": self.counts()}


class DeterministicWorldLedger:
    """Minimal cover-following object ledger; no probabilities or retraction."""

    COVER_RADIUS_M = 0.08

    def reset(self, meta: EpisodeMeta) -> None:
        self.meta, self.frame = meta, None
        self.objects = {oid: {"class": cls, "attributes": dict(attrs), "last_xyz": None}
                        for oid, cls, attrs in meta.objects}
        self.inside: dict[str, str | None] = {}
        self.events: list[str] = []

    def step(self, frame: FrameObs, detected: list) -> None:
        self.frame = frame
        for oid, obj in frame.objects.items():
            if obj.visible and obj.xyz is not None:
                self.objects[oid]["last_xyz"] = np.asarray(obj.xyz, float).copy()
                if obj.cls == "cube" and oid in self.inside:
                    self.inside.pop(oid, None)
                    self.events.append(f"t={frame.t} {oid} became visible; cover relation cleared")
        for event in detected:
            if event.kind != "disappear" or self.objects[event.args[0]]["class"] != "cube":
                continue
            cube, last = event.args[0], np.asarray(event.data["last_xyz"], float)
            bins = [(float(np.linalg.norm(obj.xyz[:2] - last[:2])), oid)
                    for oid, obj in frame.objects.items()
                    if obj.cls == "container" and obj.visible and obj.xyz is not None]
            best = min(bins) if bins else None
            cover = best[1] if best is not None and best[0] < self.COVER_RADIUS_M else None
            self.inside[cube] = cover
            relation = f"covered by {cover}" if cover else "cover unknown"
            self.events.append(f"t={frame.t} {cube} disappeared; {relation}")

    def text(self) -> str:
        lines = ["MEMORY REPRESENTATION: deterministic ledger (no probabilities, calibration, or retraction)",
                 "Objects with persistent visual-track identifiers:"]
        for oid, record in self.objects.items():
            obj = self.frame.objects.get(oid) if self.frame is not None else None
            if obj is not None and obj.visible and obj.xyz is not None:
                xyz = ", ".join(f"{float(v):.3f}" for v in obj.xyz)
                state = f"visible at {xyz}; {_point(self.frame, oid)}"
            elif record["last_xyz"] is not None:
                state = "hidden/not visible; last at " + ", ".join(f"{float(v):.3f}" for v in record["last_xyz"])
            else:
                state = "unknown; never observed"
            lines.append(f"  [{oid}] {record['class']}: {state}")
        lines.append("Deterministic cover ledger (the relation follows the container track):")
        cubes = [oid for oid, record in self.objects.items() if record["class"] == "cube"]
        lines.extend(f"  [{cube}] -> {self.inside.get(cube) or 'unknown'}" for cube in cubes)
        lines.append("Relevant event history:")
        lines.extend("  " + event for event in self.events[-30:])
        return "\n".join(lines)

    def beliefs(self) -> list[dict]:
        return [{"cube": cube, "answer": self.inside.get(cube), "p_max": 1.0,
                 "probs": {self.inside.get(cube): 1.0}}
                for cube, record in self.objects.items() if record["class"] == "cube"]


class FactorialMemory:
    """Common interface used by ledger_* and arbiter_* factorial conditions."""

    def __init__(self, kind: str, params: Params):
        if kind not in {"ledger", "arbiter"}:
            raise ValueError(kind)
        self.kind, self.params = kind, params
        self.world = DeterministicWorldLedger() if kind == "ledger" else ArbiterKBSystem(params)
        self.task_state = NormalizedTaskState(params)
        self.frame: FrameObs | None = None

    def reset(self, meta: EpisodeMeta) -> None:
        self.task_state.reset(meta)
        self.world.reset(meta)
        self.frame = None

    def feed(self, frames: list[FrameObs]) -> None:
        for frame in frames:
            detected = self.task_state.step(frame)
            if self.kind == "ledger":
                self.world.step(frame, detected)
            else:
                self.world.step(frame)
            self.frame = frame

    def text(self) -> str:
        body = self.world.text() if self.kind == "ledger" else controller_view(self.world.kb)
        points = [f"  [{oid}] {obj.cls} at pixel [{obj.uv[1]:.0f}, {obj.uv[0]:.0f}]"
                  for oid, obj in (self.frame.objects.items() if self.frame else [])
                  if obj.visible and obj.uv is not None]
        return "\n".join([self.task_state.text(), body, "Visible object-to-pixel lookup:", *points])

    def beliefs(self) -> list[dict]:
        if self.kind == "ledger":
            return self.world.beliefs()
        rows = []
        for oid, obj in self.world.kb.objects.items():
            if obj.cls != "cube":
                continue
            distribution = self.world.believe_inside(oid)
            answer = distribution.argmax()
            rows.append({"cube": oid, "answer": answer[0], "p_max": answer[1],
                         "probs": distribution.probs})
        return rows

    def audit(self) -> dict:
        return {"memory_kind": self.kind, "task_state": self.task_state.audit()}
