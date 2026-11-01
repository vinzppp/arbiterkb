"""Deterministic memory arms of the "memory or policy?" study (main NOTES 7327cf1, "Arms").

M-ledger: task-agnostic, no LLM. Per object its last observed pixel and visibility; cover relations
from observation; the executed-option log with outcomes (execution phase only); counts of executed
options per type; demonstration events in a separate block labelled "demonstration". It holds no
recommendation and no task progress beyond counts of what was executed.

M-KB: the frozen ArbiterKB through the phase-safe adapter (demo-inclusive progress removed), with an
execution-only action count line in place of the removed progress line.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .detectors import Detectors
from .factorial_memory import NormalizedTaskState, phase_safe_arbiter_view
from .params import Params
from .systems.arbiter_kb import ArbiterKBSystem
from .types import EpisodeMeta, FrameObs
from .views import controller_view

COVER_RADIUS_M = 0.08
DEMO_KINDS = ("pick_up", "release", "press", "hold_above", "hand_pause")
# MEMI's MARKERS minus "target": drawn where the hand is or was, so never "the object the hand is at".
TRACE_CLASSES = {"waypoint", "highlighted target", "highlighted area"}


def _px(uv) -> str:
    return f"[{uv[1]:.0f}, {uv[0]:.0f}]"


def _lookup(frame: FrameObs | None) -> list[str]:
    return [f"  [{oid}] {obj.cls} at pixel {_px(obj.uv)}" for oid, obj in (frame.objects.items() if frame else [])
            if obj.visible and obj.uv is not None]


def _xyz(v) -> str:
    return "(" + ", ".join(f"{float(x):.2f}" for x in v) + ")"


class PreregLedger:
    """The preregistered deterministic ledger; the same code for every task."""

    def __init__(self, params: Params):
        self.params = params

    def reset(self, meta: EpisodeMeta) -> None:
        self.objects = {oid: {"cls": cls, "uv": None, "t": None, "xyz": None} for oid, cls, _ in meta.objects}
        self.detectors = {True: Detectors(self.params.detect), False: Detectors(self.params.detect)}
        self.frame: FrameObs | None = None
        self.inside: dict[str, str | None] = {}
        self.demo: list[str] = []
        self.log: list[str] = []
        self.counts: Counter = Counter()
        self.pending: tuple[str, str] | None = None

    def feed(self, frames: list[FrameObs]) -> None:
        for frame in frames:
            self._step(frame)

    def _step(self, frame: FrameObs) -> None:
        self.frame = frame
        for oid, obj in frame.objects.items():
            if obj.visible and obj.uv is not None:
                self.objects[oid].update(uv=np.asarray(obj.uv, float).copy(), t=frame.t,
                                         xyz=None if obj.xyz is None else np.asarray(obj.xyz, float).copy())
                if obj.cls == "cube" and self.inside.pop(oid, None) is not None and frame.is_video:
                    self.demo.append(f"t={frame.t} {oid} seen again")
        for event in self.detectors[bool(frame.is_video)].step(frame):
            if event.kind == "disappear":
                self._cover(event, frame)
            elif frame.is_video and event.kind in DEMO_KINDS:
                self.demo.append(self._demo_line(event, frame))

    def _cover(self, event, frame: FrameObs) -> None:
        cube = event.args[0]
        if self.objects.get(cube, {}).get("cls") != "cube":
            return
        last = np.asarray(event.data["last_xyz"], float)
        bins = [(float(np.linalg.norm(o.xyz[:2] - last[:2])), oid) for oid, o in frame.objects.items()
                if o.cls == "container" and o.visible and o.xyz is not None]
        best = min(bins) if bins else None
        self.inside[cube] = best[1] if best is not None and best[0] < COVER_RADIUS_M else None
        if frame.is_video:
            self.demo.append(f"t={frame.t} {cube} disappeared; "
                             + (f"covered by {self.inside[cube]}" if self.inside[cube] else "cover unknown"))

    def _demo_line(self, event, frame: FrameObs) -> str:
        ee = np.asarray(event.data.get("ee", frame.ee_xyz), float)
        near = [(float(np.linalg.norm(np.asarray(o["xyz"]) - ee)), oid) for oid, o in self.objects.items()
                if o["xyz"] is not None and o["cls"] not in TRACE_CLASSES]
        side = f" ({event.data['side']} side)" if "side" in event.data else ""
        where = f"; nearest object {min(near)[1]}" if near else ""
        return f"t={frame.t} {event.kind.replace('_', ' ')}{side}, hand at {_xyz(ee)}{where}"

    def chose(self, command: dict, menu: list) -> None:
        """Called with each command before it executes (choice and point as sent)."""
        label = str(command.get("choice"))
        option = next((o for o in menu or [] if str(o.get("label")) == label), {})
        point = command.get("point")
        self.pending = (label, f"{label}: {option.get('action', '?')}" + (f" at {list(point)}" if point else ""))

    def executed(self, result: str, status: str) -> None:
        """Called after the executor returns, with the harness's result and the environment's status."""
        if self.pending is None:
            return
        label, text = self.pending
        self.counts[text.split(" at [")[0]] += 1
        self.log.append(f"{len(self.log) + 1}. {text} -> {result}" + (f", status {status}" if status else ""))
        self.pending = None

    def text(self) -> str:
        lines = ["MEMORY: deterministic ledger (no probabilities; records only what was observed or executed)",
                 "Objects (last observed pixel [y, x]):"]
        for oid, o in self.objects.items():
            now = self.frame.objects.get(oid) if self.frame is not None else None
            if now is not None and now.visible and now.uv is not None:
                lines.append(f"  [{oid}] {o['cls']}: visible at pixel {_px(now.uv)}")
            elif o["uv"] is not None:
                lines.append(f"  [{oid}] {o['cls']}: not visible now; last seen t={o['t']} at pixel {_px(o['uv'])}")
            else:
                lines.append(f"  [{oid}] {o['cls']}: never observed")
        lines.append("Cover relations (observed; a covered cube moves with its container):")
        lines += [f"  [{c}] -> {self.inside[c] or 'unknown'}" for c in self.inside] or ["  none"]
        lines.append("DEMONSTRATION (events seen in the demonstration video; not executed by the robot):")
        lines += ["  " + x for x in self.demo] or ["  none"]
        lines.append("Executed options (execution phase only, oldest first):")
        lines += ["  " + x for x in self.log] or ["  none"]
        lines.append("Executed option counts:")
        lines += [f"  {k}: {n}" for k, n in self.counts.items()] or ["  none"]
        return "\n".join(lines)


class KBMemory:
    """Frozen ArbiterKB through the phase-safe adapter; execution-only action counts replace its progress line."""

    def __init__(self, params: Params):
        self.system, self.counts = ArbiterKBSystem(params), NormalizedTaskState(params)

    def reset(self, meta: EpisodeMeta) -> None:
        self.system.reset(meta)
        self.counts.reset(meta)
        self.frame: FrameObs | None = None

    def feed(self, frames: list[FrameObs]) -> None:
        for frame in frames:
            self.counts.step(frame)
            self.system.step(frame)
            self.frame = frame

    def chose(self, command: dict, menu: list) -> None:
        pass

    def executed(self, result: str, status: str) -> None:
        pass

    def text(self) -> str:
        c = self.counts.counts()
        head = ("EXECUTION COUNTS (execution phase only; the demonstration is not counted): "
                f"pick-ups {c['pick_up']}, releases {c['release']}, button presses {c['press']}, "
                f"holds {c['hold_above']}.")
        body = phase_safe_arbiter_view(controller_view(self.system.kb)).replace(
            "use the normalized state above", "use the execution counts above")
        return "\n".join([head, body, "Visible object-to-pixel lookup:", *_lookup(self.frame)])
