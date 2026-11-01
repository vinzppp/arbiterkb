"""Leakage-safe structured observations for the RoboMME common harness.

The adapter sees semantic labels, visible centroids and poses, but never simulator
actor identities.  Repeated objects are associated only by geometric continuity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import EpisodeMeta, FrameObs, ObjObs

CORE_CLASSES = {"container", "cube", "target"}  # classes the frozen KB rules reason about; others are kept as-is
STATIC_CLASSES = {"target", "button"}  # never move in RoboMME (MEMI scene.py: "Targets do not move")
STATIC_GATE_PX = 15.0  # growing tracks: a static track never takes a detection farther than this from where it was


@dataclass(frozen=True)
class Detection:
    label: str
    cls: str
    xyz: np.ndarray
    uv: np.ndarray
    quality: float = 1.0


def _base_id(label: str, cls: str) -> str:
    words = label.lower().split()
    if cls == "container":
        return "bin"
    if cls == "target":
        return "target"
    if cls not in CORE_CLASSES:
        return "_".join(words)
    color = next((x for x in ("red", "blue", "green") if x in words), None)
    return f"cube_{color}" if color else "cube"


def _groups(samples: list[list[Detection]], expected: list[tuple[str, str]] | None = None) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for key in expected or []:
        counts[key] = max(counts.get(key, 0), 1)
    for detections in samples:
        now: dict[tuple[str, str], int] = {}
        for d in detections:
            key = (d.label, d.cls)
            now[key] = now.get(key, 0) + 1
        for key, n in now.items():
            counts[key] = max(counts.get(key, 0), n)
    return counts


def _ids(samples: list[list[Detection]], expected: list[tuple[str, str]] | None = None) -> dict[tuple[str, str], list[str]]:
    counts, used, out = _groups(samples, expected), set(), {}
    for key, n in sorted(counts.items()):
        base = _base_id(*key)
        names = [base] if n == 1 and base.startswith("cube_") else [f"{base}_{i}" for i in range(n)]
        names = [x if x not in used else f"{x}_{len(used)}" for x in names]
        used.update(names)
        out[key] = names
    return out


def _distance(track: ObjObs | None, detection: Detection) -> float:
    if track is None or track.xyz is None or track.uv is None:
        return 1e9
    return float(np.linalg.norm(track.uv - detection.uv) + 50 * np.linalg.norm(track.xyz - detection.xyz))


def _static_cost(track: ObjObs | None, detection: Detection) -> float:
    """A static object's last seen place is where it still is: far from it, the detection is another object."""
    if track is None or track.uv is None:
        return 1e9
    if float(np.linalg.norm(track.uv - detection.uv)) > STATIC_GATE_PX:
        return 2e9
    return _distance(track, detection)


def _match(names: list[str], previous: dict[str, ObjObs], detections: list[Detection],
           last: dict[str, ObjObs] | None = None) -> dict[str, Detection]:
    """Greedy nearest assignment. With `last` (growing tracks, static class), costs come from each track's last
    sighting and a jump beyond STATIC_GATE_PX costs more than starting a fresh track."""
    cost = (lambda name, d: _static_cost(last.get(name), d)) if last is not None else (
        lambda name, d: _distance(previous.get(name), d))
    pairs = sorted((cost(name, d), name, i) for name in names for i, d in enumerate(detections))
    assigned, seen_names, seen_detections = {}, set(), set()
    for _, name, i in pairs:
        if name not in seen_names and i not in seen_detections:
            assigned[name], seen_names, seen_detections = detections[i], seen_names | {name}, seen_detections | {i}
    return assigned


class VisualTracks:
    """Nearest-neighbour visual tracks. The vocabulary is fixed from the reset samples unless `grow`, in which
    case a type first seen later gets new tracks, and so does an extra copy of a non-core type."""

    def __init__(self, samples: list[list[Detection]], expected: list[tuple[str, str]] | None = None,
                 grow: bool = False):
        self.by_group, self.previous, self.t, self.grow = _ids(samples, expected), {}, 0, grow
        self.last: dict[str, ObjObs] = {}

    def _extend(self, grouped: dict[tuple[str, str], list[Detection]]) -> None:
        used = {name for names in self.by_group.values() for name in names}
        for key, found in sorted(grouped.items()):
            names, base = self.by_group.setdefault(key, []), _base_id(*key)
            if key[1] in STATIC_CLASSES:
                self._extend_static(names, found, base, used)
            if names and key[1] in CORE_CLASSES:
                continue
            i = len(names)
            while len(names) < len(found):
                while f"{base}_{i}" in used:
                    i += 1
                names.append(f"{base}_{i}")
                used.add(names[-1])

    def _extend_static(self, names: list[str], found: list[Detection], base: str, used: set) -> None:
        """Enough tracks for every static detection that is far from all places this type was seen."""
        spots = [self.last[n].uv for n in names if n in self.last]
        fresh = [n for n in names if n not in self.last]
        far = sum(all(float(np.linalg.norm(d.uv - s)) > STATIC_GATE_PX for s in spots) for d in found)
        i = len(names)
        while len(fresh) < far:
            while f"{base}_{i}" in used:
                i += 1
            names.append(f"{base}_{i}")
            used.add(names[-1])
            fresh.append(names[-1])

    def objects(self) -> list[tuple[str, str, dict]]:
        out = []
        for (label, cls), names in self.by_group.items():
            attrs = {"color": label.split()[0]} if cls == "cube" and label.split()[0] in {"red", "blue", "green"} else {}
            out += [(name, cls, attrs) for name in names]
        return out

    def meta(self, task: str, episode: int, split: str, goal: str) -> EpisodeMeta:
        return EpisodeMeta("robomme_common", episode, split, goal, self.objects(), None, None)

    def step(self, detections: list[Detection], ee_xyz, gripper_width: float, is_video: bool) -> FrameObs:
        grouped: dict[tuple[str, str], list[Detection]] = {}
        for d in detections:
            grouped.setdefault((d.label, d.cls), []).append(d)
        if self.grow:
            self._extend(grouped)
        visible = {name: d for key, names in self.by_group.items()
                   for name, d in _match(names, self.previous, grouped.get(key, []),
                                         self.last if self.grow and key[1] in STATIC_CLASSES else None).items()}
        objects = self._objects(visible)
        self.last.update({name: obj for name, obj in objects.items() if obj.visible})
        frame = FrameObs(self.t, objects, np.asarray(ee_xyz, float), float(gripper_width), is_video)
        self.previous, self.t = objects, self.t + 1
        return frame

    def _objects(self, visible: dict[str, Detection]) -> dict[str, ObjObs]:
        classes = {name: cls for (_, cls), names in self.by_group.items() for name in names}
        out = {}
        for name, cls in classes.items():
            d = visible.get(name)
            out[name] = ObjObs(name, cls, d is not None, None if d is None else d.xyz.copy(),
                               None if d is None else d.uv.copy(), 0.0 if d is None else d.quality)
        return out


def scene_digest(samples: list[list[Detection]]) -> str:
    """Hashable canonical scene content, deliberately excluding hidden actor ids."""
    import hashlib
    import json

    rows = [[(d.label, d.cls, np.round(d.xyz, 4).tolist(), np.round(d.uv, 1).tolist())
             for d in sorted(ds, key=lambda x: (x.label, *x.uv))] for ds in samples]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
