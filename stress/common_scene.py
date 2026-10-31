"""Bridge MEMI's captured simulator rows to ArbiterKB's leakage-safe tracks."""
from __future__ import annotations

import copy
import re

import numpy as np

from arbiter_kb.common_observation import Detection, VisualTracks, scene_digest


def _label(record: dict, task: str) -> tuple[str, str] | None:
    from memi.plan.rollout.scene import classify_object_type

    name = record.get("actor_name", "")
    if task == "VideoRepick" and str(name).startswith("bin_"):
        name = "cube_" + str(name)[4:]
    label = classify_object_type(name, task)
    if label == "container":
        return label, "container"
    if label == "target":
        return label, "target"
    if label and (label.endswith("cube") or label == "cube"):
        return label, "cube"
    return None


def detections(row: dict, task: str) -> list[Detection]:
    out = []
    for record in row.get("visible_object_records") or []:
        kind = _label(record, task)
        xyz = (record.get("actor_pose") or {}).get("xyz")
        yx = record.get("visible_centroid_yx")
        if kind and xyz and len(xyz) >= 3 and yx and len(yx) >= 2:
            out.append(Detection(kind[0], kind[1], np.asarray(xyz[:3], float),
                                 np.asarray([yx[1], yx[0]], float)))
    return out


def degraded(rows: list[dict], mode: str, seed: int, dropout: float, jitter_px: float) -> list[dict]:
    if mode == "clean":
        return copy.deepcopy(rows)
    out = copy.deepcopy(rows)
    rng = np.random.default_rng(seed)
    for row in out:
        records = row.get("visible_object_records") or []
        if mode == "dropout":
            row["visible_object_records"] = [r for r in records if rng.random() >= dropout]
        elif mode == "jitter":
            for record in records:
                yx = record.get("visible_centroid_yx")
                if yx and len(yx) >= 2:
                    record["visible_centroid_yx"] = (np.asarray(yx) + rng.normal(0, jitter_px, 2)).round().tolist()
        else:
            raise ValueError(f"unknown observation mode {mode!r}")
    return out


class SceneStream:
    def __init__(self, task: str, reset_rows: list[dict], goal: str, episode: int, split: str):
        self.task = task
        samples = [detections(row, task) for row in reset_rows]
        expected = [(f"{color} cube", "cube") for color in ("red", "blue", "green")
                    if re.search(rf"\b{color}\s+(?:cube|block)\b", goal.lower())]
        self.tracks = VisualTracks(samples, expected)
        self.meta = self.tracks.meta(task, episode, split, goal)
        self.digest = scene_digest(samples)
        self.raw_to_track, self.inside, self.previous_raw = {}, {}, {}

    def feed(self, rows: list[dict]) -> list:
        frames = []
        for row in rows:
            arm = row.get("arm") or {}
            ds = detections(row, self.task)
            self._cover_update(row)
            frame = self.tracks.step(ds, arm.get("xyz", [0, 0, 0]), arm.get("opening", .08),
                                     bool(row.get("is_video_demo")))
            self._map_raw(row, frame)
            frames.append(frame)
        return frames

    def _records(self, row: dict) -> list[tuple[dict, Detection]]:
        out = []
        for record in row.get("visible_object_records") or []:
            one = detections({"visible_object_records": [record]}, self.task)
            if one:
                out.append((record, one[0]))
        return out

    def _cover_update(self, row: dict) -> None:
        now = {str(r.get("actor_name")): (r, d) for r, d in self._records(row)}
        for raw, (_, cube) in self.previous_raw.items():
            if cube.cls != "cube" or raw in now:
                continue
            bins = [(name, d) for name, (_, d) in self.previous_raw.items() if d.cls == "container"]
            if bins:
                best, distance = min(((name, float(np.linalg.norm(d.xyz[:2] - cube.xyz[:2]))) for name, d in bins),
                                     key=lambda x: x[1])
                if distance < .08:
                    self.inside[raw] = best
        self.previous_raw = now

    def _map_raw(self, row: dict, frame) -> None:
        for record, d in self._records(row):
            candidates = [(float(np.linalg.norm(o.uv - d.uv)), name) for name, o in frame.objects.items()
                          if o.visible and o.cls == d.cls and o.uv is not None]
            if candidates:
                self.raw_to_track[str(record.get("actor_name"))] = min(candidates)[1]

    def score_beliefs(self, beliefs: list[dict]) -> list[dict]:
        from arbiter_kb.metrics import brier

        out = []
        cube_raw = {self.raw_to_track.get(raw): raw for raw in self.inside}
        bins = [name for name, cls, _ in self.meta.objects if cls == "container"]
        for row in beliefs:
            raw = cube_raw.get(row["cube"])
            truth = self.raw_to_track.get(self.inside.get(raw, ""))
            if truth is None:
                continue
            out.append({**row, "truth": truth, "correct": row["answer"] == truth,
                        "brier": brier(row["probs"], truth, bins)})
        return out
