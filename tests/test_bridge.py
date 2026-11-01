"""The ledger/KB bridge keeps every object type MEMI's observation names (orchestrator, after G0).

Uses the real MEMI package when it is importable (inside the RoboMME image); otherwise a stub
with the few MEMI names the bridge calls, copied from MEMI bb6c37c (memi/plan/rollout/scene.py,
segments.py), so the bridge logic is still checked on the Mac.
"""
import importlib.util
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from arbiter_kb.common_observation import Detection, VisualTracks


def _stub_memi():
    scene = types.ModuleType("memi.plan.rollout.scene")

    def classify_object_type(name, task):
        name, task = str(name).lower(), (task or "").lower()
        if "button_cap" in name:
            return "button"
        if "button_base" in name:
            return None
        if "cube" in name:
            return next((f"{c} cube" for c in ("red", "blue", "green") if c in name), "cube")
        if "target" in name:
            return "target"
        if "bin" in name:
            return "container"
        if "position_highlight" in name:
            return "waypoint" if task in {"routestick", "patternlock"} else None
        if "highlight_disk" in name:
            return "highlighted target" if task in {"routestick", "patternlock"} else "highlighted area"
        return None

    scene.classify_object_type = classify_object_type
    scene.REDUCE_TO_FRONT = {"waypoint"}
    scene._actor_index = lambda n: int(m.group(1)) if (m := re.search(r"(\d+)\s*$", str(n))) else -1
    segments = types.ModuleType("memi.plan.rollout.segments")
    segments._RENAMED = {"VideoRepick": (re.compile(r"^bin_"), "cube_")}
    return {"memi": types.ModuleType("memi"), "memi.plan": types.ModuleType("memi.plan"),
            "memi.plan.rollout": types.ModuleType("memi.plan.rollout"),
            "memi.plan.rollout.scene": scene, "memi.plan.rollout.segments": segments}


@pytest.fixture
def scene(monkeypatch):
    if importlib.util.find_spec("memi") is None:
        for name, module in _stub_memi().items():
            monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "common_scene", Path(__file__).resolve().parents[1] / "stress/common_scene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rec(name, yx, xyz=(0.0, 0.0, 0.0)):
    return {"actor_name": name, "visible_centroid_yx": list(yx), "actor_pose": {"xyz": list(xyz)}}


def test_bridge_keeps_buttons_targets_and_highlights(scene):
    row = {"visible_object_records": [rec("button_cap_0", (56, 138)), rec("button_base_0", (58, 138)),
                                      rec("cube_red_0", (90, 60)), rec("target_0", (94, 84)),
                                      rec("highlight_disk_0", (100, 100))]}
    got = sorted((d.label, d.cls) for d in scene.detections(row, "SwingXtimes"))
    assert got == [("button", "button"), ("highlighted area", "highlighted area"),
                   ("red cube", "cube"), ("target", "target")]


def test_bridge_keeps_only_the_front_waypoint_as_memi_does(scene):
    row = {"visible_object_records": [rec("position_highlight_3", (10, 10)), rec("position_highlight_7", (20, 20)),
                                      rec("highlight_disk_1", (30, 30))]}
    got = sorted((d.label, tuple(d.uv)) for d in scene.detections(row, "RouteStick"))
    assert got == [("highlighted target", (30.0, 30.0)), ("waypoint", (20.0, 20.0))]


def test_bridge_follows_memis_videorepick_rename(scene):
    row = {"visible_object_records": [rec("bin_0", (78, 117))]}
    assert [(d.label, d.cls) for d in scene.detections(row, "VideoRepick")] == [("cube", "cube")]


def test_tracks_name_non_core_classes_by_label():
    def det(label, cls, x):
        return Detection(label, cls, np.array([x, 0.0, 0.0]), np.array([x * 100, 5.0]))

    tracks = VisualTracks([[det("button", "button", 0.1), det("highlighted target", "highlighted target", 0.2),
                            det("target", "target", 0.3), det("target", "target", 0.4)]])
    assert sorted(tracks.objects()) == [("button_0", "button", {}),
                                        ("highlighted_target_0", "highlighted target", {}),
                                        ("target_0", "target", {}), ("target_1", "target", {})]
    frame = tracks.step([det("button", "button", 0.1)], (0, 0, 0), 0.08, False)
    assert frame.objects["button_0"].visible and not frame.objects["target_0"].visible
