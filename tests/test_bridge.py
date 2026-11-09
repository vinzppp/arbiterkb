"""The ledger/KB bridge keeps every object type MEMI's observation names (orchestrator, after G0).

The bridge calls MEMI's own naming (classify_object_type, _covers, the VideoRepick rename), so these
tests need the real MEMI package and run inside the RoboMME image:
    docker run --rm --platform linux/amd64 --network none -v "$PWD:/workspace/ArbiterKB" \
      arbiterkb-robomme-cpu:bb6c37c python -m pytest -q tests/test_bridge.py
On the Mac (no MEMI) only the VisualTracks test runs.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from arbiter_kb.common_observation import Detection, VisualTracks

needs_memi = pytest.mark.skipif(importlib.util.find_spec("memi") is None, reason="MEMI is only in the RoboMME image")


@pytest.fixture
def scene():
    spec = importlib.util.spec_from_file_location(
        "common_scene", Path(__file__).resolve().parents[1] / "stress/common_scene.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rec(name, yx, xyz=(0.0, 0.0, 0.0)):
    return {"actor_name": name, "visible_centroid_yx": list(yx), "actor_pose": {"xyz": list(xyz)}}


@needs_memi
def test_memi_named_bridge_keeps_buttons_targets_and_highlights(scene):
    row = {"visible_object_records": [rec("button_cap_0", (56, 138)), rec("button_base_0", (58, 138)),
                                      rec("cube_red_0", (90, 60)), rec("target_0", (94, 84)),
                                      rec("highlight_disk_0", (150, 150))]}
    got = sorted((d.label, d.cls) for d in scene.detections(row, "SwingXtimes", "memi_named"))
    assert got == [("button", "button"), ("highlighted area", "highlighted area"),
                   ("red cube", "cube"), ("target", "target")]


@needs_memi
def test_highlight_over_a_target_is_named_as_memi_names_it(scene):
    targets = scene.target_spots([{"visible_object_records": [rec("target_0", (94, 86))]}], "SwingXtimes")
    row = {"visible_object_records": [rec("highlight_disk_0", (94, 86))]}
    assert [d.label for d in scene.detections(row, "SwingXtimes", "memi_named", targets)] == ["highlighted target"]


@needs_memi
def test_core_bridge_is_the_frozen_behaviour(scene):
    row = {"visible_object_records": [rec("button_cap_0", (56, 138)), rec("cube_red_0", (90, 60)),
                                      rec("target_0", (94, 84)), rec("bin_1", (20, 20)), rec("highlight_disk_0", (5, 5))]}
    got = sorted((d.label, d.cls) for d in scene.detections(row, "SwingXtimes"))
    assert got == [("container", "container"), ("red cube", "cube"), ("target", "target")]
    vr = {"visible_object_records": [rec("bin_0", (78, 117))]}
    assert [(d.label, d.cls) for d in scene.detections(vr, "VideoRepick")] == [("cube", "cube")]


@needs_memi
def test_memi_named_keeps_only_the_front_waypoint(scene):
    row = {"visible_object_records": [rec("position_highlight_3", (10, 10)), rec("position_highlight_7", (20, 20))]}
    got = [(d.label, tuple(d.uv)) for d in scene.detections(row, "RouteStick", "memi_named")]
    assert got == [("waypoint", (20.0, 20.0))]


@needs_memi
def test_type_first_seen_after_reset_gets_a_track(scene):
    reset = [{"visible_object_records": [rec("target_0", (94, 86), (0.1, 0.0, 0.0))]}]
    stream = scene.SceneStream("SwingXtimes", reset, "unit goal", 0, "val", "memi_named")
    later = {"visible_object_records": [rec("target_0", (94, 86), (0.1, 0.0, 0.0)),
                                        rec("highlight_disk_0", (94, 86), (0.1, 0.0, 0.0))]}
    frame = stream.feed([later])[0]
    assert frame.objects["highlighted_target_0"].visible
    core = scene.SceneStream("SwingXtimes", reset, "unit goal", 0, "val")
    assert set(core.feed([later])[0].objects) == {"target_0"}


def test_tracks_grow_only_when_asked():
    def det(label, cls, x):
        return Detection(label, cls, np.array([x, 0.0, 0.0]), np.array([x * 100, 5.0]))

    reset = [[det("target", "target", 0.3)]]
    fixed, grow = VisualTracks(reset), VisualTracks(reset, grow=True)
    later = [det("target", "target", 0.3), det("button", "button", 0.1), det("target", "target", 0.5)]
    assert set(fixed.step(later, (0, 0, 0), 0.08, False).objects) == {"target_0"}
    objects = grow.step(later, (0, 0, 0), 0.08, False).objects
    assert set(objects) == {"target_0", "target_1", "button_0"} and objects["button_0"].cls == "button"
    assert float(objects["target_0"].uv[0]) == 30.0 and float(objects["target_1"].uv[0]) == 50.0


def test_growing_static_tracks_do_not_jump_to_another_target():
    """RouteStick: 5 targets, at most 4 visible; when one is hidden its track must not take another target."""
    def tgt(x):
        return Detection("target", "target", np.array([x, 0.0, 0.01]), np.array([x * 1000, 50.0]))

    xs = [0.10, 0.15, 0.20, 0.25, 0.30]
    reset = [[tgt(x) for x in xs[:4]]]
    tracks = VisualTracks(reset, grow=True)
    first = tracks.step([tgt(x) for x in xs[:4]], (0, 0, 0), 0.08, True)
    at = {name: float(o.uv[0]) for name, o in first.objects.items()}
    later = tracks.step([tgt(x) for x in xs[1:]], (0, 0, 0), 0.08, True)
    for name, o in later.objects.items():
        if name in at and o.visible:
            assert abs(float(o.uv[0]) - at[name]) <= 15.0
    assert sum(o.visible for o in later.objects.values()) == 4 and len(later.objects) == 5
    hidden = [n for n, o in later.objects.items() if not o.visible]
    assert hidden == [n for n, x in at.items() if x == 100.0]


@needs_memi
def test_reset_highlight_seen_before_its_target_is_labelled_once(scene):
    reset = [{"visible_object_records": [rec("highlight_disk_0", (94, 86), (0.1, 0.0, 0.0))]},
             {"visible_object_records": [rec("target_0", (94, 86), (0.1, 0.0, 0.0))]}]
    stream = scene.SceneStream("RouteStick", reset, "unit goal", 0, "val", "memi_named")
    assert sorted(cls for _, cls, _ in stream.meta.objects) == ["highlighted target", "target"]
