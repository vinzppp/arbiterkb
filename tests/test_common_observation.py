import numpy as np

from arbiter_kb.common_memory import DeterministicMemory
from arbiter_kb.common_observation import Detection, VisualTracks, scene_digest
from arbiter_kb.params import Params


def d(label, cls, xyz, uv):
    return Detection(label, cls, np.array(xyz, float), np.array(uv, float))


def test_visual_tracks_use_geometry_not_source_identity():
    first = [d("container", "container", [0, 0, .03], [20, 50]),
             d("container", "container", [.2, 0, .03], [200, 50]),
             d("red cube", "cube", [0, 0, .02], [20, 50])]
    bank = VisualTracks([first])
    f0 = bank.step(first, [0, 0, .2], .08, True)
    second = [d("container", "container", [.19, 0, .03], [190, 50]),
              d("container", "container", [.01, 0, .03], [30, 50])]
    f1 = bank.step(second, [0, 0, .2], .08, True)
    assert f0.objects["bin_0"].uv[0] == 20
    assert f1.objects["bin_0"].uv[0] == 30
    assert not f1.objects["cube_red"].visible


def test_scene_digest_ignores_input_order():
    a = d("container", "container", [0, 0, 0], [10, 20])
    b = d("red cube", "cube", [1, 0, 0], [30, 40])
    assert scene_digest([[a, b]]) == scene_digest([[b, a]])


def test_goal_declared_hidden_cube_gets_anonymous_track():
    tracks = VisualTracks([[]], [("blue cube", "cube")])
    assert ("cube_blue", "cube", {"color": "blue"}) in tracks.objects()


def test_deterministic_memory_arms_share_frames():
    sample = [d("container", "container", [0, 0, .03], [20, 50]),
              d("red cube", "cube", [0, 0, .02], [20, 50])]
    bank = VisualTracks([sample])
    meta = bank.meta("VideoUnmaskSwap", 0, "val", "pick the red cube's container")
    frames = [bank.step(sample, [0, 0, .2], .08, True), bank.step(sample[:1], [0, 0, .2], .08, True)]
    for arm in ("soft", "arbiter"):
        memory = DeterministicMemory(arm, Params())
        memory.reset(meta)
        memory.feed(frames)
        assert "bin_0" in memory.text()
        assert memory.beliefs()[0]["cube"] == "cube_red"
