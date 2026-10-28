"""GuardedBound baselines on hand-made frames: which passes / jumps trigger the haircut, and the tie rule."""
import numpy as np
import pytest

from arbiter_kb.systems.baselines import GuardedBound
from arbiter_kb.types import EpisodeMeta, FrameObs, ObjObs

OBJS = [("bin_a", "container", {}), ("bin_b", "container", {}), ("cube_g", "cube", {"color": "g"})]
META = EpisodeMeta("hand", 0, "train", "pick up the container that hides the g cube", OBJS, None, None)


def frame(t, a, b, cube=None):
    o = {"bin_a": ObjObs("bin_a", "container", a is not None, None if a is None else np.array([*a, 0.03]), None, 1.0),
         "bin_b": ObjObs("bin_b", "container", b is not None, None if b is None else np.array([*b, 0.03]), None, 1.0),
         "cube_g": ObjObs("cube_g", "cube", cube is not None, None if cube is None else np.array([*cube, 0.017]), None, 1.0)}
    return FrameObs(t, o, np.zeros(3), 0.08, True)


def cover(t_end=10):
    """cube under bin_a's spot until t=5, then both containers appear (bin_a over the cube)."""
    return [frame(t, None, None, (0, 0)) for t in range(5)] + [frame(t, (0, 0), (0.1, 0)) for t in range(5, t_end)]


def swap(h, t0=10, n=10):
    """bin_a and bin_b exchange places on mirrored lanes y = ±h·sin(πα): closest 3D distance 2h."""
    out = []
    for k in range(n + 1):
        al = k / n
        y = h * np.sin(np.pi * al)
        out.append(frame(t0 + k, (0.1 * al, y), (0.1 - 0.1 * al, -y)))
    return out + [frame(t0 + n + 1 + k, (0.1, 0), (0, 0)) for k in range(3)]


def run(mode, frames):
    g = GuardedBound(mode)
    g.reset(META)
    for f in frames:
        g.step(f)
    return g


def test_bound_follows_the_true_container_through_the_pass():
    g = run("overlap_haircut", cover() + swap(0.025))
    assert g.inner.believe_inside("cube_g").argmax()[0] == "bin_a"


@pytest.mark.parametrize("h, fires", [(0.025, False), (0.01, True)])  # closest 3D distance 0.05 / 0.02 m
def test_overlap_haircut_fires_only_below_d_ov(h, fires):
    g = run("overlap_haircut", cover() + swap(h))
    d = g.believe_inside("cube_g")
    assert ("cube_g" in g.doubt) == fires
    assert d.argmax() == (("bin_a", 0.5) if fires else ("bin_a", 1.0))  # 50/50 keeps Bound's answer


def test_haircut_fires_on_any_pass_with_the_cubes_container():
    g = run("haircut", cover() + swap(0.025))
    assert g.doubt["cube_g"][:2] == ("bin_a", "bin_b")
    assert g.believe_inside("cube_g").probs == {"bin_a": 0.5, "bin_b": 0.5}


@pytest.mark.parametrize("step, fires", [(0.04, True), (0.02, False)])
def test_jump_detector_threshold(step, fires):
    frames = cover() + [frame(t, (step, 0), (0.1, 0)) for t in range(10, 13)]
    g = run("jump", frames)
    assert ("cube_g" in g.doubt) == fires


def test_stale_doubt_is_ignored_once_the_answer_leaves_the_pair():
    g = run("haircut", cover())
    g.doubt["cube_g"] = ("bin_x", "bin_y", 7)
    assert g.believe_inside("cube_g").probs == {"bin_a": 1.0}


def test_explain_looks_up_the_observations():
    from arbiter_kb.debug import explain
    g = run("overlap_haircut", cover() + swap(0.01))
    text = explain(g.trace_rows(), "cube_g")
    assert "o23:bin_a: observation bin_a" in text and "o4:cube_g: observation cube_g" in text
    assert "NOT FOUND" in explain(g.trace_rows(), "cube_missing")
