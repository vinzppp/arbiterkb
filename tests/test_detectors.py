import numpy as np

from arbiter_kb.detectors import Detectors
from arbiter_kb.params import DetectParams
from arbiter_kb.types import FrameObs, ObjObs


def f(t, a, b, vis_cube=True):
    objs = {"bin_a": ObjObs("bin_a", "container", True, np.array([*a, 0.03]), None, 1.0),
            "bin_b": ObjObs("bin_b", "container", True, np.array([*b, 0.03]), None, 1.0),
            "cube_g": ObjObs("cube_g", "cube", vis_cube, np.array([0, 0, 0.017]) if vis_cube else None, None, 1.0)}
    return FrameObs(t, objs, np.zeros(3), 0.08, True)


def run(frames):
    det, out = Detectors(DetectParams()), []
    for fr in frames:
        out += det.step(fr)
    return out


def test_disappear_carries_last_position_and_observation():
    ev = run([f(0, (0, 0), (1, 1)), f(1, (0, 0), (1, 1), vis_cube=False)])
    d = [e for e in ev if e.kind == "disappear"]
    assert d and d[0].data["obs"] == ["o0:cube_g"] and d[0].data["last_xyz"][2] == 0.017


def test_near_pass_needs_both_moving():
    # bin_a moves past a STILL bin_b at 1 cm: no near-pass (3D threshold 0.10 m, no camera)
    frames = [f(t, (0.01 * t, 0), (0.05, 0.01)) for t in range(10)] + [f(10, (0.09, 0), (0.05, 0.01))]
    assert not [e for e in run(frames) if e.kind == "near_pass"]


def test_near_pass_emitted_with_min_distance_when_pass_ends():
    frames = [f(t, (0.01 * t, 0), (0.1 - 0.01 * t, 0.02)) for t in range(11)]
    frames += [f(11 + k, (0.1, 0), (0.0, 0.02)) for k in range(2)]  # both stop -> pass closes
    np_ev = [e for e in run(frames) if e.kind == "near_pass"]
    assert len(np_ev) == 1
    assert np_ev[0].data["min_d"] < 0.03 and 0 <= np_ev[0].data["closeness"] < 1
