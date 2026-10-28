"""M3: R3 grasp / place, geometric relations, the event log and the controller view, on hand-made frames."""
import numpy as np

from arbiter_kb.params import Params
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem
from arbiter_kb.types import EpisodeMeta, FrameObs, ObjObs
from arbiter_kb.views import ask, controller_view, event_log, facts, geometric

OBJS = [("bin_a", "container", {}), ("cube_r", "cube", {"color": "red"}), ("cube_b", "cube", {"color": "blue"})]
META = EpisodeMeta("hand", 0, "train", "put the red cube on the container", OBJS, None, None)
BIN, BLUE = np.array([0.1, 0.1, 0.03]), np.array([0.0, -0.1, 0.017])


def frame(t, ee, width, red, red_vis=True):
    o = {"bin_a": ObjObs("bin_a", "container", True, BIN.copy(), None, 1.0),
         "cube_r": ObjObs("cube_r", "cube", red_vis, np.asarray(red, float) if red_vis else None, None, 1.0),
         "cube_b": ObjObs("cube_b", "cube", True, BLUE.copy(), None, 1.0)}
    return FrameObs(t, o, np.asarray(ee, float), width, False)


def pick_place(hide_in_hand=False):
    """ee goes to the red cube at (0, 0.1), closes, carries it onto bin_a, opens."""
    red0 = np.array([0.0, 0.1, 0.017])
    fs = [frame(t, [0, 0.1, 0.2 - 0.03 * t], 0.08, red0) for t in range(6)]
    fs.append(frame(6, [0, 0.1, 0.02], 0.04, red0))  # close on the cube
    for k in range(1, 11):  # carry: the cube rides in the fingers
        ee = np.array([0.01 * k, 0.1, 0.02 + 0.006 * k])
        fs.append(frame(6 + k, ee, 0.04, ee, red_vis=not hide_in_hand))
    top = BIN + [0, 0, 0.03]
    fs.append(frame(17, top, 0.04, top, red_vis=not hide_in_hand))
    fs.append(frame(18, top + [0, 0, 0.05], 0.08, top))  # open above bin_a, the cube stays on it
    return fs


def run(frames):
    s = ArbiterKBSystem(Params())
    s.reset(META)
    for f in frames:
        s.step(f)
    return s.kb


def test_grasp_then_place_on_the_object_below():
    kb = run(pick_place())
    ev = [(e.type, e.args) for e in kb.events]
    assert ("grasp", ("cube_r",)) in ev and ("place", ("cube_r", "bin_a")) in ev
    assert kb.count("grasp", "cube_r") == 1
    assert [e.kind for e in kb.evidence if e.kind.startswith("grip")] == ["grip_close", "grip_open"]


def test_held_object_out_of_sight_follows_the_end_effector_and_is_not_covered():
    s = ArbiterKBSystem(Params())
    s.reset(META)
    fs = pick_place(hide_in_hand=True)
    for f in fs[:12]:
        s.step(f)
    o = s.kb.objects["cube_r"]
    assert np.allclose(o.pose.value, fs[11].ee_xyz) and o.pose.source == "inferred"
    assert s.kb.inside("cube_r") is None  # vanishing in the fingers is not a cover
    assert (("Held", "cube_r"), 1.0) in facts(s.kb)


def test_geometric_relations():
    kb = run(pick_place())
    rel = {f for f, _ in geometric(kb)}
    assert ("On", "cube_r", "bin_a") in rel  # the cube rests on top of bin_a after the place
    assert ("LeftOf", "bin_a", "cube_b") in rel  # +y is the robot's left
    assert not any(f[0] == "On" and f[1] == "cube_b" for f in rel)
    assert ask(kb, ("On", "?x", "bin_a"))[0][0] == {"?x": "cube_r"}


def test_event_log_reads_from_to_and_cites_ids():
    kb = run(pick_place())
    log = event_log(kb)
    assert any("gripper closed" in x for x in log) and any("gripper opened" in x for x in log)
    assert any("event grasp(cube_r)" in x for x in log) and any("event place(cube_r, bin_a)" in x for x in log)
    assert all(x.startswith("[e") or x.startswith("[v") for x in log)
    ts = [int(x.split("t=")[1].split()[0]) for x in log]
    assert ts == sorted(ts)


def test_controller_view_has_goal_progress_and_fits_the_budget():
    v = controller_view(run(pick_place()))
    assert v.startswith("Goal: put the red cube on the container")
    assert "grasps 1" in v and "places 1" in v and "[cube_r] red cube" in v
    assert len(v) / 4 < 1500  # rough token count (DEVELOPMENT §3.8 budget)


def test_hidden_cube_pose_is_not_printed_as_a_coordinate():
    from arbiter_kb.sources.synthetic import SyntheticSource

    src = SyntheticSource(lane_offset=0.0, swap_steps=20)
    ep = src.episodes("train", 1)[0]
    s = ArbiterKBSystem(Params())
    s.reset(src.meta(ep))
    for f in src.frames(ep):
        s.step(f)
    v = controller_view(s.kb)
    assert all("see Inside" in line for line in v.splitlines() if "cube, hidden" in line)
