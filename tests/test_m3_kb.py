"""M3: R3 grasp / place, geometric relations, the event log and the controller view, on hand-made frames."""
import numpy as np

from dataclasses import replace

from arbiter_kb.debug import explain_item
from arbiter_kb.detectors import Detectors
from arbiter_kb.params import DetectParams, Params, RuleParams
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
    rel = {f for f, _, _ in geometric(kb)}
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
    hidden = [line for line in v.splitlines() if "cube, hidden" in line]
    assert hidden and all("see Inside" in line for line in hidden)


def put_back_container():
    """bin_a covers the red cube (t=3), is grasped (t=6), lifted, lowered back and released (t=9)."""
    red = np.array([0.1, 0.1, 0.017])
    fs = [frame(t, [0, -0.2, 0.3], 0.08, red) for t in range(3)]
    fs.append(frame(3, [0, -0.2, 0.3], 0.08, red, red_vis=False))  # bin_a (at BIN) now over the cube's spot
    fs[-1].objects["cube_r"] = fs[-1].objects["cube_r"].__class__("cube_r", "cube", False, None, None, 1.0)
    for t, ee, w in [(4, BIN + [0, 0, 0.1], 0.08), (5, BIN, 0.08), (6, BIN, 0.04), (7, BIN + [0, 0, 0.05], 0.04),
                     (8, BIN, 0.04), (9, BIN + [0, 0, 0.02], 0.08)]:
        f = frame(t, ee, w, red, red_vis=False)
        fs.append(f)
    return fs


def test_container_put_back_is_not_placed_on_its_hidden_cube():
    kb = run(put_back_container())
    ev = [(e.type, e.args) for e in kb.events]
    assert ("cover", ("bin_a", "cube_r")) in ev and ("grasp", ("bin_a",)) in ev
    assert ("place", ("bin_a", "table")) in ev and ("place", ("bin_a", "cube_r")) not in ev


def test_no_grasp_when_nothing_is_within_r_grasp():
    fs = [frame(t, [0.3, 0.3, 0.3], 0.08 if t < 3 else 0.04, [0, 0.1, 0.017]) for t in range(6)]
    kb = run(fs)
    assert [e.kind for e in kb.evidence if e.kind.startswith("grip")] == ["grip_close"]
    assert not [e for e in kb.events if e.type == "grasp"]


def test_held_cube_lost_from_view_is_held_not_visible():
    fs = pick_place()[:9] + [frame(t, [0.03 + 0.01 * t, 0.1, 0.05], 0.04, None, red_vis=False) for t in range(9, 12)]
    kb = run(fs)
    assert kb.objects["cube_r"].visibility.value == "held"
    f = facts(kb)
    assert (("Held", "cube_r"), 1.0) in f and not any(x[0] == ("Visible", "cube_r") for x in f)


def test_gripper_hysteresis():
    d, flips = Detectors(DetectParams()), []
    for t, w in enumerate([0.08, 0.069, 0.074, 0.071, 0.076, 0.074, 0.069]):
        ev = d._gripper(frame(t, [0, 0, 0.2], w, [0, 0, 0.017]))
        flips += [(t, e.kind) for e in ev]
    assert flips == [(1, "grip_close"), (4, "grip_open"), (6, "grip_close")]


def test_move_evidence_carries_where_and_when_it_started():
    from arbiter_kb.sources.synthetic import SWAP_T0, SyntheticSource

    src = SyntheticSource(lane_offset=0.0, swap_steps=20, n_swaps=1)
    ep = src.episodes("train", 1)[0]
    s = ArbiterKBSystem(Params())
    s.reset(src.meta(ep))
    for f in src.frames(ep):
        s.step(f)
    starts = [e for e in s.kb.evidence if e.kind == "move_start"]  # containers only
    assert starts and all("xyz" in e.data and e.data["t_at"] == e.t - 1 for e in starts)
    assert min(e.data["t_at"] for e in starts) >= SWAP_T0 - 1
    assert any(f"(t={starts[0].data['t_at']}-" in x for x in event_log(s.kb))


def test_grasp_explains_down_to_the_end_effector_observation():
    s = ArbiterKBSystem(Params())
    s.reset(META)
    for f in pick_place():
        s.step(f)
    grasp = next(e for e in s.kb.events if e.type == "grasp")
    text = explain_item(s.trace_rows(), grasp.id)
    assert "o6:ee: observation ee at t=6" in text and "NOT FOUND" not in text


def test_view_uses_the_given_parameters():
    kb = run(pick_place())
    near = lambda p: "Near(" in controller_view(kb, p)  # noqa: E731
    assert near(replace(RuleParams(), r_near=0.5)) and not near(replace(RuleParams(), r_near=0.001))


def test_event_log_keeps_covers_when_it_drops_old_lines():
    from arbiter_kb.sources.synthetic import SyntheticSource

    src = SyntheticSource(lane_offset=0.0, swap_steps=20, n_swaps=3)
    ep = src.episodes("train", 1)[0]
    s = ArbiterKBSystem(Params())
    s.reset(src.meta(ep))
    for f in src.frames(ep):
        s.step(f)
    log = event_log(s.kb, last=4)
    assert log[0].endswith("earlier lines omitted)")
    assert sum(" covered by " in x for x in log) == 3 and not any("disappeared" in x for x in log)


def test_cover_is_one_line_with_the_containment_rule_and_relation_lines_are_citable():
    from arbiter_kb.sources.synthetic import SyntheticSource
    from arbiter_kb.views import citable_ids

    src = SyntheticSource(lane_offset=0.0, swap_steps=20, n_swaps=1)
    ep = src.episodes("train", 1)[0]
    s = ArbiterKBSystem(Params())
    s.reset(src.meta(ep))
    for f in src.frames(ep):
        s.step(f)
    v = controller_view(s.kb)
    covers = [x for x in v.splitlines() if " covered by " in x]
    assert len(covers) == 3 and all("from then on it moves with bin_" in x and "at (" in x for x in covers)
    assert "disappeared" not in v  # the cover line carries the place; no separate 'disappeared at' line
    ids = citable_ids(s.kb)
    assert all(f"[Inside({c})]" in v and f"Inside({c})" in ids for c in ("cube_red", "cube_green", "cube_blue"))
