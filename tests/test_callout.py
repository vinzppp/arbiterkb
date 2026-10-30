"""arbiter_kb/callout.py: the layer-1 scoring rules of NOTES "Layer 1 on real scenes"."""
from arbiter_kb.callout import (holds, label, objects, pointable, right_object, right_sequence, score, summarize,
                               went_back)

SWAP_GOAL = "pick up the container hiding the blue cube"
TRUTH = {"bin_0": [0.0, 0.0, 0.03, 0.0], "bin_1": [0.0, 0.15, 0.03, 0.0], "bin_2": [0.0, -0.15, 0.03, 0.0],
         "cube_blue": [0.001, 0.15, 0.017, 0.0], "cube_red": [0.0, -0.149, 0.017, 0.0],
         "table_scene": [0.0, 0.0, -0.02, 0.0], "goal_site": [0.1, 0.1, 0.1, 0.0],
         "on_base": [-0.6, 0.0, 0.05, 0.0], "off_table": [0.9, 0.0, 0.02, 0.0]}


def act(x, y, z, g=0.0):
    return {"action": [x, y, z, 3.14, 0.0, 0.0, g], "frames": 10, "status": "ongoing"}


def test_objects_filters_furniture_base_and_off_table():
    o = objects(TRUTH)
    assert set(o) == {"bin_0", "bin_1", "bin_2", "cube_blue", "cube_red"}


def test_parts_merge_for_distances_but_every_cube_is_kept():
    o = objects({"button": [0.12, 0.009, 0.007, 0], "cap_link": [0.12, 0.009, 0.014, 0], "bin_0": [0.2, 0.1, 0.053, 0],
                 "cube_green_0": [0.2, 0.1, 0.017, 0], "cube_blue_0": [0.2, 0.1, 0.017, 0]})
    assert len(o) == 5  # two cubes at one position (VideoUnmaskSwap val ep 9's listing) both stay
    assert set(pointable(o)) == {"button", "bin_0"}
    assert right_object("pick up the container hiding the blue cube", o)[0] == "bin_0"


def test_right_object_is_the_hiding_container_or_the_cube():
    o = objects(TRUTH)
    assert right_object(SWAP_GOAL, o)[0] == "bin_1"
    assert right_object("place the blue cube on the target", o)[0] == "cube_blue"
    assert right_object("pick up the cube", o)[0] is None
    assert right_object("place the green cube on the target", o)[0] is None


def test_covered_cube_is_left_out_of_the_distances():
    assert set(pointable(objects(TRUTH))) == {"bin_0", "bin_1", "bin_2"}


def test_colour_must_name_a_cube():
    o = objects({"bin_red": [0.0, 0.0, 0.053, 0], "cube_red_0": [0.0, 0.0, 0.017, 0], "bin_1": [0.2, 0.0, 0.053, 0]})
    assert right_object("pick up the container hiding the red cube", o)[0] == "bin_red"


def test_right_wrong_uncertain_and_no_action():
    o = objects(TRUTH)
    right = right_object(SWAP_GOAL, o)[0]
    assert score([act(0.0, 0.16, 0.2)], o, right)["category"] == "right"
    assert score([act(0.0, 0.0, 0.2)], o, right)["category"] == "wrong"
    assert score([act(0.0, 0.2, 0.2)], o, right)["category"] == "wrong"  # nearest is right, but 5 cm off
    assert score([act(0.0, 0.075, 0.2)], o, right)["category"] == "uncertain"  # halfway between bin_0 and bin_1
    assert score([{"answer": "x", "problems": ["p"]}], o, right)["category"] == "no_action"
    unknown = score([act(0.0, 0.16, 0.2), act(0.0, 0.15, 0.03, -1.0)], o, None)  # fallback: no right object, but
    assert unknown["category"] == "unknown" and unknown["d_right"] is None  # d1 and the close dz still reported
    assert abs(unknown["d1"] - 0.01) < 1e-9 and unknown["close_object"] == "bin_1"


def test_first_close_dz_and_retries_skipped():
    o = objects(TRUTH)
    steps = [{"answer": "x", "problems": ["p"]}, act(0.0, 0.15, 0.2, 1.0), act(0.0, 0.15, 0.04, 0.0),
             act(0.0, 0.15, 0.035, -1.0), act(0.0, 0.0, 0.03, -1.0)]
    r = score(steps, o, "bin_1")
    assert r["category"] == "right" and r["close_object"] == "bin_1"
    assert abs(r["close_dz"] - 0.005) < 1e-9


def test_summary_counts_and_medians():
    o = objects(TRUTH)
    rows = [{"task": "T", "arm": "K", "model": "m", **score([act(0.0, 0.15 + d, 0.2, -1.0)], o, "bin_1")}
            for d in (0.0, 0.01, 0.1)]
    s = summarize(rows)[0]
    assert (s["n"], s["right"], s["wrong"], s["closes"]) == (3, 2, 1, 3)
    assert s["d_right_median"] == 0.01 and s["d_right_max"] == 0.1


def test_right_sequence_two_containers():
    o = objects(TRUTH)
    goal = "pick up the container hiding the blue cube, finally pick up another container hiding the red cube"
    assert right_sequence(goal, o) == ["bin_1", "bin_2"]


def test_holds_track_lift_until_open():
    o = objects(TRUTH)
    steps = [act(0.0, 0.15, 0.2, 1.0), act(0.0, 0.15, 0.04, -1.0), act(0.0, 0.15, 0.3), act(0.0, 0.0, 0.1, 1.0),
             act(0.0, 0.0, 0.035, -1.0), act(0.0, 0.0, 0.06)]
    hs = holds(steps, o)
    assert [(h["object"], h["lift"]) for h in hs] == [("bin_1", 0.3), ("bin_0", 0.06)]


def test_labels_first_rule_wins():
    o = objects(TRUTH)
    seq = ["bin_1", "bin_2"]
    lift = lambda x, y, z: [act(x, y, 0.04, -1.0), act(x, y, z), act(x, y, z, 1.0)]  # noqa: E731
    right_high = holds(lift(0.0, 0.15, 0.3), o)
    assert label({"end_reason": "env_error"}, right_high, seq, {0: 0.25})[0] == "planning failure"
    assert label({"end_reason": "env:fail"}, holds(lift(0.0, 0.0, 0.3), o), seq, {})[0] == "wrong object at close"
    relift = holds(lift(0.0, 0.15, 0.3) + lift(0.0, 0.15, 0.3), o)  # the same right object twice is not wrong
    assert label({"end_reason": "env:fail"}, relift, seq, {0: 0.25})[0] == "grasp missed (inferred)"
    assert label({"end_reason": "model_answer:declared_complete"}, right_high, seq, {})[0] == "premature done"
    assert label({"end_reason": "env:fail"}, holds(lift(0.0, 0.15, 0.15), o), seq, {0: 0.25})[0] == "lift too low"
    assert label({"end_reason": "env:timeout"}, [], seq, {})[0] == "other"


def test_went_back_after_the_put_down():
    o = objects(TRUTH)
    seq = ["bin_1", "bin_2"]
    pick1 = [act(0.0, 0.15, 0.04, -1.0), act(0.0, 0.15, 0.04, -1.0), act(0.0, 0.15, 0.25), act(0.2, 0.15, 0.06, 1.0)]
    back = pick1 + [act(0.2, 0.15, 0.04, -1.0), act(0.2, 0.15, 0.25)]  # re-lifts the first bin where it was put
    onward = pick1 + [act(0.0, -0.15, 0.04, -1.0), act(0.0, -0.15, 0.25)]
    assert went_back(back, o, seq)[0] == "yes"
    assert went_back(onward, o, seq)[0] == "no"  # the re-close inside pick 1 is not a go-back
    assert went_back(pick1[:3] + [act(0.0, -0.14, 0.06, 1.0)], o, seq)[0] == "ambiguous"  # put down on bin_2
    assert went_back(back, o, ["bin_1"])[0] == "n/a"


def test_went_back_free_needs_no_object_poses():
    from arbiter_kb.callout import went_back_free
    st = lambda x, y, z, g, s="ongoing": {"action": [x, y, z, 3.14, 0, 0, g], "status": s}  # noqa: E731
    pick1 = [st(0.0, 0.15, 0.04, -1), st(0.0, 0.15, 0.25, 0), st(0.2, 0.15, 0.06, 1)]
    assert went_back_free(pick1 + [st(0.21, 0.15, 0.04, -1), st(0.21, 0.15, 0.25, 0, "fail")])[0] == "yes"
    assert went_back_free(pick1 + [st(0.0, -0.15, 0.04, -1), st(0.0, -0.15, 0.25, 0, "success")])[0] == "no"
    wrong_first = [st(0.0, 0.15, 0.04, -1), st(0.0, 0.15, 0.25, 0, "fail")]  # the env rejected the first lift
    assert went_back_free(wrong_first)[0] == "n/a"
