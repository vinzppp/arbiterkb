"""arbiter_kb/callout.py: the layer-1 scoring rules of NOTES "Layer 1 on real scenes"."""
from arbiter_kb.callout import objects, right_object, score, summarize, uncovered

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


def test_objects_merges_one_actor_under_two_names():
    o = objects({"cube": [0.1, 0.1, 0.02, 0], "cubes[0]": [0.1, 0.1, 0.02, 0], "bin": [0.2, 0.1, 0.03, 0]})
    assert set(o) == {"cube", "bin"}


def test_right_object_is_the_hiding_container_or_the_cube():
    o = objects(TRUTH)
    assert right_object(SWAP_GOAL, o)[0] == "bin_1"
    assert right_object("place the blue cube on the target", o)[0] == "cube_blue"
    assert right_object("pick up the cube", o)[0] is None
    assert right_object("place the green cube on the target", o)[0] is None


def test_covered_cube_is_left_out_of_the_distances():
    assert set(uncovered(objects(TRUTH))) == {"bin_0", "bin_1", "bin_2"}


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
