import numpy as np
import pytest

from arbiter_kb.beliefs import Distribution
from arbiter_kb.kb import KB, AddEvent, AddEvidence, MixDist, Retract, SetDist, SetPose, SetVis
from arbiter_kb.views import ask, text_view, unify

OBJS = [("bin_1", "container", {}), ("bin_2", "container", {}), ("bin_3", "container", {}),
        ("cube_green", "cube", {"color": "green"})]


def three_cup_kb():
    kb = KB(OBJS)
    e1, = kb.apply([AddEvidence(32, "disappear", ("cube_green",), {"obs": ["o31:cube_green"]})])
    e2, = kb.apply([AddEvidence(90, "near_pass", ("bin_1", "bin_2"), {"obs": ["o90:bin_1"]})])
    d = Distribution({"bin_1": 0.55, "bin_2": 0.45}, 90, [e1, e2], {"bin_1": [e1], "bin_2": [e1, e2]})
    kb.apply([SetDist("Inside", "cube_green", d), SetVis("cube_green", "hidden", 0.9, 32, [e1])])
    for i, b in enumerate(["bin_1", "bin_2", "bin_3"]):
        kb.apply([SetPose(b, [0.1 * i, 0, 0.03], 1.0, 90, "observed", [f"o90:{b}"]),
                  SetVis(b, "visible", 1.0, 90, [f"o90:{b}"])])
    return kb, e1, e2


def test_ask_inside_returns_probabilities():
    kb, _, _ = three_cup_kb()
    got = ask(kb, ("Inside", "cube_green", "?c"))
    assert [g[0] for g in got] == [{"?c": "bin_1"}, {"?c": "bin_2"}]
    assert [g[1] for g in got] == pytest.approx([0.55, 0.45])
    assert ask(kb, ("Hidden", "?x")) == [({"?x": "cube_green"}, 0.9)]


def test_unify_rejects_conflicts():
    assert unify(("Inside", "?x", "?x"), ("Inside", "a", "b")) is None
    assert unify(("Inside", "?x", "b"), ("Inside", "a", "b")) == {"?x": "a"}


def test_retract_cascade_three_levels():
    kb, e1, e2 = three_cup_kb()
    v1, = kb.apply([AddEvent(32, "cover", ("bin_1", "cube_green"), 0.9, [e1])])
    v2, = kb.apply([AddEvent(40, "swap", ("bin_1", "bin_2"), 0.9, [v1])])  # level 2 depends on level 1
    kb.apply([SetPose("cube_green", [0, 0, 0.02], 0.8, 41, "inferred", [v2])])  # level 3
    kb.apply([Retract(e1)])
    assert kb.inside("cube_green").probs == {None: 1.0}  # every hypothesis rested on e1
    ev = {e.id: e for e in kb.events}
    assert ev[v1].conf == pytest.approx(0.45) and ev[v2].conf == pytest.approx(0.45)
    assert kb.objects["cube_green"].pose.conf == pytest.approx(0.4)


def test_mix_op_and_log():
    kb, e1, e2 = three_cup_kb()
    kb.pop_log()
    kb.apply([MixDist("Inside", "cube_green", "bin_1", "bin_2", 0.5, 91, e2)], "R6")
    assert kb.inside("cube_green").probs["bin_1"] == pytest.approx(0.5)
    log = kb.pop_log()
    assert log[0]["op"] == "MixDist" and log[0]["rule"] == "R6" and log[0]["before"]["bin_1"] == 0.55


def test_count_comes_from_events():
    kb, e1, _ = three_cup_kb()
    kb.apply([AddEvent(1, "pick", ("bin_1",), 1.0, [e1]), AddEvent(2, "pick", ("bin_1",), 1.0, [e1])])
    assert kb.count("pick", "bin_1") == 2 and kb.count("pick", "bin_2") == 0


def test_check_catches_unknown_support():
    kb = KB(OBJS)
    with pytest.raises(AssertionError):
        kb.apply([SetDist("Inside", "cube_green", Distribution({"bin_1": 1.0}, 0, ["e99"]))])


def test_check_catches_inferred_without_support():
    kb = KB(OBJS)
    with pytest.raises(AssertionError):
        kb.apply([SetPose("cube_green", [0, 0, 0], 0.5, 0, "inferred", [])])


def test_text_view_short():
    kb, _, _ = three_cup_kb()
    tv = text_view(kb)
    assert "[cube_green] green cube, hidden since t=0; inside bin_1 (0.55) | bin_2 (0.45)" in tv
    assert len(tv.split()) < 200
    assert np.isfinite(kb.objects["bin_2"].pose.value).all()
