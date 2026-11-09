import pytest

from arbiter_kb.beliefs import Distribution
from arbiter_kb.canonical_probe import hard_pick_probe, project_identity, threshold_pick_probe


def test_projection_has_complete_candidate_space_and_unknown():
    state = project_identity("cube_red", Distribution({"bin_1": .7, None: .3}, 4, ["e4"]),
                             ["bin_0", "bin_1", "bin_2"], {"bin_1": (10, 20)})
    assert state.candidates == ("bin_0", "bin_1", "bin_2", None)
    assert state.probs() == {"bin_0": 0.0, "bin_1": .7, "bin_2": 0.0, None: .3}
    assert sum(state.probs().values()) == pytest.approx(1)


def test_same_semantic_state_decides_same_regardless_of_provenance():
    dist = Distribution.point("bin_1", 8, ["o8:bin_1"])
    a = project_identity("cube_red", dist, ["bin_0", "bin_1"], {"bin_1": (12, 34)}, {"arm": "ledger"})
    b = project_identity("cube_red", dist, ["bin_0", "bin_1"], {"bin_1": (12, 34)}, {"arm": "arbiter"})
    assert hard_pick_probe(a) == hard_pick_probe(b)


def test_belief_can_change_a_legal_pick_without_arm_specific_logic():
    points = {"bin_0": (10, 10), "bin_1": (20, 20)}
    a = project_identity("cube_red", Distribution({"bin_0": .9, "bin_1": .1}, 1), list(points), points)
    b = project_identity("cube_red", Distribution({"bin_0": .1, "bin_1": .9}, 1), list(points), points)
    assert hard_pick_probe(a)["target"] == "bin_0"
    assert hard_pick_probe(b)["target"] == "bin_1"
    assert hard_pick_probe(a)["legal"] and hard_pick_probe(b)["legal"]


def test_unknown_and_low_confidence_do_not_invent_information_action():
    unknown = project_identity("cube_red", Distribution.point(None, 2, []), ["bin_0"], {"bin_0": (1, 2)})
    assert hard_pick_probe(unknown)["reason"] == "unknown_or_not_visible"
    ambiguous = project_identity("cube_red", Distribution({"bin_0": .55, None: .45}, 2),
                                 ["bin_0"], {"bin_0": (1, 2)})
    action = threshold_pick_probe(ambiguous, .8)
    assert not action["legal"]
    assert action["reason"] == "ambiguous_no_legal_information_action"
