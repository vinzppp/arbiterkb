import math

import pytest

from arbiter_kb.beliefs import Distribution


def total(d):
    return sum(d.probs.values())


def test_from_distances_normalised_and_ordered():
    d = Distribution.from_distances({"a": 0.0, "b": 0.02}, 0.02, 5, {"a": ["e1"], "b": ["e2"]})
    assert abs(total(d) - 1) < 1e-12
    assert d.argmax()[0] == "a"
    assert d.probs["a"] / d.probs["b"] == pytest.approx(math.exp(0.5))


def test_no_candidate_is_unknown():
    d = Distribution.from_distances({}, 0.02, 5, {})
    assert d.probs == {None: 1.0}


def test_eps_none_floor():
    d = Distribution.from_distances({"a": 0.0}, 0.02, 0, {"a": ["e1"]}, eps_none=0.01)
    assert d.probs[None] == pytest.approx(0.01) and abs(total(d) - 1) < 1e-12


def test_swap_mix_moves_mass_and_keeps_sum():
    d = Distribution({"a": 0.9, "b": 0.1}, 0, ["e1"], {"a": ["e1"]})
    m = d.swap_mix("a", "b", 0.25, 1, "e9")
    assert m.probs["a"] == pytest.approx(0.75 * 0.9 + 0.25 * 0.1)
    assert m.probs["b"] == pytest.approx(0.75 * 0.1 + 0.25 * 0.9)
    assert abs(total(m) - 1) < 1e-12 and "e9" in m.support
    assert d.probs["a"] == 0.9, "operations must not mutate"


def test_swap_mix_w_half_is_uniform_between_pair():
    m = Distribution({"a": 1.0}, 0).swap_mix("a", "b", 0.5, 1, "e1")
    assert m.probs["a"] == pytest.approx(0.5) and m.probs["b"] == pytest.approx(0.5)


def test_flatten_toward_uniform():
    d = Distribution({"a": 1.0}, 0).flatten(0.5, ["a", "b"], 1)
    assert d.probs == pytest.approx({"a": 0.75, "b": 0.25})


def test_drop_removes_backed_mass_and_empty_becomes_unknown():
    d = Distribution({"a": 0.6, "b": 0.4}, 0, ["e1", "e2"], {"a": ["e1"], "b": ["e2"]})
    assert d.drop("e1", 1).probs == pytest.approx({"b": 1.0})
    assert d.drop("e1", 1).drop("e2", 2).probs == {None: 1.0}


def test_drop_of_distribution_level_support_halves_confidence():
    d = Distribution({"a": 1.0}, 0, ["e7"], {"a": []})
    assert d.drop("e7", 1).max_prob() < 1.0


def test_entropy():
    assert Distribution({"a": 1.0}, 0).entropy() == 0
    assert Distribution({"a": 0.5, "b": 0.5}, 0).entropy() == pytest.approx(math.log(2))


def test_retract_after_mix_keeps_the_other_hypothesis():
    """Code review: retracting one container's cover evidence after a mix must not wipe everything."""
    d = Distribution({"a": .9, "b": .1}, 0, ["e1", "e2"], {"a": ["e1"], "b": ["e2"]}).swap_mix("a", "b", 0.1, 1, "e3")
    assert d.drop("e1", 2).probs == pytest.approx({"b": 1.0})


def test_mix_into_empty_hypothesis_inherits_support():
    d = Distribution({"a": 1.0}, 0, ["e1"], {"a": ["e1"]}).swap_mix("a", "b", 0.3, 1, "e2")
    assert d.hyp_support["b"] == ["e1"]


def test_none_hypothesis_rests_on_the_trigger():
    from arbiter_kb.beliefs import Distribution
    d = Distribution.from_distances({"a": 0.0}, 0.02, 1, {"a": ["e1", "o1:a"]}, eps_none=0.01, none_support=["e1"])
    assert d.hyp_support[None] == ["e1"]
