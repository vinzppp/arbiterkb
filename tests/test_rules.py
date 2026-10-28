"""Each rule on hand-made frames (E3), including the negative cases."""
import numpy as np
import pytest

from arbiter_kb.beliefs import Distribution
from arbiter_kb.kb import KB, AddEvidence, MixDist, SetDist, SetPose, SetVis
from arbiter_kb.params import RuleParams
from arbiter_kb.rules import r1_cover, r2_follow, r6_swap, r7_decay
from arbiter_kb.types import FrameObs, ObjObs

OBJS = [("bin_a", "container", {}), ("bin_b", "container", {}), ("cube_g", "cube", {"color": "g"})]


def frame(t, bins: dict, cube=None):
    objs = {b: ObjObs(b, "container", xy is not None, None if xy is None else np.array([*xy, 0.03]), None, 1.0)
            for b, xy in bins.items()}
    objs["cube_g"] = ObjObs("cube_g", "cube", cube is not None,
                            None if cube is None else np.array([*cube, 0.017]), None, 1.0)
    return FrameObs(t, objs, np.zeros(3), 0.08, True)


def disappear(kb, t, xy):
    ev, = kb.apply([AddEvidence(t, "disappear", ("cube_g",), {"last_xyz": [*xy, 0.017], "obs": []})])
    return kb.evidence[-1]


def test_r1_two_candidates_closest_wins():
    kb = KB(OBJS)
    ev = disappear(kb, 32, (0.0, 0.0))
    ops = r1_cover(kb, [ev], frame(32, {"bin_a": (0.01, 0), "bin_b": (0.03, 0)}), RuleParams())
    d = next(o.dist for o in ops if isinstance(o, SetDist))
    assert d.argmax()[0] == "bin_a" and set(d.probs) == {"bin_a", "bin_b"}
    assert d.probs["bin_a"] / d.probs["bin_b"] == pytest.approx(np.exp((0.03**2 - 0.01**2) / (2 * 0.02**2)))


def test_r1_no_candidate_is_unknown_with_support():
    kb = KB(OBJS)
    ev = disappear(kb, 32, (0.0, 0.0))
    ops = r1_cover(kb, [ev], frame(32, {"bin_a": (0.2, 0), "bin_b": None}), RuleParams())
    d = next(o.dist for o in ops if isinstance(o, SetDist))
    vis = next(o for o in ops if isinstance(o, SetVis))
    assert d.probs == {None: 1.0} and d.support == [ev.id] and vis.state == "unknown"


def test_r1_small_sigma_does_not_underflow():
    """Code-review BLOCK: d/σ = 8 must still give the candidate all the mass."""
    kb = KB(OBJS)
    ev = disappear(kb, 32, (0.0, 0.0))
    ops = r1_cover(kb, [ev], frame(32, {"bin_a": (0.04, 0), "bin_b": None}), RuleParams(sigma_cover=0.005))
    assert next(o.dist for o in ops if isinstance(o, SetDist)).probs == {"bin_a": 1.0}


def kb_hidden_in(bin_id):
    kb = KB(OBJS)
    ev = disappear(kb, 32, (0.0, 0.0))
    kb.apply([SetDist("Inside", "cube_g", Distribution.point(bin_id, 32, [ev.id])),
              SetVis("cube_g", "hidden", 1.0, 32, [ev.id])])
    return kb


def near_pass(kb, closeness):
    kb.apply([AddEvidence(90, "near_pass", ("bin_a", "bin_b"), {"closeness": closeness, "obs": []})])
    return kb.evidence[-1]


def test_r6_weight_formula():
    kb = kb_hidden_in("bin_a")
    ev = near_pass(kb, 0.4)
    mix = [o for o in r6_swap(kb, [ev], frame(90, {"bin_a": (0, 0), "bin_b": (0.1, 0)}), RuleParams()) if isinstance(o, MixDist)]
    assert len(mix) == 1 and mix[0].w == pytest.approx(0.5 * (1 - 0.4))


def test_r6_no_mix_when_not_close():
    kb = kb_hidden_in("bin_a")
    ev = near_pass(kb, 1.0)  # closeness = 1 means at the threshold: w = 0
    assert not [o for o in r6_swap(kb, [ev], frame(90, {}), RuleParams()) if isinstance(o, MixDist)]


def test_r6_ignores_visible_cubes():
    kb = kb_hidden_in("bin_a")
    kb.apply([SetVis("cube_g", "visible", 1.0, 89, ["o89:cube_g"])])
    ev = near_pass(kb, 0.0)
    assert not [o for o in r6_swap(kb, [ev], frame(90, {}), RuleParams()) if isinstance(o, MixDist)]


def test_r7_off_at_lambda_one_and_decays_below():
    kb = kb_hidden_in("bin_a")
    assert r7_decay(kb, [], frame(40, {}), RuleParams()) == []
    ops = r7_decay(kb, [], frame(42, {}), RuleParams(lam=0.9))
    assert ops[0].f == pytest.approx(0.9 ** 10) and ops[0].plausible == ["bin_a", "bin_b"]


def test_r2_follows_probability_weighted_pose():
    kb = kb_hidden_in("bin_a")
    kb.apply([SetPose("bin_a", [0.1, 0, 0.03], 1.0, 40, "observed", ["o40:bin_a"]),
              SetPose("bin_b", [0.3, 0, 0.03], 1.0, 40, "observed", ["o40:bin_b"]),
              SetDist("Inside", "cube_g", Distribution({"bin_a": 0.75, "bin_b": 0.25}, 40, kb.inside("cube_g").support))])
    ops = r2_follow(kb, [], frame(40, {}), RuleParams())
    assert np.allclose(ops[0].xyz, [0.15, 0, 0.03]) and ops[0].source == "inferred"
