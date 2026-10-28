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
    assert d.argmax()[0] == "bin_a" and set(d.probs) == {"bin_a", "bin_b", None}
    assert d.probs[None] == pytest.approx(RuleParams().eps_none)
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
    ops = r1_cover(kb, [ev], frame(32, {"bin_a": (0.04, 0), "bin_b": None}), RuleParams(sigma_cover=0.005, eps_none=0.0))
    assert next(o.dist for o in ops if isinstance(o, SetDist)).probs == {"bin_a": 1.0}


def kb_hidden_in(bin_id):
    kb = KB(OBJS)
    ev = disappear(kb, 32, (0.0, 0.0))
    kb.apply([SetDist("Inside", "cube_g", Distribution.point(bin_id, 32, [ev.id])),
              SetVis("cube_g", "hidden", 1.0, 32, [ev.id])])
    return kb


def near_pass(kb, min_3d):
    kb.apply([AddEvidence(90, "near_pass", ("bin_a", "bin_b"), {"closeness": 0.5, "min_3d": min_3d, "obs": []})])
    return kb.evidence[-1]


@pytest.mark.parametrize("d, w", [(0.0, 0.5), (0.02, 0.5 * np.exp(-1)), (0.12, 0.5 * np.exp(-6))])
def test_r6_v15_weight(d, w):
    """w = w_max·exp(-min_3d/d0): 0.5 at contact, 0.1839 at 2 cm, 0.0012 at 12 cm (defaults)."""
    kb = kb_hidden_in("bin_a")
    ev = near_pass(kb, d)
    mix = [o for o in r6_swap(kb, [ev], frame(90, {"bin_a": (0, 0), "bin_b": (0.1, 0)}), RuleParams()) if isinstance(o, MixDist)]
    assert len(mix) == 1 and mix[0].w == pytest.approx(w)


def test_r6_uses_3d_distance_not_image_closeness():
    kb = kb_hidden_in("bin_a")
    ev = near_pass(kb, 0.05)
    ev.data["closeness"] = 0.0  # the image says touching; 3D says 5 cm apart
    mix = [o for o in r6_swap(kb, [ev], frame(90, {}), RuleParams()) if isinstance(o, MixDist)]
    assert mix[0].w == pytest.approx(0.5 * np.exp(-2.5))


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


def test_r7_no_decay_in_the_frame_the_distribution_changed():
    kb = kb_hidden_in("bin_a")  # set at t = 32
    assert r7_decay(kb, [], frame(32, {}), RuleParams(lam=0.9)) == []


def test_swap_weight_must_be_a_probability():
    with pytest.raises(AssertionError):
        Distribution({"a": 0.7, "b": 0.3}, 0).swap_mix("a", "b", 1.5, 1, "e1")


def test_joint_exclusive_pinned_cube_frees_the_other():
    from arbiter_kb.beliefs import joint_exclusive
    out = joint_exclusive({"x": Distribution({"a": 0.5, "b": 0.5}, 0, ["e1"]),
                           "y": Distribution({"b": 1.0}, 0, ["e2"])}, 1)
    assert out["x"].probs == pytest.approx({"a": 1.0}) and out["y"].probs == pytest.approx({"b": 1.0})


def test_joint_exclusive_would_overconfide_after_a_mix():
    """Why R5 runs only on cover frames: after one ambiguous swap (w = 0.2) the truth is a mixture of
    two permutations, so x: a .8. Treating the marginals as independent sharpens it to about .94."""
    from arbiter_kb.beliefs import joint_exclusive
    x = Distribution({"a": 1.0}, 0).swap_mix("a", "b", 0.2, 1, "e1")
    y = Distribution({"b": 1.0}, 0).swap_mix("a", "b", 0.2, 1, "e1")
    assert x.probs["a"] == pytest.approx(0.8)
    assert joint_exclusive({"x": x, "y": y}, 2)["x"].probs["a"] == pytest.approx(0.64 / 0.68)


def test_r5_runs_only_when_a_cube_was_just_covered():
    from arbiter_kb.rules import r5_exclusive
    kb = KB(OBJS + [("cube_r", "cube", {"color": "r"})])
    e1 = disappear(kb, 32, (0.0, 0.0))
    e2, = kb.apply([AddEvidence(32, "disappear", ("cube_r",), {"last_xyz": [0.1, 0, 0.017], "obs": []})])
    kb.apply([SetDist("Inside", "cube_g", Distribution({"bin_a": 0.6, "bin_b": 0.4}, 32, [e1.id])),
              SetDist("Inside", "cube_r", Distribution({"bin_b": 1.0}, 32, [e2])),
              SetVis("cube_g", "hidden", 1.0, 32, [e1.id]), SetVis("cube_r", "hidden", 1.0, 32, [e2])])
    ops = r5_exclusive(kb, kb.evidence[:2], frame(32, {}), RuleParams())
    assert {o.subj: o.dist.argmax() for o in ops}["cube_g"] == ("bin_a", pytest.approx(1.0))
    assert r5_exclusive(kb, [], frame(33, {}), RuleParams()) == []


def test_joint_exclusive_infeasible_returns_none_and_r5_records_a_conflict():
    from arbiter_kb.beliefs import joint_exclusive
    from arbiter_kb.rules import r5_exclusive
    assert joint_exclusive({"x": Distribution({"b": 1.0}, 0), "y": Distribution({"b": 1.0}, 0)}, 1) is None
    kb = KB(OBJS + [("cube_r", "cube", {"color": "r"})])
    e1 = disappear(kb, 32, (0.0, 0.0))
    e2, = kb.apply([AddEvidence(32, "disappear", ("cube_r",), {"last_xyz": [0.1, 0, 0.017], "obs": []})])
    kb.apply([SetDist("Inside", "cube_g", Distribution({"bin_b": 1.0}, 32, [e1.id])),
              SetDist("Inside", "cube_r", Distribution({"bin_b": 1.0}, 32, [e2])),
              SetVis("cube_g", "hidden", 1.0, 32, [e1.id]), SetVis("cube_r", "hidden", 1.0, 32, [e2])])
    ops = r5_exclusive(kb, kb.evidence[:2], frame(32, {}), RuleParams())
    assert len(ops) == 1 and ops[0].kind == "exclusive_conflict" and set(ops[0].args) == {"cube_g", "cube_r"}


def test_r5_leaves_earlier_and_mixed_cubes_alone():
    """cube_g was covered earlier and then mixed 50/50 by a contact swap; a new cover pins cube_r to bin_b.
    Joint-normalising would snap cube_g to bin_a with certainty (overconfident), so R5 must leave it alone."""
    from arbiter_kb.rules import r5_exclusive
    kb = KB(OBJS + [("cube_r", "cube", {"color": "r"})])
    e1 = disappear(kb, 32, (0.0, 0.0))
    kb.apply([SetDist("Inside", "cube_g", Distribution.point("bin_a", 32, [e1.id])),
              SetVis("cube_g", "hidden", 1.0, 32, [e1.id])])
    kb.apply(r6_swap(kb, [near_pass(kb, 0.0)], frame(90, {}), RuleParams()))
    assert kb.inside("cube_g").probs == pytest.approx({"bin_a": 0.5, "bin_b": 0.5})
    e2, = kb.apply([AddEvidence(95, "disappear", ("cube_r",), {"last_xyz": [0.1, 0, 0.017], "obs": []})])
    kb.apply([SetDist("Inside", "cube_r", Distribution({"bin_b": 1.0}, 95, [e2])),
              SetVis("cube_r", "hidden", 1.0, 95, [e2])])
    assert r5_exclusive(kb, [kb.evidence[-1]], frame(95, {}), RuleParams()) == []
