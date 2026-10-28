"""DEVELOPMENT §6 M1 done-when (d), part 2: a synthetic scenario where R5 (one cube per container)
beats OverlapHaircut. The cover itself is ambiguous: two containers 4 cm apart, and the red cube's last
sighting is 2.5 cm off (position noise), nearer the wrong container. Bound - and so OverlapHaircut, which
sees no pass - puts BOTH cubes in bin_a, confidently. R5 uses the green cube, firmly in bin_a, to move
the red one to bin_b. No swap happens: this is propagation, not a better swap feature."""
import numpy as np
import pytest

import arbiter_kb.systems.arbiter_kb as akb
from arbiter_kb.metrics import brier
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem
from arbiter_kb.systems.baselines import GuardedBound
from arbiter_kb.types import EpisodeMeta, FrameObs, ObjObs

OBJS = [("bin_a", "container", {}), ("bin_b", "container", {}),
        ("cube_g", "cube", {"color": "g"}), ("cube_r", "cube", {"color": "r"})]
META = EpisodeMeta("hand", 0, "train", "pick up the container that hides the red cube", OBJS, None, None)
TRUTH = {"cube_g": "bin_a", "cube_r": "bin_b"}
BINS = ["bin_a", "bin_b"]


def frames():
    def f(t, covered):
        o = {"bin_a": ObjObs("bin_a", "container", covered, np.array([0.0, 0, 0.03]) if covered else None, None, 1.0),
             "bin_b": ObjObs("bin_b", "container", covered, np.array([0.04, 0, 0.03]) if covered else None, None, 1.0),
             "cube_g": ObjObs("cube_g", "cube", not covered, None if covered else np.array([0.0, 0, 0.017]), None, 1.0),
             "cube_r": ObjObs("cube_r", "cube", not covered, None if covered else np.array([0.015, 0, 0.017]), None, 1.0)}
        return FrameObs(t, o, np.zeros(3), 0.08, True)
    return [f(t, t >= 5) for t in range(12)]


def answer(system):
    system.reset(META)
    for fr in frames():
        system.step(fr)
    out = {}
    for cube, truth in TRUTH.items():
        d = system.believe_inside(cube)
        out[cube] = (d.argmax(), brier({str(k): v for k, v in d.probs.items()}, truth, BINS))
    return out


def test_r5_beats_overlap_haircut_on_an_ambiguous_cover(monkeypatch):
    oh = answer(GuardedBound("overlap_haircut"))
    kb = answer(ArbiterKBSystem())
    assert oh["cube_r"][0] == ("bin_a", 1.0)  # confidently wrong: a silent failure
    assert kb["cube_r"][0][0] == "bin_b" and kb["cube_r"][0][1] == pytest.approx(0.79, abs=0.01)  # below τ_hi: honest, not overconfident
    assert kb["cube_r"][1] < oh["cube_r"][1] and kb["cube_g"][0][0] == "bin_a"
    monkeypatch.setattr(akb, "RULES", [r for r in akb.RULES if r[0] != "R5"])
    no_r5 = answer(ArbiterKBSystem())
    assert no_r5["cube_r"][0][0] == "bin_a"  # without R5 ArbiterKB is wrong too (p ≈ 0.6): R5 is what helps
