"""The KB in words (narrative.py): places relative to the robot, container contents, moves joined into exchanges."""
from arbiter_kb.narrative import exchanges, narrative
from arbiter_kb.params import Params
from arbiter_kb.sources.synthetic import SyntheticSource
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem


def kb_after(ep: int):
    src = SyntheticSource(lane_offset="mix", swap_steps="mix")
    s = ArbiterKBSystem(Params())
    s.reset(src.meta(ep))
    for f in src.frames(ep):
        s.step(f)
    return s.kb, src.ground_truth(ep)


def test_words_name_contents_places_and_exchanges():
    kb, gt = kb_after(100)
    text = narrative(kb)
    assert "Containers from the robot's left to its right:" in text and "nearest to the robot" in text
    for cube, b in gt.cube_to_bin.items():
        assert f"the {cube[5:]} cube" in text and f"under {b}" in text
    assert "exchanged places" in text and "(task item)" in text
    # back-to-back exchanges stay separate: no group holds more than two containers here
    assert all(len(objs) <= 2 for _, _, objs in exchanges(kb))
