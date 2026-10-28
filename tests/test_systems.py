from arbiter_kb.params import Params
from arbiter_kb.sources.synthetic import SyntheticSource
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem
from arbiter_kb.systems.baselines import TrackerSystem


def run_to_end(system, src, ep):
    system.reset(src.meta(ep))
    for f in src.frames(ep):
        system.step(f)
    return system


def test_all_systems_find_the_container_on_clean_input():
    src = SyntheticSource()
    for ep in src.episodes("train", 5):
        gt = src.ground_truth(ep)
        cube = gt.targets[0]
        for s in (TrackerSystem("bound"), ArbiterKBSystem()):
            assert run_to_end(s, src, ep).believe_inside(cube).argmax()[0] == gt.cube_to_bin[cube]


def test_arbiterkb_plan_has_support_and_waypoints():
    src = SyntheticSource()
    ep = src.episodes("train", 1)[0]
    s = run_to_end(ArbiterKBSystem(), src, ep)
    step = s.plan(src.ground_truth(ep).targets[0])
    assert step.skill == "pick" and step.waypoints.shape == (4, 7) and step.support
    assert list(step.waypoints[:, 6]) == [1.0, 0.0, -1.0, 0.0]


def test_rules_do_not_mutate_kb():
    from arbiter_kb.rules import RULES
    src = SyntheticSource()
    ep = src.episodes("train", 1)[0]
    s = ArbiterKBSystem()
    s.reset(src.meta(ep))
    frames = list(src.frames(ep))
    for f in frames[:40]:
        s.step(f)
    before = s.kb.snapshot()
    for _, rule in RULES:
        rule(s.kb, s.kb.evidence[-3:], frames[40], Params().rule)
    assert s.kb.snapshot() == before
