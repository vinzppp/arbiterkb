from arbiter_kb.debug import explain, timeline
from arbiter_kb.evaluate import run
from arbiter_kb.params import Params
from arbiter_kb.sources.corrupt import CorruptedSource
from arbiter_kb.sources.synthetic import SyntheticSource
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem


def test_explain_reaches_observations_for_every_episode():
    src = CorruptedSource(SyntheticSource(), p=1.0)
    for ep in src.episodes("train", 5):
        s = ArbiterKBSystem()
        _, traces, _ = run([s], src, [ep], Params(), "id_swap")
        cube = src.ground_truth(ep).targets[0]
        text = explain(traces[("arbiterkb", ep)], cube)
        assert "observation" in text and "NOT FOUND" not in text, text
        assert abs(sum(timeline(traces[("arbiterkb", ep)], cube)[-1][1].values()) - 1) < 1e-6


def test_explain_every_scored_row_all_systems_m1_sources():
    """E2 (M1): every scored row of every system, under each M1 corruption, explains down to an observation."""
    from arbiter_kb.sources.realistic import NoisySource, OverlapSwapSource
    from arbiter_kb.systems.baselines import GuardedBound, TrackerSystem

    base = SyntheticSource(lane_offset="mix", swap_steps="mix")
    eps = base.episodes("train", 6)
    for cond, src in [("none", base), ("id_swap_overlap", OverlapSwapSource(base, 1.0)),
                      ("pos_noise", NoisySource(base))]:
        systems = [TrackerSystem("frozen"), TrackerSystem("bound"), GuardedBound("haircut"),
                   GuardedBound("overlap_haircut"), GuardedBound("jump"), ArbiterKBSystem()]
        rows, traces, _ = run(systems, src, eps, Params(), cond)
        assert len(rows) == len(eps) * len(systems)
        for r in rows:
            text = explain(traces[(r["system"], r["episode"])], r["cube"], r["t_star"])
            assert "observation" in text and "NOT FOUND" not in text, (cond, r["system"], text)
