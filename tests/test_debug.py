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
