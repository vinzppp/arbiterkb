"""VLM annotator: checks drop bad items one by one, beliefs carry support, no numbers ever enter the KB."""
import json

import numpy as np

from arbiter_kb.annotate import annotate, check, to_ops
from arbiter_kb.kb import KB, AddEvidence
from arbiter_kb.views import ask, controller_view

OBJS = [("bin_a", "container", {}), ("bin_b", "container", {}), ("cube_r", "cube", {"color": "red"})]


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def chat(self, messages, max_tokens=0):
        self.calls.append(messages)
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


def kb_with_evidence():
    kb = KB(OBJS)
    kb.apply([AddEvidence(5, "disappear", ("cube_r",), {"last_xyz": [0, 0, 0.017], "obs": []})])
    return kb


def test_check_keeps_good_items_and_drops_bad_ones_individually():
    kb = kb_with_evidence()
    ans = {"relations": [
        {"subject": "cube_r", "predicate": "inside", "object": "bin_a", "strength": "clear", "cites": ["e1"]},
        {"subject": "cube_r", "predicate": "inside", "object": "bin_z", "strength": "clear", "cites": []},
        {"subject": "cube_r", "predicate": "on", "object": "bin_b", "strength": "sure", "cites": []},
        {"subject": "cube_r", "predicate": "near", "object": "bin_b", "strength": "clear", "cites": ["e99"]},
        {"subject": "cube_r", "predicate": "near", "object": "bin_b", "strength": "clear", "cites": [], "dist": 0.03}],
        "objects": [{"id": "bin_a", "description": "blue bowl upside down"}, {"id": "bin_q", "description": "x"}]}
    rels, descs, errs = check(ans, kb)
    assert [r["object"] for r in rels] == ["bin_a"] and [d["id"] for d in descs] == ["bin_a"]
    assert len(errs) == 5 and any("numbers" in e for e in errs) and any("cites unknown" in e for e in errs)


def test_text_only_answer_writes_relations_with_cites_and_no_descriptions():
    kb = kb_with_evidence()
    llm = FakeLLM({"relations": [{"subject": "cube_r", "predicate": "inside", "object": "bin_a",
                                  "strength": "ambiguous", "cites": ["e1"]}],
                   "objects": [{"id": "bin_a", "description": "blue bowl"}]})
    rep = annotate(kb, llm, t=10)
    b = kb.facts[("inside", "cube_r", "bin_a")]
    assert rep["written"] == 1 and b.conf == 0.5 and b.source == "vlm" and b.support == ["e1"]
    assert ("description", "bin_a") not in kb.facts  # needs an image
    assert ask(kb, ("inside", "cube_r", "?c"))[0] == ({"?c": "bin_a"}, 0.5)
    assert "inside(cube_r, bin_a) 0.50 (vlm) [e1]" in controller_view(kb)


def test_with_an_image_beliefs_cite_the_keyframe_and_descriptions_show_in_the_view():
    kb = kb_with_evidence()
    llm = FakeLLM({"relations": [], "objects": [{"id": "bin_a", "description": "blue bowl, upside down"}]})
    rep = annotate(kb, llm, t=12, rgb=np.zeros((8, 8, 3), np.uint8))
    assert rep["image"] and kb.facts[("description", "bin_a")].support == ["o12:image"]
    assert 'looks like "blue bowl, upside down"' in controller_view(kb)
    assert any(p.get("type") == "image_url" for p in llm.calls[0][1]["content"])


def test_garbage_answer_changes_nothing():
    kb = kb_with_evidence()
    rep = annotate(kb, FakeLLM("I think the cube is in bin_a at x=0.1"), t=3)
    assert rep["written"] == 0 and rep["problems"] == ["unparseable answer"] and kb.facts == {}
    assert to_ops([], [], 3, None) == []


def test_som_draws_ids_and_hidden_cube_hypotheses_on_containers():
    from arbiter_kb.params import Params
    from arbiter_kb.som import hypotheses, som_image
    from arbiter_kb.sources.synthetic import SyntheticSource
    from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem

    src = SyntheticSource(lane_offset=0.0, swap_steps=20, n_swaps=1)
    ep = src.episodes("train", 1)[0]
    s = ArbiterKBSystem(Params())
    s.reset(src.meta(ep))
    frames = list(src.frames(ep))
    for f in frames:
        s.step(f)
    hyp = hypotheses(s.kb)
    assert hyp and all(x.endswith(tuple("0123456789")) for v in hyp.values() for x in v)
    blank = np.full((256, 256, 3), 255, np.uint8)
    out = som_image(blank, frames[-1], s.kb)
    assert out.shape == blank.shape and out.dtype == np.uint8
    u, v = frames[-1].objects["bin_0"].uv.astype(int)
    patch = out[max(v - 20, 0):v, max(u - 15, 0):u + 15]
    assert (patch != 255).any()  # a label was drawn just above bin_0
