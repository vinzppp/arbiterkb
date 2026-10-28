"""VLM keyframe call (lister + annotator, v1.9): one call; checks drop bad items one by one, beliefs carry support,
no numbers ever enter the KB."""
import json

import numpy as np

from arbiter_kb.keyframe import (MAX_OBJECTS, check_annotations, check_clues, keyframe_call, mark_task_items, phrases,
                                 to_ops)
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
    rels, descs, errs = check_annotations(ans, kb)
    assert [r["object"] for r in rels] == ["bin_a"] and [d["id"] for d in descs] == ["bin_a"]
    assert len(errs) == 5 and any("numbers" in e for e in errs) and any("cites unknown" in e for e in errs)


def test_clue_objects_check_phrases_and_cap():
    good, errs = check_clues([
        {"name": "Cube", "color": "red", "count": 1, "task_item": True},
        {"name": "container", "color": "white", "count": 3, "task_item": False},
        {"name": "the big robot arm", "color": "gray", "count": 1},
        {"name": "cube", "color": "blue", "count": "two"},
        {"name": "cube", "color": "green", "count": MAX_OBJECTS}])
    assert [o["name"] for o in good] == ["cube", "container"] and len(errs) == 3  # bad name, bad count, over the cap
    assert phrases(good) == ["red cube.", "white container.", "white container.", "white container."]
    assert check_clues(None)[1] == ["clue_objects missing"]


def test_clue_objects_colour_not_repeated_and_task_items_from_the_instruction():
    objs = [{"name": "blue cube", "color": "blue", "count": 1, "task_item": False},
            {"name": "cube", "color": "green", "count": 1, "task_item": False},
            {"name": "container", "color": "gray", "count": 3, "task_item": False}]
    assert phrases(objs)[:2] == ["blue cube.", "green cube."]
    marked = mark_task_items(objs, "pick up the container that hides the green cube")
    assert [o["task_item"] for o in marked] == [False, True, True]


def test_one_call_returns_clue_objects_and_writes_beliefs_citing_the_keyframe():
    kb = kb_with_evidence()
    kb.task["goal"] = "pick up the container that hides the red cube"
    llm = FakeLLM({"clue_objects": [{"name": "cube", "color": "red", "count": 1, "task_item": False},
                                    {"name": "container", "color": "white", "count": 2, "task_item": False}],
                   "relations": [{"subject": "cube_r", "predicate": "inside", "object": "bin_a",
                                  "strength": "ambiguous", "cites": ["e1"]}],
                   "objects": [{"id": "bin_a", "description": "blue bowl, upside down"}]})
    rep = keyframe_call(kb, llm, t=12, rgb=np.zeros((8, 8, 3), np.uint8))
    assert len(llm.calls) == 1 and any(p.get("type") == "image_url" for p in llm.calls[0][1]["content"])
    assert rep["phrases"] == ["red cube.", "white container.", "white container."] and rep["problems"] == []
    assert all(o["task_item"] for o in rep["clue_objects"])  # both named by the goal
    b = kb.facts[("inside", "cube_r", "bin_a")]
    assert rep["written"] == 2 and b.conf == 0.5 and b.source == "vlm" and b.support == ["o12:image", "e1"]
    assert ask(kb, ("inside", "cube_r", "?c"))[0] == ({"?c": "bin_a"}, 0.5)
    v = controller_view(kb)
    assert "[inside(cube_r, bin_a)] 0.50 (vlm)" in v and 'looks like "blue bowl, upside down"' in v


def test_first_frame_with_no_tracked_objects_only_lists():
    kb = KB([])
    kb.task["goal"] = "pick up the red cube"
    llm = FakeLLM({"clue_objects": [{"name": "cube", "color": "red", "count": 1, "task_item": True}],
                   "relations": [{"subject": "cube_r", "predicate": "on", "object": "table", "strength": "clear"}],
                   "objects": []})
    rep = keyframe_call(kb, llm, t=0, rgb=np.zeros((8, 8, 3), np.uint8))
    assert "pick up the red cube" in llm.calls[0][1]["content"][0]["text"]
    assert rep["phrases"] == ["red cube."] and rep["written"] == 0 and len(rep["problems"]) == 1 and kb.facts == {}


def test_garbage_answer_changes_nothing():
    kb = kb_with_evidence()
    rep = keyframe_call(kb, FakeLLM("I think the cube is in bin_a at x=0.1"), t=3, rgb=np.zeros((8, 8, 3), np.uint8))
    assert rep["written"] == 0 and rep["problems"] == ["unparseable answer"] and kb.facts == {}
    assert rep["phrases"] == [] and to_ops([], [], 3, "o3:image") == []


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


def test_som_labels_do_not_overlap():
    from arbiter_kb.som import place

    taken = []
    for u, v in [(100, 100), (110, 102), (105, 98), (120, 100)]:  # four objects within 20 px
        place(u, v, "bin_0\nred? 0.92", taken, 256, 256)
    boxes = taken
    assert all(not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
               for i, a in enumerate(boxes) for b in boxes[i + 1:])
