"""VLM keyframe call (v1.10: image-only listing) and the Set-of-Mark frame. Relations and descriptions: test_review."""
import json

import numpy as np

from arbiter_kb.keyframe import MAX_OBJECTS, check_clues, list_objects, mark_task_items, phrases


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def chat(self, messages, max_tokens=0):
        self.calls.append(messages)
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


def test_clue_objects_check_phrases_and_cap():
    good, errs = check_clues([
        {"name": "Cube", "color": "red", "count": 1, "task_item": True},
        {"name": "container", "color": "white", "count": 3, "task_item": False},
        {"name": "the big robot arm", "color": "gray", "count": 1},
        {"name": "cube", "color": "blue", "count": "two"},
        {"name": "cube", "color": "green", "count": MAX_OBJECTS}])
    assert [o["name"] for o in good] == ["cube", "container"] and len(errs) == 3  # bad name, bad count, over the cap
    assert phrases(good) == ["red cube.", "white container.", "white container.", "white container."]
    assert check_clues(None)[1] == ["unparseable answer"]


def test_clue_objects_colour_not_repeated_and_task_items_from_the_instruction():
    objs = [{"name": "blue cube", "color": "blue", "count": 1, "task_item": False},
            {"name": "cube", "color": "green", "count": 1, "task_item": False},
            {"name": "container", "color": "gray", "count": 3, "task_item": False}]
    assert phrases(objs)[:2] == ["blue cube.", "green cube."]
    marked = mark_task_items(objs, "pick up the container that hides the green cube")
    assert [o["task_item"] for o in marked] == [False, True, True]


def test_listing_sees_only_the_image_and_the_instruction():
    llm = FakeLLM({"objects": [{"name": "cube", "color": "red", "count": 1, "task_item": True}]})
    out = list_objects(llm, np.zeros((8, 8, 3), np.uint8), "pick up the red cube")
    (msg,) = llm.calls[0]
    assert msg["role"] == "user" and "pick up the red cube" in msg["content"][0]["text"]
    assert [p["type"] for p in msg["content"]] == ["text", "image_url"]  # no memory text
    assert out["phrases"] == ["red cube."] and out["problems"] == []
    assert list_objects(FakeLLM("no json"), np.zeros((8, 8, 3), np.uint8), "x")["problems"] == ["unparseable answer"]


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
