"""VLM review call (v1.10, with the annotator's work): one call over a window of frames; every item names frames
it rests on and is checked alone; results are evidence or beliefs with source vlm, shown in the event log."""
import json

import numpy as np

from arbiter_kb.kb import KB, AddEvidence
from arbiter_kb.review import review, window_lines
from arbiter_kb.views import ask, controller_view

OBJS = [("bin_a", "container", {}), ("bin_b", "container", {}), ("cube_r", "cube", {"color": "red"})]
IMG = np.zeros((8, 8, 3), np.uint8)
FRAMES = [(10, IMG), (12, IMG), (14, IMG)]


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def chat(self, messages, max_tokens=0):
        self.calls.append(messages)
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


def kb_with_window():
    kb = KB(OBJS)
    kb.task["goal"] = "pick up the container that hides the red cube"
    kb.apply([AddEvidence(5, "disappear", ("cube_r",), {"last_xyz": [0, 0, 0.017], "obs": []}),
              AddEvidence(12, "near_pass", ("bin_a", "bin_b"), {"min_3d": 0.06, "obs": []})])
    return kb


def test_window_holds_only_the_events_in_the_frames_time_range():
    lines = window_lines(kb_with_window(), 10, 14)
    assert len(lines) == 1 and lines[0].startswith("[e2] t=12")


def test_one_call_checks_relations_descriptions_identity_and_summary():
    kb = kb_with_window()
    llm = FakeLLM({
        "checks": [{"id": "e2", "verdict": "contradict", "frames": [12]}, {"id": "e1", "verdict": "confirm",
                                                                          "frames": [10]}],
        "missing": [{"what": "the robot touched bin_b", "objects": ["bin_b"], "frames": [14]}],
        "relations": [{"subject": "cube_r", "predicate": "inside", "object": "bin_a", "strength": "ambiguous",
                       "frames": [10, 14]},
                      {"subject": "cube_r", "predicate": "on", "object": "bin_b", "strength": "clear", "frames": [99]}],
        "objects": [{"id": "bin_a", "description": "white cup, upside down", "frames": [14]}],
        "identity": [{"mark": "bin_a", "now": "bin_b", "frames": [10, 14]}],
        "summary": "two white cups slid past each other; nothing was picked up"})
    rep = review(kb, llm, FRAMES, ask_identity=["bin_a"])
    sent = llm.calls[0][1]["content"]
    assert sum(p["type"] == "image_url" for p in sent) == 3 and "[e2] t=12" in sent[0]["text"]
    assert "[e1]" not in sent[0]["text"] and "Identity question" in sent[0]["text"]
    # e1 lies outside the window and the relation with frame 99 was never shown: both dropped
    assert len(rep["problems"]) == 2 and rep["written"] == 6
    b = kb.facts[("inside", "cube_r", "bin_a")]
    assert b.conf == 0.5 and b.source == "vlm" and b.support == ["o10:image", "o14:image"]
    assert ask(kb, ("inside", "cube_r", "?c"))[0] == ({"?c": "bin_a"}, 0.5)
    v = controller_view(kb)
    assert "the video contradicts [e2] (vlm)" in v and "seen in the video, not recorded: the robot touched" in v
    assert "bin_a now looks like bin_b (vlm)" in v and "video summary, t=10-14 (vlm): two white cups" in v
    assert 'looks like "white cup, upside down"' in v


def test_numbers_unknown_ids_and_garbage_are_dropped():
    kb = kb_with_window()
    rep = review(kb, FakeLLM({"relations": [{"subject": "cube_r", "predicate": "near", "object": "bin_q",
                                              "strength": "clear", "frames": [10]}],
                              "identity": [{"mark": "cube_r", "now": "bin_a", "frames": [10]}],
                              "summary": "the cube moved 3 cm"}), FRAMES)
    assert rep["written"] == 0 and len(rep["problems"]) == 3 and kb.facts == {}
    rep = review(kb, FakeLLM("the cups swapped at x=0.1"), FRAMES)
    assert rep["written"] == 0 and rep["problems"] == ["unparseable answer"]
