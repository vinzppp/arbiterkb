"""The ask call (ask.py): one closed question, one marked frame, no memory text; checked answers only."""
import json

import numpy as np

from arbiter_kb.ask import ask
from arbiter_kb.kb import KB
from arbiter_kb.views import controller_view

IMG = np.zeros((8, 8, 3), np.uint8)


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def chat(self, messages, max_tokens=0):
        self.calls.append(messages)
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


def kb():
    k = KB([("bin_a", "container", {}), ("cube_r", "cube", {"color": "red"})])
    k.task["goal"] = "pick up the container that hides the red cube"
    return k


def test_the_prompt_has_one_image_and_no_memory_text():
    llm = FakeLLM({"answer": "container"})
    ask(kb(), llm, "occluder", "cube_r", 7, IMG)
    (msg,) = llm.calls[0]
    assert [p["type"] for p in msg["content"]] == ["text", "image_url"]
    assert "cube_r" in msg["content"][0]["text"] and "Goal" not in msg["content"][0]["text"]


def test_occluder_answer_is_evidence_shown_in_the_log():
    k = kb()
    rep = ask(k, FakeLLM({"answer": "robot_arm"}), "occluder", "cube_r", 7, IMG)
    assert rep["written"] == 1 and k.evidence[-1].data["obs"] == ["o7:image"]
    assert "cube_r hidden by robot arm (vlm)" in controller_view(k)
    assert ask(k, FakeLLM({"answer": "a lid"}), "occluder", "cube_r", 8, IMG)["written"] == 0


def test_appearance_becomes_a_description_and_bad_text_is_dropped():
    k = kb()
    ask(k, FakeLLM({"description": "white cup, upside down", "color": "white"}), "appearance", "bin_a", 9, IMG)
    assert 'looks like "white cup, upside down (white)"' in controller_view(k)
    rep = ask(k, FakeLLM({"description": "a cup 3 cm wide", "color": "white"}), "appearance", "bin_a", 10, IMG)
    assert rep["written"] == 0 and rep["problem"]
    assert ask(k, FakeLLM({"description": "cannot_tell", "color": "cannot_tell"}), "appearance", "bin_a", 11,
               IMG)["written"] == 0
