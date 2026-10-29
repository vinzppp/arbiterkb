"""Goal counter (goal_counter.py): one parse per goal, checked; the KB counts its own actions against it."""
import json

from arbiter_kb.goal_counter import parse_goal, progress_line
from arbiter_kb.kb import KB, AddEvidence


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer

    def call(self, messages, max_tokens=0, reasoning=None):
        return (self.answer if isinstance(self.answer, str) else json.dumps(self.answer)), {}


def test_parse_is_checked():
    assert parse_goal(FakeLLM({"count": "release", "target": 4, "then": "press"}), "x") == \
        {"count": "release", "target": 4, "then": "press"}
    assert parse_goal(FakeLLM({"count": "jump", "target": 4}), "x") is None
    assert parse_goal(FakeLLM({"count": "release", "target": True}), "x") is None
    assert parse_goal(FakeLLM("no json"), "x") is None


def test_progress_counts_the_kbs_own_actions():
    kb = KB([])
    ee = {"ee": [0, 0, 0.02], "obs": []}
    kb.apply([AddEvidence(1, "pick_up", (), ee), AddEvidence(5, "release", (), {**ee, "how": "placed"}),
              AddEvidence(7, "hold_above", (), {**ee, "side": "left"})])
    assert progress_line(kb, {"count": "release", "target": 2}) == "Goal progress: put-downs 1 of 2."
    kb.apply([AddEvidence(9, "release", (), {**ee, "how": "placed"})])
    assert progress_line(kb, {"count": "release", "target": 2}).endswith("2 of 2 — the repetitions are done.")
    assert progress_line(kb, {"count": "hold_left", "target": 3}) == "Goal progress: holds on the left 1 of 3."
    assert progress_line(kb, None) == ""


def test_the_last_object_wins_over_a_draft():
    text = 'I think `{"count": "pick_up", "target": 2}` ... no, final:\n{"count": "hold_left", "target": 3, "then": "x"}'
    assert parse_goal(FakeLLM(text), "x")["count"] == "hold_left"


def test_the_default_view_carries_the_counter_and_v3_does_not():
    from arbiter_kb.views import controller_view

    kb = KB([])
    kb.task.update(goal="pick it up two times", counter={"count": "release", "target": 2, "then": "press"})
    kb.apply([AddEvidence(5, "release", (), {"ee": [0, 0, 0.02], "how": "placed", "obs": []})])
    v4, v3 = controller_view(kb), controller_view(kb, version="v3")
    assert "Goal progress: put-downs 1 of 2." in v4 and "Goal progress" not in v3
    kb.task["counter"] = None
    assert controller_view(kb) == v3  # no counter, no containers: v4 is v3 word for word


def test_parse_answer_takes_the_last_top_level_object():
    from arbiter_kb.controller import parse_answer

    assert parse_answer('draft {"next": "a"} final {"next": "b", "w": {"x": 1}}') == {"next": "b", "w": {"x": 1}}
    assert parse_answer("no json here") is None
