from arbiter_kb.kb import KB
from arbiter_kb.ops import AddEvidence
from arbiter_kb.self_memory import changes, intent, outcome, record, self_view
from arbiter_kb.memtool import run_tool, working_memory


def _kb() -> KB:
    kb = KB([("bin_0", "container", {}), ("cube_red", "cube", {"color": "red"})])
    kb.task["goal"] = "pick up the container that hides the red cube"
    return kb


def test_intent_from_the_waypoint():
    kb = _kb()
    assert intent(kb, [0, 0, 0.02], True) == "pick_up"
    kb.apply([AddEvidence(3, "pick_up", (), {"obs": []})])
    assert intent(kb, [0, 0, 0.2], False) == "release" and intent(kb, [0, 0, 0.2], True) == "move"


def test_outcome_done_not_done_and_instead():
    kb = _kb()
    d = record(kb, 5, [0, 0, 0.02], True, declared="pick_up")
    assert outcome(kb, d) == "not done yet"
    kb.apply([AddEvidence(9, "press", (), {"obs": []})])
    assert outcome(kb, d).startswith("not done; instead press")
    kb.apply([AddEvidence(12, "pick_up", (), {"obs": []})])
    assert outcome(kb, d).startswith("done (pick_up at t=12")


def test_self_view_lists_changes_first_and_is_empty_before_a_decision():
    kb = _kb()
    assert self_view(kb) == ""
    record(kb, 5, None, False, declared="move")
    v = self_view(kb)
    assert v.startswith("Since your last decision (t=5):") and "nothing" in v and "t=5: move -> moved" in v
    assert changes(kb, kb.task["decisions"][-1]) == []


def test_memtool_working_memory_and_unknown_queries():
    kb = _kb()
    kb.objects["bin_0"].pose.value = [0.0, 0.1, 0.05]
    w = working_memory(kb)
    assert "Working memory (now):" in w and "history:" in w and "bin_0" in w
    assert run_tool(kb, "history", "nope").startswith("unknown object")
    assert run_tool(kb, "fly", "").startswith("unknown tool")


def test_changes_name_a_put_down_on_the_table():
    from arbiter_kb.ops import AddEvent
    kb = _kb()  # closed-loop smoke run: place(cube, "table") raised a KeyError in name()
    d = record(kb, 5, [0, 0, 0.2], False, declared="move")
    kb.apply([AddEvidence(8, "grip_open", (), {"obs": []})])
    kb.apply([AddEvent(8, "place", ("cube_red", "table"), 1.0, [kb.evidence[-1].id])])
    assert changes(kb, d)[0] == "t=8: place the red cube table"
