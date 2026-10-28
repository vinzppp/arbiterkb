"""Task items come from the instruction text only; the open vocabulary carries no scene counts."""
from arbiter_kb.task_items import GENERIC_SLOTS, is_task_item, open_vocabulary, parse_task_items


def test_parse_robomme_style_goals():
    assert parse_task_items("pick up the container that hides the red cube") == [
        {"phrase": "container", "noun": "container", "color": None},
        {"phrase": "red cube", "noun": "cube", "color": "red"}]
    assert [i["phrase"] for i in parse_task_items("Press the button, then put the blue block in the bowl")] == \
        ["button", "blue cube", "container"]
    assert parse_task_items("wave") == []


def test_open_vocabulary_has_no_scene_counts_and_containers_first():
    v = open_vocabulary(parse_task_items("pick up the container that hides the green cube"))
    assert v[:GENERIC_SLOTS["container"]] == ["container."] * GENERIC_SLOTS["container"]
    assert "green cube." in v and v.count("cube.") == GENERIC_SLOTS["cube"]
    assert open_vocabulary([]) == ["container."] * 5 + ["cube."] * 4  # the same for every episode


def test_is_task_item_matches_noun_and_colour():
    items = parse_task_items("pick up the container that hides the red cube")
    assert is_task_item({"color": "red"}, "cube", items) and not is_task_item({"color": "blue"}, "cube", items)
    assert is_task_item({}, "container", items)  # "the container" names every container


def test_controller_view_marks_task_items():
    from arbiter_kb.params import Params
    from arbiter_kb.sources.synthetic import SyntheticSource
    from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem
    from arbiter_kb.views import controller_view

    src = SyntheticSource()
    ep = src.episodes("train", 1)[0]
    s = ArbiterKBSystem(Params())
    s.reset(src.meta(ep))
    for f in src.frames(ep):
        s.step(f)
    v = controller_view(s.kb)
    target = src.ground_truth(ep).targets[0]
    assert "Task items (named by the goal): container, " in v
    assert [line for line in v.splitlines() if line.strip().startswith(f"[{target}]")][0].count("(task item)") == 1
    others = [c for c in ("cube_red", "cube_green", "cube_blue") if c != target]
    assert not any("(task item)" in line for line in v.splitlines() for c in others if line.strip().startswith(f"[{c}]"))


def test_clue_objects_check_phrases_and_cap():
    import json

    import numpy as np

    from arbiter_kb.clue_objects import MAX_OBJECTS, check, list_objects, phrases

    good, errs = check({"objects": [
        {"name": "Cube", "color": "red", "count": 1, "task_item": True},
        {"name": "container", "color": "white", "count": 3, "task_item": False},
        {"name": "the big robot arm", "color": "gray", "count": 1},
        {"name": "cube", "color": "blue", "count": "two"},
        {"name": "cube", "color": "green", "count": MAX_OBJECTS}]})
    assert [o["name"] for o in good] == ["cube", "container"] and len(errs) == 3  # bad name, bad count, over the cap
    assert phrases(good) == ["red cube.", "white container.", "white container.", "white container."]
    assert check("no json")[1] == ["unparseable answer"]

    class Fake:
        def chat(self, messages, max_tokens=0):
            assert "pick up the red cube" in messages[0]["content"][0]["text"]
            return json.dumps({"objects": [{"name": "cube", "color": "red", "count": 1, "task_item": True}]})

    out = list_objects(Fake(), np.zeros((8, 8, 3), np.uint8), "pick up the red cube")
    assert out["phrases"] == ["red cube."] and out["problems"] == []


def test_clue_objects_colour_not_repeated_and_task_items_from_the_instruction():
    from arbiter_kb.clue_objects import mark_task_items, phrases

    objs = [{"name": "blue cube", "color": "blue", "count": 1, "task_item": False},
            {"name": "cube", "color": "green", "count": 1, "task_item": False},
            {"name": "container", "color": "gray", "count": 3, "task_item": False}]
    assert phrases(objs)[:2] == ["blue cube.", "green cube."]
    marked = mark_task_items(objs, "pick up the container that hides the green cube")
    assert [o["task_item"] for o in marked] == [False, True, True]
