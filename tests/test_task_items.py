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



def test_a_placing_goal_looks_for_targets_not_containers():
    v = open_vocabulary(parse_task_items("place the red cube on the first target it was previously placed on"))
    assert v[:GENERIC_SLOTS["container"]] == ["target."] * GENERIC_SLOTS["container"] and "container." not in v
    assert open_vocabulary(parse_task_items("pick up the container that hides the green cube"))[0] == "container."
    assert open_vocabulary(parse_task_items("put the cube on the blue target"))[0] == "target."
