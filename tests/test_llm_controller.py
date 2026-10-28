"""M4: the OpenRouter client (fake transport, no network) and the controller's prompt, parsing and checks."""
import json

import numpy as np
import pytest

from arbiter_kb import controller as C
from arbiter_kb.llm import LLM, BudgetExceeded, count_images, load_env

KEY = "sk-test-not-a-real-key"


def fake(cost=0.01, text='{"waypoint": [0.1, 0.0, 0.2, -3.1416, 0, 0], "gripper": 1}'):
    calls = []

    def transport(body, key):
        calls.append((body, key))
        return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost}}
    return transport, calls


def llm(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    tr, calls = fake(**kw)
    return LLM(cache_dir=tmp_path, max_usd=0.025, transport=tr), calls


def test_cache_hit_costs_nothing_and_key_is_never_stored(tmp_path, monkeypatch):
    m, calls = llm(tmp_path, monkeypatch)
    msgs = [{"role": "user", "content": "hi"}]
    a, b = m.chat(msgs), m.chat(msgs)
    assert a == b and len(calls) == 1 and m.usage["hits"] == 1 and m.usage["usd"] == pytest.approx(0.01)
    assert calls[0][1] == KEY and calls[0][0]["provider"]["allow_fallbacks"] is False
    assert all(KEY not in p.read_text() for p in tmp_path.iterdir())


def test_a_cache_hit_reports_the_original_calls_usage(tmp_path, monkeypatch):
    """Review G_M4: a replayed run wrote 0 tokens / 0 s / $0; now only the spend is 0."""
    m, calls = llm(tmp_path, monkeypatch)
    msgs = [{"role": "user", "content": "hi"}]
    _, first = m.call(msgs)
    _, again = m.call(msgs)
    assert len(calls) == 1 and (first["hits"], again["hits"]) == (0, 1)
    same = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "images", "seconds")
    assert {k: first[k] for k in same} == {k: again[k] for k in same} and again["prompt_tokens"] == 100
    assert (first["usd"], again["usd"], again["cached_usd"]) == (pytest.approx(0.01), 0.0, pytest.approx(0.01))


def test_budget_cap_refuses_the_call_that_would_pass_it(tmp_path, monkeypatch):
    m, calls = llm(tmp_path, monkeypatch, cost=0.02)
    m.chat([{"role": "user", "content": "a"}])  # price unknown: allowed
    with pytest.raises(BudgetExceeded):
        m.chat([{"role": "user", "content": "b"}])  # 0.02 + 0.02 > 0.025
    assert len(calls) == 1 and m.usage["usd"] <= 0.025


def test_budget_is_shared_across_threads(tmp_path, monkeypatch):
    """Review G_M4: $0.25 cap at $0.10 a call spent $0.40. One shared client now stops at the cap."""
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    tr, calls = fake(cost=0.10)
    m = LLM(cache_dir=tmp_path, max_usd=0.25, transport=tr)
    m.chat([{"role": "user", "content": "first"}])  # learns the price

    def one(i):
        try:
            return m.chat([{"role": "user", "content": str(i)}])
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, range(8)))
    assert m.usage["usd"] <= 0.25 + 1e-9 and len(calls) == 2


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("arbiter_kb.llm.load_env", lambda: {})
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY missing"):
        LLM(cache_dir=tmp_path, transport=fake()[0]).chat([{"role": "user", "content": "x"}])


def test_load_env_parses_without_printing(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# c\nOPENROUTER_API_KEY='abc'\n\nARBITER_MAX_USD=1.5\n")
    assert load_env(p) == {"OPENROUTER_API_KEY": "abc", "ARBITER_MAX_USD": "1.5"}


@pytest.mark.parametrize("arm", C.ARMS)
def test_system_prompts_share_the_robot_text(arm):
    s = C.system_prompt(arm)
    assert s.startswith("You control a simulated Franka Panda") and '"cites"' in s
    assert "moves with that container" in s  # the same rule for every arm
    assert ("event log" in s) == (arm == "K")
    assert ("no depth" in s) == (arm in ("H", "S"))


def test_user_content_has_images_and_state():
    rgb = np.zeros((8, 8, 3), np.uint8)
    parts = C.user_content("K", "pick it", "[e1] t=3 ...", [(rgb, "Current front"), (rgb, "Current wrist")], {"tcp": [0, 0, 0]})
    assert count_images([{"role": "user", "content": parts}]) == 2
    assert parts[1]["text"].startswith("[e1]") and parts[0]["text"] == "Task goal: pick it" and parts[-1]["text"].startswith("Robot state")


@pytest.mark.parametrize("text, kind", [
    ('```json\n{"waypoint": [0.1, 0, 0.2, -3.14, 0, 0], "gripper": -1}\n```', None),
    ('{"waypoint": null}', "declared_complete"), ("no json here", "unparseable"),
    ('{"waypoint": [0.1, 0, 0.2], "gripper": 1}', "invalid_action"),
    ('{"waypoint": [0.1, 0, 0.2, 0, 0, 0], "gripper": 2}', "invalid_action")])
def test_parse_and_to_action(text, kind):
    a, k = C.to_action(C.parse_answer(text))
    assert k == kind and (a is None) == (kind is not None)


def test_check_workspace_reach_and_ids():
    ok = np.array([0.0, 0.1, 0.1, -3.14, 0, 0, 1])
    known = {"bin_0", "e3", "o173:bin_0", "o9:ee"}
    assert C.check(ok, {"target": "bin_0", "cites": ["e3", "o173:bin_0", "o9:ee"]}, known) == []
    assert any("o5:bin_9" in e for e in C.check(ok, {"cites": ["o5:bin_9"]}, known))
    far = np.array([0.55, 0.55, 0.05, 0, 0, 0, 0])
    errs = C.check(far, {"target": "bin_9", "cites": ["e99"]}, {"bin_0"})
    assert any("reach" in e for e in errs) and any("unknown target" in e for e in errs) and any("e99" in e for e in errs)
    assert any("outside" in e for e in C.check(np.array([0, 0, -0.1, 0, 0, 0, 0]), {}, set()))


def test_invented_observation_ids_are_rejected():
    """Review G_M4: o<t>:<object> used to pass for any t; only ids the prompt printed may be cited."""
    text = ("[bin_0] container, visible at 0.1, 0.0, 0.03, last seen t=173 [o173:bin_0]\n"
            "[Inside(cube_red)] bin_0 0.92 | elsewhere 0.08  [e10,e15,o32:bin_0]\n[On(cube_red, bin_0)] 0.9 [o40:ee]")
    known = C.shown_ids(text, {"bin_0", "bin_1", "cube_red"})
    assert {"bin_0", "cube_red", "o173:bin_0", "e10", "e15", "o32:bin_0", "On(cube_red, bin_0)", "o40:ee"} <= known
    assert "bin_1" not in known  # an object the text never names
    ok = np.array([0.0, 0.1, 0.1, -3.14, 0, 0, 1])
    for bad in ("o99999:bin_0", "o174:bin_0", "o173:image", "o32:bin_1", 5):
        assert C.check(ok, {"target": "bin_0", "cites": ["e10", bad]}, known) == [f"unknown cite {bad!r}"]


def test_target_distance_is_logged_only():
    a = np.array([0.1, 0.1, 0.2, 0, 0, 0, 0])
    assert C.target_distance(a, {"target": "bin_0"}, lambda k: np.array([0.1, 0.13, 0.03])) == pytest.approx(0.03)
    assert C.target_distance(a, {"target": None}, lambda k: None) is None


def test_raw_tracker_log_samples_and_keeps_the_last_frame():
    from arbiter_kb.sources.synthetic import SyntheticSource

    src = SyntheticSource()
    frames = list(src.frames(src.episodes("train", 1)[0]))
    log = C.raw_tracker_log(frames, stride=10).splitlines()
    assert log[0].startswith("t=0:") and log[-1].startswith(f"t={frames[-1].t}:") and "hidden" in log[-1]
    assert "Inside" not in "\n".join(log) and json.dumps(log)


def test_k0_table_and_s_sentence_carry_no_history():
    from arbiter_kb.sources.synthetic import SyntheticSource

    src = SyntheticSource()
    last = list(src.frames(src.episodes("train", 1)[0]))[-1]
    table = C.object_table(last)
    assert "bin_0 (container) at" in table and "cube" not in table  # cubes are hidden now: K0 cannot know them
    assert C.one_sentence({"cube_red": "bin_0", "cube_blue": None}) == "cube_red is under bin_0."
    assert C.one_sentence({}) == "No object is hidden."
