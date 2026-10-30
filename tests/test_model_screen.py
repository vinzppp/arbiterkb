"""scripts/model_screen.py and screen_rules.py: the stage rules on hand-made answers, malformed answers score 0 and
never raise, the selected episodes' states, and the whole scoring path with a fake model (two passes, resume)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model_screen as M  # noqa: E402
import screen_rules as R  # noqa: E402

A, B, HAND = [-0.12, 0.07], [0.11, -0.03], [0.0, 0.0]
TRUTH = {"A": A, "B": B, "hand": HAND, "others": {"bin_0": B, "bin_2": [-0.07, -0.08]}}


def _ans(xy, z, g, wp=True):
    return json.dumps({"waypoint": [*xy, z, -3.1416, 0, 0] if wp else None, "gripper": g, "target": "bin_1"})


@pytest.mark.parametrize("stage,text,ok", [
    (1, _ans(A, 0.2, 1), 1), (1, _ans(A, 0.2, -1), 0), (1, _ans([A[0] + 0.04, A[1]], 0.2, 0), 0),
    (1, _ans(A, 0.06, -1), 1),  # direct grasp from home
    (1, _ans(A, 0.08, 0), 0),  # neither above (z < 0.10) nor a grasp
    (2, _ans(A, R.GRASP_Z, -1), 1), (2, _ans(A, 0.10, -1), 0), (2, _ans([A[0] + 0.015, A[1]], 0.04, -1), 0),
    (2, _ans(A, 0.06, 0), 1),  # descend first, close next
    (3, _ans(A, 0.20, 0), 1), (3, _ans(A, 0.14, 0), 0), (3, _ans(A, 0.20, 1), 0),
    (4, _ans([-0.12, 0.22], 0.07, 1), 1), (4, _ans([-0.08, -0.05], 0.07, 1), 0), (4, _ans([-0.12, 0.22], 0.20, 1), 0),
    (5, _ans(B, 0.2, 0), 1), (5, _ans(A, 0.2, 0), 0), (6, _ans(B, R.GRASP_Z, -1), 1),
    (7, _ans(B, 0, 0, wp=False), 1), (7, _ans(B, 0.2, 0), 0), (3, _ans(A, 0, 0, wp=False), 0)])
def test_stage_rules(stage, text, ok):
    truth = {**TRUTH, "hand": A if stage in (2, 3, 4) else HAND}
    assert R.score(stage, truth, text)["passed"] == ok


def test_malformed_answers_score_zero_and_never_raise():
    for text in (None, "", "prose only", '{"waypoint": "up"}', '{"waypoint": [1, 2]}', '{"waypoint": [0,0,0.2,0,0,0],'
                 ' "gripper": 5}', '{"waypoint": [0,0,0.2,0,0,0], "gripper": -1, "target": ["x"]}', "[1, 2, 3]"):
        r = R.score(2, TRUTH, text)
        assert r["passed"] == 0 and isinstance(r["reason"], str)
    assert R.score(1, TRUTH, _ans([0.9, 0.0], 0.2, 1))["reason"].startswith("x=")  # outside the workspace


@pytest.fixture(scope="module")
def items():
    return M.select(M.SyntheticSource(swap_steps="mix", lane_offset="mix"), M.load_params(), 100, 2)


def test_states(items):
    assert len(items) == 14 and [x["stage"] for x in items[:7]] == list(range(1, 8))
    text = ["\n".join(p["text"] for p in x["messages"][1]["content"]) for x in items[:7]]
    assert '"finger_positions_m": [0.005, 0.005]' in text[2]  # stage 3 is asked after the scripted close on A
    assert "pick_up bin" in text[3] and "-> done (pick_up" in text[3]  # the KB saw that grasp (SCRIPT_GRASP_Z)
    assert items[2]["truth"]["hand"] == items[2]["truth"]["A"]
    assert items[3]["A"] not in items[3]["truth"]["others"] and items[4]["A"] in items[4]["truth"]["others"]
    assert all(R.score(x["stage"], x["truth"], R.canonical(x))["passed"] for x in items)


def test_mock_run_two_passes_and_resume(items, tmp_path):
    cand = ("mock/oracle", "mock", None, None, 0.0, 0.0, 1000)
    rows = M.run_model(cand, items, 2, 1.0, R.mock_transport(items), tmp_path)
    s = M.summarize(rows, cand)
    assert s["calls"] == 28 and s["errors"] == 0 and s["pass_all"] == round(12 / 14, 3) and s["agreement"] == 1.0
    assert s["stage_pass"]["2_grasp_A"] == 0.5 and s["valid_json"] == round(13 / 14, 3)
    M.write(tmp_path / "out", rows, s)
    assert "premature done" in (tmp_path / "out" / "report.md").read_text()
    again = M.run_model(cand, items, 2, 0.0, R.mock_transport(items), tmp_path)  # cap 0: only cache replays
    assert [r["passed"] for r in again] == [r["passed"] for r in rows] and all(r["error"] is None for r in again)


def test_budget_rows_are_not_scored(items, tmp_path):
    cand = ("mock/oracle", "mock", None, None, 0.0, 0.0, 1000)
    rows = M.run_model(cand, items[:3], 1, 0.0, R.mock_transport(items), tmp_path)
    assert all(r["error"] == "budget" and r["passed"] is None for r in rows)
    assert M.summarize(rows, cand)["answered"] == 0


def test_choice_uses_stages_1_to_6_and_reports_stage_7_apart():
    rows = [{"stage": s, "passed": int(s != 7), "valid_json": 1} for s in range(1, 8) for _ in range(10)]
    sel = R.selection(rows)
    assert sel["calls_1_6"] == 60 and sel["pass_1_6"] == 1.0 and sel["stage7_done"] == 0.0 and sel["meets"]
    low = [{**r, "passed": 0} if r["stage"] == 3 and i % 10 < 5 else r for i, r in enumerate(rows)]  # stage 3: 0.5
    assert not R.selection(low)["meets"]
    s = lambda m, g, act, usd, meets: {"model": m, "group": g, "active_B": act, "usd_per_call": usd, "meets": meets}  # noqa: E731
    assert R.choose([s("big", "small", 17, 0.001, True), s("cheap", "small", 10, 0.002, True),
                     s("fail", "small", 3, 0.0001, False), s("ceil", "strong", None, 0.1, True)])["model"] == "cheap"
    assert R.choose([s("a", "small", 10, 0.002, True), s("b", "small", 10, 0.001, True)])["model"] == "b"
    assert R.choose([s("x", "strong", None, 0.1, True)]) is None
