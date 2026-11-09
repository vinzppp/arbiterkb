"""scripts/k2_offline.py's scoring and pairing (eval review): a malformed answer scores 0 and never raises, ids in
brackets match, and the comparison uses only calls both arms answered."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import k2_offline as K  # noqa: E402

ITEM = {"condition": "clean", "episode": 1, "call": 1, "right": "bin_1", "A": "bin_1", "B": "bin_0", "xy": [0.1, 0.2]}


def _ans(target, wp):
    return json.dumps({"description": "go", "waypoint": wp, "gripper": 0, "target": target})


def test_malformed_answers_score_zero_and_brackets_match():
    assert K.score(ITEM, "K", _ans("[bin_1]", [0.1, 0.2, 0.17, -3.14, 0, 0]), {})["both"] == 1
    for text in (None, "no json", _ans("bin_1", ["a", 0, 0, 0, 0, 0]), _ans(["bin_1"], [0.1, 0.2, 0.17, 0, 0, 0]),
                 _ans("bin_1", [0.1, 0.2])):
        assert K.score(ITEM, "K", text, {})["both"] == 0


def test_pairs_need_both_arms():
    ok = [K.score({**ITEM, "episode": e, "call": c}, a, _ans("bin_1", [0.1, 0.2, 0.17, 0, 0, 0]), {})
          for e in (1, 2) for c in (1, 2) for a in K.ARMS]
    lost = K.score({**ITEM, "episode": 3, "call": 1}, "K", None, {}, "budget")
    alone = K.score({**ITEM, "episode": 3, "call": 1}, "K2", _ans("bin_1", [0.1, 0.2, 0.17, 0, 0, 0]), {})
    p = K.paired(ok + [lost, alone], "clean")
    assert p["pairs"] == 4 and p["unpaired"] == 1 and p["episodes"] == 2 and p["diff"] == 0.0
