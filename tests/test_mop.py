"""Memory arms and fairness audit of the "memory or policy?" study."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from arbiter_kb.mop_memory import KBMemory, PreregLedger
from arbiter_kb.params import Params
from arbiter_kb.types import EpisodeMeta, FrameObs, ObjObs

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = [("bin_0", "container", {}), ("cube_red", "cube", {"color": "red"}), ("button_0", "button", {})]
META = EpisodeMeta("unit", 0, "val", "unit goal", OBJECTS, None, None)
MENU = [{"label": "a", "action": "pick up the cube", "need_parameter": True},
        {"label": "b", "action": "press the button", "need_parameter": False}]


def frame(t, cube, video, ee=(0.0, 0.0, 0.2), width=0.08):
    def obj(oid, cls, xyz, uv):
        return ObjObs(oid, cls, xyz is not None, None if xyz is None else np.asarray(xyz, float),
                      None if uv is None else np.asarray(uv, float), 1.0)
    return FrameObs(t, {"bin_0": obj("bin_0", "container", (0.0, 0.0, 0.03), (10, 20)),
                        "cube_red": obj("cube_red", "cube", cube, (12, 22) if cube is not None else None),
                        "button_0": obj("button_0", "button", (0.2, 0.0, 0.01), (30, 40))},
                    np.asarray(ee, float), width, video)


def grasp(t0, video):
    return [frame(t0 + i, (0.0, 0.0, 0.017), video, ee=(0.0, 0.0, 0.017), width=w)
            for i, w in enumerate((0.08, 0.03, 0.0366, 0.0366))]


def test_ledger_separates_demonstration_from_executed_options():
    ledger = PreregLedger(Params())
    ledger.reset(META)
    ledger.feed(grasp(0, video=True))
    ledger.chose({"choice": "b", "point": None}, MENU)
    ledger.executed("executed", "ongoing")
    text = ledger.text()
    demo, executed = text.split("DEMONSTRATION")[1].split("Executed options")
    assert "pick up" in demo and "nearest object cube_red" in demo
    assert "1. b: press the button -> executed, status ongoing" in executed
    assert "b: press the button: 1" in executed and "pick up the cube: 1" not in executed
    assert "[button_0] button: visible at pixel [40, 30]" in text


def test_ledger_records_cover_and_never_recommends():
    ledger = PreregLedger(Params())
    ledger.reset(META)
    ledger.feed([frame(0, (0.0, 0.0, 0.017), True), frame(1, None, True)])
    text = ledger.text()
    assert "[cube_red] -> bin_0" in text
    spec = importlib.util.spec_from_file_location("mop_fairness", ROOT / "scripts/mop_fairness.py")
    fairness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fairness)
    assert not fairness.RECOMMEND.findall(text)


def test_kb_memory_counts_execution_only_and_keeps_buttons():
    kb = KBMemory(Params())
    kb.reset(META)
    kb.feed(grasp(0, video=True))
    kb.feed([frame(4, (0.0, 0.0, 0.017), False, ee=(0.0, 0.0, 0.017), width=0.08)])
    kb.feed(grasp(5, video=False))
    text = kb.text()
    world, rest = text.split("DEMONSTRATION")
    demo, other = rest.split("Other observed objects (outside")
    assert text.startswith("EXECUTION COUNTS (execution phase only; the demonstration is not counted): pick-ups 1,")
    assert "\nProgress:" not in text and "picked up an object" not in world
    assert demo.count("picked up an object") == 1 and "(pick-up #1)" in demo
    assert "[button_0] button: visible at pixel [40, 30]" in other
    assert "button_0" not in kb.system.kb.objects


def _fairness():
    spec = importlib.util.spec_from_file_location("mop_fairness", ROOT / "scripts/mop_fairness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe(fairness, mutate=None):
    scaffold, task = "SCAFFOLD", "VideoUnmaskSwap"
    notes = {"fh": "", "ld": "\nLD", "mm": "\nMM", "kb": "\nKB"}
    prompts = {}
    for mem in ("fh", "ld", "mm", "kb"):
        for pol in ("g", "m"):
            system = scaffold + notes[mem] + fairness.POLICY_HEAD + fairness.policy_text(task, pol)
            user = "turn" + (fairness.MEMORY_HEAD + f"payload {mem}" if mem != "fh" else "") + fairness.ANSWER + "{...}."
            prompts[f"{mem}_{pol}"] = [{"system": system, "messages": [{"role": "user", "content": user}]}]
    prompts["memi"] = [{"system": "stock", "messages": []}]
    if mutate:
        mutate(prompts)
    return {"task": task, "prompts": prompts, "stock_memi_system_ok": True}


def test_fairness_passes_identical_parts_and_catches_each_violation():
    fairness = _fairness()
    assert fairness.audit_task(_probe(fairness))["fails"] == []

    def memory_differs(p):
        p["ld_m"][0]["messages"][0]["content"] += " extra"

    def policy_differs(p):
        p["kb_m"][0]["system"] += " extra"

    def recommends(p):
        for arm in ("kb_g", "kb_m"):
            content = p[arm][0]["messages"][0]["content"]
            p[arm][0]["messages"][0]["content"] = content.replace("payload kb", "payload kb; pick bin_2 next")

    assert any(f.startswith("M1") for f in fairness.audit_task(_probe(fairness, memory_differs))["fails"])
    assert any(f.startswith("P1") for f in fairness.audit_task(_probe(fairness, policy_differs))["fails"])
    assert any(f.startswith("R1") for f in fairness.audit_task(_probe(fairness, recommends))["fails"])


def test_matched_policy_files_are_the_g0_split():
    summary = json.loads((ROOT / "results/memory_or_policy/g0/memi_split/split_summary.json").read_text())
    fairness = _fairness()
    import hashlib
    for task, row in summary["tasks"].items():
        assert hashlib.sha256(fairness.policy_text(task, "m").encode()).hexdigest() == row["policy_matched_sha256"]


def test_ledger_demo_events_skip_markers_drawn_at_the_hand():
    meta = EpisodeMeta("unit", 0, "val", "unit goal", OBJECTS + [("waypoint_0", "waypoint", {})], None, None)
    ledger = PreregLedger(Params())
    ledger.reset(meta)
    frames = grasp(0, video=True)
    for f in frames:
        f.objects["waypoint_0"] = ObjObs("waypoint_0", "waypoint", True, np.asarray(f.ee_xyz, float),
                                         np.asarray((50.0, 50.0)), 1.0)
    ledger.feed(frames)
    demo = ledger.text().split("DEMONSTRATION")[1].split("Executed options")[0]
    assert "nearest object cube_red" in demo and "nearest object waypoint_0" not in demo
