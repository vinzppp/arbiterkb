"""Frozen fairness and semantics checks for the memory x policy pilot."""
import importlib.util
from pathlib import Path

import numpy as np

from arbiter_kb.factorial_memory import (DeterministicWorldLedger, FactorialMemory,
                                         NormalizedTaskState)
from arbiter_kb.params import Params
from arbiter_kb.types import EpisodeMeta, FrameObs, ObjObs

POLICY_SPEC = importlib.util.spec_from_file_location(
    "common_policy", Path(__file__).resolve().parents[1] / "stress/common_policy.py")
POLICY = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY)
factorial_policy_sha, factorial_policy_text = POLICY.factorial_policy_sha, POLICY.factorial_policy_text


OBJECTS = [("bin_a", "container", {}), ("cube_red", "cube", {"color": "red"})]
META = EpisodeMeta("unit", 0, "val", "unit goal", OBJECTS, None, None)


def obj(oid, cls, xyz=None, uv=None):
    visible = xyz is not None
    return ObjObs(oid, cls, visible, None if xyz is None else np.asarray(xyz, float),
                  None if uv is None else np.asarray(uv, float), 1.0)


def frame(t, cube, bin_at=(0.0, 0.0, 0.03), ee=(0.0, 0.0, 0.2), width=0.08, video=True):
    return FrameObs(t, {
        "bin_a": obj("bin_a", "container", bin_at, (10, 20)),
        "cube_red": obj("cube_red", "cube", cube, (12, 22) if cube is not None else None),
    }, np.asarray(ee, float), width, video)


def test_factorial_policy_bytes_depend_only_on_policy_factor_and_task():
    for task in ("VideoRepick", "VideoUnmaskSwap"):
        for kind in ("generic", "matched"):
            text = factorial_policy_text(task, kind)
            # Both memory arms receive this exact byte string; no arm is an input.
            assert factorial_policy_sha(task, kind) == factorial_policy_sha(task, kind)
            assert text.encode() == factorial_policy_text(task, kind).encode()
    assert factorial_policy_text("VideoRepick", "generic") == factorial_policy_text(
        "VideoUnmaskSwap", "generic")


def test_ledger_records_cover_follows_track_and_clears_on_reappearance():
    params, ledger, state = Params(), DeterministicWorldLedger(), NormalizedTaskState(Params())
    ledger.reset(META)
    state.reset(META)
    visible = frame(0, (0.01, 0.00, 0.017))
    hidden = frame(1, None)
    ledger.step(visible, state.step(visible))
    ledger.step(hidden, state.step(hidden))
    assert ledger.inside == {"cube_red": "bin_a"}
    moved = frame(2, None, bin_at=(0.25, 0.10, 0.03))
    ledger.step(moved, state.step(moved))
    assert "[cube_red] -> bin_a" in ledger.text()
    again = frame(3, (0.25, 0.10, 0.017), bin_at=(0.25, 0.10, 0.03))
    ledger.step(again, state.step(again))
    assert ledger.inside == {}
    assert ledger.beliefs()[0]["answer"] is None


def test_ledger_explicitly_keeps_unknown_when_no_cover_is_supported():
    ledger, state = DeterministicWorldLedger(), NormalizedTaskState(Params())
    ledger.reset(META)
    state.reset(META)
    visible = frame(0, (0.30, 0.30, 0.017))
    hidden = frame(1, None)
    ledger.step(visible, state.step(visible))
    ledger.step(hidden, state.step(hidden))
    assert ledger.inside["cube_red"] is None
    assert "[cube_red] -> unknown" in ledger.text()


def test_normalized_task_state_excludes_demo_from_execution_progress():
    state = NormalizedTaskState(Params())
    state.reset(META)
    # The demo close identifies the cube, but must never increment execution progress.
    state.step(frame(0, (0.0, 0.0, 0.017), ee=(0.0, 0.0, 0.017), width=0.08, video=True))
    state.step(frame(1, (0.0, 0.0, 0.017), ee=(0.0, 0.0, 0.017), width=0.01, video=True))
    assert state.demo_target == "cube_red"
    assert state.counts() == {"pick_up": 0, "release": 0, "press": 0, "hold_above": 0}

    # A settled execution grasp is counted by the separate execution detector.
    for t, width in enumerate((0.08, 0.03, 0.0366, 0.0366), 2):
        state.step(frame(t, (0.0, 0.0, 0.017), ee=(0.0, 0.0, 0.017), width=width, video=False))
    assert state.counts()["pick_up"] == 1
    assert state.audit()["demo_target"] == "cube_red"


def test_arbiter_adapter_suppresses_demo_progress_from_legacy_view():
    """Regression for the mixed demo/execution progress path present on main."""
    memory = FactorialMemory("arbiter", Params())
    memory.reset(META)
    demo = [frame(t, (0.0, 0.0, 0.017), ee=(0.0, 0.0, 0.017), width=width, video=True)
            for t, width in enumerate((0.08, 0.03, 0.0366, 0.0366))]
    memory.feed(demo)

    # The frozen KB path has one detector for both phases and therefore counts
    # the demonstration action.  The research adapter must expose only the
    # shared execution-only count.
    from arbiter_kb.views import controller_view
    assert "pick-ups 1" in controller_view(memory.world.kb)
    fixed = memory.text()
    assert "Execution progress: pick-ups 0" in fixed
    assert "\nProgress:" not in fixed
    assert "picked up an object" not in fixed
