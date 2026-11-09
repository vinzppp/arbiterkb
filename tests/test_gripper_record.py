"""NOTES "Grasps above z_low": a grasp from the settled finger opening, not the hand's height; K's decision record
written after the step with what executed; a pick_up intent only on a change from open to closed."""
import json

import numpy as np

from arbiter_kb.closed_loop import Memory, run
from arbiter_kb.detectors import Detectors
from arbiter_kb.kb import KB
from arbiter_kb.params import DetectParams, Params
from arbiter_kb.self_memory import intent, outcome, record
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem
from arbiter_kb.types import FrameObs


class StepEnv:
    """Each step returns `frames` frames (0 = a step that executed nothing) and the scripted status."""

    def __init__(self, script):
        self.script, self.k = list(script), 0

    def obs(self, n):
        img = np.zeros((4, 4, 3), np.uint8)
        return {"front_rgb_list": [img] * n, "wrist_rgb_list": [img] * max(n, 1), "eef_state_list": [[0.0] * 6],
                "gripper_state_list": [[0.04, 0.04]], "wrist_camera_extrinsic_list": [np.zeros((3, 4))],
                "front_camera_extrinsic_list": [np.zeros((3, 4))]}

    def info(self, status="ongoing"):
        return {"task_goal": ["pick up the red cube"], "elapsed_steps": self.k, "status": status,
                "wrist_camera_intrinsic": np.eye(3), "front_camera_intrinsic": np.eye(3)}

    def reset(self):
        return self.obs(7), self.info()

    def step(self, action):
        frames, status = self.script[self.k]
        self.k += 1
        return self.obs(frames), 0, status == "success", False, self.info(status)


def kinds(path):
    """Actions from a hand path [(z, opening)], one frame each."""
    d, out = Detectors(DetectParams()), []
    for t, (z, w) in enumerate(path):
        out += [e for e in d.step(FrameObs(t, {}, np.array([0.0, 0.0, z]), w, False))
                if e.kind in ("pick_up", "release", "press")]
    return [(e.kind, e.data.get("how")) for e in out]


def test_a_container_grasped_at_0062_is_a_pick_up():
    held = [(0.2, 0.08), (0.062, 0.08), (0.062, 0.03), (0.062, 0.012), (0.062, 0.012), (0.2, 0.012), (0.2, 0.08)]
    assert kinds(held) == [("pick_up", None), ("release", "dropped")]


def test_an_empty_close_is_not_a_pick_up_at_any_height():
    for z in (0.02, 0.059):  # the dataset's recoveries closed above the cube and settled at 0.000
        empty = [(0.2, 0.08), (z, 0.08), (z, 0.03), (z, 0.0), (z, 0.0), (0.2, 0.0), (0.2, 0.08)]
        assert ("pick_up", None) not in kinds(empty)


def test_a_cube_settles_after_a_few_frames():
    cube = [(0.02, 0.08), (0.02, 0.06), (0.02, 0.045), (0.02, 0.0366), (0.02, 0.0366), (0.1, 0.0366)]
    assert kinds(cube) == [("pick_up", None)]


def test_pick_up_intent_only_on_open_to_closed():
    kb = KB([("bin_0", "container", {})])
    assert intent(kb, [0, 0, 0.2], True) == "pick_up"
    record(kb, 1, [0, 0, 0.2], True)
    assert intent(kb, [0, 0, 0.2], True) == "move"  # a keep while closed and empty
    assert intent(kb, [0, 0, 0.03], True) == "press"  # low: a press, as before


def test_a_step_that_did_not_execute_is_recorded_so_and_keeps_the_gripper():
    def perceive(obs):
        return [FrameObs(0, {}, np.zeros(3), 0.08, False)]
    perceive.objects = lambda: []
    mem = Memory("K", ArbiterKBSystem(Params()), perceive)
    wp = lambda g: json.dumps({"description": "x", "waypoint": [0.0, 0.0, 0.2, -3.1416, 0, 0], "gripper": g})  # noqa
    answers = iter([wp(-1), wp(1), wp(0)])
    run(StepEnv([(0, "ongoing"), (5, "ongoing"), (5, "success")]), "K", lambda m, e: (next(answers), {}), mem)
    kb = mem.system.kb
    ds = kb.task["decisions"]
    assert [(d["gripper_closed"], d["executed"]) for d in ds] == [(False, False), (False, True), (False, True)]
    assert outcome(kb, ds[0]) == "not executed: the robot did not move"
