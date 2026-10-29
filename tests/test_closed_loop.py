"""The closed loop (closed_loop.py) on a fake RoboMME env with the same observation keys: endings, arm H's chained
conversation, the fresh per-step request of the memory arms, one retry on check() problems."""
import json

import numpy as np

from arbiter_kb.closed_loop import NOOP_TEXT, Memory, frame_indices, run
from arbiter_kb.params import Params
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem
from arbiter_kb.types import FrameObs


class FakeEnv:
    """Each step returns `frames` frames (0 = a no-op) and the scripted status."""

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
        if status == "error":
            return None, 0, False, False, {"status": "error", "error_message": "no plan"}
        return self.obs(frames), 0, status in ("success", "fail"), False, self.info(status)


def scripted(answers):
    sent = []

    def ask(msgs, effort):
        sent.append(msgs)
        return answers[min(len(sent) - 1, len(answers) - 1)], {"prompt_tokens": 10, "usd": 0.001}
    return ask, sent


WP = json.dumps({"description": "go", "waypoint": [0.0, 0.0, 0.2, -3.1416, 0, 0], "gripper": 0})


def test_success_and_h_chains_the_conversation():
    ask, sent = scripted([WP, WP])
    row = run(FakeEnv([(5, "ongoing"), (5, "success")]), "H", ask)
    assert row["success"] and row["end_reason"] == "env:success" and row["model_calls"] == 2
    assert len(sent[1]) == 4  # system, first user turn, assistant, follow-up


def test_endings_declared_complete_noops_env_error():
    done = json.dumps({"description": "done", "waypoint": None, "gripper": 0})
    assert run(FakeEnv([]), "H", scripted([done])[0])["end_reason"] == "model_answer:declared_complete"
    ask, sent = scripted([WP])
    row = run(FakeEnv([(0, "ongoing")] * 3), "H", ask)
    assert row["end_reason"] == "limit:3_consecutive_noops" and NOOP_TEXT in json.dumps(sent[-1][-1]["content"])
    assert run(FakeEnv([(0, "error")]), "H", scripted([WP])[0])["end_reason"] == "env_error"


def test_memory_arm_gets_a_fresh_request_with_its_self_view():
    def perceive(obs):
        return [FrameObs(0, {}, np.zeros(3), 0.08, False)]
    perceive.objects = lambda: []
    mem = Memory("K", ArbiterKBSystem(Params()), perceive)
    ask, sent = scripted([WP, WP, WP])
    run(FakeEnv([(5, "ongoing"), (5, "ongoing"), (5, "success")]), "K", ask, mem)
    assert all(len(m) == 2 for m in sent)  # system + one user message, every step
    assert "Since your last decision" in json.dumps(sent[1][1]["content"])


def test_one_retry_on_check_problems_then_execute():
    far = json.dumps({"description": "far", "waypoint": [2.0, 0.0, 0.2, -3.1416, 0, 0], "gripper": 0})
    ask, sent = scripted([far, WP])
    row = run(FakeEnv([(5, "success")]), "H", ask)
    assert row["success"] and row["model_calls"] == 2 and "not executed" in json.dumps(sent[1][-1]["content"])


def test_frame_indices_stride_and_last():
    assert frame_indices(8) == [0, 3, 6, 7] and frame_indices(1) == [0]


def test_memory_arm_retry_sends_the_problem_with_a_new_request():
    def perceive(obs):
        return [FrameObs(0, {}, np.zeros(3), 0.08, False)]
    perceive.objects = lambda: []
    far = json.dumps({"description": "far", "waypoint": [2.0, 0.0, 0.2, -3.1416, 0, 0], "gripper": 0})
    ask, sent = scripted([far, WP])
    row = run(FakeEnv([(5, "success")]), "K", ask, Memory("K", ArbiterKBSystem(Params()), perceive))
    assert row["success"] and len(sent) == 2 and "not executed" in json.dumps(sent[1][1]["content"])
    assert sent[0] != sent[1]  # a different request, so a cache cannot hand back the same answer
