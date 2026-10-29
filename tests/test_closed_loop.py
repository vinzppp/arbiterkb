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


def test_h_frames_go_into_grids_and_every_frame_is_kept():
    from arbiter_kb.closed_loop import GRID, grid, h_content
    f = [np.full((4, 4, 3), k, np.uint8) for k in range(1, 6)]
    g = grid(f)
    assert g.shape == (4 * GRID, 4 * GRID, 3) and g[0, 4, 0] == 2 and g[4, 0, 0] == 4 and g[-1, -1, 0] == 0
    env = FakeEnv([])
    obs, info = env.reset()
    obs["front_rgb_list"] = [np.zeros((4, 4, 3), np.uint8)] * 1078  # VideoPlaceOrder's opening video
    images = [p for p in h_content(obs, info, "Video so far", True) if p["type"] == "image_url"]
    assert len(images) == 40 + 2  # 360 sampled frames in 40 grids, then the current front frame and the wrist frame


def test_r_targets_come_from_the_frames_and_never_seen_slots_are_left_out():
    from arbiter_kb.types import ObjObs

    def perceive(obs):  # closed-loop smoke run: every R target was "unknown" (the log prints bare ids)
        return [FrameObs(0, {"bin_0": ObjObs("bin_0", "container", True, np.zeros(3), None, 1.0),
                             "cube_s9": ObjObs("cube_s9", "cube", False, None, None, 1.0)}, np.zeros(3), 0.08, False)]
    perceive.objects = lambda: []
    tgt = json.dumps({"description": "go", "waypoint": [0.0, 0.0, 0.2, -3.1416, 0, 0], "gripper": 0, "target": "bin_0",
                      "cites": ["t=20: bin_0 0.008, 0.018, 0.020"]})  # a whole log line: R has no ids to cite
    ask, sent = scripted([tgt])
    row = run(FakeEnv([(5, "success")]), "R", ask, Memory("R", None, perceive))
    assert row["success"] and row["model_calls"] == 1
    log = json.dumps(sent[0][1]["content"])
    assert "bin_0" in log and "cube_s9" not in log
