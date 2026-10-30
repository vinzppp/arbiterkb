"""The closed loop (closed_loop.py) on a fake RoboMME env with the same observation keys: endings, arm H's chained
conversation, the fresh per-step request of the memory arms, one retry on check() problems."""
import json

import numpy as np

from arbiter_kb.closed_loop import NOOP_TEXT, Memory, fingers_closed, frame_indices, gripper_closed, run
from arbiter_kb.params import Params
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem
from arbiter_kb.types import EpisodeMeta, FrameObs, ObjObs


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
    assert row["steps"][0]["description"] == "go" and "target" not in row["steps"][0]  # what the model said, logged


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


def test_a_limit_raised_by_ask_ends_the_episode_as_a_failure():
    from arbiter_kb.closed_loop import Limit

    def ask(msgs, effort):
        raise Limit("episode_usd")
    row = run(FakeEnv([]), "H", ask)
    assert row["end_reason"] == "limit:episode_usd" and not row["success"] and row["model_calls"] == 0


def test_fixed_first_layout_reaches_the_fresh_arms_only():
    def perceive(obs):
        return [FrameObs(0, {}, np.zeros(3), 0.08, False)]
    perceive.objects = lambda: []
    ask, sent = scripted([WP, WP])
    run(FakeEnv([(5, "ongoing"), (5, "success")]), "K", ask, Memory("K", ArbiterKBSystem(Params()), perceive),
        layout="fixed_first", breakpoint=True)
    firsts = [m[1]["content"][:2] for m in sent]
    assert firsts[0] == firsts[1] and firsts[0][0]["text"] == "Task goal: pick up the red cube"
    assert firsts[0][1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    ask, sent = scripted([WP])
    run(FakeEnv([(5, "success")]), "H", ask, layout="fixed_first", breakpoint=True)
    assert "prompt_cache_breakpoint" not in json.dumps(sent)


def test_gripper_keep_holds_the_state_before_it():
    """NOTES "Bug: keep recorded as open": gripper 0 while holding was recorded as open, read as a release."""
    assert [gripper_closed(c, b) for c, b in ((-1, False), (1, True), (0, True), (0, False))] == [True, False, True,
                                                                                                    False]
    assert not fingers_closed({"gripper_state_list": [[0.04, 0.04]]})
    assert fingers_closed({"gripper_state_list": [[0.002, 0.002]]})

    def perceive(obs):
        return [FrameObs(0, {}, np.zeros(3), 0.08, False)]
    perceive.objects = lambda: []
    mem = Memory("K", ArbiterKBSystem(Params()), perceive)
    wp = lambda g: json.dumps({"description": "x", "waypoint": [0.0, 0.0, 0.2, -3.1416, 0, 0], "gripper": g})  # noqa
    ask, _ = scripted([wp(-1), wp(0), wp(0), wp(1), wp(0)])
    run(FakeEnv([(5, "ongoing")] * 4 + [(5, "success")]), "K", ask, mem)
    closed = [d["gripper_closed"] for d in mem.system.kb.task["decisions"]]
    assert closed == [True, True, True, False, False]  # with the bug: [True, False, False, False, False]


# ---- arm K2 (view_k2.py): the initial block and earlier log messages never change, so each request starts with the
# previous one; the current state comes last; changes are logged once.


def _frames(t0: int, n: int, x0: float, x1: float):
    """bin_0 slides from x0 to x1; the red cube is visible from t=2 on."""
    out = []
    for k in range(n):
        t = t0 + k
        x = x0 + (x1 - x0) * k / max(n - 1, 1)
        objs = {"bin_0": ObjObs("bin_0", "container", True, np.array([x, 0.1, 0.03]), None, 1.0),
                "cube_red": ObjObs("cube_red", "cube", t >= 2, np.array([0.2, -0.1, 0.02]) if t >= 2 else None,
                                   None, 1.0)}
        out.append(FrameObs(t, objs, np.array([0.0, 0.0, 0.45]), 0.08, False))
    return out


def _memory(batches):
    it = iter(batches)

    def perceive(obs):
        return next(it, [])
    perceive.objects = lambda: [("bin_0", "container", {}), ("cube_red", "cube", {"color": "red"})]
    return Memory("K2", ArbiterKBSystem(Params()), perceive)


def test_each_request_starts_with_the_previous_one_and_ends_with_the_current_state():
    mem = _memory([_frames(0, 12, 0.0, 0.0), _frames(12, 12, 0.0, 0.15), _frames(24, 12, 0.15, 0.15)])
    ask, sent = scripted([WP, WP, WP])
    run(FakeEnv([(5, "ongoing"), (5, "ongoing"), (5, "success")]), "K2", ask, mem, breakpoint=True)
    users = [m[1:] for m in sent]
    for a, b in zip(users, users[1:]):
        assert b[:len(a) - 1] == a[:-1]  # every earlier message is byte-identical: a cacheable prefix
    last = users[-1]
    assert last[0]["content"][0]["text"] == "Task goal: pick up the red cube"
    assert last[0]["content"][2]["text"].startswith("Initial state (t=0")
    assert last[-1]["content"][0]["text"].startswith("Now, t=")
    assert all(m["content"][-1].get("prompt_cache_breakpoint") for m in last[:-1])
    assert "prompt_cache_breakpoint" not in json.dumps(last[-1])
    log = "\n".join(m["content"][0]["text"] for m in last[1:-1])
    assert "bin_0 moved by (0.15" in log and log.count("cube_red seen at") == 1


def test_initial_lists_only_first_frame_objects_and_current_lists_the_goal_items():
    from arbiter_kb.view_k2 import K2View
    s = ArbiterKBSystem(Params())
    s.reset(EpisodeMeta("t", 0, "train", "pick up the container that hides the red cube",
                        [("bin_0", "container", {}), ("cube_red", "cube", {"color": "red"})], None, None))
    for f in _frames(0, 6, 0.0, 0.0):
        s.step(f)
    a, logs, c = K2View().update(s)
    assert "cube_red" not in a and "[bin_0] container, seen at (0.000, 0.100, 0.030)" in a  # red: first seen at t=2
    assert "1. red cube [cube_red]: visible" in c and "hand at (0.000, 0.000, 0.450), finger opening 0.080 m, holding nothing" in c


def test_k2_logs_the_hold_numbers_decisions_and_belief_ids_and_r4_clears_the_lifted_bin():
    """A cube covered under bin_0, another under bin_1 with some mass on bin_0; bin_0 is grasped, lifted, the first
    cube is seen at bin_0's spot (R4: the second cube loses its bin_0 mass), then bin_0 is released."""
    from arbiter_kb.self_memory import record
    from arbiter_kb.view_k2 import K2View
    s = ArbiterKBSystem(Params())
    s.reset(EpisodeMeta("t", 0, "train", "pick up the container that hides the red cube",
                        [("bin_0", "container", {}), ("bin_1", "container", {}), ("cube_red", "cube", {"color": "red"}),
                         ("cube_blue", "cube", {"color": "blue"})], None, None))
    b0, b1, red, blue = np.array([0.0, 0.1, 0.03]), np.array([0.0, -0.1, 0.03]), np.array([0.0, 0.1, 0.017]), \
        np.array([0.0, -0.1, 0.017])

    def frame(t, ee, width, bin0, cubes_seen, red_seen=False):
        o = {"bin_0": ObjObs("bin_0", "container", t >= 3, bin0 if t >= 3 else None, None, 1.0),
             "bin_1": ObjObs("bin_1", "container", t >= 3, b1 if t >= 3 else None, None, 1.0),
             "cube_red": ObjObs("cube_red", "cube", cubes_seen or red_seen, red if cubes_seen or red_seen else None,
                                None, 1.0),
             "cube_blue": ObjObs("cube_blue", "cube", cubes_seen, blue if cubes_seen else None, None, 1.0)}
        return FrameObs(t, o, np.asarray(ee, float), width, False)
    view = K2View()
    for t in range(6):
        s.step(frame(t, [0, 0, 0.45], 0.08, b0, t < 3))
    from arbiter_kb.beliefs import Distribution
    from arbiter_kb.kb import SetDist
    d = s.kb.relations[("Inside", "cube_blue")]
    s.kb.apply([SetDist("Inside", "cube_blue", Distribution({"bin_1": 0.8, "bin_0": 0.2}, 5, d.support,
                                                           {"bin_1": d.support, "bin_0": d.support}))], "test")
    view.update(s)
    record(s.kb, 5, np.array([0.0, 0.1, 0.03, -3.1416, 0, 0]), True, "bin_0", ())
    path = [([0, 0.1, 0.03], 0.01, b0), ([0, 0.1, 0.03], 0.01, b0)] + \
        [([0, 0.1, z], 0.01, np.array([0, 0.1, z])) for z in (0.1, 0.2, 0.25)] + \
        [([0.2, 0.2, 0.04], 0.01, np.array([0.2, 0.2, 0.04])), ([0.2, 0.2, 0.04], 0.08, np.array([0.2, 0.2, 0.04]))]
    for k, (ee, w, bpos) in enumerate(path, 6):
        s.step(frame(k, ee, w, bpos, False, red_seen=k >= 9))
    a, logs, c = view.update(s)
    log = logs[-1]
    assert "[d1] t=5 you: waypoint (0.000, 0.100, 0.030), gripper closed after it, target bin_0" in log
    assert any(v.type == "reveal" and v.args == ("bin_0", "cube_red") for v in s.kb.events)
    assert "bin_0" not in s.kb.relations[("Inside", "cube_blue")].probs  # R4: bin_0 hid the red cube
    assert "Inside(cube_blue): bin_1" in log and "replaces [b" in log
    assert "highest hand z while held 0.250" in log and "t=12 [bin_0] released, hand at (0.200, 0.200, 0.040)" in log
    assert "1. red cube [cube_red]: seen under [bin_0]; [bin_0]: grasped t=" in c
    from arbiter_kb.views import controller_view  # K: the revealed cube is a fact, not an open belief
    k = controller_view(s.kb)
    assert "[Inside(cube_red)] was under bin_0, revealed t=" in k and "Inside(cube_red)] bin_0" not in k
    assert "hid the red cube (revealed t=" in k and "may hold the red cube" not in k
    assert "the red cube (task item): visible; it was under bin_0 (revealed t=" in k


def test_k_lists_a_cube_seen_again_without_a_reveal_as_a_fact():
    from arbiter_kb.views import controller_view
    s = ArbiterKBSystem(Params())
    s.reset(EpisodeMeta("t", 0, "train", "pick up the red cube",
                        [("bin_0", "container", {}), ("cube_red", "cube", {"color": "red"})], None, None))
    for t in range(12):  # the cube is covered at t=3 by bin_0, seen again at t=8 at another spot, nothing held
        cube = t < 3 or t >= 8
        objs = {"bin_0": ObjObs("bin_0", "container", True, np.array([0.0, 0.1, 0.03]), None, 1.0),
                "cube_red": ObjObs("cube_red", "cube", cube, np.array([0.0, 0.1 if t < 3 else 0.3, 0.02])
                                   if cube else None, None, 1.0)}
        s.step(FrameObs(t, objs, np.array([0.0, 0.0, 0.45]), 0.08, False))
    k = controller_view(s.kb)
    assert "[Inside(cube_red)] seen again at t=8, now at 0.000, 0.300, 0.020; the earlier belief (bin_0" in k
    assert "may hold the red cube" not in k


def test_moves_below_the_noise_threshold_are_not_reported():
    from arbiter_kb.motion import reported
    assert not reported([0, 0, 0], [0.03, 0.02, 0.0]) and reported([0, 0, 0], [0.04, 0.02, 0.0])
