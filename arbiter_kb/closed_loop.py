"""The closed loop of arXiv 2610.17350 (github.com/jackbjed/embodied-astra-robomme, harness/astra_waypoint_eval_v3.py,
commit 79dcc7f, MIT), with the memory module as the only switch between arms (DEVELOPMENT §3.8, M5).

Kept from theirs: the episode loop (reset; ask; parse; step the env with a 7-vector waypoint + gripper; stop on
terminated / truncated / status success, fail, timeout or error), the endings that count as failures (declared complete
with "waypoint": null; an unparseable or invalid answer; an env error; 3 consecutive actions that execute no steps;
150 model calls), the no-op message, the frame stride (one front frame every 3 control steps plus the last), the state
text (TCP pose, finger positions, elapsed steps, camera calibration), and the system prompt (controller.py).
Ours: arm H chains the whole conversation by re-sending it (OpenRouter has no previous_response_id; the content is
the same); every other arm gets a fresh request per step with its memory text and the CURRENT front and wrist frames;
the KB reads the same stride-3 frames H sees (perceive), and its view starts with self_memory's section; our check()
problems get one retry, as an executor error would.
`env` is RoboMME's BenchmarkEnvBuilder(..., action_space="waypoint", max_steps=1300) env on the pod; the Mac tests use
a fake env with the same observation keys.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np

from .controller import (check, image_part, parse_answer, raw_tracker_log, shown_ids, system_prompt, text_part,
                         to_action, user_content)
from .self_memory import record, self_view
from .views import controller_view

MAX_CALLS, MAX_NOOPS, STRIDE = 150, 3, 3
GRID = 3  # arm H: sampled frames go GRID x GRID per image. OpenRouter's pre-flight counts 1,536 tokens per image
# (about 149 images per request at 32k output), while the model uses ~66 per 256-px frame; VideoPlaceOrder's opening
# video alone is 360 frames (smoke run). Ours, not 2610.17350's (they sent single frames); every frame is kept.
NOOP_TEXT = ("The last action executed no steps: the robot did not move and no time passed, so the scene is unchanged. "
             "Choose the next action.")  # theirs, verbatim
TERMINAL = ("success", "fail", "timeout", "error")
CLOSED_WIDTH = 0.01  # m: total finger opening below this reads as a closed gripper (Panda: ~0.08 open, ~0 closed)


def gripper_closed(command: float, before: bool) -> bool:
    """The gripper state a waypoint leaves: -1 closes, +1 opens, 0 keeps the state before it. Reading 0 as open
    recorded every "keep" while holding as a release in K's self-memory (NOTES "Bug: keep recorded as open")."""
    return True if command < 0 else False if command > 0 else before


def fingers_closed(obs: dict) -> bool:
    """The robot's own gripper state, from the finger positions (before the first open / close command)."""
    return float(np.sum(np.reshape(_np(obs["gripper_state_list"][-1]), -1)[:2])) < CLOSED_WIDTH


class Limit(Exception):
    """Raised by `ask` to end the episode as a failure, "limit:<reason>" (the pod runner's per-episode spend cap)."""


def frame_indices(n: int, stride: int = STRIDE) -> list[int]:
    """One frame in every `stride`, plus the last (current) frame (theirs)."""
    return sorted(set(range(0, n, stride)) | {n - 1})


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def state(obs: dict, info: dict, first: bool = False) -> dict:
    """Their state text: TCP pose, finger positions, elapsed control steps, camera calibration."""
    r = lambda a, nd=4: np.round(np.asarray(_np(a), float), nd).tolist()  # noqa: E731
    s = {"tcp_pose_xyz_rpy": r(np.reshape(_np(obs["eef_state_list"][-1]), -1)),
         "finger_positions_m": r(np.reshape(_np(obs["gripper_state_list"][-1]), -1)),
         "elapsed_control_steps": int(np.reshape(_np(info.get("elapsed_steps", 0)), -1)[-1]),
         "wrist_camera_K": r(np.reshape(_np(info["wrist_camera_intrinsic"]), (3, 3)), 2),
         "wrist_camera_extrinsic_world_to_camera": r(np.reshape(_np(obs["wrist_camera_extrinsic_list"][-1]), (3, 4)))}
    if first:
        s["front_camera_K"] = r(np.reshape(_np(info["front_camera_intrinsic"]), (3, 3)), 2)
        s["front_camera_extrinsic_world_to_camera"] = r(np.reshape(_np(obs["front_camera_extrinsic_list"][-1]), (3, 4)))
    return s


def grid(frames: list, g: int = GRID) -> np.ndarray:
    """g x g frames in reading order (left to right, top to bottom) in one image; empty cells black."""
    h, w = frames[0].shape[:2]
    out = np.zeros((g * h, g * w, 3), np.uint8)
    for k, f in enumerate(frames[:g * g]):
        out[k // g * h:(k // g + 1) * h, k % g * w:(k % g + 1) * w] = f
    return out


def h_content(obs: dict, info: dict, label: str, first: bool = False) -> list:
    """Arm H (theirs, with our grids): the sampled front frames of the video / the motion, GRID x GRID per image, the
    current front frame alone, the current wrist frame, the state."""
    fr, idx = obs["front_rgb_list"], frame_indices(len(obs["front_rgb_list"]))
    frames = [_np(fr[i]).astype(np.uint8) for i in idx]
    out = [text_part(f"{label}: front-camera frames {idx} of {len(fr)}, {GRID} x {GRID} per image in reading order "
                     "(left to right, top to bottom; black cells are empty), then the current front frame alone:")]
    out += [image_part(grid(frames[k:k + GRID * GRID])) for k in range(0, len(frames), GRID * GRID)]
    return out + [image_part(frames[-1]), text_part("Current wrist camera:"),
                  image_part(_np(obs["wrist_rgb_list"][-1]).astype(np.uint8)),
                  text_part("Robot state and calibration: " + json.dumps(state(obs, info, first)))]


class Memory:
    """What a non-H arm reads, rebuilt every step from the frames so far. `perceive(obs)` -> FrameObs for each
    sampled new frame (on the pod: the perception worker + sources.online.OnlineObs, or simulator truth)."""

    def __init__(self, arm: str, system, perceive: Callable):
        self.arm, self.system, self.perceive, self.frames = arm, system, perceive, []

    def feed(self, obs: dict) -> None:
        for f in self.perceive(obs):
            self.frames.append(f)
            if self.system is not None:
                self.system.step(f)

    def ids(self) -> set[str]:
        """The objects perception has seen at least once (R's log and its target check; never-seen slots are left out,
        as in K's view)."""
        return {k for f in self.frames for k, o in f.objects.items() if o.visible}

    def text(self) -> str:
        if self.arm == "R":
            return ("Raw tracker log (every 10th frame):\n" + raw_tracker_log(self.frames, keep=self.ids())
                    if self.frames else "")
        kb = self.system.kb
        return "\n\n".join(x for x in (self_view(kb), controller_view(kb, self.system.params.rule)) if x)


def run(env, arm: str, ask: Callable, memory: Memory | None = None, effort: dict | None = None, layout: str = "v1",
        breakpoint: bool = False) -> dict:
    """One episode. ask(messages, reasoning) -> (text, usage). Returns their result row plus our usage.
    layout / breakpoint: the fresh arms' request layout (controller.user_content); H is unaffected."""
    return Episode(env, arm, ask, memory, effort, layout, breakpoint).run()


class Episode:
    def __init__(self, env, arm: str, ask: Callable, memory: Memory | None, effort: dict | None, layout: str = "v1",
                 breakpoint: bool = False):
        self.env, self.arm, self.ask, self.memory, self.effort = env, arm, ask, memory, effort
        self.layout, self.breakpoint = layout, breakpoint
        self.steps, self.usage, self.calls, self.noops, self.retried, self.note = [], [], 0, 0, False, []
        self.status, self.end = "ongoing", None

    def run(self) -> dict:
        self.obs, self.info = self.env.reset()
        self.closed = fingers_closed(self.obs)
        self.goal, t0, m = self.info["task_goal"][0], time.time(), self.memory
        if m is not None:
            if m.system is not None:
                m.system.reset(memory_meta(m, self.goal))
            m.feed(self.obs)
        self.sys_msg = {"role": "system", "content": system_prompt(self.arm)}
        self.convo = [{"role": "user", "content": [text_part(f"Task goal: {self.goal}")]
                       + h_content(self.obs, self.info, "Video so far", True)}]
        while self.status == "ongoing":
            self.turn()
        tot = {k: round(sum(float(x.get(k, 0)) for x in self.usage), 4)
               for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "usd", "seconds")}
        return {"arm": self.arm, "goal": self.goal, "status": self.status, "success": self.status == "success",
                "end_reason": self.end, "model_calls": self.calls, "steps": self.steps, "usage": tot,
                "episode_s": round(time.time() - t0, 1)}

    def finish(self, status: str, end: str) -> None:
        self.status, self.end = status, end

    def turn(self) -> None:
        mem = self.memory.text() if self.memory is not None else ""
        user = self.convo if self.arm == "H" else [{"role": "user", "content": _fresh(
            self.arm, self.goal, mem, self.obs, self.info, self.layout, self.breakpoint) + self.note}]
        self.note = []
        try:
            text, u = self.ask([self.sys_msg] + user, self.effort)
        except Limit as e:
            return self.finish("limit", f"limit:{e}")
        self.calls, answer = self.calls + 1, parse_answer(text)
        self.usage.append(u)
        action, kind = to_action(answer)
        problems = _problems(action, answer, mem, self.memory)
        if problems and not self.retried:  # one retry with the problems, for every arm alike
            self.retried = True
            self.steps.append({"answer": text[-400:], "problems": problems})
            self.reply(text, [text_part("Your last answer was not executed: " + "; ".join(problems) + ". Answer again.")],
                       retry=True)
        elif action is None:
            self.steps.append({"answer": text[-400:], "invalid_kind": kind})
            self.finish(kind, f"model_answer:{kind}")
        else:
            self.retried = False
            self.execute(action, answer or {}, text)
        if self.status == "ongoing" and self.calls >= MAX_CALLS:
            self.finish("model_call_limit", f"limit:{MAX_CALLS}_model_calls")

    def reply(self, text: str, content: list, retry: bool = False) -> None:
        """Arm H keeps the conversation; the other arms start fresh each step, so a retry's note (and their own last
        answer) rides on the next fresh request (the smoke run found it was dropped and the same request resent)."""
        if self.arm == "H":
            self.convo = self.convo + [{"role": "assistant", "content": text}, {"role": "user", "content": content}]
        elif retry:
            self.note = [text_part("Your last answer was:\n" + text[-1500:])] + content

    def execute(self, action: np.ndarray, answer: dict, text: str) -> None:
        m = self.memory
        if m is not None and m.system is not None:
            record(m.system.kb, m.frames[-1].t if m.frames else 0, action[:6], gripper_closed(action[6], self.closed),
                   answer.get("target"), answer.get("cites") or ())
        self.closed = gripper_closed(action[6], self.closed)
        # what the model said it was doing, for reading failures later (the Astra reading had only retries' text)
        said = {k: answer[k] for k in ("description", "target") if answer.get(k) is not None}
        new_obs, _, term, trunc, new_info = self.env.step(action)
        if new_obs is None or (new_info or {}).get("status") == "error":
            self.steps.append({"action": action.tolist(), "error": (new_info or {}).get("error_message"), **said})
            return self.finish("error", "env_error")
        n = len(new_obs.get("front_rgb_list") or [])
        self.noops = self.noops + 1 if n == 0 else 0
        if n:
            self.obs, self.info = new_obs, new_info
            if m is not None:
                m.feed(self.obs)
        st = self.info.get("status", "unknown")
        self.steps.append({"action": action.tolist(), "frames": n, "status": st, **said})
        last = {**self.obs, "front_rgb_list": self.obs["front_rgb_list"][-1:]}
        self.reply(text, h_content(self.obs, self.info, "Frames recorded during that motion") if n else
                   [text_part(NOOP_TEXT)] + h_content(last, self.info, "Current"))
        if n and (bool(term) or bool(trunc) or st in TERMINAL):
            self.finish(st, f"env:{st}")
        elif self.noops >= MAX_NOOPS:
            self.finish("no_progress", f"limit:{MAX_NOOPS}_consecutive_noops")


def _problems(action, answer, mem: str, memory: Memory | None) -> list[str]:
    """check() for every arm; target / cites are checked only against ids a memory text printed (H has none). R's log
    prints bare ids, so its objects come from the frames (smoke run: every R target was "unknown", one retry each)."""
    if action is None:
        return []
    objs = [] if memory is None else memory.ids() if memory.system is None else list(memory.system.kb.objects)
    errs = check(action, answer, shown_ids(mem, objs))
    if memory is not None and memory.system is None:  # R's log lines have no ids to cite (smoke run: R cited whole
        errs = [e for e in errs if not e.startswith("unknown cite")]  # lines, and half its calls were retries)
    return errs if mem else [e for e in errs if not e.startswith(("unknown target", "unknown cite"))]


def _fresh(arm: str, goal: str, mem: str, obs: dict, info: dict, layout: str = "v1", breakpoint: bool = False) -> list:
    cur = [(_np(obs["front_rgb_list"][-1]).astype(np.uint8), "Current front camera"),
           (_np(obs["wrist_rgb_list"][-1]).astype(np.uint8), "Current wrist camera")]
    return user_content(arm, goal, mem, cur, state(obs, info, True), layout, breakpoint)


def memory_meta(memory: Memory, goal: str):
    from .types import EpisodeMeta
    objects = memory.perceive.objects() if hasattr(memory.perceive, "objects") else []
    return EpisodeMeta("robomme_closed_loop", 0, "test", goal, objects, None, None)
