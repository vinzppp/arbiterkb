"""Closed-loop harness skeleton (DEVELOPMENT §6 M4 item 2, NOTES "B"): the loop that M5 runs on the pod, with the
memory module as the only switch between arms. One step: the KB has read every frame so far; the controller gets the
prompt (for arm K: the self view FIRST, then the controller view); its answer is checked, recorded in the KB
(self_memory.record) and sent to the environment; the new frames go into the KB.

Not yet a port of the 2610.17350 harness: its prompt skeleton, feedback text and completion / failure rules are to be
copied from its released code (github.com/jackbjed/embodied-astra-robomme) before M5. `Env` is what the pod's
RoboMME wrapper must provide; `ReplayEnv` replays a recorded dataset episode, ignoring the actions, so the loop and the
self memory can be checked on the Mac.
"""
from __future__ import annotations

from typing import Callable, Protocol

from .self_memory import outcome, record, self_view
from .sources import lerobot as L
from .types import EpisodeMeta, FrameObs


class Env(Protocol):
    def reset(self) -> tuple[EpisodeMeta, list[FrameObs]]: ...
    def step(self, waypoint, gripper_closed: bool) -> tuple[list[FrameObs], bool]: ...  # (new frames, done)


class ReplayEnv:
    """A recorded RoboMME dataset episode: a decision at the start of each execution subgoal; step() returns the
    frames up to `slack` frames into the next subgoal (or the end), whatever the action was."""

    def __init__(self, ep: dict, slack: int = 8):
        self.ep, self.slack = ep, slack
        self.segs = [s for s in L.segments(ep["simple"]) if not ep["demo"][s[0]]]
        self.T, self.k, self.t = len(ep["tcp"]), 0, 0

    def reset(self) -> tuple[EpisodeMeta, list[FrameObs]]:
        self.k, self.t = 0, self.segs[0][0] if self.segs else self.T
        return L.meta(self.ep), list(L.frames(self.ep, self.t))

    def step(self, waypoint, gripper_closed: bool) -> tuple[list[FrameObs], bool]:
        self.k += 1
        end = min(self.segs[self.k][0] + self.slack, self.T) if self.k < len(self.segs) else self.T
        new = [f for f in L.frames(self.ep, end) if f.t >= self.t]
        self.t = end
        return new, self.k >= len(self.segs)

    def subgoal(self) -> str:
        """The current subgoal's label (evaluation and scripted controllers only)."""
        return self.segs[self.k][2] if self.k < len(self.segs) else ""


Decide = Callable[[object, str], dict]  # (system, prompt text) -> {"waypoint", "gripper_closed", "target", ...}


def run_episode(env: Env, system, decide: Decide, view: Callable[[object], str], max_calls: int = 30) -> list[dict]:
    """The loop; returns one row per decision: t, intent, the outcome once the next frames are in, prompt chars."""
    meta, frames = env.reset()
    system.reset(meta)
    for f in frames:
        system.step(f)
    rows, done, t = [], False, frames[-1].t if frames else 0
    while not done and len(rows) < max_calls:
        prompt = "\n\n".join(x for x in (self_view(system.kb), view(system.kb)) if x)
        ans = decide(system, prompt)
        d = record(system.kb, t, ans.get("waypoint"), bool(ans.get("gripper_closed")), ans.get("target"),
                   ans.get("cites", ()), ans.get("intent"))
        frames, done = env.step(ans.get("waypoint"), bool(ans.get("gripper_closed")))
        for f in frames:
            system.step(f)
        t = frames[-1].t if frames else t
        rows.append({"t": d["t"], "intent": d["intent"], "outcome": outcome(system.kb, d), "prompt_chars": len(prompt),
                     "self_chars": len(self_view(system.kb))})
    return rows
