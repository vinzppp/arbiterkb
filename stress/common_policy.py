"""Policy adapters for the four arms of the RoboMME common harness."""
from __future__ import annotations

from collections import Counter

from arbiter_kb.common_memory import DeterministicMemory

MEMORY_NOTE = """

MEMORY-ARM OVERRIDE
Each request in this arm is a fresh conversation: earlier user and assistant
turns are not present. Instead, at every decision you receive a MEMORY SNAPSHOT
built only from the same visible semantic observations printed in the steps.
Its anonymous ids come from geometric visual association, never simulator
actor identity. Use the snapshot as the episode history; the latest steps
remain authoritative for what is visible now. Answer in the original JSON
format with choice and point only.
"""


class StructuredPolicy:
    def __init__(self, task: str, episode: int, settings, arm: str, params):
        from memi.llm.rollout_client import RolloutClient
        from memi.plan.prompts.fullhist import SYSTEM

        self.task, self.episode, self.arm = task, episode, arm
        self.chat = RolloutClient(settings.model, settings.effort, settings.max_tokens, settings.timeout)
        self.model, self.effort, self.system = settings.model, settings.effort, SYSTEM + MEMORY_NOTE
        self.memory, self.calls, self.n = DeterministicMemory(arm, params), [], 0

    def start(self, task, goal, menu, has_demo=False):
        self.task, self.goal, self.n, self.has_demo = task, goal, 0, bool(has_demo)

    def reset_memory(self, meta):
        self.memory.reset(meta)

    def observe(self, frames):
        self.memory.feed(frames)

    def decide(self, segment, menu):
        from memi.plan.policy_turn import parse_answer
        from memi.plan.prompts.fullhist import decide_turn

        user = decide_turn(self.n, segment, menu, self.goal, demo=self.has_demo and self.n == 0)
        user += "\n\nMEMORY SNAPSHOT\n" + self.memory.text()
        text, rec = self.chat.send(self.system, [{"role": "user", "content": user}], self._purpose())
        cmd, status = parse_answer(text, menu)
        self.calls.append({"kind": "decide", "index": self.n, "status": status, "raw": text,
                           "command": cmd, "beliefs": self.memory.beliefs(), **rec})
        self.n += 1
        return cmd

    def _purpose(self):
        return f"oracle.common_{self.arm}.{self.task}.episode{self.episode}"

    def summary(self):
        from memi.llm.rollout_client import TOKEN_FIELDS

        usage = {k: sum(int(c.get(k) or 0) for c in self.calls) for k in TOKEN_FIELDS}
        usage["latency"] = round(sum(float(c.get("latency") or 0) for c in self.calls), 3)
        return {"name": self.arm, "model": self.model, "effort": self.effort,
                "calls": dict(Counter(c["kind"] for c in self.calls)),
                "statuses": dict(Counter(c["status"] for c in self.calls)), "usage": usage, "log": self.calls}


class ScriptedPolicy:
    """No-LLM executor preflight: choose the first menu item and its first visible point."""

    def __init__(self, task: str, episode: int, arm: str):
        self.task, self.episode, self.arm, self.n = task, episode, arm, 0
        self.max_decisions = 1

    def start(self, task, goal, menu, has_demo=False):
        self.n = 0

    def observe(self, frames):
        pass

    def decide(self, segment, menu):
        import re
        from memi.plan.policy_turn import needs_point

        choice = menu[0] if menu else {}
        points = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", segment)
        self.n += 1
        return {"choice": choice.get("label"), "point": [float(x) for x in points[-1]] if points and needs_point(choice) else None}

    def summary(self):
        return {"name": f"scripted_{self.arm}", "model": "none", "effort": "none", "usage": {}, "log": []}


def make_policy(arm: str, backend: str, task: str, episode: int, settings, prompts: str, params):
    if backend == "scripted":
        return ScriptedPolicy(task, episode, arm)
    if arm in ("fullhist", "memi"):
        from memi.plan.rollout.policies import make

        settings.policy_prompt = prompts if arm == "memi" else ""
        policy = make("memory" if arm == "memi" else "fullhist", task, episode, settings)
        # MEMI's stock policies do not expose the experimental arm name.  Keep
        # it on the adapter so result provenance matches the frozen invocation.
        policy.arm = arm
        return policy
    return StructuredPolicy(task, episode, settings, arm, params)
