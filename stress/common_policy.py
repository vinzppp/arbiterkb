"""Policy adapters for the four arms of the RoboMME common harness."""
from __future__ import annotations

import hashlib
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

GENERIC_FACTORIAL_POLICY = """Use the full task objective and legal action menu in the user turn. The demonstration
shows task-relevant history but is not execution: never count demonstration actions as actions completed by the
agent. Use only the current observation interval, the memory snapshot, and execution feedback. Treat an executed
action as progress only when the observations and environment feedback support that it succeeded. Complete the full
objective, including its stated stopping condition; do not finish early. Do not invent object identity when memory
says unknown. This is a competent generic policy, not a task-specific action program."""

MATCHED_FACTORIAL_POLICIES = {
    "VideoRepick": """Read the required number N of execution cycles from the goal: 'again' means 1, 'two times'
means 2, and 'three times' means 3; use any other explicit count as written. Demonstration actions never count as
execution. One execution cycle is a successful A pickup of the demonstration-grasped cube followed by a successful B
put-down. Start with A on the current visible position of the demonstration-grasped cube. If the execution state
shows the cube is held, choose B. Increment completed execution cycles only after the corresponding successful
execution put-down. After a put-down, choose A when completed cycles are below N and choose C only when completed
cycles equal N. A and B update progress only from successful execution feedback. B and C need no point. Memory may
identify the cube or say unknown; this policy supplies no object identity or simulator-ground-truth location.""",
    "VideoUnmaskSwap": """Read the requested cube colors and their order from the goal. Demonstration actions never
count as execution. Use the memory's cover relation and persistent object records to locate the container currently
hiding each requested cube after all demonstrated movements; this policy supplies no container identity or location.
With an empty hand, choose A on the current visible center of the container for the next requested cube. A successful
execution pickup completes that requested target. If another requested target remains, choose B to put the held
container down, then choose A for the next target. A successful pickup of the final requested container completes the
goal; do not deliberately put the final container down. Update completed targets only from successful execution
feedback. If the required cover relation is unknown, do not invent certainty; use the best supported visible target
or the safest legal fallback.""",
}


def factorial_policy_text(task: str, policy_kind: str) -> str:
    if policy_kind == "generic":
        return GENERIC_FACTORIAL_POLICY
    if policy_kind == "matched" and task in MATCHED_FACTORIAL_POLICIES:
        return MATCHED_FACTORIAL_POLICIES[task]
    raise ValueError(f"no factorial policy for {task}/{policy_kind}")


def factorial_policy_sha(task: str, policy_kind: str) -> str:
    return hashlib.sha256(factorial_policy_text(task, policy_kind).encode()).hexdigest()


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


class FactorialPolicy:
    """Fresh-call policy with independently selected memory and policy factors."""

    def __init__(self, task: str, episode: int, settings, arm: str, params):
        from memi.llm.rollout_client import RolloutClient
        from memi.plan.prompts.fullhist import SYSTEM
        from arbiter_kb.factorial_memory import FactorialMemory

        memory_kind, policy_kind = arm.split("_", 1)
        self.task, self.episode, self.arm = task, episode, arm
        self.memory_kind, self.policy_kind = memory_kind, policy_kind
        self.policy_text = factorial_policy_text(task, policy_kind)
        self.system = SYSTEM + MEMORY_NOTE + "\n\nFACTORIAL POLICY\n" + self.policy_text
        self.chat = RolloutClient(settings.model, settings.effort, settings.max_tokens, settings.timeout)
        self.model, self.effort = settings.model, settings.effort
        self.memory, self.calls, self.n = FactorialMemory(memory_kind, params), [], 0

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
                           "command": cmd, "beliefs": self.memory.beliefs(),
                           "memory_audit": self.memory.audit(), "policy_kind": self.policy_kind,
                           "policy_sha256": factorial_policy_sha(self.task, self.policy_kind), **rec})
        self.n += 1
        return cmd

    def _purpose(self):
        return f"oracle.factorial_{self.arm}.{self.task}.episode{self.episode}"

    def summary(self):
        from memi.llm.rollout_client import TOKEN_FIELDS

        usage = {key: sum(int(call.get(key) or 0) for call in self.calls) for key in TOKEN_FIELDS}
        usage["latency"] = round(sum(float(call.get("latency") or 0) for call in self.calls), 3)
        return {"name": self.arm, "memory_kind": self.memory_kind, "policy_kind": self.policy_kind,
                "policy_sha256": factorial_policy_sha(self.task, self.policy_kind),
                "system_sha256": hashlib.sha256(self.system.encode()).hexdigest(),
                "model": self.model, "effort": self.effort,
                "calls": dict(Counter(call["kind"] for call in self.calls)),
                "statuses": dict(Counter(call["status"] for call in self.calls)),
                "usage": usage, "log": self.calls}


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
    if arm in {"ledger_generic", "ledger_matched", "arbiter_generic", "arbiter_matched"}:
        return FactorialPolicy(task, episode, settings, arm, params)
    from mop_policy import ARMS as MOP_ARMS, MopPolicy

    if arm in MOP_ARMS:
        return MopPolicy(task, episode, settings, arm, params, prompts)
    return StructuredPolicy(task, episode, settings, arm, params)
