"""The nine arms of the "memory or policy?" study (main NOTES 7327cf1, "Arms").

Arm ids: <memory>_<policy> with memory fh (MEMI's FullHist conversation), ld (preregistered
deterministic ledger), mm (MEMI's fields, carried by the model), kb (frozen ArbiterKB, phase-safe)
and policy g (generic) / m (MEMI's subgoal text after the mechanical split); plus "memi" (stock).

Every factorial arm sends: system = MEMI's FullHist scaffold + MEMORY_NOTES[memory] + POLICY_HEAD +
policy text.  Fresh-call arms (ld, mm, kb) append MEMORY_HEAD + memory text to MEMI's decide turn;
fh keeps MEMI's one-conversation history.  So the policy text depends only on (task, policy) and
the memory note and payload only on the memory and the observations (fairness audit:
scripts/mop_fairness.py).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from common_policy import GENERIC_FACTORIAL_POLICY, MEMORY_NOTE

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "results/memory_or_policy/g0/memi_split"
MEMORIES, POLICIES = ("fh", "ld", "mm", "kb"), ("g", "m")
ARMS = tuple(f"{m}_{p}" for m in MEMORIES for p in POLICIES) + ("memi",)
POLICY_HEAD = "\n\nTASK POLICY\n"
# P-generic = factorial P0 with one phrase changed (orchestrator ruling 2026-11-01): FullHist has no snapshot.
assert GENERIC_FACTORIAL_POLICY.count("the memory snapshot") == 1
GENERIC_POLICY = GENERIC_FACTORIAL_POLICY.replace("the memory snapshot", "the memory given to you")
MEMORY_HEAD = "\n\nMEMORY\n"

MM_NOTE = """

MEMORY-ARM OVERRIDE
Each request in this arm is a fresh conversation: earlier user and assistant
turns are not present. Instead, you keep a memory between decisions. Under
MEMORY, POLICY PROMPT FOR MEMORY declares each memory field and how to keep it,
and MEMORY CARRIED IN gives the values you wrote at the previous decision step.
The latest steps remain authoritative for what is visible now. Besides choice
and point, answer `memory_updates`: a list with one {"field", "value"} entry per
declared field whose value changes at this decision step; a field not listed
keeps its value, an empty value clears it, and a step that changes nothing
answers []. Answer with JSON only:
{"choice": "<label>", "point": [y, x] or null, "memory_updates": [{"field": "...", "value": "..."}]}
"""
MEMORY_NOTES = {"fh": "", "ld": MEMORY_NOTE, "kb": MEMORY_NOTE, "mm": MM_NOTE}
ANSWER_HEAD = "\n\nAnswer with "
# The last line of a fresh-call turn: MEMI's own for ld / kb; for mm the same with memory_updates, so the
# turn and the system note ask for one format.
ANSWER_LINES = {"ld": 'Answer with {"choice": "<label>", "point": [y, x] or null}.',
                "kb": 'Answer with {"choice": "<label>", "point": [y, x] or null}.',
                "mm": 'Answer with {"choice": "<label>", "point": [y, x] or null, '
                      '"memory_updates": [{"field": "...", "value": "..."}]}.'}


def policy_text(task: str, policy: str) -> str:
    """P-generic (factorial P0, one phrase changed) or P-matched (MEMI's subgoal text, split rules 1-2, G0 output)."""
    if policy == "g":
        return GENERIC_POLICY
    if policy == "m":
        return (SPLIT_DIR / task / "policy_matched.txt").read_text()
    raise ValueError(policy)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def system_text(task: str, memory: str, policy: str) -> str:
    from memi.plan.prompts.fullhist import SYSTEM

    return SYSTEM + MEMORY_NOTES[memory] + POLICY_HEAD + policy_text(task, policy)


class MEMIFieldsMemory:
    """M-MEMI: MEMI's fields (name + description, unchanged) and the values the model carries."""

    def __init__(self, task: str, prompts_file: str):
        import json

        from memi.plan.policy_turn import entry

        self.entry = entry(json.loads(Path(prompts_file).read_text())[task])
        self.values: dict = {}

    def reset(self, meta) -> None:
        self.values = {}

    def feed(self, frames) -> None:
        pass

    def chose(self, command, menu) -> None:
        pass

    def executed(self, result, status) -> None:
        pass

    def update(self, answer_text: str) -> dict:
        from memi.llm.parse import parse_object
        from memi.plan.policy_turn import carry, updates
        from memi.plan.variables import field_names

        changes = updates(parse_object(answer_text) if answer_text else None)
        self.values = carry(field_names(self.entry), self.values, changes)
        return changes

    def text(self) -> str:
        from memi.plan.policy_turn import memory_text
        from memi.text import MEMORY

        line = memory_text(self.entry, self.values)
        return ("POLICY PROMPT FOR MEMORY\n" + self.entry[MEMORY].rstrip()
                + "\nMEMORY CARRIED IN\n" + (f"  {line}" if line else "  (empty)"))


def make_memory(memory: str, task: str, params, prompts_file: str):
    from arbiter_kb.mop_memory import KBMemory, PreregLedger

    if memory == "ld":
        return PreregLedger(params)
    if memory == "kb":
        return KBMemory(params)
    if memory == "mm":
        return MEMIFieldsMemory(task, prompts_file)
    raise ValueError(memory)


class MopPolicy:
    """One factorial arm. fh keeps a conversation; ld / mm / kb send a fresh request with a memory block."""

    def __init__(self, task: str, episode: int, settings, arm: str, params, prompts_file: str):
        from memi.llm.rollout_client import RolloutClient

        self.memory_kind, self.policy_kind = arm.split("_")
        self.task, self.episode, self.arm = task, episode, arm
        self.system = system_text(task, self.memory_kind, self.policy_kind)
        self.chat = RolloutClient(settings.model, settings.effort, settings.max_tokens, settings.timeout)
        self.model, self.effort = settings.model, settings.effort
        self.memory = None if self.memory_kind == "fh" else make_memory(self.memory_kind, task, params, prompts_file)
        self.calls, self.n, self.messages = [], 0, []

    def start(self, task, goal, menu, has_demo=False):
        self.task, self.goal, self.n, self.has_demo, self.messages = task, goal, 0, bool(has_demo), []

    def reset_memory(self, meta):
        if self.memory is not None:
            self.memory.reset(meta)

    def observe(self, frames):
        if self.memory is not None:
            self.memory.feed(frames)

    def executed(self, command, result, status):
        if self.memory is not None:
            self.memory.executed(result, status)

    def _turn(self, segment, menu) -> list[dict]:
        from memi.plan.prompts.fullhist import decide_turn

        demo = self.has_demo and self.n == 0
        if self.memory_kind == "fh":
            goal = "" if self.messages else self.goal
            self.messages.append({"role": "user", "content": decide_turn(self.n, segment, menu, goal, demo=demo)})
            return list(self.messages)
        turn = decide_turn(self.n, segment, menu, self.goal, demo=demo)
        body = turn[:turn.rindex(ANSWER_HEAD)]
        user = body + MEMORY_HEAD + self.memory.text() + "\n\n" + ANSWER_LINES[self.memory_kind]
        return [{"role": "user", "content": user}]

    def decide(self, segment, menu):
        from memi.plan.policy_turn import parse_answer

        messages = self._turn(segment, list(menu or []))
        memory_before = self.memory.text() if self.memory is not None else None
        text, rec = self.chat.send(self.system, messages, f"oracle.mop_{self.arm}.{self.task}.episode{self.episode}")
        if self.memory_kind == "fh" and text:
            self.messages.append({"role": "assistant", "content": text})
        changes = self.memory.update(text) if self.memory_kind == "mm" else None
        cmd, status = parse_answer(text, list(menu or []))
        if self.memory is not None:
            self.memory.chose(cmd, menu)
        self.calls.append({"kind": "decide", "index": self.n, "status": status, "raw": text, "command": cmd,
                           "memory_before": memory_before, "memory_updates": changes,
                           "policy_sha256": sha(policy_text(self.task, self.policy_kind)),
                           "memory_note_sha256": sha(MEMORY_NOTES[self.memory_kind]), **rec})
        self.n += 1
        return cmd

    def summary(self):
        from memi.llm.rollout_client import TOKEN_FIELDS

        usage = {k: sum(int(c.get(k) or 0) for c in self.calls) for k in TOKEN_FIELDS}
        usage["latency"] = round(sum(float(c.get("latency") or 0) for c in self.calls), 3)
        return {"name": self.arm, "memory_kind": self.memory_kind, "policy_kind": self.policy_kind,
                "policy_sha256": sha(policy_text(self.task, self.policy_kind)),
                "system_sha256": sha(self.system), "model": self.model, "effort": self.effort,
                "models_returned": sorted({c.get("model_returned", "") for c in self.calls} - {""}),
                "calls": dict(Counter(c["kind"] for c in self.calls)),
                "statuses": dict(Counter(c["status"] for c in self.calls)), "usage": usage, "log": self.calls}
