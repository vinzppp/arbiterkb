"""No-LLM probe of the nine "memory or policy?" arms on one scripted trajectory (fairness audit input).

One val episode runs through MEMI's capture and `multi_choice` executor, driven by the G0 scripted
choice.  At every decision all nine arms build their real request with a stub chat that returns the
scripted answer, so every arm sees the same observations and history.  Writes prompts.json:
{arm: [{"system", "messages"} per decision]} plus the episode outcome.  Run with --network none.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]

from g0_preflight import MEMI_PROMPTS, ProbePolicy, StubChat  # noqa: E402


class MopProbe(ProbePolicy):
    """ProbePolicy's scripted executor, with the nine study arms as the prompt paths."""

    def __init__(self, task: str, episode: int, memi_root: Path, max_decisions: int):
        import memi.llm.rollout_client as rollout_client
        import memi.plan.rollout.policies as memi_policies

        rollout_client.RolloutClient = StubChat
        memi_policies.RolloutClient = StubChat
        from arbiter_kb.params import Params
        from memi.plan.rollout.policies import PolicySettings
        from common_policy import make_policy
        from mop_policy import ARMS

        params = Params.load(ROOT / "params.json")
        prompts = str(memi_root / MEMI_PROMPTS)
        settings = PolicySettings(model="none", effort="none", policy_prompt=prompts)
        self.task, self.max_decisions, self.n = task, max_decisions, 0
        self.arms = {arm: make_policy(arm, "api", task, episode, settings, prompts, params) for arm in ARMS}
        self.commands, self.frame_classes, self.kb_raw_progress = [], {}, []

    def reset_memory(self, meta):
        self.meta_objects = [(oid, cls) for oid, cls, _ in meta.objects]
        for policy in self.arms.values():
            if hasattr(policy, "reset_memory"):
                policy.reset_memory(meta)

    def observe(self, frames):
        for policy in self.arms.values():
            if hasattr(policy, "observe"):
                policy.observe(frames)

    def executed(self, command, result, status):
        for policy in self.arms.values():
            if hasattr(policy, "executed"):
                policy.executed(command, result, status)

    def decide(self, segment, menu):
        cmd = self._command(segment, menu)
        StubChat.answer = json.dumps({**cmd, "memory_updates": []})
        for policy in self.arms.values():
            policy.decide(segment, menu)
        self.commands.append(cmd)
        self.n += 1
        return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--memi-root", type=Path, default=Path("/opt/memi"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-decisions", type=int, default=6)
    a = ap.parse_args()
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("the probe is model-free; refuse to run with an API key in the environment")
    sys.path.insert(0, str(a.memi_root))
    from common_episode import run_episode

    probe = MopProbe(a.task, a.episode, a.memi_root, a.max_decisions)
    cfg = {"mode": "clean", "seed": 0, "dropout": 0.0, "jitter_px": 0.0, "bridge": "memi_named"}
    log = run_episode(a.task, a.episode, "val", probe, cfg, max_decisions=a.max_decisions)
    from memi.plan.prompts import POLICY_SYSTEM

    stock = probe.arms["memi"].chat.prompts
    out = {"task": a.task, "episode": a.episode, "outcome": log.outcome, "commands": probe.commands,
           "stock_memi_system_ok": bool(stock) and all(p["system"] == POLICY_SYSTEM for p in stock),
           "objects": probe.meta_objects,
           "prompts": {arm: policy.chat.prompts for arm, policy in probe.arms.items()}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.task}_val{a.episode}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"task": a.task, "outcome": log.outcome, "decisions": len(probe.commands),
                      "calls": {k: len(v.chat.prompts) for k, v in probe.arms.items()}}))


if __name__ == "__main__":
    main()
