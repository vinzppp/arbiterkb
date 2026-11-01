"""G0 no-LLM preflight for the "memory or policy?" study (main NOTES 7327cf1, gate G0).

One RoboMME episode per task through MEMI's capture and `multi_choice` executor.  The
executor is driven by a scripted choice (menu option i mod len(menu), its first available
point or else the last pixel printed in the segment), never by a model.  At every decision
the same segment is also pushed through each arm's prompt path with a stub chat that
returns the scripted answer, so FullHist, stock MEMI and the ledger/KB adapters build their
real prompts without any network call (run the container with --network none).

Writes one JSON per task: timings, CPU time, outcome, object classes MEMI's capture names
versus the classes the ArbiterKB bridge keeps, each arm's prompt sizes and hashes, and the
first and last prompt of every arm as text files for reading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]

MEMI_PROMPTS = "artifacts/gpt/policy_prompts/optimized_luna.json"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class StubChat:
    """Stands in for MEMI's RolloutClient; records prompts and returns the scripted answer."""

    answer = "{}"

    def __init__(self, *args, **kwargs):
        self.model, self.effort, self.prompts = "none", "none", []

    def send(self, system, messages, purpose):
        self.prompts.append({"system": system, "messages": [dict(m) for m in messages]})
        return StubChat.answer, {"model": "none", "effort": "none", "model_returned": "none", "latency": 0.0}


def cpu_seconds() -> float:
    own, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


class ProbePolicy:
    """Scripted executor policy that also builds every arm's prompt from the same inputs."""

    arm = "g0_probe"

    def __init__(self, task: str, episode: int, memi_root: Path, max_decisions: int):
        import memi.llm.rollout_client as rollout_client
        import memi.plan.rollout.policies as memi_policies

        rollout_client.RolloutClient = StubChat
        memi_policies.RolloutClient = StubChat
        from arbiter_kb.params import Params
        from memi.plan.rollout.policies import PolicySettings, make
        from common_policy import FactorialPolicy

        params = Params.load(ROOT / "params.json")
        settings = PolicySettings(model="none", effort="none", policy_prompt=str(memi_root / MEMI_PROMPTS))
        self.task, self.max_decisions, self.n = task, max_decisions, 0
        self.arms = {"fullhist": make("fullhist", task, episode, settings),
                     "memi": make("memory", task, episode, settings),
                     "ledger": FactorialPolicy(task, episode, settings, "ledger_generic", params),
                     "arbiter": FactorialPolicy(task, episode, settings, "arbiter_generic", params)}
        self.commands, self.frame_classes, self.kb_raw_progress = [], Counter(), []

    def start(self, task, goal, menu, has_demo=False):
        self.goal, self.has_demo = goal, has_demo
        for policy in self.arms.values():
            policy.start(task, goal, menu, has_demo)

    def reset_memory(self, meta):
        self.meta_objects = [(oid, cls) for oid, cls, _ in meta.objects]
        for name in ("ledger", "arbiter"):
            self.arms[name].reset_memory(meta)

    def observe(self, frames):
        for frame in frames:
            self.frame_classes.update(obj.cls for obj in frame.objects.values() if obj.visible)
        for name in ("ledger", "arbiter"):
            self.arms[name].observe(frames)

    def _command(self, segment, menu):
        from memi.plan.policy_turn import needs_point

        choice = menu[self.n % len(menu)] if menu else {}
        point = None
        if needs_point(choice):
            available = [a for a in (choice.get("available") or []) if isinstance(a, dict)]
            given = next((a.get("point") for a in available if a.get("point")), None)
            pixels = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", segment)
            point = list(given) if given else ([float(x) for x in pixels[-1]] if pixels else [0.0, 0.0])
        return {"choice": choice.get("label"), "point": point}

    def decide(self, segment, menu):
        from arbiter_kb.views import controller_view

        cmd = self._command(segment, menu)
        StubChat.answer = json.dumps({**cmd, "memory_updates": []})
        for policy in self.arms.values():
            policy.decide(segment, menu)
        raw = controller_view(self.arms["arbiter"].memory.world.kb)
        self.kb_raw_progress.append([line.strip() for line in raw.splitlines()
                                     if line.strip().startswith(("Progress:", "Goal progress:"))])
        self.commands.append(cmd)
        self.n += 1
        return cmd

    def prompts(self) -> dict:
        out = {}
        for name, policy in self.arms.items():
            sent = policy.chat.prompts
            texts = [p["system"] + "\n\n" + "\n\n".join(m["content"] for m in p["messages"]) for p in sent]
            out[name] = {"calls": len(sent), "chars": [len(t) for t in texts],
                         "system_sha256": sha(sent[0]["system"]) if sent else None,
                         "first": texts[0] if texts else "", "last": texts[-1] if texts else ""}
        return out

    def summary(self):
        return {"name": self.arm, "model": "none", "effort": "none", "usage": {}, "log": []}


def capture_classes(rows: list[dict], task: str) -> Counter:
    from memi.plan.rollout.scene import classify_object_type

    seen = Counter()
    for row in rows:
        for record in row.get("visible_object_records") or []:
            seen[str(classify_object_type(record.get("actor_name"), task))] += 1
    return seen


def run(task: str, episode: int, split: str, memi_root: Path, out: Path, max_decisions: int) -> dict:
    import common_episode
    from common_episode import run_episode
    from common_scene import SceneStream

    kept_rows = []
    original_feed = SceneStream.feed

    def feed(self, rows):
        kept_rows.extend(rows)
        return original_feed(self, rows)

    SceneStream.feed = feed
    timings, original = [], {}
    for phase in ("_start", "_reset"):
        original[phase] = getattr(common_episode, phase)

        def timed(*args, _phase=phase, **kwargs):
            t0, c0 = time.time(), cpu_seconds()
            try:
                return original[_phase](*args, **kwargs)
            finally:
                timings.append({"phase": _phase.strip("_"), "wall_s": round(time.time() - t0, 2),
                                "cpu_s": round(cpu_seconds() - c0, 2)})
        setattr(common_episode, phase, timed)

    policy = ProbePolicy(task, episode, memi_root, max_decisions)
    cfg = {"mode": "clean", "seed": 0, "dropout": 0.0, "jitter_px": 0.0}
    t0, c0 = time.time(), cpu_seconds()
    log = run_episode(task, episode, split, policy, cfg, max_decisions=max_decisions)
    wall, cpu = time.time() - t0, cpu_seconds() - c0
    prompts = policy.prompts()
    task_dir = out / f"{task}_{split}{episode}"
    task_dir.mkdir(parents=True, exist_ok=True)
    for name, p in prompts.items():
        (task_dir / f"{name}_first_prompt.txt").write_text(p.pop("first"))
        (task_dir / f"{name}_last_prompt.txt").write_text(p.pop("last"))
    ledger, arbiter = policy.arms["ledger"].memory, policy.arms["arbiter"].memory
    (task_dir / "ledger_memory_final.txt").write_text(ledger.text())
    (task_dir / "arbiter_memory_final.txt").write_text(arbiter.text())
    decisions = log.decisions
    row = {
        "task": task, "split": split, "episode": episode, "outcome": log.outcome, "success": log.success,
        "error": (log.error or "")[-1500:] if getattr(log, "error", None) else "",
        "goal": log.goal, "menu": log.menu, "demo_rows": log.demo_rows, "reset_digest": log.reset_digest,
        "decisions": len(decisions),
        "commands": policy.commands,
        "decision_results": [d.get("result") for d in decisions],
        "decision_seconds": [d.get("seconds") for d in decisions],
        "captured_rows_per_decision": [d.get("captured") for d in decisions],
        "phase_timings": timings, "episode_wall_s": round(wall, 2), "episode_cpu_s": round(cpu, 2),
        "cpu_utilisation_cores": round(cpu / wall, 3) if wall else None,
        "capture_object_types": dict(capture_classes(kept_rows, task)),
        "bridge_classes_visible_frames": dict(policy.frame_classes),
        "bridge_meta_objects": getattr(policy, "meta_objects", []),
        "normalized_task_state_after_run": arbiter.task_state.audit(),
        "kb_raw_progress_lines_per_decision": policy.kb_raw_progress,
        "prompts": prompts,
        "capture_errors": list(getattr(log, "capture_errors", []) or []),
    }
    (task_dir / "g0_preflight.json").write_text(json.dumps(row, indent=1, default=str))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--memi-root", type=Path, default=Path("/opt/memi"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-decisions", type=int, default=6)
    a = ap.parse_args()
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("G0 preflight is model-free; refuse to run with an API key in the environment")
    sys.path.insert(0, str(a.memi_root))
    row = run(a.task, a.episode, a.split, a.memi_root, a.out, a.max_decisions)
    print(json.dumps({k: row[k] for k in ("task", "episode", "outcome", "decisions", "episode_wall_s",
                                          "episode_cpu_s", "cpu_utilisation_cores")}))


if __name__ == "__main__":
    main()
