"""Closed-loop pilot on RoboMME (DEVELOPMENT M5; NOTES "Closed loop"). ON THE POD, sg-main venv (the perception worker
runs in sg-percep, started by sg_pilot). One episode per (task, episode, arm): the 2610.17350 loop
(arbiter_kb/closed_loop.py) with arm H (the image history) or a memory arm (R, K) whose KB reads every 3rd front frame
through the real perception pipeline (stress/perceive.slot_row -> arbiter_kb/sources/online.OnlineObs).

  python stress/closed_loop_pod.py --tasks VideoUnmaskSwap --episodes 0 --arms K --dataset val --out /workspace/cl/smoke
  python stress/closed_loop_pod.py --tasks VideoUnmaskSwap,VideoPlaceOrder --episodes $(seq 0 9) --arms H,R,K \\
         --dataset val --out /workspace/cl/pilot --max-usd 5

--vocab open (default): the detector's words are generic phrases plus the task's own (no simulator scene); scene:
G's privileged word list (a reference only). Results: <out>/<task>_ep<e>_<arm>/result.json, resumable; the dataset
split is recorded (test reads need a frozen params.json and go in the ledger, as everywhere else).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SG_PILOT = Path(os.environ.get("SG_PILOT_DIR", "/workspace/sg/notebooks/pilot"))
sys.path[:0] = [str(SG_PILOT), str(ROOT), str(ROOT / "scripts"), str(ROOT / "stress")]

from arbiter_kb import ledger  # noqa: E402
from arbiter_kb.closed_loop import STRIDE, Limit, Memory, run  # noqa: E402
from arbiter_kb.llm import LLM, BudgetExceeded  # noqa: E402
from arbiter_kb.sources.online import OnlineObs  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402

MODEL = "qwen/qwen3.6-35b-a3b"  # takes images (OpenRouter); every arm uses it unless --model
PROVIDER = {"qwen/": "CoreWeave", "openai/": "OpenAI"}  # --provider auto; others untested, name one
EFFORT = {"default": None, "off": {"enabled": False}}


def make_env(task: str, episode: int, dataset: str):
    """Their env (waypoint, 1300 steps) plus the depth and calibration the perception needs (never shown to the model)."""
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    return BenchmarkEnvBuilder(env_id=task, dataset=dataset, action_space="waypoint", gui_render=False,
                               max_steps=1300).make_env_for_episode(
        episode, include_front_depth=True, include_front_camera_extrinsic=True, include_front_camera_intrinsic=True,
        include_wrist_camera_extrinsic=True, include_wrist_camera_intrinsic=True)


class Perceive:
    """obs -> FrameObs for every STRIDE-th new front frame (the frames arm H sees), through the real pipeline."""

    def __init__(self, goal: str, vocab: str, episode_num: int, K_front: np.ndarray):
        import sg_pilot as sg
        from perceive import open_config
        if vocab != "open":
            raise SystemExit("--vocab scene needs the simulator scene; not wired for the closed loop yet")
        cfg, _ = open_config(sg, goal)
        self.handle = sg.make_robomme_processor(cfg)
        n_bins = sum(c.startswith("container") for c in self.handle.classes)
        self.online, self.ep, self.seen = OnlineObs(self.handle.classes, n_bins), episode_num, 0
        self.K = np.asarray(K_front, np.float32).reshape(3, 3)

    def objects(self):
        return self.online.objects()

    def __call__(self, obs: dict) -> list:
        from perceive import slot_row
        fr = obs["front_rgb_list"]
        out = []
        for i in range(len(fr)):
            t = self.seen + i
            if t % STRIDE and i != len(fr) - 1:
                continue
            E = np.asarray(obs["front_camera_extrinsic_list"][i], np.float32).reshape(3, 4)
            depth = np.asarray(obs["front_depth_list"][i], np.float32).squeeze() / 1000.0  # mm -> m
            row, _, _ = slot_row(self.handle, np.asarray(fr[i], np.uint8), depth, self.K, E, t, self.ep)
            ee = np.asarray(obs["eef_state_list"][i], float).reshape(-1)[:3]
            width = float(np.sum(np.asarray(obs["gripper_state_list"][i], float).reshape(-1)[:2]))
            out.append(self.online.step(row, ee, width))
        self.seen += len(fr)
        return out


def episode(task: str, ep: int, arm: str, a, llm, params) -> dict:
    memory = None
    if arm != "H":
        # the word list needs the goal, the depth back-projection K. Only the first reset() of a fresh env gives the
        # episode's own scene (a second reset on the same env gives another; smoke run), so probe on a throwaway env.
        probe = make_env(task, ep, a.dataset)
        info = probe.reset()[1]
        probe.close()
        p = Perceive(info["task_goal"][0], a.vocab, ep, info["front_camera_intrinsic"])
        memory = Memory(arm, ArbiterKBSystem(params) if arm != "R" else None, p)
    env = make_env(task, ep, a.dataset)
    qwen = a.model.startswith("qwen/")  # Qwen's thinking settings; any other model runs on its API defaults
    sampling = {"top_p": a.top_p, "top_k": a.top_k, "presence_penalty": a.presence_penalty} if qwen else None
    start = llm.usage["usd"]

    def ask(m, eff):
        if llm.usage["usd"] - start >= a.max_usd_episode:  # one runaway episode must not spend the run's budget
            raise Limit("episode_usd")
        return llm.call(m, temperature=a.temperature if qwen else None, seed=0 if qwen else None,
                        max_tokens=a.max_tokens, reasoning=eff, sampling=sampling)
    row = run(env, arm, ask, memory, EFFORT[a.effort])
    env.close()
    return {**row, "task": task, "episode": ep, "dataset": a.dataset, "effort": a.effort, "vocab": a.vocab}


def main() -> None:
    from run import git_state, load_params
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="VideoUnmaskSwap")
    ap.add_argument("--episodes", nargs="+", type=int, required=True)
    ap.add_argument("--arms", default="H,R,K")
    ap.add_argument("--dataset", choices=["val", "test"], default="val")
    ap.add_argument("--effort", choices=list(EFFORT), default="default")
    ap.add_argument("--vocab", choices=["open", "scene"], default="open")
    ap.add_argument("--out", default="/workspace/cl/pilot")
    ap.add_argument("--max-usd", dest="max_usd", type=float, default=5.0)
    ap.add_argument("--max-usd-episode", dest="max_usd_episode", type=float, default=3.0,
                    help="an episode that has spent this much ends as limit:episode_usd")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--provider", default="auto",  # one for every arm; DeepInfra takes at most 30 images (H: ~150);
                    help="OpenRouter provider, no fallbacks; auto: by the model's family (PROVIDER)")  # Parasail took
    # 160 but was rate-limited for minutes; CoreWeave took 160, answered fastest and gave no 429 in the smoke run
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=32000,  # 4000 cut H's and R's reasoning;
                    help="completion cap per call, every arm alike")  # 16000 cut R's on VideoPlaceOrder (no loop, smoke run)
    # Qwen's recommended thinking-mode sampling, every arm alike: at temperature 0 (greedy) R's and K's reasoning looped
    # until the cap (smoke run: 450 lines, 69 distinct), as Qwen warns; 2610.17350 used its API defaults (T 1, top-p 0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--top-k", dest="top_k", type=int, default=20)
    ap.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=1.5,  # loop probe, smoke run:
                    help="Qwen's anti-repetition setting")  # 0 / 15 first turns looped with it, 3 / 15 without
    a = ap.parse_args()
    if a.provider == "auto":
        a.provider = next((v for k, v in PROVIDER.items() if a.model.startswith(k)), None) or \
            sys.exit(f"--provider needed for {a.model}")
    if not a.model.startswith("qwen/"):  # recorded as sent: the API defaults
        a.temperature = a.top_p = a.top_k = a.presence_penalty = None
    params, (commit, dirty) = load_params(), git_state()
    if a.dataset == "test":
        if not ledger.was_frozen(params.hash()):
            sys.exit("freeze params first (run.py --freeze)")
        ledger.record("READ", params.hash(), f"RoboMME test closed loop {a.tasks} {a.episodes} {a.arms}")
    llm = LLM(model=a.model, max_usd=a.max_usd, provider={"order": [a.provider], "allow_fallbacks": False,
                                                         "require_parameters": True})
    for task in a.tasks.split(","):
        for ep in a.episodes:
            for arm in a.arms.split(","):
                d = Path(a.out) / f"{task}_ep{ep}_{arm}"
                if (d / "result.json").exists():
                    continue
                d.mkdir(parents=True, exist_ok=True)
                try:
                    row = episode(task, ep, arm, a, llm, params)
                except BudgetExceeded as ex:  # the run's budget: every later episode would fail the same way
                    print(json.dumps({"task": task, "episode": ep, "arm": arm, "stopped": repr(ex)}), flush=True)
                    return print("usage", llm.usage)
                except Exception as ex:  # recorded unscored, rerun on resume (theirs)
                    row = {"task": task, "episode": ep, "arm": arm, "status": "harness_error", "success": None,
                           "error": repr(ex)}
                if row.get("status") == "limit":  # a spend cap, not the policy: unscored (H costs most, so a scored
                    row["success"] = None  # cap would favour the memory arms for a budget reason)
                row.update(commit=commit, dirty=dirty, params_hash=params.hash(), model=a.model, provider=a.provider,
                           max_usd_episode=a.max_usd_episode,
                           max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                           presence_penalty=a.presence_penalty)
                if row.get("status") != "harness_error":
                    (d / "result.json").write_text(json.dumps(row, indent=1, default=str))
                print(json.dumps({k: row.get(k) for k in ("task", "episode", "arm", "status", "end_reason",
                                                          "model_calls", "usage", "error") if k in row},
                                 default=str), flush=True)
    print("usage", llm.usage)


if __name__ == "__main__":
    main()
