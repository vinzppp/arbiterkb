"""K vs K2 offline (NOTES "K2: initial state, change log, current state"; the rule is fixed there before any call).

Synthetic VideoUnmaskSwap train episodes (as Batch A: clean opening videos, text only) with a two-cube goal, then the
scripted two-pick continuation of scripts/two_pick.py (open loop: the robot follows the script whatever the model
answers, so both arms read memories of the same events). Before each of the seven scripted actions both arms get one
request built as the closed loop builds it (K: controller.user_content, layout v1; K2: view_k2.request), without
frames. Scored per call: `target` names the container the goal needs now (bin A before its pick is done, calls 1-3;
bin B after A was put down, calls 5-7; call 4, A just lifted, is reported only) and the waypoint's xy is within 3 cm
of that container's true centre. Ground truth is used for scoring only.

  python scripts/k2_offline.py --print                 # episode 0's requests, no API call
  python scripts/k2_offline.py --call --max-usd 1.5    # Qwen, the closed loop's Qwen settings
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run import git_state, load_params  # noqa: E402
from two_pick import Robot, plan  # noqa: E402
from arbiter_kb.closed_loop import Memory  # noqa: E402
from arbiter_kb.controller import parse_answer, system_prompt, user_content  # noqa: E402
from arbiter_kb.llm import LLM, BudgetExceeded  # noqa: E402
from arbiter_kb.sources.synthetic import SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.view_k2 import K2View, request as k2_request  # noqa: E402

ARMS, SCORED, NEAR = ("K", "K2"), (1, 2, 3, 5, 6, 7), 0.03
QWEN = {"model": "qwen/qwen3.6-35b-a3b", "provider": "CoreWeave", "temperature": 0.6, "max_tokens": 32000,
        "sampling": {"top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}}  # stress/closed_loop_pod.py's Qwen settings


def episode(src: SyntheticSource, ep: int, params) -> list[dict]:
    """Seven calls: per call both arms' messages, the container the goal needs now and its true xy."""
    meta, gt = src.meta(ep), src.ground_truth(ep)
    c1 = gt.targets[0]
    c2 = next(c for c in gt.cube_to_bin if c != c1)
    goal = f"pick up the container that hides the {c1[5:]} cube, then the container that hides the {c2[5:]} cube"
    system, rng = ArbiterKBSystem(params), np.random.default_rng([ep, 7])
    system.reset(dataclasses.replace(meta, task_goal=goal))
    mem, view = Memory("K", system, lambda obs: []), K2View()
    for f in src.frames(ep):
        mem.frames.append(f)
        system.step(f)
    truth = src._episode(ep).xyz  # evaluation only: where each container really is at the end of the video
    A, B = gt.cube_to_bin[c1], gt.cube_to_bin[c2]
    robot = Robot(mem.frames[-1], rng, {b: (c, np.array([*truth[b][-1][:2], 0.017])) for c, b in gt.cube_to_bin.items()})
    out = []
    for k, (target, wp, g) in enumerate(plan(A, B, truth[A][-1], truth[B][-1]), 1):
        st = robot.state()
        k2, _ = k2_request(view, system, goal, [], st, False, [])
        right = A if k <= 3 else B
        out.append({"episode": ep, "call": k, "A": A, "B": B, "right": right, "xy": robot.pos[right][:2].tolist(),
                    "K": [{"role": "system", "content": system_prompt("K")},
                          {"role": "user", "content": user_content("K", goal, mem.text(), [], st)}],
                    "K2": [{"role": "system", "content": system_prompt("K2")}] + k2})
        for f in robot.act(system.kb, mem.frames[-1].t, target, wp, g):
            mem.frames.append(f)
            system.step(f)
    return out


def score(item: dict, arm: str, text: str, usage: dict) -> dict:
    ans = parse_answer(text) or {}
    wp = ans.get("waypoint")
    d = float(np.linalg.norm(np.subtract(wp[:2], item["xy"]))) if isinstance(wp, list) and len(wp) == 6 else None
    tgt = ans.get("target")
    return {"episode": item["episode"], "call": item["call"], "arm": arm, "right": item["right"], "A": item["A"],
            "B": item["B"], "target": tgt, "target_ok": int(tgt == item["right"]), "d_xy": None if d is None else
            round(d, 4), "both": int(tgt == item["right"] and d is not None and d <= NEAR),
            "prompt_tokens": usage.get("prompt_tokens", 0), "usd": usage.get("usd", 0.0), "answer": text[-300:]}


def paired(rows: list[dict], calls=SCORED, n_boot: int = 10000) -> dict:
    """Per episode mean of `both` over `calls`; K2 - K, with a 95% paired bootstrap CI over episodes."""
    per = {}
    for r in rows:
        if r["call"] in calls:
            per.setdefault(r["episode"], {}).setdefault(r["arm"], []).append(r["both"])
    eps = [e for e, v in per.items() if all(a in v for a in ARMS)]
    diff = np.array([np.mean(per[e]["K2"]) - np.mean(per[e]["K"]) for e in eps])
    rng = np.random.default_rng(0)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)] if len(diff) else [0.0]
    return {"n_episodes": len(eps), "K": float(np.mean([np.mean(per[e]["K"]) for e in eps])) if eps else None,
            "K2": float(np.mean([np.mean(per[e]["K2"]) for e in eps])) if eps else None,
            "diff": float(diff.mean()) if len(diff) else None,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def summarize(rows: list[dict]) -> dict:
    main_ = paired(rows)
    call4 = {a: {k: sum(r["arm"] == a and r["call"] == 4 and r["target"] == r[k] for r in rows) for k in ("A", "B")}
             for a in ARMS}
    return {"primary": main_, "calls_5_7": paired(rows, (5, 6, 7)), "call_4_target": call4,
            "prompt_tokens_mean": {a: float(np.mean([r["prompt_tokens"] for r in rows if r["arm"] == a] or [0]))
                                   for a in ARMS},
            "usd": round(sum(r["usd"] for r in rows), 4),
            "rule": "K2 may replace K for a closed-loop run only if the primary CI's lower bound is above -0.05",
            "passes": bool(main_["ci95"][0] > -0.05)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--first", type=int, default=100)  # Batch A's train seeds start at 100
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--call", action="store_true")
    ap.add_argument("--max-usd", dest="max_usd", type=float, default=1.5)
    a = ap.parse_args()
    params, (commit, dirty) = load_params(), git_state()
    src = SyntheticSource(swap_steps="mix", lane_offset="mix")
    items = [x for ep in range(a.first, a.first + a.episodes) for x in episode(src, ep, params)]
    if a.print:
        for x in items[:7]:
            for arm in ARMS:
                print(f"=== episode {x['episode']} call {x['call']} arm {arm} (right: {x['right']})")
                print("\n\n".join(p["text"] for m in x[arm][1:] for p in m["content"] if p["type"] == "text"))
    if not a.call:
        return
    llm = LLM(model=QWEN["model"], max_usd=a.max_usd,
              provider={"order": [QWEN["provider"]], "allow_fallbacks": False, "require_parameters": True})
    out = ROOT / "results" / "runs" / f"k2_offline_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)

    def one(job):
        x, arm = job
        try:
            text, u = llm.call(x[arm], temperature=QWEN["temperature"], seed=0, max_tokens=QWEN["max_tokens"],
                               sampling=QWEN["sampling"])
        except BudgetExceeded:
            return None
        return score(x, arm, text, u)
    with ThreadPoolExecutor(8) as pool:
        rows = [r for r in pool.map(one, [(x, arm) for x in items for arm in ARMS]) if r is not None]
    write(out, rows, {"commit": commit, "dirty": dirty, "params_hash": params.hash(), "args": vars(a), "qwen": QWEN,
                      "usage": llm.usage})


def write(out: Path, rows: list[dict], meta: dict) -> None:
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(summary, indent=1), "\n", out)


if __name__ == "__main__":
    main()
