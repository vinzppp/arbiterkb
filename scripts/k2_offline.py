"""K vs K2 offline (NOTES "K vs K2 offline, redesigned so that every scored call depends on memory"; the rule is
fixed there before any call).

Synthetic VideoUnmaskSwap train episodes (as Batch A: text only) with a two-cube goal, then the probe script of
scripts/two_pick.py (open loop: the robot follows the script whatever the model answers, so both arms read memories
of the same events). At five states, each with the hand not above the container the goal needs, both arms get one
request built as the closed loop builds it (K: controller.user_content, layout v1; K2: view_k2.request), without
frames. Two conditions: clean, and noisy (every visible object, every frame, plus a residual drawn from real perception
on still containers, data/g_perceived_still_residuals.npy from scripts/still_noise.py). Scored: `target` names the
right container and the waypoint's xy is within 3 cm of its true centre. Ground truth is used for scoring only.

  python scripts/k2_offline.py --print [--noisy]      # episode 100's five requests per arm, written to
                                                       # results/runs/k2_print/ep100_<clean|noisy>.txt; no API call
  python scripts/k2_offline.py --call --max-usd 3      # Qwen, the closed loop's Qwen settings
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
from two_pick import Robot, noisy, probe_plan  # noqa: E402
from arbiter_kb.closed_loop import Memory  # noqa: E402
from arbiter_kb.controller import parse_answer, system_prompt, to_action, user_content  # noqa: E402
from arbiter_kb.llm import LLM, BudgetExceeded  # noqa: E402
from arbiter_kb.sources.synthetic import SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.view_k2 import K2View, request as k2_request  # noqa: E402

ARMS, CONDITIONS, NEAR, MARGIN = ("K", "K2"), ("clean", "noisy"), 0.03, -0.05
RESID = ROOT / "data" / "g_perceived_still_residuals.npy"
QWEN = {"model": "qwen/qwen3.6-35b-a3b", "provider": "CoreWeave", "temperature": 0.6, "max_tokens": 32000,
        "sampling": {"top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}}  # stress/closed_loop_pod.py's Qwen settings


def episode(src: SyntheticSource, ep: int, params, condition: str, resid) -> list[dict]:
    """The five asked calls: both arms' messages, the right container and its true xy."""
    meta, gt = src.meta(ep), src.ground_truth(ep)
    c1 = gt.targets[0]
    c2 = next(c for c in gt.cube_to_bin if c != c1)
    goal = f"pick up the container that hides the {c1[5:]} cube, then the container that hides the {c2[5:]} cube"
    rng, pool = np.random.default_rng([ep, 7]), resid if condition == "noisy" else None
    system = ArbiterKBSystem(params)
    system.reset(dataclasses.replace(meta, task_goal=goal))
    mem, view = Memory("K", system, lambda obs: []), K2View()
    for f in src.frames(ep):
        mem.frames.append(noisy(f, rng, pool))
        system.step(mem.frames[-1])
    truth = {k: v[-1] for k, v in src._episode(ep).xyz.items()}  # evaluation only: positions after the video
    A, B = gt.cube_to_bin[c1], gt.cube_to_bin[c2]
    robot = Robot(mem.frames[-1], rng, {b: (c, np.array([*truth[b][:2], 0.017])) for c, b in gt.cube_to_bin.items()},
                  pool)
    robot.pos = {k: (truth[k].copy() if robot.pos[k] is not None else None) for k in robot.pos}  # the truth, not noise
    out = []
    for k, (right, target, wp, g) in enumerate(probe_plan(A, B, robot.pos), 1):
        if right is not None:
            st = robot.state()
            k2, _ = k2_request(view, system, goal, [], st, False, [])
            out.append({"condition": condition, "episode": ep, "call": len(out) + 1, "A": A, "B": B, "right": right,
                        "xy": robot.pos[right][:2].tolist(),
                        "K": [{"role": "system", "content": system_prompt("K")},
                              {"role": "user", "content": user_content("K", goal, mem.text(), [], st)}],
                        "K2": [{"role": "system", "content": system_prompt("K2")}] + k2})
        if wp is None:
            break
        for f in robot.act(system.kb, mem.frames[-1].t, target, wp, g):
            mem.frames.append(f)
            system.step(f)
    return out


def score(item: dict, arm: str, text: str | None, usage: dict, error: str | None = None) -> dict:
    """One row; an answer that is not a valid action scores 0 and never raises."""
    ans = parse_answer(text or "") or {}
    action, _ = to_action(ans)
    tgt = ans.get("target")
    tgt = tgt.strip(" []") if isinstance(tgt, str) else None
    d = None if action is None else float(np.linalg.norm(action[:2] - np.asarray(item["xy"])))
    return {**{k: item[k] for k in ("condition", "episode", "call", "right", "A", "B")}, "arm": arm, "target": tgt,
            "target_ok": int(tgt == item["right"]), "d_xy": None if d is None else round(d, 4),
            "both": int(tgt == item["right"] and d is not None and d <= NEAR), "error": error,
            "prompt_tokens": usage.get("prompt_tokens", 0), "usd": usage.get("usd", 0.0),
            "answer": (text or "")[-300:]}


def paired(rows: list[dict], condition: str, calls=(1, 2, 3, 4, 5), n_boot: int = 10000) -> dict:
    """Only (episode, call) pairs with an answer from both arms; per episode mean of `both`; K2 - K with a 95% paired
    bootstrap CI over episodes."""
    got = {(r["episode"], r["call"], r["arm"]): r["both"] for r in rows
           if r["condition"] == condition and r["call"] in calls and r["error"] is None}
    keys = {(e, c) for e, c, _ in got}
    pairs = sorted(k for k in keys if all((*k, a) in got for a in ARMS))
    per: dict = {}
    for e, c in pairs:
        for a in ARMS:
            per.setdefault(e, {}).setdefault(a, []).append(got[(e, c, a)])
    diff = np.array([np.mean(v["K2"]) - np.mean(v["K"]) for v in per.values()])
    rng = np.random.default_rng(0)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)] if len(diff) else [np.nan]
    return {"pairs": len(pairs), "unpaired": len(keys) - len(pairs), "episodes": len(per),
            **{a: float(np.mean([np.mean(v[a]) for v in per.values()])) if per else None for a in ARMS},
            "diff": float(diff.mean()) if len(diff) else None,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def summarize(rows: list[dict], expected: int) -> dict:
    by = {c: paired(rows, c) for c in CONDITIONS}
    return {"primary": by, "calls_3_5": {c: paired(rows, c, (3, 4, 5)) for c in CONDITIONS},
            "call_4_names_A": {f"{c}/{a}": sum(r["target"] == r["A"] for r in rows if r["condition"] == c and
                                               r["arm"] == a and r["call"] == 4) for c in CONDITIONS for a in ARMS},
            "prompt_tokens_mean": {a: float(np.mean([r["prompt_tokens"] for r in rows if r["arm"] == a] or [0]))
                                   for a in ARMS},
            "calls_expected": expected, "rows": len(rows), "errors": sum(r["error"] is not None for r in rows),
            "usd": round(sum(r["usd"] for r in rows), 4),
            "rule": f"K2 passes only if the CI lower bound is above {MARGIN} in both conditions",
            "passes": all(by[c]["pairs"] and by[c]["ci95"][0] > MARGIN for c in CONDITIONS)}


def ask_all(llm: LLM, items: list[dict]) -> list[dict]:
    def one(job):
        x, arm = job
        try:
            text, u = llm.call(x[arm], temperature=QWEN["temperature"], seed=0, max_tokens=QWEN["max_tokens"],
                               sampling=QWEN["sampling"])
            return score(x, arm, text, u)
        except BudgetExceeded:
            return score(x, arm, None, {}, "budget")
        except Exception as ex:  # an API error: recorded, unpaired, reported
            return score(x, arm, None, {}, repr(ex)[:200])
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, [(x, arm) for x in items for arm in ARMS]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--first", type=int, default=100)  # Batch A's train seeds start at 100
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--noisy", action="store_true", help="--print the noisy condition instead of the clean one")
    ap.add_argument("--call", action="store_true")
    ap.add_argument("--max-usd", dest="max_usd", type=float, default=3.0)
    a = ap.parse_args()
    params, (commit, dirty) = load_params(), git_state()
    src, resid = SyntheticSource(swap_steps="mix", lane_offset="mix"), np.load(RESID)
    items = [x for c in CONDITIONS for ep in range(a.first, a.first + a.episodes) for x in episode(src, ep, params, c,
                                                                                                     resid)]
    if a.print:
        show(items, a.first, "noisy" if a.noisy else "clean")
    if not a.call:
        return
    llm = LLM(model=QWEN["model"], max_usd=a.max_usd,
              provider={"order": [QWEN["provider"]], "allow_fallbacks": False, "require_parameters": True})
    out = ROOT / "results" / "runs" / f"k2_offline_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    rows = ask_all(llm, items)
    write(out, rows, summarize(rows, 2 * len(items)), {"commit": commit, "dirty": dirty, "params_hash": params.hash(),
                                                        "args": vars(a), "qwen": QWEN, "usage": llm.usage})


def show(items: list[dict], ep: int, condition: str) -> None:
    """Every asked call of one episode, both arms, full text (each user message separated by a blank line)."""
    lines = []
    for x in (x for x in items if x["episode"] == ep and x["condition"] == condition):
        for arm in ARMS:
            lines.append(f"=== episode {ep} {condition} call {x['call']} arm {arm} (right: {x['right']})")
            lines.append("\n\n".join(p["text"] for m in x[arm][1:] for p in m["content"] if p["type"] == "text"))
    path = ROOT / "results" / "runs" / "k2_print" / f"ep{ep}_{condition}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(path)


def write(out: Path, rows: list[dict], summary: dict, meta: dict) -> None:
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(summary, indent=1), "\n", out)


if __name__ == "__main__":
    main()
