"""Does a richer KB text make facts easier to extract? (NOTES "Does a richer KB text make the facts easier to
extract?": the rule was fixed before the run.) Synthetic VideoUnmaskSwap, train seeds 100-129; arms R (raw tracker
log), K (controller view), KR (narrative.py); 8 questions per episode in one call; truth from the generator.

  python scripts/extract_test.py --print          # free: one prompt per arm
  python scripts/extract_test.py --call [--keep-as extract_kb]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from review_test import wilson  # noqa: E402
from arbiter_kb.controller import parse_answer, raw_tracker_log  # noqa: E402
from arbiter_kb.llm import LLM  # noqa: E402
from arbiter_kb.narrative import BASE, narrative  # noqa: E402
from arbiter_kb.sources.synthetic import COVER_T, HOLD, SWAP_T0, SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.views import controller_view  # noqa: E402

EFFORT = {"default": None, "off": {"enabled": False}}
SEEDS = range(100, 130)
ROBOT = "The robot's base is at (-0.615, 0) and it faces +x; the robot's left is +y."


def truth(src: SyntheticSource, ep: int) -> dict:
    e, gt = src._episode(ep), src.ground_truth(ep)  # evaluation only (E5b)
    bins = [o for o, cls, _ in e.meta.objects if cls == "container"]
    steps = src.settings(ep)[1]
    swaps = [[b for b in bins if np.linalg.norm(e.xyz[b][SWAP_T0 + (k + 1) * steps - 1, :2] -
                                                 e.xyz[b][SWAP_T0 + k * steps, :2]) > 0.01]
             for k in range((len(e.xyz[bins[0]]) - SWAP_T0 - HOLD) // steps)]
    x = random.Random(ep).choice(bins)
    final = {b: e.xyz[b][-1, :2] for b in bins}
    nearest = min(bins, key=lambda b: np.linalg.norm(final[b] - BASE))
    hidden = {b: c for c, b in gt.cube_to_bin.items()}
    target = gt.targets[0]
    return {**{f"q1_{c[5:]}": b for c, b in gt.cube_to_bin.items()}, "q2": sum(x in s for s in swaps), "q2_bin": x,
            "q3": max(bins, key=lambda b: final[b][1]), "q4": sorted(swaps[-1]),
            "q5": hidden[nearest][5:] if nearest in hidden else "none", "q6": "yes" if gt.moved[target] else "no",
            "target": target[5:]}


def questions(t: dict) -> str:
    cols = [k[3:] for k in t if k.startswith("q1_")]
    q = {**{f"q1_{c}": f"which container hides the {c} cube now? (container id)" for c in cols},
         "q2": f"how many times did {t['q2_bin']} move? (integer)",
         "q3": "which container is now farthest to the robot's left? (container id)",
         "q4": "which two containers took part in the most recent exchange of places? (list of two container ids)",
         "q5": "which cube is hidden in the container nearest the robot now? (its colour, or none)",
         "q6": f"has the container that hides the {t['target']} cube moved since it covered the cube? (yes or no)"}
    return "Answer with JSON only, one key per question:\n" + json.dumps(q, indent=1)


def memory(arm: str, src: SyntheticSource, ep: int, params) -> str:
    frames = list(src.frames(ep))
    if arm == "R":
        return f"Goal: {src.meta(ep).task_goal}\nRaw tracker log (every 10th frame):\n" + raw_tracker_log(frames)
    s = ArbiterKBSystem(params)
    s.reset(src.meta(ep))
    for f in frames:
        s.step(f)
    return controller_view(s.kb, params.rule) if arm == "K" else narrative(s.kb)


def right(key: str, got, want) -> bool:
    if key == "q4":
        return isinstance(got, list) and sorted(map(str, got)) == want
    if key == "q2":
        return str(got).strip() == str(want)
    g = str(got).lower()
    if key == "q5":
        return (re.search(r"red|green|blue|none|yellow|purple|orange", g) or [None])[0] == want
    return g.strip() == want


def jobs(params) -> list[dict]:
    src, out = SyntheticSource(lane_offset="mix", swap_steps="mix"), []
    for ep in SEEDS:
        t = truth(src, ep)
        for arm in ("R", "K", "KR"):
            msgs = [{"role": "system", "content": "You answer questions about a robot's table-top scene from its "
                                                  "memory. " + ROBOT},
                    {"role": "user", "content": memory(arm, src, ep, params) + "\n\n" + questions(t)}]
            out += [{"ep": ep, "arm": arm, "effort": eff, "msgs": msgs, "truth": t} for eff in EFFORT]
    return out


def score(job: dict, text: str) -> dict:
    ans, t = parse_answer(text), job["truth"]
    ans = ans if isinstance(ans, dict) else {}
    keys = [k for k in t if k.startswith("q") and k != "q2_bin"]
    return {"ep": job["ep"], "arm": job["arm"], "effort": job["effort"],
            **{k: right(k, ans.get(k), t[k]) for k in keys}, "answers": ans}


def report(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k.startswith("q")]
    out = {}
    for arm in ("R", "K", "KR"):
        for eff in EFFORT:
            rs = [r for r in rows if r["arm"] == arm and r["effort"] == eff]
            per = {k: wilson(sum(r[k] for r in rs), len(rs)) for k in keys}
            out[f"{arm}/{eff}"] = {"all": wilson(sum(r[k] for r in rs for k in keys), len(rs) * len(keys)), **per}
    out["paired"] = {f"{a}-{b}/{eff}": paired(rows, a, b, eff, keys) for eff in EFFORT
                     for a, b in (("KR", "R"), ("KR", "K"), ("K", "R"))}
    return out


def paired(rows, a, b, eff, keys) -> list[float]:
    acc = {(r["arm"], r["ep"]): np.mean([r[k] for k in keys]) for r in rows if r["effort"] == eff}
    d = np.array([acc[(a, ep)] - acc[(b, ep)] for ep in SEEDS if (a, ep) in acc and (b, ep) in acc])
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return [round(float(d.mean()), 3), round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--call", action="store_true", help="required to spend money (180 calls, cap $2)")
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    params, (commit, dirty) = load_params(), git_state()
    todo = jobs(params)
    if a.print:
        for j in todo[:6:2]:
            print(f"===== {j['arm']}\n{j['msgs'][1]['content']}\n")
        print(len(todo), "requests;", json.dumps(todo[0]["truth"]))
        return
    if not a.call:
        sys.exit("pass --print (free) or --call")
    llm = LLM(model="qwen/qwen3.6-35b-a3b", max_usd=2.0,
              provider={"order": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True})
    with ThreadPoolExecutor(8) as pool:
        texts = list(pool.map(lambda j: llm.call(j["msgs"], max_tokens=4000, reasoning=EFFORT[j["effort"]])[0], todo))
    rows = [score(j, t) for j, t in zip(todo, texts)]
    out = {"report": report(rows), "rows": rows, "usage": llm.usage,
           "meta": {"commit": commit, "dirty": dirty, "params_hash": params.hash(), "seeds": [SEEDS[0], SEEDS[-1]]}}
    print(json.dumps(out["report"], indent=1), json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                           for k, v in llm.usage.items()}))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
