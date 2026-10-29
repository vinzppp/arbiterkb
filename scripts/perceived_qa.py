"""Does the KB-in-words gain survive real perception? (NOTES, same title: the rule was fixed before the run.)
G's perceived VideoUnmaskSwap episodes, train only (episodes 0-9 of each setting); arms R (raw tracker log), K (the
latest view, v4), KR (narrative.py); the questions of extract_test.py plus a confidence per Q1 answer; truth from the
simulator's tracks, mapped to the perceived ids at the last frame. Row "ref": the cubes' visibility from the
simulator (privileged), everything else perceived.

  python scripts/perceived_qa.py --print                    # free: one prompt per arm, truth coverage
  python scripts/perceived_qa.py --call [--keep-as perceived_qa]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from extract_test import EFFORT, ROBOT, questions, right  # noqa: E402
from video_windows import wilson  # noqa: E402
from arbiter_kb.controller import parse_answer, raw_tracker_log  # noqa: E402
from arbiter_kb.llm import LLM  # noqa: E402
from arbiter_kb.narrative import BASE, narrative  # noqa: E402
from arbiter_kb.sg_compat import F_X  # noqa: E402
from arbiter_kb.sources.perceived import PerceivedSource, _seen  # noqa: E402
from arbiter_kb.swapstats import identities  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.views import controller_view  # noqa: E402

ARMS = ("R", "K", "KR")
OCCLUSION = {"r": 0.08, "dz": 0.01}  # chosen on G train by cover recall (0.907 >= the 0.90 gate), NOTES


def slot_on(ids: np.ndarray, node: int, look: int = 5) -> int | None:
    """The container slot sitting on oracle container `node` at the end (the last `look` frames)."""
    for t in range(len(ids) - 1, len(ids) - 1 - look, -1):
        hit = [s for s in range(ids.shape[1]) if ids[t, s] == node]
        if hit:
            return hit[0]
    return None


def truth(d: dict) -> dict:
    """Q1-Q6 in perceived ids; questions whose true container has no slot on it at the end are left out."""
    m, per, ora = d["meta"], d["perceived"], d["oracle"]
    nb, T = m["n_bins"], len(per)
    bins = list(range(nb))
    ids = identities(per, ora, bins, bins)
    end = {n: slot_on(ids, n) for n in bins}
    name = lambda n: f"bin_{end[n]}" if end[n] is not None else None  # noqa: E731
    xy = ora[:, :, F_X:F_X + 2]
    cubes = {int(c): int(b) for c, b in m["cube_to_bin"].items()}
    colour = {c: m["classes"][c].split()[0] for c in cubes}
    t = {f"q1_{colour[c]}": name(b) for c, b in cubes.items()}
    moved = [[n for n in bins if np.linalg.norm(xy[min(t1, T - 1), n] - xy[t0, n]) > 0.01] for t0, t1 in m["schedule"]]
    x = random.Random(d["name"]).choice([n for n in bins if end[n] is not None])
    nearest = min(bins, key=lambda n: np.linalg.norm(xy[-1, n] - BASE))
    target = min(cubes)
    t_cover = max(k for k in range(T) if _seen(ora, k, target))
    t.update(q2=sum(x in s for s in moved), q2_bin=f"bin_{end[x]}", q3=name(max(bins, key=lambda n: xy[-1, n, 1])),
             q4=sorted(name(n) for n in moved[-1]) if all(end[n] is not None for n in moved[-1]) else None,
             q5=next((colour[c] for c, b in cubes.items() if b == nearest), "none") if end[nearest] is not None else None,
             q6="yes" if np.linalg.norm(xy[-1, cubes[target]] - xy[t_cover, cubes[target]]) > 0.03 else "no",
             target=colour[target])
    covered_at = {c: max(k for k in range(T) if _seen(ora, k, c)) + 1 for c in cubes}
    then = {c: slot_on(ids[:covered_at[c] + 6], b) for c, b in cubes.items()}  # the slot on it just after the cover
    t["swapped"] = {colour[c]: then[c] != end[b] for c, b in cubes.items()}  # the tracker moved its id since
    return {k: v for k, v in t.items() if v is not None}


def memory(arm: str, src: PerceivedSource, ep: int, params) -> str:
    frames = list(src.frames(ep))
    if arm == "R":
        return f"Goal: {src.meta(ep).task_goal}\nRaw tracker log (every 10th frame):\n" + raw_tracker_log(frames)
    s = ArbiterKBSystem(params)
    s.reset(src.meta(ep))
    for f in frames:
        s.step(f)
    return controller_view(s.kb, params.rule) if arm == "K" else narrative(s.kb)


def ask(t: dict) -> str:
    q = json.loads(questions(t).split("\n", 1)[1])
    q = {k: v for k, v in q.items() if k in t}
    q["q1_conf"] = "for each q1 answer, how sure are you (0 to 1)? (object: colour -> number)"
    return "Answer with JSON only, one key per question:\n" + json.dumps(q, indent=1)


def jobs(params, relinked: bool = False) -> list[dict]:
    out = []
    rows = (("relinked", PerceivedSource(**OCCLUSION, relinked=True)),) if relinked else \
        (("perceived", PerceivedSource(**OCCLUSION)), ("ref", PerceivedSource(oracle_cubes=True)))
    for row, src in rows:
        for ep in src.episodes("train"):
            d = src.load(ep)
            t = truth(d)
            for arm in ARMS if not relinked else ("R", "K"):
                msgs = [{"role": "system", "content": "You answer questions about a robot's table-top scene from "
                                                      "its memory. " + ROBOT},
                        {"role": "user", "content": memory(arm, src, ep, params) + "\n\n" + ask(t)}]
                out += [{"ep": d["name"], "row": row, "arm": arm, "effort": eff, "msgs": msgs, "truth": t}
                        for eff in EFFORT]
    return out


def score(job: dict, text: str) -> dict:
    ans, t = parse_answer(text), job["truth"]
    ans = ans if isinstance(ans, dict) else {}
    conf = ans.get("q1_conf") if isinstance(ans.get("q1_conf"), dict) else {}
    keys = [k for k in t if k.startswith("q") and k != "q2_bin"]
    q1 = {k[3:]: {"right": right(k, ans.get(k), t[k]), "swapped": t["swapped"][k[3:]],
                  "conf": _num(conf.get(k[3:]))} for k in keys if k.startswith("q1_")}
    return {"ep": job["ep"], "row": job["row"], "arm": job["arm"], "effort": job["effort"],
            **{k: right(k, ans.get(k), t[k]) for k in keys}, "q1": q1, "answers": ans}


def _num(v) -> float | None:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return None


def auroc(conf: list[float], ok: list[bool]) -> float | None:
    pos, neg = [c for c, y in zip(conf, ok) if y], [c for c, y in zip(conf, ok) if not y]
    if not pos or not neg:
        return None
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def report(rows: list[dict]) -> dict:
    out = {}
    for row in sorted({r["row"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows}):
            for eff in EFFORT:
                rs = [r for r in rows if (r["row"], r["arm"], r["effort"]) == (row, arm, eff)]
                q1 = [v for r in rs for v in r["q1"].values()]
                stock = [v["right"] for r in rs if r["ep"].startswith("stock") for v in r["q1"].values()]
                stress = [v["right"] for r in rs if not r["ep"].startswith("stock") for v in r["q1"].values()]
                sw = [v for v in q1 if v["swapped"] and v["conf"] is not None]
                other = [k for k in rs[0] if k.startswith("q") and not k.startswith("q1")] if rs else []
                out[f"{row}/{arm}/{eff}"] = {
                    "q1": wilson(sum(v["right"] for v in q1), len(q1)), "q1_stock": wilson(sum(stock), len(stock)),
                    "q1_stress": wilson(sum(stress), len(stress)),
                    "q1_swapped": wilson(sum(v["right"] for v in q1 if v["swapped"]), sum(v["swapped"] for v in q1)),
                    "auroc_swapped": auroc([v["conf"] for v in sw], [v["right"] for v in sw]),
                    "auroc_all": auroc([v["conf"] for v in q1 if v["conf"] is not None],
                                       [v["right"] for v in q1 if v["conf"] is not None]),
                    **{k: round(float(np.mean([r[k] for r in rs if k in r])), 3) for k in other}}
        for eff in EFFORT:
            for a, b in [p for p in (("KR", "R"), ("K", "R"), ("KR", "K")) if {p[0], p[1]} <= {r["arm"] for r in rows}]:
                out[f"{row}/{a}-{b}/{eff}"] = {part: paired(rows, row, a, b, eff, part)
                                               for part in ("all", "stock", "stress")}
    return out


def paired(rows, row, a, b, eff, part) -> list[float] | None:
    keep = lambda ep: part == "all" or ep.startswith("stock") == (part == "stock")  # noqa: E731
    acc = {(r["arm"], r["ep"]): np.mean([v["right"] for v in r["q1"].values()]) for r in rows
           if r["row"] == row and r["effort"] == eff and r["q1"] and keep(r["ep"])}
    d = np.array([acc[(a, ep)] - acc[(b, ep)] for (arm, ep) in acc if arm == a and (b, ep) in acc])
    if not len(d):
        return None
    boot = np.random.default_rng(0).choice(d, (2000, len(d))).mean(1)
    return [round(float(d.mean()), 3), round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--call", action="store_true", help="required to spend money (1,080 calls; cap $2)")
    ap.add_argument("--keep-as", dest="keep_as", default="")
    ap.add_argument("--relinked", action="store_true", help="container tracks re-linked by motion (relink.py); R, K")
    a = ap.parse_args()
    params, (commit, dirty) = load_params(), git_state()
    todo = jobs(params, a.relinked)
    if a.print:
        for j in todo[:6:2]:
            print(f"===== {j['arm']}\n{j['msgs'][1]['content']}\n")
        ts = list({j["ep"] + j["row"]: j["truth"] for j in todo}.values())
        print(len(todo), "requests; questions kept per episode:", np.mean([sum(k.startswith("q") for k in t) for t in ts]),
              "; q1 cubes swapped:", sum(v for t in ts for v in t["swapped"].values()), "of",
              sum(len(t["swapped"]) for t in ts))
        return
    if not a.call:
        sys.exit("pass --print (free) or --call")
    llm = LLM(model="qwen/qwen3.6-35b-a3b", max_usd=2.0,
              provider={"order": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True})
    with ThreadPoolExecutor(8) as pool:
        texts = list(pool.map(lambda j: llm.call(j["msgs"], max_tokens=4000, reasoning=EFFORT[j["effort"]])[0], todo))
    rows = [score(j, t) for j, t in zip(todo, texts)]
    out = {"report": report(rows), "rows": rows, "usage": llm.usage,
           "meta": {"commit": commit, "dirty": dirty, "params_hash": params.hash(), "occlusion": OCCLUSION, "relinked": a.relinked}}
    print(json.dumps(out["report"], indent=1), json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                           for k, v in llm.usage.items()}))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
