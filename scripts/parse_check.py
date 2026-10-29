"""Goal-parse accuracy, v1 (goal_counter as first run: first-to-last brace, no extra rules) against the current
goal_counter (v2: last JSON object, two general rules; v3: + the back-and-forth rule), on all 100 episodes of each counting task (NOTES "Goal parse v2"). Scored on the target
number; the kind may be any action whose count equals the target's at the decision points.

  python scripts/parse_check.py --call [--keep-as goal_parse]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state  # noqa: E402
from arbiter_kb import goal_counter as G  # noqa: E402
from arbiter_kb.controller import parse_answer  # noqa: E402
from arbiter_kb.llm import LLM  # noqa: E402
from arbiter_kb.sources import lerobot as L  # noqa: E402

KEY = {"PickXtimes": ("onto the target", {"pick_up", "release"}), "BinFill": ("into the bin", {"pick_up", "release"}),
       "SwingXtimes": ("left-side target", {"hold_left", "hold_right"})}
V1_RULES = (" If the instruction gives no number, the target is 1. A motion repeated back and forth\nN times means N "
            "holds on each side. When the instruction moves an object back and forth between two places, count\nthe "
            "holds, whether or not it gives a number.")


def v1(llm, goal: str) -> dict | None:
    text = llm.call([{"role": "user", "content": G.PROMPT.replace(V1_RULES, "").format(goal=goal)}], max_tokens=4000)[0]
    spec = parse_answer(text)
    return spec if isinstance(spec, dict) and spec.get("count") in G.KINDS and isinstance(spec.get("target"), int) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", action="store_true")
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    if not a.call:
        sys.exit("pass --call (about 600 short calls)")
    llm = LLM(model="qwen/qwen3.6-35b-a3b", max_usd=0.5,
              provider={"order": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True})
    rows = []
    for task, (key, kinds) in KEY.items():
        eps = [L.load(p) for p in L.episode_files(task)]
        with ThreadPoolExecutor(8) as pool:
            olds = list(pool.map(lambda e: v1(llm, e["meta"]["task"]), eps))
            news = list(pool.map(lambda e: G.parse_goal(llm, e["meta"]["task"]), eps))
        for e, o, n in zip(eps, olds, news):
            tgt = sum(key in s[2] for s in L.segments(e["simple"]) if not e["demo"][s[0]])
            ok = lambda sp: bool(sp) and sp["count"] in kinds and sp["target"] == tgt  # noqa: E731
            rows.append({"task": task, "episode": e["meta"]["episode"], "truth": tgt, "v1": o, "v2": n,
                         "v1_ok": ok(o), "v2_ok": ok(n)})
    rep = {t: {v: [sum(r[f"{v}_ok"] for r in rows if r["task"] == t), sum(r["task"] == t for r in rows)]
               for v in ("v1", "v2")} for t in KEY}
    print(json.dumps(rep), json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in llm.usage.items()}))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps({"report": rep, "rows": rows, "usage": llm.usage,
                                                    "meta": dict(zip(("commit", "dirty"), git_state()))}, indent=1))


if __name__ == "__main__":
    main()
