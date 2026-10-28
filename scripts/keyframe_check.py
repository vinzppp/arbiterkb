"""Does the merged keyframe call (arbiter_kb/keyframe.py, spec v1.9) still list the objects right? The same 4 real
VideoUnmaskSwap frames as the lister benchmark (results/latest/vlm_list_bench), the real VLM (ArbiterParams), two
memories: none (the first frame) and a synthetic KB with 7 objects (a full memory in the prompt; its ids do not match
the picture, so relations are not scored, only counted).

  python scripts/keyframe_check.py --call [--keep-as keyframe_check]

Scoring as in NOTES "Which small VLM can list the objects?": a frame is right when the total count is right, the
coloured cubes are exactly right, and no other kinds of object are named (white / gray containers called cubes pass).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from arbiter_kb.kb import KB  # noqa: E402
from arbiter_kb.keyframe import keyframe_call  # noqa: E402
from arbiter_kb.llm import vlm_client  # noqa: E402
from arbiter_kb.sources.synthetic import SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402

FRAMES = ROOT / "results" / "latest" / "vlm_list_bench"
TRUTH = {"f21": {"red": 1, "green": 1, "blue": 1, "plain": 0}, "f56": {"plain": 3}, "f105": {"plain": 3},
         "f325": {"green": 1, "plain": 3}}
KINDS = {"cube", "block", "container", "cup", "bin", "box", "bowl", "cover", "lid", "mug"}
PLAIN = {"", "white", "gray", "grey", "silver"}
GOAL = "pick up the container that hides the green cube"


def right(objs: list[dict], truth: dict) -> bool:
    got: dict = {}
    for o in objs:
        if o["name"].split()[-1].rstrip("s") not in KINDS:
            return False
        key = "plain" if o["color"] in PLAIN else o["color"]
        got[key] = got.get(key, 0) + o["count"]
    return {k: v for k, v in got.items() if v} == {k: v for k, v in truth.items() if v}


def memories(params) -> dict:
    empty = KB([])
    empty.task["goal"] = GOAL
    src = SyntheticSource(lane_offset=0.0, swap_steps=20, n_swaps=1)
    ep = src.episodes("train", 1)[0]
    s = ArbiterKBSystem(params)
    s.reset(src.meta(ep))
    for f in src.frames(ep):
        s.step(f)
    s.kb.task["goal"] = GOAL
    return {"none": empty, "full": s.kb}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", action="store_true", help="required to spend money (8 calls, under $0.01)")
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    if not a.call:
        sys.exit("pass --call")
    params, (commit, dirty) = load_params(), git_state()
    llm, rows = vlm_client(params.arbiter, max_usd=0.2), []
    for mem, kb in memories(params).items():
        for f, truth in TRUTH.items():
            rgb = (mpimg.imread(FRAMES / f"{f}.png")[..., :3] * 255).astype(np.uint8)
            rep = keyframe_call(kb, llm, t=0, rgb=rgb)
            rows.append({"memory": mem, "frame": f, "right": right(rep["clue_objects"], truth),
                         "objects": [f"{o['count']} {o['color']} {o['name']}" for o in rep["clue_objects"]],
                         "written": rep["written"], "problems": rep["problems"]})
            print(json.dumps(rows[-1]), flush=True)
    out = {"rows": rows, "usage": llm.usage, "meta": {"commit": commit, "dirty": dirty, "params_hash": params.hash(),
                                                     "model": params.arbiter.vlm_model, "goal": GOAL}}
    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in llm.usage.items()}))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
