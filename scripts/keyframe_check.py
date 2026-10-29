"""Does the keyframe call (arbiter_kb/keyframe.py) list the objects right? The same 4 real VideoUnmaskSwap frames as
the lister benchmark (results/latest/vlm_list_bench), the real VLM (ArbiterParams). v1.9's merged prompt (listing +
annotation, with the memory in the prompt) was checked with an earlier version of this script
(archive/results/keyframe_check, commit 409476c); v1.10 lists from the image alone.

  python scripts/keyframe_check.py --call [--keep-as keyframe_check_v110]

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
from arbiter_kb.keyframe import list_objects  # noqa: E402
from arbiter_kb.llm import vlm_client  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", action="store_true", help="required to spend money (4 calls, under $0.01)")
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    if not a.call:
        sys.exit("pass --call")
    params, (commit, dirty) = load_params(), git_state()
    llm, rows = vlm_client(params.arbiter, max_usd=0.2), []
    for f, truth in TRUTH.items():
        rgb = (mpimg.imread(FRAMES / f"{f}.png")[..., :3] * 255).astype(np.uint8)
        rep = list_objects(llm, rgb, GOAL)
        rows.append({"frame": f, "right": right(rep["objects"], truth), "problems": rep["problems"],
                     "objects": [f"{o['count']} {o['color']} {o['name']}" for o in rep["objects"]]})
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
