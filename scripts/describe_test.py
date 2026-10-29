"""What can a small VLM see in a window of frames, with no memory text? (NOTES "What can a small VLM see in a
window": the rule was fixed before the run.) The same 60 windows as review_test.py; the only overlay is a mark at
the hand (robot state -> pixel by a homography fitted on other episodes). Condition A: full frames; B: 96-px crops
around the hand, enlarged.

  python scripts/describe_test.py --show                                   # free: saves one window per condition
  python scripts/describe_test.py --long --plan   # v2 windows (t-10 .. t+30); then tools/extract_frames.py --windows
                                                  # data/robomme_lerobot/review_windows_long.json --out review_frames_long
  python scripts/describe_test.py --call --cond A [--model anthropic/claude-sonnet-5.5] [--keep-as NAME]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from review_test import DATA, TASKS, kb_until, wilson, windows  # noqa: E402
from arbiter_kb.controller import image_part, parse_answer, text_part  # noqa: E402
from arbiter_kb.llm import LLM, vlm_client  # noqa: E402
from arbiter_kb.sources import lerobot as L  # noqa: E402

CHOICES = ("pick_up", "put_down", "both", "moving_only", "nothing", "cannot_tell")
CROP = 96
LONG = (10, 30, 5)  # v2 windows: t - 10 .. t + 30, every 5th frame (NOTES "Longer windows")
PROMPT = """These are frames from a robot's front camera over about SECONDS seconds, oldest first, each titled with its
frame number. A small magenta ring marks the robot's gripper. Say what you SEE; if the frames do not show it, say
cannot_tell.
Answer with JSON only:
{"happened": "pick_up|put_down|both|moving_only|nothing|cannot_tell",
 "summary": "<= 25 words: what happened in these frames",
 "relations": [{"subject": "<object, in words, e.g. red cube>", "predicate": "<one word, e.g. on, in, held_by, near>",
                "object": "<object, in words>"}]}
happened: pick_up = the gripper takes an object; put_down = it lets an object go; both = both happen;
moving_only = the arm moves but takes or releases nothing; nothing = no motion.
relations: what holds in the LAST frame, between the objects you can see (including the gripper)."""


def homography() -> np.ndarray:
    """Hand xy -> pixel (u, v), fitted (DLT) on episodes 11-30 of each task: hand xy at gripper closings against the
    grounded subgoal's '<row, column>' pixel (checked by eye on the frames)."""
    src, dst = [], []
    for task in TASKS:
        for path in L.episode_files(task, 30)[10:]:
            ep = L.load(path)
            for t in range(1, len(ep["width"])):
                m = re.search(r"<(\d+), (\d+)>", ep["grounded"][t])
                if m and ep["width"][t - 1] >= 0.07 > ep["width"][t]:
                    src.append(ep["tcp"][t][:2])
                    dst.append([float(m.group(2)), float(m.group(1))])  # labels are <row, column>
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    h = np.linalg.svd(np.asarray(rows))[2][-1].reshape(3, 3)
    res = [np.hypot(*(project(h, s) - d)) for s, d in zip(src, dst)]
    print(f"homography: {len(src)} points, median residual {np.median(res):.1f} px", flush=True)
    return h


def project(h: np.ndarray, xy) -> np.ndarray:
    p = h @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]


def mark(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """A magenta ring of radius 9 px at uv = (column, row)."""
    out = rgb.copy()
    rr, cc = np.ogrid[:out.shape[0], :out.shape[1]]
    ring = np.abs(np.hypot(cc - uv[0], rr - uv[1]) - 9) < 1.5
    out[ring] = [255, 0, 255]
    return out


def crop(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """CROP x CROP around uv (clamped to the image), enlarged to the frame size by pixel repetition + resampling."""
    n = rgb.shape[0]
    c0 = int(np.clip(uv[0] - CROP / 2, 0, n - CROP))
    r0 = int(np.clip(uv[1] - CROP / 2, 0, n - CROP))
    idx = (np.arange(n) * CROP / n).astype(int)
    return rgb[r0:r0 + CROP, c0:c0 + CROP][idx][:, idx]


def long_windows(ws: list[dict]) -> list[dict]:
    """The same events with v2 windows; those running past the episode are left out."""
    out = []
    for w in ws:
        n = len(L.load(Path(w["path"]))["tcp"])
        fr = list(range(w["t"] - LONG[0], w["t"] + LONG[1] + 1, LONG[2]))
        if fr[0] >= 0 and fr[-1] < n:
            out.append({**w, "frames": fr, "dir": "review_frames_long"})
    return out


def frames(w: dict, ep: dict, h: np.ndarray, cond: str) -> list:
    d = np.load(DATA / w.get("dir", "review_frames") / f"{w['key']}.npz")
    rgb = dict(zip(d["idx"].tolist(), d["rgb"]))
    out = []
    for t in w["frames"]:
        uv = project(h, ep["tcp"][t][:2])
        img = mark(rgb[t], uv)
        out.append((t, img if cond == "A" else crop(img, uv)))
    return out


def truth(w: dict, ep: dict, params) -> str:
    kinds = {e.kind for e in kb_until(ep, w["frames"][-1], params).evidence
             if e.kind in ("pick_up", "release") and w["frames"][0] <= e.t <= w["frames"][-1]}
    return "both" if len(kinds) == 2 else "pick_up" if "pick_up" in kinds else "put_down"


def ask(llm, fr: list) -> dict:
    content = [text_part(PROMPT.replace("SECONDS", f"{(fr[-1][0] - fr[0][0]) / 10:.1f}"))]
    for t, img in fr:
        content += [text_part(f"Frame {t}:"), image_part(img)]
    ans = parse_answer(llm.chat([{"role": "user", "content": content}], max_tokens=1500))
    return ans if isinstance(ans, dict) else {}


def plan(ws: list[dict]) -> None:
    frames_by: dict = {}
    for w in ws:
        frames_by.setdefault(w["key"], set()).update(w["frames"])
    (DATA / "review_windows_long.json").write_text(json.dumps({k: sorted(v) for k, v in frames_by.items()}))
    print(len(ws), "windows ->", DATA / "review_windows_long.json")


def row(w: dict, llm, h: np.ndarray, cond: str, params) -> dict:
    ep = L.load(Path(w["path"]))
    ans = ask(llm, frames(w, ep, h, cond))
    r = {"key": w["key"], "t": w["t"], "truth": truth(w, ep, params), "happened": ans.get("happened"),
         "summary": ans.get("summary"), "relations": ans.get("relations"),
         "labels": sorted({ep["simple"][t] for t in w["frames"]})}
    print(json.dumps(r), flush=True)
    return r


def report(rows: list[dict]) -> dict:
    ok = sum(r["happened"] == r["truth"] for r in rows)
    return {"accuracy": wilson(ok, len(rows)), "n": len(rows),
            "cannot_tell": wilson(sum(r["happened"] == "cannot_tell" for r in rows), len(rows)),
            "confusion": {f"{k[0]} -> {k[1]}": n for k, n in
                          collections.Counter((r["truth"], r["happened"]) for r in rows).most_common()},
            "truth_counts": dict(collections.Counter(r["truth"] for r in rows)), "rule": "accuracy >= 0.80"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--call", action="store_true", help="required to spend money (60 calls)")
    ap.add_argument("--cond", default="A", choices=("A", "B"))
    ap.add_argument("--model", default="")
    ap.add_argument("--max-usd", dest="max_usd", type=float, default=1.0)
    ap.add_argument("--keep-as", dest="keep_as", default="")
    ap.add_argument("--long", action="store_true", help="v2 windows (t - 10 .. t + 30)")
    ap.add_argument("--plan", action="store_true", help="with --long: write the frame list for tools/extract_frames.py")
    a = ap.parse_args()
    params, (commit, dirty), h = load_params(), git_state(), homography()
    ws = long_windows(windows(params)) if a.long else windows(params)
    if a.plan:
        plan(ws)
        return
    if a.show:
        for cond in "AB":
            w = ws[0]
            strip = np.concatenate([img for _, img in frames(w, L.load(Path(w["path"])), h, cond)], axis=1)
            mpimg.imsave(DATA / f"describe_{cond}.png", strip)
        print("saved", DATA / "describe_A.png", DATA / "describe_B.png")
        return
    if not a.call:
        sys.exit("pass --show (free) or --call")
    llm = LLM(model=a.model, max_usd=a.max_usd, provider=None) if a.model else vlm_client(params.arbiter, max_usd=a.max_usd)
    rows = [row(w, llm, h, a.cond, params) for w in ws]
    rep = report(rows)
    out = {"report": rep, "rows": rows, "usage": llm.usage, "meta": {"commit": commit, "dirty": dirty, "cond": a.cond, "long": a.long,
                                                                     "model": a.model or params.arbiter.vlm_model}}
    print(json.dumps(rep, indent=1), json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                                 for k, v in llm.usage.items()}))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
