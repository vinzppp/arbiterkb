"""Semantic actions (arbiter_kb/actions.py) against the RoboMME dataset's subgoal labels, per action type.

  python scripts/check_actions.py [--tasks PickXtimes,SwingXtimes,BinFill] [--n 100] [--keep-as NAME]

For every labelled subgoal with an expected action (pick up -> pick_up, place onto the target -> release placed,
put into the bin -> release dropped, put on the table -> release, press -> press, right / left target ->
hold_above right / left): recall = such subgoals with a matching action between their start and end + slack.
Precision = actions that fall in a subgoal expecting them. Robot state only; labels are evaluation only.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from arbiter_kb.detectors import Detectors  # noqa: E402
from arbiter_kb.sources import lerobot as L  # noqa: E402

EXPECT = [("onto the target", "release", "placed"), ("into the bin", "release", "dropped"),
          ("on the table", "release", None), ("press the button", "press", None),
          ("right-side target", "hold_above", "right"), ("left-side target", "hold_above", "left"),
          ("pick up", "pick_up", None)]
SLACK = 8  # frames after a subgoal ends that its action may still be reported (pause / press completion)


def expected(text: str) -> tuple[str, str | None] | None:
    return next(((k, v) for key, k, v in EXPECT if key in text), None)


def actions(ep: dict, params) -> list:
    det, out = Detectors(params.detect), []
    for f in L.frames(ep, len(ep["tcp"])):
        out += [e for e in det.step(f) if e.kind in ("pick_up", "release", "press", "hold_above")]
    return out


def matches(e, kind: str, detail: str | None) -> bool:
    return e.kind == kind and (detail is None or detail in (e.data.get("how"), e.data.get("side")))


def episode(ep: dict, params, rec: dict, prec: dict) -> None:
    acts, segs = actions(ep, params), L.segments(ep["simple"])
    for s, e, text in segs:
        exp = expected(text)
        if exp:
            hit = any(s <= a.t <= e + SLACK and matches(a, *exp) for a in acts)
            rec[exp][0] += hit
            rec[exp][1] += 1
    for a in acts:
        ok = any(s <= a.t <= e + SLACK and (x := expected(text)) and matches(a, *x) for s, e, text in segs)
        prec[a.kind][0] += ok
        prec[a.kind][1] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="PickXtimes,SwingXtimes,BinFill")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    params, (commit, dirty) = load_params(), git_state()
    rec, prec = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    for task in a.tasks.split(","):
        for path in L.episode_files(task, a.n):
            episode(L.load(path), params, rec, prec)
    out = {"recall": {f"{k}:{v}": [h, n, round(h / n, 3)] for (k, v), (h, n) in sorted(rec.items(), key=str)},
           "precision": {k: [h, n, round(h / n, 3)] for k, (h, n) in sorted(prec.items())},
           "meta": {"commit": commit, "dirty": dirty, "params_hash": params.hash(), "tasks": a.tasks, "n": a.n,
                    "slack_frames": SLACK}}
    print(json.dumps(out, indent=1))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
