"""B, offline (NOTES "B — the controller's actions and their outcomes"): does the self memory report the outcome of a
decision correctly? The harness loop (harness.run_episode) on recorded RoboMME counting episodes (ReplayEnv), with a
scripted controller whose declared intent is the current subgoal's action (labels: evaluation only).

  python scripts/check_self_memory.py [--first 30 --n 70] [--keep-as self_memory_check]

- right intent: the outcome should say "done" (recall per intent);
- wrong intent (control): the declared intent is replaced by a different action; the outcome should NOT say "done";
- prompt size: the self view's characters against the controller view's.
No model call; episodes 31-100 only (1-30 are kept for the final run, NOTES "Before the full rerun").
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import load_params  # noqa: E402
from check_actions import expected  # noqa: E402
from arbiter_kb.harness import ReplayEnv, run_episode  # noqa: E402
from arbiter_kb.sources import lerobot as L  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.views import controller_view  # noqa: E402

TASKS = ("PickXtimes", "SwingXtimes", "BinFill")
OTHER = {"pick_up": "press", "release": "pick_up", "press": "release", "hold_above": "release"}


def scripted(env: ReplayEnv, wrong: bool):
    def decide(system, prompt: str) -> dict:
        e = expected(env.subgoal())
        kind = e[0] if e else "move"
        if wrong and kind in OTHER:
            kind = OTHER[kind]
        return {"waypoint": None, "gripper_closed": kind in ("pick_up", "press"), "intent": kind,
                "label": env.subgoal()}
    return decide


def episode(path, params, wrong: bool) -> list[dict]:
    ep = L.load(path)
    env, s = ReplayEnv(ep), ArbiterKBSystem(params)
    labels = []
    dec = scripted(env, wrong)

    def decide(system, prompt):
        a = dec(system, prompt)
        labels.append((a["intent"], a["label"]))
        return a
    rows = run_episode(env, s, decide, lambda kb: controller_view(kb, params.rule))
    view = len(controller_view(s.kb, params.rule))
    return [{**r, "label": lab, "declared": i, "view_chars": view} for r, (i, lab) in zip(rows, labels)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=30)
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    params, rep = load_params(), {}
    for wrong in (False, True):
        c = collections.defaultdict(lambda: [0, 0])
        sizes = []
        for task in TASKS:
            for path in L.episode_files(task, a.first + a.n)[a.first:]:
                for r in episode(path, params, wrong):
                    if r["declared"] == "move":
                        continue
                    c[f"{task}/{r['declared']}"][0] += r["outcome"].startswith("done")
                    c[f"{task}/{r['declared']}"][1] += 1
                    sizes.append((r["self_chars"], r["view_chars"]))
        key = "wrong_intent_says_done" if wrong else "right_intent_says_done"
        rep[key] = {k: [v[0], v[1], round(v[0] / v[1], 3)] for k, v in sorted(c.items())}
        rep[key + "_all"] = round(sum(v[0] for v in c.values()) / max(1, sum(v[1] for v in c.values())), 3)
        if not wrong:
            rep["self_chars_median"] = int(np.median([x for x, _ in sizes]))
            rep["view_chars_median"] = int(np.median([y for _, y in sizes]))
    print(json.dumps(rep, indent=1))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
