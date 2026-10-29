"""Counting tasks from robot state only (NOTES "Beyond permanence"): at each moment an action has just finished,
does the controller know whether to repeat it or to wrap up? RoboMME dataset, PickXtimes / SwingXtimes / BinFill.

  python scripts/count_test.py --tasks PickXtimes,SwingXtimes,BinFill --n 30 --arms N,P,K,KO --effort off,default \
         --workers 8 --call --max-usd 4 [--print]

Arms (same prompt skeleton, same action list; only the memory differs):
  N   no memory: the goal and the current hand state;
  P   raw robot log: hand xyz and gripper every 10 frames since the start;
  K   the KB run on the robot state (grip and hand-pause evidence, progress counts, event log; views.controller_view);
  KO  oracle: the subgoals completed so far, from the dataset authors' labels (upper bound).
The answer is one action from a fixed list (no object positions exist without vision, so no waypoint is scored).
Truth = the action class of the next subgoal. Decision points = LAG frames after each "place / put into the bin /
reach the left-side target" subgoal ends in the execution phase (v1 used the boundary itself).
KS (v2) = K with the semantic actions (actions.py); K in v1 had only grip / pause events.
KR = the same KB in words (narrative.robot_narrative; NOTES "Counting with the KB in words"); KT = KR trimmed; KH = K + joined facts; KC = K + a goal-progress line
(goal_counter.py; the goal parsed once per episode).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from arbiter_kb import CONTRACT_VERSION  # noqa: E402
from arbiter_kb.controller import parse_answer  # noqa: E402
from arbiter_kb.goal_counter import parse_goal, progress_line  # noqa: E402
from arbiter_kb.llm import LLM, BudgetExceeded  # noqa: E402
from arbiter_kb.narrative import merged_view, robot_narrative  # noqa: E402
from arbiter_kb.sources import lerobot as L  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.views import controller_view  # noqa: E402

ACTIONS = ["pick up a cube", "place the cube on the target", "put the cube into the bin",
           "move to the top of the right-side target", "move to the top of the left-side target",
           "put the cube on the table", "press the button"]
KEYS = [("press the button", 6), ("into the bin", 2), ("on the table", 5), ("right-side target", 3),
        ("left-side target", 4), ("onto the target", 1), ("pick up", 0)]
EFFORT = {"default": None, "off": {"enabled": False}}
LAG = 5  # frames after a step's end before deciding, for every arm (v2 of this test; v1 decided at the boundary)
SYSTEM = ("You decide a robot's next step. You get the task, a memory of what the robot has done so far, and the "
          "robot's current state. Answer with JSON only: {\"next\": \"<one action, copied exactly from this list>\"}. "
          "Actions: " + json.dumps(ACTIONS))


def action_of(text: str) -> int | None:
    return next((i for k, i in KEYS if k in text), None)


def decisions(ep: dict) -> list[tuple[int, int]]:
    """[(t, truth action index)] at the start of each subgoal that follows a completed place / bin / left move."""
    segs = [s for s in L.segments(ep["simple"]) if not ep["demo"][s[0]]]
    out = []
    for prev, nxt in zip(segs, segs[1:]):
        t = nxt[0] + LAG  # a step counts as done once it is observable (the KB needs n_pause still frames)
        if action_of(prev[2]) in (1, 2, 4) and action_of(nxt[2]) is not None and t < nxt[1]:
            out.append((t, action_of(nxt[2])))
    return out


def raw_log(ep: dict, t: int, stride: int = 10) -> str:
    rows = [f"t={k}: hand ({ep['tcp'][k][0]:.3f}, {ep['tcp'][k][1]:.3f}, {ep['tcp'][k][2]:.3f}), gripper "
            f"{'closed' if ep['width'][k] < 0.07 else 'open'}" for k in list(range(0, t, stride)) + [t - 1]]
    return "Robot log (every 10th frame, 10 frames per second):\n" + "\n".join(rows)


def kb_view(ep: dict, t: int, params, words: bool = False, trim: bool = False, joined: bool = False,
            spec: dict | None = None) -> str:
    s = ArbiterKBSystem(params)
    s.reset(L.meta(ep))
    for f in L.frames(ep, t):
        s.step(f)
    if words:
        return robot_narrative(s.kb, trim)
    view = controller_view(s.kb, params.rule)
    if spec is not None:
        view = view.replace("\nProgress:", f"\n{progress_line(s.kb, spec)}\nProgress:", 1) if spec else view
    return merged_view(view, s.kb) if joined else view


def oracle(ep: dict, t: int) -> str:
    done = [x for x in L.segments(ep["simple"]) if x[1] <= t and not ep["demo"][x[0]]]
    return "Completed so far:\n" + ("\n".join(f"{i + 1}. {x[2]}" for i, x in enumerate(done)) or "nothing")


def prompt(arm: str, ep: dict, t: int, params, spec: dict | None = None) -> list:
    mem = {"N": lambda: "", "P": lambda: raw_log(ep, t), "K": lambda: kb_view(ep, t, params),
           "KR": lambda: kb_view(ep, t, params, words=True), "KT": lambda: kb_view(ep, t, params, True, True),
           "KH": lambda: kb_view(ep, t, params, joined=True), "KC": lambda: kb_view(ep, t, params, spec=spec or {}),
           "KO": lambda: oracle(ep, t)}[arm]()
    hand = ep["tcp"][t - 1]
    now = (f"Current state: frame {t}, hand at ({hand[0]:.3f}, {hand[1]:.3f}, {hand[2]:.3f}), gripper "
           f"{'closed' if ep['width'][t - 1] < 0.07 else 'open'}.")
    task = "" if mem.startswith("Goal:") else f"Task: {ep['meta']['task']}"  # the KB view already states the goal
    user = "\n\n".join(x for x in (task, mem, now) if x)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def jobs(a, params, specs: dict | None = None) -> list[dict]:
    out = []
    for task in a.tasks.split(","):
        for path in L.episode_files(task, a.first + a.n)[a.first:]:
            ep = L.load(path)
            for k, (t, truth) in enumerate(decisions(ep)):
                for arm in a.arms.split(","):
                    msgs = prompt(arm, ep, t, params, (specs or {}).get(ep["meta"]["episode"]))
                    out += [{"task": task, "episode": ep["meta"]["episode"], "k": k, "t": t, "truth": ACTIONS[truth],
                             "arm": arm, "effort": eff, "msgs": msgs, "model": a.model} for eff in a.effort.split(",")]
    return out


def goal_specs(a) -> dict:
    """Arm KC: one goal parse per episode (goal_counter.parse_goal), thinking default; needs --call (cached after)."""
    if not a.call:
        sys.exit("arm KC parses each goal with one model call: pass --call")
    llm = LLM(model=a.model, max_usd=0.2, provider={"order": [a.provider], "allow_fallbacks": False,
                                                    "require_parameters": True})
    eps = [L.load(p) for task in a.tasks.split(",") for p in L.episode_files(task, a.first + a.n)[a.first:]]
    with ThreadPoolExecutor(a.workers) as pool:
        got = list(pool.map(lambda e: parse_goal(llm, e["meta"]["task"]), eps))
    return {e["meta"]["episode"]: g for e, g in zip(eps, got)}


def score(job: dict, text: str, u: dict) -> dict:
    ans = parse_answer(text)
    got = ans.get("next") if isinstance(ans, dict) else None
    row = {k: job[k] for k in ("task", "episode", "k", "t", "arm", "effort", "truth", "model")}
    row.update(answer=got, correct=got == job["truth"], valid=got in ACTIONS)
    row.update({k: round(float(u.get(k, 0)), 6) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens",
                                                           "seconds", "usd", "cached_usd")})
    return row


def run(a) -> None:
    commit, dirty = git_state()
    params = load_params()
    specs = goal_specs(a) if "KC" in a.arms.split(",") else None
    todo = jobs(a, params, specs)
    if a.print:
        for m in todo[0]["msgs"]:
            print(f"--- {m['role']}\n{m['content']}")
        print(f"\n{len(todo)} requests")
        return
    slug = a.model.split("/")[-1].replace(".", "")  # one directory per run: two runs in one second collided (v3)
    out = ROOT / "results" / "runs" / f"count_{time.strftime('%Y%m%d_%H%M%S')}_{slug}_{os.getpid()}"
    out.mkdir(parents=True, exist_ok=False)
    prov = None if a.provider == "auto" else {"order": [a.provider], "allow_fallbacks": False, "require_parameters": True}
    llm = LLM(model=a.model, max_usd=a.max_usd, provider=prov)
    ask = lambda j: llm.call(j["msgs"], max_tokens=a.max_tokens, reasoning=EFFORT[j["effort"]])  # noqa: E731
    with open(out / "rows.csv", "w", newline="") as f, ThreadPoolExecutor(a.workers) as pool:
        w = None
        for job, fut in [(j, pool.submit(ask, j)) for j in todo]:
            try:
                row = score(job, *fut.result())
            except BudgetExceeded as e:
                print("stopped:", e)
                break
            w = w or csv.DictWriter(f, fieldnames=list(row))
            w.writerow(row) if f.tell() else (w.writeheader(), w.writerow(row))
            f.flush()
    meta = {"goal_specs": specs, "commit": commit, "dirty": dirty, "args": {k: v for k, v in vars(a).items()}, "params_hash": params.hash(),
            "contract_version": CONTRACT_VERSION, "usage": llm.usage, "data_revision": "92796ab (lerobot/robomme)"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("wrote", out.relative_to(ROOT), "usage", {k: round(v, 4) if isinstance(v, float) else v for k, v in llm.usage.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="PickXtimes,SwingXtimes,BinFill")
    ap.add_argument("--n", type=int, default=30, help="episodes per task")
    ap.add_argument("--first", type=int, default=0, help="skip the first episodes (v3: 60 -> unseen 61-100)")
    ap.add_argument("--provider", default="DeepInfra", help="pinned provider, or 'auto' for OpenRouter routing")
    ap.add_argument("--arms", default="N,P,K,KO")
    ap.add_argument("--effort", default="off,default")
    ap.add_argument("--model", default="qwen/qwen3.6-35b-a3b")
    ap.add_argument("--max-usd", dest="max_usd", type=float, default=2.0)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--print", action="store_true", help="show the first prompt and the request count; no API call")
    ap.add_argument("--call", action="store_true", help="required to spend money")
    a = ap.parse_args()
    if not (a.print or a.call):
        sys.exit("pass --print (free) or --call")
    run(a)


if __name__ == "__main__":
    main()
