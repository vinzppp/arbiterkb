"""M4 dry run (DEVELOPMENT §6 M4 item 4): the controller's first grasp waypoint on recorded train episodes,
without moving a robot. The KB is filled up to the decision frame t*; each arm then gets one request.

  python scripts/dry_run.py --source synthetic --arms K,K-L3,R --n 5 --print     # show the prompt, no API call
  python scripts/dry_run.py --source oracle --arms H,K,R --n 5 --call --max-usd 0.50

Scored per (episode, arm): parsed / invalid, the check() problems, the xy distance from the waypoint to the
TRUE target container at t* (from GroundTruth: evaluation only, never in the prompt), whether "target" names
the container the KB places there, tokens, images and $. Synthetic episodes have no images, so the arms read
text only (H needs images: oracle replays). Train split only; this is prompt development (E5c).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from arbiter_kb import controller as C  # noqa: E402
from arbiter_kb.evaluate import physical_truth  # noqa: E402
from arbiter_kb.llm import LLM  # noqa: E402
from arbiter_kb.sources.replay import OracleReplaySource  # noqa: E402
from arbiter_kb.sources.synthetic import SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.views import controller_view  # noqa: E402


def fill(src, ep, params):
    """(system with its KB at t*, frames up to t*, ground truth, target cube, t*)."""
    gt = src.ground_truth(ep)
    cube = gt.targets[0]
    t_star = gt.t_star[cube]
    s = ArbiterKBSystem(params)
    s.reset(src.meta(ep))
    frames = []
    for f in src.frames(ep):
        s.step(f)
        frames.append(f)
        if f.t == t_star:
            break
    return s, frames, gt, cube, t_star


def images(src, ep, t_star, arm) -> list:
    """[(rgb, label)]: H gets every 3rd front frame + the current front / wrist; the others only the current."""
    if not isinstance(src, OracleReplaySource):
        return []
    rep = src._load(ep)[0]
    cur = [(rep.front_rgb[t_star], "Current front camera"), (rep.wrist_rgb[t_star], "Current wrist camera")]
    if arm != "H":
        return cur
    idx = sorted(set(range(0, t_star + 1, 3)) | {t_star})
    return [(rep.front_rgb[i], f"Front frame {i}") for i in idx] + cur[1:]


def messages(arm, s, frames, src, ep, t_star) -> list:
    f = frames[-1]
    state = {"tcp_xyz": [round(float(v), 4) for v in f.ee_xyz], "finger_positions_m": [round(f.gripper_width / 2, 4)] * 2,
             "elapsed_control_steps": f.t}
    view = controller_view(s.kb, s.params.rule)
    inside = {c: d.argmax()[0] for (p, c), d in s.kb.relations.items() if p == "Inside"
              and s.kb.objects[c].visibility.value == "hidden"}
    mem = {"K": view, "K-L3": view.split("Event log")[0].rstrip(), "R": C.raw_tracker_log(frames), "H": "",
           "K0": C.object_table(frames[-1]), "S": C.one_sentence(inside)}[arm]
    content = C.user_content(arm, s.kb.task["goal"], mem, images(src, ep, t_star, arm), state)
    return [{"role": "system", "content": C.system_prompt(arm)}, {"role": "user", "content": content}]


def score(arm, ep, text, s, gt, cube, bins) -> dict:
    ans = C.parse_answer(text)
    act, kind = C.to_action(ans)
    row = {"episode": ep, "arm": arm, "kind": kind or "action", "problems": "", "xy_err_m": None,
           "target": None, "target_is_true_bin": None}
    if act is None:
        return row
    true_xy = gt.bin_xyz_at_t_star[cube][:2]
    truth_label = physical_truth(s, bins, gt.bin_xyz_at_t_star[cube])
    row.update(problems="; ".join(C.check(act, ans, s.kb.known_ids() | set(s.kb.objects))),
               xy_err_m=round(float(np.linalg.norm(act[:2] - true_xy)), 4), target=ans.get("target"),
               target_is_true_bin=ans.get("target") == truth_label)
    return row


def run(args) -> list[dict]:
    params = load_params()
    src = OracleReplaySource() if args.source == "oracle" else SyntheticSource(lane_offset="mix", swap_steps="mix")
    eps = src.episodes("train", n=args.n)
    if not eps:
        print(f"SKIPPED: no {args.source} train episodes (oracle: run scripts/copy_replays.sh)")
        return []
    llm, rows = LLM(model=args.model, max_usd=args.max_usd), []
    for ep in eps:
        s, frames, gt, cube, t_star = fill(src, ep, params)
        bins = [o for o, c, _ in src.meta(ep).objects if c == "container"]
        for arm in args.arms.split(","):
            if arm == "H" and args.source != "oracle":
                continue
            msgs = messages(arm, s, frames, src, ep, t_star)
            if args.print:
                show(msgs)
                return []
            before = dict(llm.usage)
            row = score(arm, ep, llm.chat(msgs, max_tokens=args.max_tokens), s, gt, cube, bins)
            row.update({k: round(llm.usage[k] - before[k], 6) for k in ("prompt_tokens", "completion_tokens", "images", "usd")})
            rows.append(row)
            print(row)
    return rows


def show(msgs: list) -> None:
    for m in msgs:
        parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        print(f"--- {m['role']}")
        for p in parts:
            print(p["text"] if p["type"] == "text" else "<image>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "oracle"], default="synthetic")
    ap.add_argument("--arms", default="K,K0,S,R")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="qwen/qwen3.6-35b-a3b")
    ap.add_argument("--max-usd", dest="max_usd", type=float, default=0.25)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=800)
    ap.add_argument("--print", action="store_true", help="print the first prompt and stop (no API call)")
    ap.add_argument("--call", action="store_true", help="required to spend money")
    args = ap.parse_args()
    if not (args.print or args.call):
        sys.exit("pass --print (free) or --call (spends up to --max-usd)")
    commit, dirty = git_state()
    rows = run(args)
    if rows:
        out = ROOT / "results" / "runs" / f"dry_{args.source}_{time.strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "dry.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out.relative_to(ROOT)}/dry.csv  (commit {commit}, dirty={dirty})")


if __name__ == "__main__":
    main()
