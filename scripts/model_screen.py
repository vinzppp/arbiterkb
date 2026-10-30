"""Model screen (NOTES "Model screen: rules fixed before any call"): can a model write the right waypoint at each stage
of a two-pick VideoUnmaskSwap episode when its memory is right?

Synthetic train episodes as scripts/k2_offline.py builds them (two-cube goal), clean (memory from simulator truth), the
containers at RoboMME's height (screen_rules.BIN_Z); only episodes whose memory is sure of both goal containers
(select). Seven stages, each asked once with the closed loop's K request
(controller.user_content, layout v1, no images: the synthetic source has none): above A, close on A, lift A, put A
down, above B, close on B, done (after the scripted lift of B). Between stages the robot follows two_pick.plan (open
loop) at RoboMME's heights except the grasp (SCRIPT_Z). OpenRouter's defaults (no temperature, seed,
reasoning or provider sent); every state twice, each pass with its own disk cache (a rerun resumes for free).

  python scripts/model_screen.py --print                      # the first episode's seven requests; no call
  python scripts/model_screen.py --mock                       # the whole scoring path with a fake model; no call
  python scripts/model_screen.py --call --max-usd 10 [--models id,id]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run import git_state, load_params  # noqa: E402
from screen_rules import BIN_Z, SCRIPT_GRASP_Z, STAGES, mock_transport, score  # noqa: E402
from two_pick import Robot, plan  # noqa: E402
from arbiter_kb.closed_loop import Memory  # noqa: E402
from arbiter_kb.controller import system_prompt, user_content  # noqa: E402
from arbiter_kb.llm import LLM, BudgetExceeded, post  # noqa: E402
from arbiter_kb.sources.synthetic import SyntheticSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.types import FrameObs, ObjObs  # noqa: E402

OUT, CACHE = ROOT / "results" / "runs" / "model_screen", ROOT / "results" / "llm_cache" / "model_screen"
MIN_P = 0.8  # select(): the KB must be this sure of both goal containers (fixed before any call)
SCRIPT_Z = [0.2, SCRIPT_GRASP_Z, 0.2, 0.07, 0.2, SCRIPT_GRASP_Z, 0.2]  # RoboMME's heights, but the grasp's
# id, group, total / active parameters (B; None = not published), $ per M in / out, max_tokens (min(32000, the
# model's max output on OpenRouter)); OpenRouter models API, 2026-10-30. Cheapest first: a shared cap starves the end.
CANDIDATES = [
    ("meta-llama/llama-4-scout", "small", 109, 17, 0.10, 0.30, 16384),
    ("nvidia/nemotron-3.5-super-vl", "small", 120, 12, 0.15, 0.50, 32000),
    ("google/gemma-4-31b-it", "small", 30.7, 30.7, 0.20, 0.40, 16384),
    ("qwen/qwen3.5-122b-a10b", "small", 122, 10, 0.26, 2.08, 32000),
    ("z-ai/glm-4.6v", "small", 106, 12, 0.30, 0.90, 32000),
    ("qwen/qwen3.6-27b", "small", 27, 27, 0.32, 3.20, 32000),
    ("google/gemini-4-argon", "strong", None, None, 2.00, 10.00, 32000),
    ("openai/gpt-6-sol", "strong", None, None, 2.00, 10.00, 32000),
    ("openai/gpt-6-astra", "strong", None, None, 10.00, 50.00, 32000)]


def _real_z(f: FrameObs) -> FrameObs:
    """Containers at RoboMME's bin origin height (the synthetic source puts them at 0.03)."""
    objs = {k: ObjObs(o.id, o.cls, o.visible, np.array([*o.xyz[:2], BIN_Z]) if k.startswith("bin_") and o.xyz
                      is not None else o.xyz, o.uv, o.track_q) for k, o in f.objects.items()}
    return FrameObs(f.t, objs, f.ee_xyz, f.gripper_width, f.is_video, f.rgb)


def _setup(src: SyntheticSource, ep: int, params):
    meta, gt = src.meta(ep), src.ground_truth(ep)
    c1 = gt.targets[0]
    c2 = next(c for c in gt.cube_to_bin if c != c1)
    goal = f"pick up the container that hides the {c1[5:]} cube, then the container that hides the {c2[5:]} cube"
    system = ArbiterKBSystem(params)
    system.reset(dataclasses.replace(meta, task_goal=goal))
    mem = Memory("K", system, lambda obs: [])
    for f in src.frames(ep):
        mem.frames.append(_real_z(f))
        system.step(mem.frames[-1])
    truth = {k: np.array([*v[-1][:2], BIN_Z if k.startswith("bin_") else v[-1][2]])
             for k, v in src._episode(ep).xyz.items()}  # positions after the video
    A, B = gt.cube_to_bin[c1], gt.cube_to_bin[c2]
    robot = Robot(mem.frames[-1], np.random.default_rng([ep, 7]),
                  {b: (c, np.array([*truth[b][:2], 0.017])) for c, b in gt.cube_to_bin.items()})
    robot.pos = {k: (truth[k].copy() if robot.pos[k] is not None else None) for k in robot.pos}
    return goal, system, mem, robot, A, B


def episode(src: SyntheticSource, ep: int, params) -> list[dict]:
    """The seven asked states of one episode: the K request and the truth the rules need."""
    goal, system, mem, robot, A, B = _setup(src, ep, params)
    steps, out = plan(A, B, robot.pos[A], robot.pos[B]), []
    for k in range(8):  # stage k+1 is asked before scripted action k+1; B's lift (action 7) is not asked
        if k != 6:
            stage, acted = len(out) + 1, A if len(out) < 4 else B
            out.append({"episode": ep, "stage": stage, "name": STAGES[stage], "A": A, "B": B, "truth": {
                "A": robot.pos[A][:2].tolist(), "B": robot.pos[B][:2].tolist(), "hand": robot.ee[:2].tolist(),
                "others": {b: p[:2].tolist() for b, p in robot.pos.items() if b.startswith("bin_") and b != acted}},
                "messages": [{"role": "system", "content": system_prompt("K")}, {"role": "user", "content":
                             user_content("K", goal, mem.text(), [], robot.state())}]})
        if k == 7:
            return out
        target, wp, g = steps[k]
        for f in robot.act(system.kb, mem.frames[-1].t, target, [*wp[:2], SCRIPT_Z[k]], g):
            mem.frames.append(f)
            system.step(f)
        if k == 3:  # released at 0.07: the container settles on the table
            robot.pos[A][2] = BIN_Z
    return out


def select(src: SyntheticSource, params, first: int, n: int) -> list[dict]:
    """The first n episodes from `first` on whose memory is right and sure about both goal containers: the KB's top
    Inside belief for each goal cube is the true container with p >= MIN_P (the screen tests execution, not doubt)."""
    out, ep = [], first
    while len({x["episode"] for x in out}) < n and ep < first + 100:
        items = episode(src, ep, params)
        view = "\n".join(p["text"] for p in items[0]["messages"][1]["content"] if p["type"] == "text")
        colours = re.search(r"hides the (\w+) cube, then the container that hides the (\w+) cube", view).groups()
        top = {c: (b, float(p)) for c, b, p in re.findall(r"\[Inside\(cube_(\w+)\)\] (bin_\d+) ([\d.]+)", view)}
        if all(top.get(c, ("", 0))[0] == items[0][k] and top[c][1] >= MIN_P for c, k in zip(colours, "AB")):
            out += items
        ep += 1
    return out


def ask(llm: LLM, item: dict, model: str, pass_no: int, max_tokens: int) -> dict:
    base = {"model": model, "pass": pass_no, **{k: item[k] for k in ("episode", "stage", "name", "A", "B")}}
    try:
        text, u = llm.call(item["messages"], temperature=None, seed=None, max_tokens=max_tokens)
        err = None
    except BudgetExceeded:
        text, u, err = None, {}, "budget"
    except Exception as e:  # noqa: BLE001 (an API error is recorded, never scored as the model's)
        text, u, err = None, {}, repr(e)[:200]
    row = {**base, **score(item["stage"], item["truth"], text), "error": err,
           "usd": round(u.get("usd", 0.0) + u.get("cached_usd", 0.0), 6), "seconds": round(u.get("seconds", 0.0), 2),
           "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
           "provider": _provider(llm, item, max_tokens) if err is None else None, "answer": text}
    if err:
        row.update(valid_json=None, passed=None, reason=f"not asked: {err}")
    return row


def _provider(llm: LLM, item: dict, max_tokens: int) -> str | None:
    """The provider OpenRouter routed to, from the cached response (the routing is OpenRouter's default)."""
    body = llm.body(item["messages"], None, None, None, max_tokens)
    path = llm.cache_dir / f"{hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}.json"
    return json.loads(path.read_text())["response"].get("provider") if path.exists() else None


def run_model(cand: tuple, items: list[dict], passes: int, cap: float, transport=post, cache=CACHE) -> list[dict]:
    model, *_, max_tokens = cand
    rows = []
    for p in range(1, passes + 1):
        llm = LLM(model=model, cache_dir=cache / f"pass{p}", max_usd=max(cap - sum(r["usd"] for r in rows), 0.0),
                  transport=transport, provider=None)
        llm.key = llm.key or ("mock" if transport is not post else "")
        with ThreadPoolExecutor(8) as pool:
            rows += list(pool.map(lambda x: ask(llm, x, model, p, max_tokens), items))
    return rows


def summarize(rows: list[dict], cand: tuple) -> dict:
    ok = [r for r in rows if r["error"] is None]
    by = {(r["episode"], r["stage"]): {} for r in ok}
    for r in ok:
        by[(r["episode"], r["stage"])][r["pass"]] = r["passed"]
    both = [v for v in by.values() if len(v) == 2]
    model, group, total, active, pin, pout, _ = cand
    return {"model": model, "group": group, "total_B": total, "active_B": active, "usd_per_M": [pin, pout],
            "calls": len(rows), "answered": len(ok), "errors": len(rows) - len(ok),
            "stage_pass": {f"{s}_{n}": _avg([r["passed"] for r in ok if r["stage"] == s]) for s, n in STAGES.items()},
            "pass_all": _avg([r["passed"] for r in ok]), "valid_json": _avg([r["valid_json"] for r in ok]),
            "agreement": _avg([int(len(set(v.values())) == 1) for v in both]), "pairs": len(both),
            "usd_per_call": _avg([r["usd"] for r in ok], 5), "s_per_call": _avg([r["seconds"] for r in ok], 2),
            "usd_total": round(sum(r["usd"] for r in rows), 4)}


def _avg(x: list, nd: int = 3) -> float | None:
    return round(float(np.mean(x)), nd) if x else None


def write(out: Path, rows: list[dict], summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# {summary['model']}", "", "| stage | pass rate |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["stage_pass"].items()]
    lines += ["", *(f"- {k}: {summary[k]}" for k in ("pass_all", "valid_json", "agreement", "pairs", "usd_per_call",
                                                      "s_per_call", "usd_total", "errors")), "", "## Failures", ""]
    for r in (r for r in rows if r["passed"] == 0):
        lines += [f"### ep {r['episode']} stage {r['stage']} {r['name']} pass {r['pass']}: {r['reason']}", "",
                  "```", str(r["answer"]), "```", ""]
    (out / "report.md").write_text("\n".join(lines))


def show(items: list[dict], ep: int) -> Path:
    """One episode's seven requests in full, with the truth each is scored against."""
    x0 = next(x for x in items if x["episode"] == ep)
    lines = [f"=== episode {ep}: A = {x0['A']}, B = {x0['B']}", "--- system prompt (K)", x0["messages"][0]["content"]]
    for x in (x for x in items if x["episode"] == ep):
        lines += [f"\n=== stage {x['stage']} {x['name']}; scored against {json.dumps(x['truth'])}",
                  "\n\n".join(p["text"] for p in x["messages"][1]["content"] if p["type"] == "text")]
    path = ROOT / "results" / "runs" / "model_screen_print" / f"ep{ep}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--first", type=int, default=100)  # train seeds, as k2_offline
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--call", action="store_true")
    ap.add_argument("--models", default="", help="comma-separated ids (default: every candidate)")
    ap.add_argument("--max-usd", dest="max_usd", type=float, default=10.0, help="total cap over all models")
    ap.add_argument("--run", default="screen1", help="results/runs/model_screen/<run>/<model>/")
    a = ap.parse_args()
    params, src = load_params(), SyntheticSource(swap_steps="mix", lane_offset="mix")
    items = select(src, params, a.first, a.episodes)
    if a.print:
        print(show(items, items[0]["episode"]))
    if a.mock:
        cand = ("mock/oracle", "mock", None, None, 0.0, 0.0, 1000)
        rows = run_model(cand, items, a.passes, 1.0, mock_transport(items), Path(tempfile.mkdtemp()))
        write(OUT / "mock" / "mock_oracle", rows, s := summarize(rows, cand))
        print(json.dumps(s, indent=1))
    if not a.call:
        return
    chosen = [c for c in CANDIDATES if not a.models or c[0] in a.models.split(",")]
    meta, spent = {"commit": git_state(), "params_hash": params.hash(), "args": vars(a), "candidates": chosen,
            "episodes": sorted({x["episode"] for x in items})}, 0.0
    (OUT / a.run).mkdir(parents=True, exist_ok=True)
    (OUT / a.run / "meta.json").write_text(json.dumps(meta, indent=1))
    for cand in chosen:
        rows = run_model(cand, items, a.passes, a.max_usd - spent)
        spent += sum(r["usd"] for r in rows)  # replayed calls report their original cost: a resume counts them
        write(OUT / a.run / cand[0].replace("/", "__"), rows, s := summarize(rows, cand))
        print(json.dumps({k: s[k] for k in ("model", "pass_all", "valid_json", "agreement", "usd_total")}))


if __name__ == "__main__":
    main()
