"""Phase M0 'Done when' checks (DEVELOPMENT.md §6 M0), printed as PASS / FAIL / SKIPPED lines.
Run through `python scripts/run.py --selftest` (or `make selftest`). Exit code 1 if any FAIL."""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np

from run import ROOT, evaluate, figure, load_params
from arbiter_kb import sg_compat as sg
from arbiter_kb.debug import explain
from arbiter_kb.sources.replay import OracleReplaySource
from arbiter_kb.sources.synthetic import SyntheticSource
from arbiter_kb.systems.baselines import TrackerSystem

RESULTS: list[str] = []


def check(name: str, ok: bool | None, value) -> None:
    tag = "SKIPPED" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append(tag)
    print(f"[{tag}] {name}: {value}")


def row(summary, cond, system):
    return next(r for r in summary if r["condition"] == cond and r["system"] == system)


def args(**kw):
    a = dict(source="synthetic", corrupt="none,id_swap", p=1.0, split="test", n=30, selftest=False, keep=False)
    a.update(kw)
    return argparse.Namespace(**a)


def replay_check() -> None:
    src = OracleReplaySource()
    eps = src.episodes("train", n=1)
    if not eps:
        check("(g) real oracle replays: Frozen/Bound adapters reproduce sg_pilot", None,
              f"no replays in {src.root}; run scripts/copy_replays.sh when a pod is up")
        return
    ep = eps[0]
    rep, g, full, names = src._load(ep)
    gt = src.ground_truth(ep)
    cube, t = gt.targets[0], gt.t_star[gt.targets[0]]
    node = {v: k for k, v in names.items()}[cube]
    errs = []
    for rule in ("frozen", "bound"):
        ref, _ = sg.apply_rule(g, rule, rep.gripper_state.sum(1), rep.ee_xyz, rep.K_front, rep.E_front)
        s = TrackerSystem(rule)
        s.reset(src.meta(ep))
        for f in src.frames(ep):
            s.step(f)
            if f.t == t:
                break
        ours = sg.project_points(s.xyz(cube), rep.K_front, rep.E_front[t])[0]
        theirs = sg.project_points(ref[t, node, sg.F_X:sg.F_Z + 1], rep.K_front, rep.E_front[t])[0]
        errs.append(float(np.linalg.norm(ours - theirs)))
    check("(g) real oracle replays: Frozen/Bound adapters reproduce sg_pilot (px)", max(errs) < 0.5, errs)


def selftest() -> int:
    run_dir, summary, info = evaluate(args(), load_params())
    a, b, f = row(summary, "none", "arbiterkb"), row(summary, "none", "bound"), row(summary, "none", "frozen")
    check("(b) clean accuracy ArbiterKB / Bound = 1.0", a["acc"] == 1.0 and b["acc"] == 1.0, (a["acc"], b["acc"]))
    check("(b) Frozen worse on moved than back-at-start", f["acc_moved"] < f["acc_back"], (f["acc_moved"], f["acc_back"]))
    ac, bc = row(summary, "id_swap", "arbiterkb"), row(summary, "id_swap", "bound")
    check("(c) id_swap p=1: Bound silent failure on corrupted = 1.0", bc["silent_on_corrupted"] == 1.0,
          f"{bc['silent_on_corrupted']} (n_corrupted={bc['n_corrupted']})")
    check("(c) id_swap p=1: ArbiterKB silent failure on corrupted < Bound's",
          ac["silent_on_corrupted"] < bc["silent_on_corrupted"], ac["silent_on_corrupted"])
    print(f"[INFO] (d) ArbiterKB mean confidence when right, clean: {a['conf_right']}  (< 1 by design; M1 tunes)")
    log = info["logs"]["id_swap"]
    ep = next(e for e, v in sorted(log.items()) if v["corrupted"])
    cube = SyntheticSource().ground_truth(ep).targets[0]
    text = explain(info["traces"][("id_swap_arbiterkb", ep)], cube)
    check("(e) debug explain reaches observations on a corrupted episode", "observation" in text,
          text.splitlines()[0])
    fig = figure(run_dir, info, SyntheticSource())
    check("figure written", fig is not None and fig.exists(), fig)
    hashes = []
    for _ in range(2):
        rd, _, _ = evaluate(args(n=10), load_params())
        hashes.append(hashlib.sha256((rd / "summary.csv").read_bytes()).hexdigest()[:12])
    check("(f) two identical runs give the same summary hash", hashes[0] == hashes[1], hashes)
    replay_check()
    step_ms = 1000 * float(np.median(info["step_s"]))
    check("E7 median ArbiterKB step < 1 ms", step_ms < 1.0, f"{step_ms:.3f} ms")
    heavy = [m for m in ("torch", "transformers", "openai") if m in sys.modules]
    check("no GPU / LLM libraries imported", not heavy, heavy or "none")
    print(f"selftest: {RESULTS.count('PASS')} PASS, {RESULTS.count('FAIL')} FAIL, {RESULTS.count('SKIPPED')} SKIPPED")
    return 1 if "FAIL" in RESULTS else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(selftest())
