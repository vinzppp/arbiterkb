"""Phase 'Done when' checks (DEVELOPMENT.md §6 M0 and M1), printed as PASS / FAIL / SKIPPED / INFO lines.
Run through `python scripts/run.py --selftest` (or `make selftest`). Exit code 1 if any FAIL.
Train seeds only: `make check` never reads the test split, so it adds nothing to the ledger (E5c)."""
from __future__ import annotations

import hashlib
import sys

import numpy as np

from figures import all_figures
from run import ROOT, base_source, evaluate, load_params, parse
from arbiter_kb import sg_compat as sg
from arbiter_kb.debug import explain
from arbiter_kb.sources.replay import OracleReplaySource
from arbiter_kb.systems.baselines import TrackerSystem

RESULTS: list[str] = []
M1_ARGV = ["--split", "train", "--systems", "all", "--corrupt", "none,id_swap_overlap,pos_noise", "--p", "0.5",
           "--lane", "mix", "--steps", "mix", "--n", "60"]


def check(name: str, ok: bool | None, value) -> None:
    tag = "SKIPPED" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append(tag)
    print(f"[{tag}] {name}: {value}")


def info(name: str, value) -> None:
    print(f"[INFO] {name}: {value}")


def row(summary, cond, system):
    return next(r for r in summary if r["condition"] == cond and r["system"] == system)


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


def m0_checks() -> None:
    args = parse(["--split", "train", "--p", "1.0"])
    run_dir, summary, inf = evaluate(args, load_params())
    a, b, f = row(summary, "none", "arbiterkb"), row(summary, "none", "bound"), row(summary, "none", "frozen")
    check("M0 (b) clean accuracy ArbiterKB / Bound = 1.0", a["acc"] == 1.0 and b["acc"] == 1.0, (a["acc"], b["acc"]))
    check("M0 (b) Frozen worse on moved than back-at-start", f["acc_moved"] < f["acc_back"], (f["acc_moved"], f["acc_back"]))
    ac, bc = row(summary, "id_swap", "arbiterkb"), row(summary, "id_swap", "bound")
    check("M0 (c) id_swap p=1: Bound silent failure on corrupted = 1.0", bc["silent_on_corrupted"] == 1.0,
          f"{bc['silent_on_corrupted']} (n_corrupted={bc['n_corrupted']})")
    info("M0 (c) teleport id_swap is a sanity corruption only (M1 R6 uses 3D distance, so lane 0.07 passes "
         "barely mix); ArbiterKB silent on corrupted", f"{ac['silent_on_corrupted']} vs Bound {bc['silent_on_corrupted']}")
    info("M0 (d) ArbiterKB mean confidence when right, clean", a["conf_right"])
    hashes = []
    for _ in range(2):
        rd, _, _ = evaluate(parse(["--split", "train", "--p", "1.0", "--n", "10"]), load_params())
        hashes.append(hashlib.sha256((rd / "summary.csv").read_bytes()).hexdigest()[:12])
    check("M0 (f) two identical runs give the same summary hash", hashes[0] == hashes[1], hashes)


def m1_checks() -> dict:
    args = parse(M1_ARGV)
    run_dir, summary, inf = evaluate(args, load_params())
    systems = sorted({r["system"] for r in summary})
    conds = sorted({r["condition"] for r in summary})
    check("M1 table: 6 systems x 3 conditions", len(systems) == 6 and len(conds) == 3 and len(summary) == 18,
          f"{len(systems)} systems x {len(conds)} conditions")
    jumps = [j for v in inf["logs"]["id_swap_overlap"].values() for j in v["label_jumps_m"]]
    check("M1 id_swap_overlap label jumps median <= d_ov (labels do not teleport)",
          bool(jumps) and float(np.median(jumps)) <= args.d_ov, f"n={len(jumps)} median={np.median(jumps):.4f} "
          f"max={max(jumps):.4f} m")
    bo, jd = row(summary, "id_swap_overlap", "bound"), row(summary, "id_swap_overlap", "jump_detector")
    check("M1 JumpDetector == Bound under id_swap_overlap (jumps stay below j_max)",
          (bo["acc"], bo["silent"]) == (jd["acc"], jd["silent"]), f"silent {jd['silent']} vs {bo['silent']}")
    oh = row(summary, "id_swap_overlap", "overlap_haircut")
    check("M1 OverlapHaircut silent on corrupted < Bound's", oh["silent_on_corrupted"] < bo["silent_on_corrupted"],
          f"{oh['silent_on_corrupted']} vs {bo['silent_on_corrupted']}")
    for cond in conds:
        a, o = row(summary, cond, "arbiterkb"), row(summary, cond, "overlap_haircut")
        info(f"M1 {cond}: ArbiterKB vs OverlapHaircut (coverage, silent, AURC, brier)",
             f"({a['coverage']}, {a['silent']}, {a['aurc']}, {a['brier']}) vs ({o['coverage']}, {o['silent']}, "
             f"{o['aurc']}, {o['brier']})  (plumbing only; untuned)")
    bad = []
    for r in inf["rows"]:
        tr = inf["traces"][(f"{r['condition']}_{r['system']}", r["episode"])]
        text = explain(tr, r["cube"], r["t_star"])
        if "observation" not in text or "NOT FOUND" in text:
            bad.append((r["condition"], r["system"], r["episode"]))
    check("M1 E2: explain reaches an observation for every scored row", not bad,
          f"{len(inf['rows'])} rows, {len(bad)} failing {bad[:3]}")
    r5 = sum(1 for (k, _), tr in inf["traces"].items() if k.endswith("arbiterkb")
             for x in tr for op in x["ops"] if op.get("rule") == "R5")
    info("M1 R5 joint_exclusive ops in the table run (cover frames only)", r5)
    figs = all_figures(run_dir, inf, base_source(args), args)
    check("M1 figures written (timeline, reliability, label jumps, risk-coverage)",
          len(figs) == 4 and all(f.exists() for f in figs),
          [f.name for f in figs])
    return inf


def selftest() -> int:
    m0_checks()
    inf = m1_checks()
    replay_check()
    step_ms = 1000 * float(np.median(inf["step_s"]))
    check("E7 median ArbiterKB step < 1 ms", step_ms < 1.0, f"{step_ms:.3f} ms")
    heavy = [m for m in ("torch", "transformers", "openai") if m in sys.modules]
    check("no GPU / LLM libraries imported", not heavy, heavy or "none")
    print(f"selftest: {RESULTS.count('PASS')} PASS, {RESULTS.count('FAIL')} FAIL, {RESULTS.count('SKIPPED')} SKIPPED")
    return 1 if "FAIL" in RESULTS else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(selftest())
