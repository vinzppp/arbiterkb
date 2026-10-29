"""Claim 3 on G's perceived episodes, without an LLM (NOTES "Claim 3 on G": the rule is fixed before the run).
Per Q1 cube (which container hides it, truth from perceived_qa.truth on re-linked tracks): each system's answer (its
most likely container) and that answer's probability. Systems: Bound (a point mass), SoftHaircut(feature) (R6's
formula without a KB) and ArbiterKB (R6 inside the KB), each with the pass feature min_3d (scale d0) or min_q (the
re-linking's assignment margin carried in track_q, scale q0). Metrics: accuracy, AUROC of p for right / wrong, AURC
(metrics.aurc: mean risk over coverage, ties enter together; code review 2026-10-29), overall and within stress.
--ablate (NOTES "Claim 3 attribution"): the KB as read with R6 off, no cover kernel / floor, R5 off, cover only.

  python scripts/claim3_g.py --grid                 # G train: every scale, choose per system by AURC
  python scripts/claim3_g.py --read [--keep-as claim3_g]   # G test, the frozen choice, once (ledger READ)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from perceived_qa import OCCLUSION, truth  # noqa: E402
from qa_stats import auroc  # noqa: E402
from arbiter_kb import ledger  # noqa: E402
from arbiter_kb.metrics import aurc as spec_aurc  # noqa: E402
from arbiter_kb.sources.perceived import PerceivedSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.systems.baselines import TrackerSystem  # noqa: E402
from arbiter_kb.systems.soft_haircut import SoftHaircut  # noqa: E402

GRID = {"min_3d": (0.02, 0.05, 0.10, 0.20), "min_q": (0.05, 0.10, 0.20, 0.50)}
CHOICE = ROOT / "results" / "latest" / "claim3_g" / "choice.json"


ABLATE = {"kb": ({}, ()), "kb-R6off": ({"w_max": 0.0}, ()), "kb-nokernel": ({"sigma_cover": 0.001, "eps_far": 0.01}, ()),
          "kb-R5off": ({}, ("R5",)), "kb-coveronly": ({"w_max": 0.0}, ("R5",))}


def make(kind: str, feature: str, scale: float, params, over: dict | None = None, skip: tuple = ()):
    rule = replace(params.rule, r6_feature=feature, **({"q0": scale} if feature == "min_q" else {"d0": scale}),
                   **(over or {}))
    p = replace(params, rule=rule)
    return {"bound": lambda: TrackerSystem("bound", p), "soft": lambda: SoftHaircut(p, feature),
            "kb": lambda: ArbiterKBSystem(p, skip=skip)}[kind]()


def answers(system, src, ep, d, t) -> list[dict]:
    system.reset(src.meta(ep))
    for f in src.frames(ep):
        system.step(f)
    out = []
    for k, want in t.items():
        if not k.startswith("q1_"):
            continue
        probs = system.believe_inside("cube_" + k[3:]).probs
        hyps = sorted(((b, p) for b, p in probs.items() if b), key=lambda kv: -kv[1])
        top, p = hyps[0] if hyps else (None, 0.0)
        cov = system.kb.objects["cube_" + k[3:]].t_last_seen if hasattr(system, "kb") else None
        near = cov is not None and any(e.kind == "near_pass" and e.t > cov and top in e.args for e in system.kb.evidence)
        out.append({"ep": d["name"], "stress": not d["name"].startswith("stock"), "right": top in want, "p": p,
                    "swapped": t["swapped"][k[3:]], "near": near})
    return out


def aurc(rows: list[dict]) -> float:
    return spec_aurc([{"p_max": r["p"], "correct": r["right"]} for r in rows])


def _auroc(score, label) -> float | None:
    v = auroc(score, label)
    return None if v is None else round(v, 3)


def metrics(rows: list[dict]) -> dict:
    s, k = [r for r in rows if r["stress"]], [r for r in rows if r["swapped"] is not None]
    return {"n": len(rows), "acc": round(float(np.mean([r["right"] for r in rows])), 3),
            "auroc": _auroc([r["p"] for r in rows], [r["right"] for r in rows]), "aurc": round(aurc(rows), 4),
            "auroc_stress": _auroc([r["p"] for r in s], [r["right"] for r in s]), "aurc_stress": round(aurc(s), 4),
            "auroc_clean_vs_swapped": _auroc([r["p"] for r in k], [not r["swapped"] for r in k]),
            "auroc_setting_bit": _auroc([0 if r["stress"] else 1 for r in rows], [r["right"] for r in rows]),
            "auroc_no_near_pass": _auroc([0 if r["near"] else 1 for r in rows], [r["right"] for r in rows])}


def auroc_of(rows: list[dict]) -> float:
    v = auroc([r["p"] for r in rows], [r["right"] for r in rows])
    return 0.5 if v is None else v


def ablate(split: str, params, out_dir: Path, commit: str, dirty: bool) -> None:
    """NOTES "Claim 3 attribution": post-hoc on test (ledger READ); nothing is tuned here."""
    if split == "test":
        ledger.record("READ", params.hash(), "G perceived test, claim 3 attribution ablations (post-hoc), claim3_g.py")
    settings = {"bound": ("bound", "min_3d", 0.02), "soft/min_3d/0.02": ("soft", "min_3d", 0.02)}
    settings |= {n: ("kb", "min_3d", 0.02, over, skip) for n, (over, skip) in ABLATE.items()}
    rows = run(split, settings, params)
    table = {n: metrics(r) for n, r in rows.items()}
    pairs = {f"{n}-kb": {"aurc": paired(rows[n], rows["kb"], aurc), "auroc": paired(rows[n], rows["kb"], auroc_of)}
             for n in ABLATE if n != "kb"}
    pairs["kb-soft"] = {"aurc": paired(rows["kb"], rows["soft/min_3d/0.02"], aurc)}
    res = {"split": split, "table": table, "pairs": pairs, "commit": commit, "dirty": dirty}
    (out_dir / f"ablate_{split}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def paired(a: list[dict], b: list[dict], fn) -> list[float]:
    eps = sorted({r["ep"] for r in a})
    ga, gb = ({e: [r for r in x if r["ep"] == e] for e in eps} for x in (a, b))
    rng, ds = np.random.default_rng(0), []
    for _ in range(1000):
        pick = [eps[i] for i in rng.integers(0, len(eps), len(eps))]
        ds.append(fn([r for e in pick for r in ga[e]]) - fn([r for e in pick for r in gb[e]]))
    return [round(fn(a) - fn(b), 4), round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)]


def run(split: str, settings: dict, params) -> dict:
    """settings: name -> (kind, feature, scale) or (kind, feature, scale, overrides, skipped rules)."""
    src = PerceivedSource(**OCCLUSION, relinked=True)
    eps = [(ep, src.load(ep)) for ep in src.episodes(split)]
    eps = [(ep, d, truth(d)) for ep, d in eps]
    return {name: [r for ep, d, t in eps for r in answers(make(cfg[0], cfg[1], cfg[2], params, *cfg[3:]), src, ep, d, t)]
            for name, cfg in settings.items()}


def grid(params, out_dir: Path, commit: str, dirty: bool) -> None:
    settings = {"bound": ("bound", "min_3d", 0.02)}
    settings |= {f"{k}/{f}/{s}": (k, f, s) for k in ("soft", "kb") for f, ss in GRID.items() for s in ss}
    rows = run("train", settings, params)
    table = {n: metrics(r) for n, r in rows.items()}
    choice = {f"{k}/{f}": min((n for n in table if n.startswith(f"{k}/{f}/")), key=lambda n: (table[n]["aurc"], n))
              for k in ("soft", "kb") for f in GRID}
    CHOICE.write_text(json.dumps(choice, indent=1))
    (out_dir / "grid.json").write_text(json.dumps({"table": table, "choice": choice, "commit": commit,
                                                   "dirty": dirty}, indent=1))
    print(json.dumps({"table": table, "choice": choice}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--ablate", default="", help="train | test (post-hoc, ledger READ): the attribution ablations")
    ap.add_argument("--keep-as", dest="keep_as", default="claim3_g")
    a = ap.parse_args()
    params, (commit, dirty) = load_params(), git_state()
    out_dir = ROOT / "results" / "latest" / a.keep_as
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.grid:
        grid(params, out_dir, commit, dirty)
    if a.ablate:
        ablate(a.ablate, params, out_dir, commit, dirty)
    if a.read:
        choice = json.loads(CHOICE.read_text())
        if not ledger.was_frozen(params.hash()):
            sys.exit("freeze params first (run.py --freeze)")
        ledger.record("READ", params.hash(), "G perceived test, claim 3 (claim3_g.py), " + json.dumps(choice))
        settings = {"bound": ("bound", "min_3d", 0.02)} | {n: tuple(n.split("/")[:2]) + (float(n.split("/")[2]),)
                                                           for n in choice.values()}
        rows = run("test", settings, params)
        table = {n: metrics(r) for n, r in rows.items()}
        pairs = {}
        for f in GRID:
            k, s = choice[f"kb/{f}"], choice[f"soft/{f}"]
            pairs[f"kb-soft/{f}"] = {"aurc": paired(rows[k], rows[s], aurc),
                                     "auroc": paired(rows[k], rows[s], auroc_of)}
        pairs["kb_min_q-kb_min_3d"] = {"aurc": paired(rows[choice["kb/min_q"]], rows[choice["kb/min_3d"]], aurc)}
        res = {"table": table, "pairs": pairs, "choice": choice, "commit": commit, "dirty": dirty,
               "params_hash": params.hash()}
        (out_dir / "read.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
