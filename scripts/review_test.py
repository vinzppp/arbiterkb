"""The review call (arbiter_kb/review.py) on real RoboMME video (NOTES "Review call on real video": the rule was fixed
before the run). Per window: the KB from robot state up to the window's end, plus ONE planted false event; the VLM
checks the window's events against 7 frames. Scored: true events confirmed, planted ones falsely confirmed.

  python scripts/review_test.py --plan            # free: writes data/robomme_lerobot/review_windows.json
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -I tools/extract_frames.py \
      --windows data/robomme_lerobot/review_windows.json
  python scripts/review_test.py --call [--keep-as review_video]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import git_state, load_params  # noqa: E402
from arbiter_kb.kb import AddEvidence  # noqa: E402
from arbiter_kb.llm import vlm_client  # noqa: E402
from arbiter_kb.review import review  # noqa: E402
from arbiter_kb.sources import lerobot as L  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402

TASKS, N_EP, PER_EP, HALF, STEP, PLANT_DT = ("VideoRepick", "VideoPlaceOrder"), 10, 3, 12, 4, 8
DATA = ROOT / "data" / "robomme_lerobot"
ACTS = ("pick_up", "release")


def kb_until(ep: dict, t: int, params):
    s = ArbiterKBSystem(params)
    s.reset(L.meta(ep))
    for f in L.frames(ep, t + 1):
        s.step(f)
    return s.kb


def windows(params) -> list[dict]:
    """[{key, t, kind, frames}]: up to PER_EP actions per episode (seed 0) with a full window inside the episode."""
    rng, out = random.Random(0), []
    for task in TASKS:
        for path in L.episode_files(task, N_EP):
            ep = L.load(path)
            n = len(ep["tcp"])
            acts = [e for e in kb_until(ep, n - 1, params).evidence if e.kind in ACTS and HALF <= e.t < n - HALF]
            for e in sorted(rng.sample(acts, min(PER_EP, len(acts))), key=lambda e: e.t):
                out.append({"key": f"{task}_{ep['meta']['episode']:04d}", "path": str(path), "t": e.t, "kind": e.kind,
                            "frames": list(range(e.t - HALF, e.t + HALF + 1, STEP))})
    return out


def plant(kb, ep: dict, w: dict) -> str:
    """One false event at t + PLANT_DT: a release after a pick-up, or a pick-up after a release."""
    t = w["t"] + PLANT_DT
    ee = [round(float(v), 4) for v in ep["tcp"][t]]
    data = {"ee": ee, "how": "placed", "obs": [f"o{t}:ee"]} if w["kind"] == "pick_up" else {"ee": ee, "obs": [f"o{t}:ee"]}
    return kb.apply([AddEvidence(t, "release" if w["kind"] == "pick_up" else "pick_up", (), data)], "planted")[0]


def one(w: dict, llm, params) -> dict:
    ep = L.load(Path(w["path"]))
    kb = kb_until(ep, w["frames"][-1], params)
    fake = plant(kb, ep, w)
    d = np.load(DATA / "review_frames" / f"{w['key']}.npz")
    rgb = dict(zip(d["idx"].tolist(), d["rgb"]))
    rep = review(kb, llm, [(t, rgb[t]) for t in w["frames"]])
    verdict = {e.args[0]: e.data["verdict"] for e in kb.evidence if e.kind == "vlm_check"}
    true_ids = [e.id for e in kb.evidence if e.kind in ACTS and w["frames"][0] <= e.t <= w["frames"][-1] and e.id != fake]
    summ = next((e.data["text"] for e in kb.evidence if e.kind == "vlm_summary"), None)
    label = sorted({ep["simple"][t] for t in w["frames"]})
    return {"key": w["key"], "t": w["t"], "kind": w["kind"], "true": [verdict.get(i, "none") for i in true_ids],
            "planted": verdict.get(fake, "none"), "missing": sum(e.kind == "vlm_missing" for e in kb.evidence),
            "problems": len(rep["problems"]), "summary": summ, "labels": label}


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0, 0.0]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 3), round(c - h, 3), round(c + h, 3)]


def report(rows: list[dict]) -> dict:
    out = {}
    for name, vals in (("true", [v for r in rows for v in r["true"]]), ("planted", [r["planted"] for r in rows])):
        c = collections.Counter(vals)
        out[name] = {k: [c[k], len(vals), wilson(c[k], len(vals))] for k in ("confirm", "contradict", "cannot_tell", "none")}
    t, p = out["true"]["confirm"][2][0], out["planted"]["confirm"][2][0]
    out["usable"] = {"true_confirm": t, "planted_false_confirm": p, "rule": "planted <= 0.10 and true >= 0.80",
                     "pass": p <= 0.10 and t >= 0.80}
    out["missing_items"], out["problems"] = sum(r["missing"] for r in rows), sum(r["problems"] for r in rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--call", action="store_true", help="required to spend money (about 60 calls, cap $0.50)")
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    params, (commit, dirty) = load_params(), git_state()
    ws = windows(params)
    if a.plan:
        frames: dict = {}
        for w in ws:
            frames.setdefault(w["key"], set()).update(w["frames"])
        (DATA / "review_windows.json").write_text(json.dumps({k: sorted(v) for k, v in frames.items()}))
        print(len(ws), "windows in", len(frames), "episodes ->", DATA / "review_windows.json")
        return
    if not a.call:
        sys.exit("pass --plan (free) or --call")
    llm = vlm_client(params.arbiter, max_usd=0.5)
    rows = [one(w, llm, params) for w in ws]
    for r in rows:
        print(json.dumps(r), flush=True)
    out = {"report": report(rows), "rows": rows, "usage": llm.usage,
           "meta": {"commit": commit, "dirty": dirty, "params_hash": params.hash(), "model": params.arbiter.vlm_model}}
    print(json.dumps(out["report"], indent=1), json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                                           for k, v in llm.usage.items()}))
    if a.keep_as:
        d = ROOT / "results" / "latest" / a.keep_as
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
