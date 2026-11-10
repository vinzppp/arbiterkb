"""Phase G on the Mac (CPU): read the pod's per-episode perception outputs and measure swaps.

  python scripts/measure_swaps.py --dir data/g_perceived      # npz files from stress/perceive.py

Writes results/g_passes.csv (one row per pass, swapped or not, with its features and swap kind),
results/g_summary.json and results/g_signature.json (id_swap_measured's parameters), both with a
provenance block (commit, args, sha256 of the inputs: E4), figures/g_features.png and figures/g_roc.png,
and prints the per-setting table and the gate decision (DEVELOPMENT §6 G).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.evaluate import write_csv  # noqa: E402
from arbiter_kb.plots import g_features  # noqa: E402
from arbiter_kb.sg_compat import wilson  # noqa: E402
from arbiter_kb.swapstats import FEATURES, KINDS, SWAP_DIRECTION, episode_passes, separability, signature  # noqa: E402
from run import git_state  # noqa: E402

GATE_RATE, GATE_AUC = 0.02, 0.98


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return {"perceived": z["perceived"], "oracle": z["oracle"], "bbox": z["bbox"], "meta": meta}


def provenance(files: list[Path], args) -> dict:
    """Where these numbers come from (E4): commit, args, and one sha256 over every input file."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + hashlib.sha256(f.read_bytes()).digest())
    commit, dirty = git_state()
    return {"commit": commit, "dirty": dirty, "args": vars(args), "n_files": len(files), "inputs_sha256": h.hexdigest()}


def per_setting(rows: list[dict], summaries: list[dict]) -> list[dict]:
    out = []
    for s in sorted({x["setting"] for x in summaries}):
        rs = [r for r in rows if r["setting"] == s]
        k, n = sum(r["swapped"] for r in rs), len(rs)
        lo, hi = wilson(k, n) if n else (np.nan, np.nan)
        acc = [x["identity_accuracy"] for x in summaries if x["setting"] == s]
        out.append({"setting": s, "episodes": len(acc), "identity_accuracy": round(float(np.nanmean(acc)), 4),
                    "passes": n, "swaps": k, "swap_rate": round(k / n, 4) if n else np.nan,
                    "swap_lo": round(max(0.0, lo), 4), "swap_hi": round(min(1.0, hi), 4),
                    "lost": sum(r["lost"] for r in rs),
                    **{f"swap_{k}": sum(r["kind"] == k for r in rs) for k in KINDS},
                    "unmatched": sum(x["n_unmatched"] for x in summaries if x["setting"] == s)})
    return out


def gate(table: list[dict], sep: dict) -> str:
    if all(not (t["swap_rate"] >= GATE_RATE) for t in table):
        return (f"FALLBACK: swaps in < {GATE_RATE:.0%} of passes in every setting. Claim 3 cannot be tested "
                "with real errors on RoboMME: switch to the plan's fallback or a harder task.")
    msg = "SWAPS EXIST: define id_swap_measured from g_passes.csv (rate per setting, feature distributions)."
    if np.isfinite(sep["best_auc"]) and sep["best_auc"] >= GATE_AUC:
        msg += (f" BUT the rule '{sep['best_rule']}' separates swapped from clean passes with AUC "
                f"{sep['best_auc']:.3f} >= {GATE_AUC}: a threshold rule solves detection. Write this in NOTES "
                "before M2; the belief layer must show value through propagation and the arbiter.")
    return msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "g_perceived"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    args = ap.parse_args()
    files = sorted(Path(args.dir).glob("*.npz"))
    if not files:
        sys.exit(f"no perception outputs in {args.dir} (provide the saved perception outputs with --dir)")
    rows, summaries = [], []
    for f in files:
        ep = load(f)
        r, s = episode_passes(ep)
        tag = {"setting": ep["meta"]["setting"], "seed": ep["meta"]["seed"], "file": f.name}
        rows += [{**tag, **x} for x in r]
        summaries.append({**tag, **s})
    table = per_setting(rows, summaries)
    sep = separability(rows) if rows else {"best_auc": np.nan, "best_rule": None}
    write_csv(Path(args.out) / "g_passes.csv", rows)
    prov = provenance(files, args)
    summary = {"settings": table, "separability": sep, "episodes": summaries, "gate": gate(table, sep),
               "provenance": prov}
    (Path(args.out) / "g_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if any(r["swapped"] for r in rows) and not all(r["swapped"] for r in rows):
        (Path(args.out) / "g_signature.json").write_text(json.dumps({**signature(rows), "provenance": prov}, indent=1))
        for p in g_features(rows, FEATURES, SWAP_DIRECTION, sep["single"], Path(args.figures)):
            print("wrote", p)
    cols = list(table[0])
    print(" | ".join(f"{c:>17}" for c in cols))
    for t in table:
        print(" | ".join(f"{t[c]!s:>17}" for c in cols))
    print("single-feature AUC:", {k: round(v, 3) for k, v in sep.get("single", {}).items()})
    print("best rule (<= 2 features):", sep["best_rule"], sep["best_auc"])
    print("GATE:", summary["gate"])


if __name__ == "__main__":
    main()
