"""ArbiterKB vs each baseline (DEVELOPMENT §6 M2 (c)), from two finished runs of scripts/run.py:

  python scripts/compare.py --calib results/runs/<train run> --eval results/runs/<eval run> \
         [--ref arbiterkb] [--base soft_haircut,overlap_haircut,bound] [--keep-as NAME]

--calib only sets each baseline's matched-coverage threshold θ (clean rows); it must be a train run.
--eval gives everything else: AURC with episode-bootstrap CIs, the paired AURC difference, and
coverage / silent rate (Wilson) with the reference at τ_hi and the baseline at θ.
Writes compare.csv + compare.json into the eval run dir (and results/latest/NAME with --keep-as).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb import compare as C  # noqa: E402
from arbiter_kb.params import Params  # noqa: E402


def matched(calib: list[dict], ref: str, base: str, tau: float) -> dict:
    target = C.coverage(C.select(calib, "none", ref), tau)
    clean = C.select(calib, "none", base)
    theta = C.threshold_for_coverage(clean, target)
    return {"target_cov": round(target, 4), "theta": theta, "calib_cov": round(C.coverage(clean, theta), 4)}


def one_row(rows: list[dict], cond: str, ref: str, base: str, tau: float, m: dict) -> dict:
    r, b = C.select(rows, cond, ref), C.select(rows, cond, base)
    ra, ba, d = C.bootstrap_aurc(r), C.bootstrap_aurc(b), C.paired_aurc_diff(r, b)
    rs, bs = C.silent_at(r, tau), C.silent_at(b, m["theta"])
    out = {"condition": cond, "ref": ref, "base": base, "n": len(r),
           "ref_aurc": ra[0], "ref_aurc_lo": ra[1], "ref_aurc_hi": ra[2],
           "base_aurc": ba[0], "base_aurc_lo": ba[1], "base_aurc_hi": ba[2],
           "diff_aurc": d[0], "diff_lo": d[1], "diff_hi": d[2], **m,
           "ref_cov": C.coverage(r, tau), "ref_silent": rs[0], "ref_silent_lo": rs[1], "ref_silent_hi": rs[2],
           "base_cov": C.coverage(b, m["theta"]), "base_silent": bs[0], "base_silent_lo": bs[1],
           "base_silent_hi": bs[2], "same_answers": sum(x["pred"] == y["pred"] for x, y in zip(
               sorted(r, key=_key), sorted(b, key=_key)))}
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def _key(r: dict) -> tuple:
    return r["episode"], r["cube"]


def table(calib_dir: Path, eval_dir: Path, ref: str, bases: list[str]) -> tuple[list[dict], dict]:
    calib, rows = C.load_rows(calib_dir), C.load_rows(eval_dir)
    cmeta, emeta = (json.loads((d / "meta.json").read_text()) for d in (calib_dir, eval_dir))
    if cmeta["args"]["split"] != "train":
        sys.exit(f"--calib must be a train run; {calib_dir} has split={cmeta['args']['split']}")
    tau = Params.load(eval_dir / "params.json").monitor.tau_hi
    out = []
    for base in bases:
        m = matched(calib, ref, base, tau)
        out += [one_row(rows, c, ref, base, tau, m) for c in sorted({r["condition"] for r in rows})]
    meta = {"calib": str(calib_dir.relative_to(ROOT)), "eval": str(eval_dir.relative_to(ROOT)),
            "calib_seeds": [min(r["episode"] for r in calib), max(r["episode"] for r in calib)],
            "eval_seeds": [min(r["episode"] for r in rows), max(r["episode"] for r in rows)],
            "eval_split": emeta["args"]["split"], "eval_first": emeta["args"].get("first", 0),
            "eval_n": emeta["args"]["n"], "params_hash": [cmeta["params_hash"], emeta["params_hash"]],
            "commit": [cmeta["commit"], emeta["commit"]], "dirty": [cmeta["dirty"], emeta["dirty"]],
            "tau_hi": tau, "n_boot": C.N_BOOT, "boot_seed": C.SEED, "resample": "episodes"}
    return out, meta


def write(out: list[dict], meta: dict, dirs: list[Path]) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "compare.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        (d / "compare.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def show(out: list[dict]) -> None:
    cols = ["condition", "base", "n", "ref_aurc", "base_aurc", "diff_aurc", "diff_lo", "diff_hi", "target_cov",
            "calib_cov", "ref_cov", "ref_silent", "base_cov", "base_silent", "same_answers"]
    print(" | ".join(f"{c:>11}" for c in cols))
    for r in out:
        print(" | ".join(f"{r[c]!s:>11}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--ref", default="arbiterkb")
    ap.add_argument("--base", default="soft_haircut,overlap_haircut,bound")
    ap.add_argument("--keep-as", dest="keep_as", default="")
    a = ap.parse_args()
    calib, ev = a.calib.resolve(), a.eval.resolve()
    out, meta = table(calib, ev, a.ref, a.base.split(","))
    write(out, meta, [ev] + ([ROOT / "results" / "latest" / a.keep_as] if a.keep_as else []))
    show(out)
    print("wrote", ev.relative_to(ROOT) / "compare.csv")


if __name__ == "__main__":
    main()
