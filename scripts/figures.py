"""Run figures (pure functions of a run's rows / traces / corruption log):
belief timeline of one corrupted episode, reliability diagram, label-jump histogram."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from run import ROOT  # noqa: E402
from arbiter_kb.metrics import risk_coverage  # noqa: E402
from arbiter_kb.plots import belief_timeline  # noqa: E402
from arbiter_kb.sources.synthetic import COVER_T  # noqa: E402


def out_dir(info: dict, run_dir: Path) -> Path:
    return ROOT / "figures" if info.get("keep") is not None else run_dir / "figures"


def timeline_figure(run_dir: Path, info: dict, base, args) -> Path | None:
    """A corrupted episode (id_swap_overlap preferred), the one where ArbiterKB was least confident."""
    cond = next((c for c in ("id_swap_overlap", "id_swap") if info["logs"].get(c)), None)
    if cond is None:
        return None
    log = info["logs"][cond]
    corrupted = [e for e, v in sorted(log.items()) if v.get("corrupted")]
    if not corrupted or ("%s_arbiterkb" % cond, corrupted[0]) not in info["traces"]:
        return None
    pm = {r["episode"]: r["p_max"] for r in info["rows"] if r["condition"] == cond and r["system"] == "arbiterkb"}
    ep = min(corrupted, key=lambda e: (pm.get(e, 1.0), e))
    gt, meta = base.ground_truth(ep), base.meta(ep)
    cube = gt.targets[0]
    bins = [o for o, c, _ in meta.objects if c == "container"]
    if cond == "id_swap":
        t_c, (a, b) = log[ep]["t"], log[ep]["ids"]
    else:
        t_c, a, b, _ = log[ep]["swaps"][0]
    title = f"{cond} episode {ep}: ids of {a} and {b} swap at t={t_c}; true container at cover {gt.cube_to_bin[cube]}"
    tr = info["traces"]
    strips = {s: tr[(f"{cond}_{s}", ep)] for s in ("bound", "frozen") if (f"{cond}_{s}", ep) in tr}
    marks = {"cover": COVER_T, "corruption": t_c, "t*": gt.t_star[cube]}
    path = out_dir(info, run_dir) / f"belief_timeline_{cond}_{args.split}_p{args.p}.png"
    return belief_timeline(tr[(f"{cond}_arbiterkb", ep)], strips, cube, gt.cube_to_bin[cube], bins, marks, path, title)


def reliability_figure(run_dir: Path, info: dict, args, n_bins: int = 10) -> Path | None:
    """Accuracy vs reported confidence p_max, one panel per condition, one line per system."""
    rows = info["rows"]
    conds = sorted({r["condition"] for r in rows}, key=lambda c: [r["condition"] for r in rows].index(c))
    if not rows:
        return None
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 4), squeeze=False)
    edges = np.linspace(0, 1, n_bins + 1)
    for ax, cond in zip(axes[0], conds):
        for s in dict.fromkeys(r["system"] for r in rows):
            sel = [r for r in rows if r["condition"] == cond and r["system"] == s]
            p = np.array([r["p_max"] for r in sel])
            c = np.array([bool(r["correct"]) for r in sel])
            k = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
            xs = [p[k == i].mean() for i in range(n_bins) if (k == i).any()]
            ys = [c[k == i].mean() for i in range(n_bins) if (k == i).any()]
            ax.plot(xs, ys, "o-", ms=3, label=s)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set(title=cond, xlabel="reported confidence p_max", ylabel="accuracy", xlim=(0, 1.02), ylim=(-0.02, 1.02))
    axes[0][0].legend(fontsize=7)
    path = out_dir(info, run_dir) / f"reliability_{args.split}_p{args.p}.png"
    return _save(fig, path)


def jump_figure(run_dir: Path, info: dict, args, params_d_ov: float | None = None) -> Path | None:
    """Histogram of how far a label moves in the frame its id swaps (id_swap_overlap)."""
    log = info["logs"].get("id_swap_overlap")
    if not log:
        return None
    jumps = [j for v in log.values() for j in v.get("label_jumps_m", [])]
    if not jumps:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(jumps, bins=20, color="tab:blue")
    ax.axvline(args.d_ov, color="r", lw=1, label=f"d_ov = {args.d_ov} m")
    ax.set(xlabel="label jump at the swap frame (m)", ylabel="count",
           title=f"id_swap_overlap: {len(jumps)} label jumps, median {np.median(jumps):.3f} m")
    ax.legend(fontsize=7)
    return _save(fig, out_dir(info, run_dir) / f"label_jumps_{args.split}_p{args.p}.png")


def risk_coverage_figure(run_dir: Path, info: dict, args) -> Path | None:
    """Risk (error rate among answered rows) vs coverage (share answered), one panel per condition.
    Lower and further right is better; the area under each curve is the AURC column."""
    rows = info["rows"]
    if not rows:
        return None
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 4), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        for s in dict.fromkeys(r["system"] for r in rows):
            pts = risk_coverage([r for r in rows if r["condition"] == cond and r["system"] == s])
            ax.step([0] + [c for c, _ in pts], [pts[0][1]] + [r for _, r in pts], where="post", label=s)
        ax.set(title=cond, xlabel="coverage (share of rows answered)", ylabel="risk (error rate when answering)",
               xlim=(0, 1.02))
    axes[0][0].legend(fontsize=7)
    return _save(fig, out_dir(info, run_dir) / f"risk_coverage_{args.split}_p{args.p}.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def all_figures(run_dir: Path, info: dict, base, args) -> list[Path]:
    figs = [timeline_figure(run_dir, info, base, args), reliability_figure(run_dir, info, args),
            jump_figure(run_dir, info, args), risk_coverage_figure(run_dir, info, args)]
    return [f for f in figs if f is not None]
