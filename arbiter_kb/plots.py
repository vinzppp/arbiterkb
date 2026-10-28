"""Figures. Every plot is a pure function of traces / rows and saves to figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .debug import index, timeline  # noqa: E402


def belief_timeline(arb_rows: list[dict], baselines: dict[str, list[dict]], cube: str, truth: str,
                    bins: list[str], marks: dict[str, int], path: Path, title: str = "") -> Path:
    """Stacked area of ArbiterKB's Inside(cube) over time, with the baselines' one-hot answers as
    strips below. marks: label -> frame (cover, corruption, t*)."""
    tl = timeline(arb_rows, cube)
    ts = np.array([t for t, _ in tl])
    keys = bins + ["None"]
    P = np.array([[p.get(k, 0.0) for k in keys] for _, p in tl])
    colors = plt.cm.tab10(np.arange(len(keys)))
    fig, axes = plt.subplots(1 + len(baselines), 1, figsize=(9, 3 + 0.6 * len(baselines)), sharex=True,
                             gridspec_kw={"height_ratios": [5] + [1] * len(baselines)})
    axes = np.atleast_1d(axes)
    axes[0].stackplot(ts, P.T, labels=[f"{k}{' (true at cover)' if k == truth else ''}" for k in keys], colors=colors)
    axes[0].set_ylabel(f"ArbiterKB\nP(Inside {cube})")
    axes[0].legend(loc="upper left", fontsize=7, ncol=len(keys))
    ev, _, _ = index(arb_rows)
    for eid, (t, a) in ev.items():
        if a["kind"] == "near_pass":
            axes[0].axvline(t, color="k", lw=0.6, ls=":")
    for ax, (name, rows) in zip(axes[1:], baselines.items()):
        pred = [r["pred"].get(cube) for r in rows]
        idx = [keys.index(str(p)) if str(p) in keys else len(keys) - 1 for p in pred]
        ax.scatter([r["t"] for r in rows], np.zeros(len(rows)), c=colors[idx], marker="s", s=8)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    for label, t in marks.items():
        for ax in axes:
            ax.axvline(t, color="r", lw=1)
        axes[0].text(t, 1.02, label, color="r", fontsize=7, ha="center")
    axes[0].set_title(title, fontsize=8)
    axes[-1].set_xlabel("frame")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
