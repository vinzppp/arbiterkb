"""Annotated current frame (Set-of-Mark, DEVELOPMENT §2.2 step 7): the current RGB with each visible object's id
drawn at its pixel position, and on each container the hidden cubes the KB places there with their
probability ("red? 0.55"). It ties the ids in the KB text to the objects in the picture, for the annotator and
the controller. Pixel positions come from the current observation table (FrameObs.uv); the drawing uses
matplotlib's Agg backend (no window), and labels sit just above each object so they do not cover it.
"""
from __future__ import annotations

import numpy as np

from .kb import KB
from .types import FrameObs


def hypotheses(kb: KB, min_p: float = 0.1) -> dict[str, list[str]]:
    """{container: ["red? 0.55", ...]} from the Inside distributions of hidden cubes."""
    out: dict[str, list[str]] = {}
    for (pred, cube), d in kb.relations.items():
        if pred != "Inside" or kb.objects[cube].visibility.value != "hidden":
            continue
        name = kb.objects[cube].attributes.get("color", cube)
        for c, p in d.probs.items():
            if c is not None and p >= min_p:
                out.setdefault(c, []).append(f"{name}? {p:.2f}")
    return out


def som_image(rgb: np.ndarray, frame: FrameObs, kb: KB) -> np.ndarray:
    """[H, W, 3] uint8, the same size as rgb."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(rgb)
    ax.set_axis_off()
    hyp = hypotheses(kb)
    for oid, o in frame.objects.items():
        if not o.visible or o.uv is None:
            continue
        u, v = float(o.uv[0]), float(o.uv[1])
        label = "\n".join([oid] + hyp.get(oid, []))
        ax.plot([u], [v], marker="o", markersize=3, color="yellow")
        ax.text(u, max(v - 6, 0), label, fontsize=6, color="black", ha="center", va="bottom",
                bbox={"boxstyle": "round,pad=0.15", "fc": "yellow", "ec": "none", "alpha": 0.8})
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return out
