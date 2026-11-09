"""Annotated current frame (Set-of-Mark, DEVELOPMENT §2.2 step 7): the current RGB with each visible object's id
drawn at its pixel position, and on each container the hidden cubes the KB places there with their
probability ("red? 0.55"). It ties the ids in the KB text to the objects in the picture, for the keyframe call and
the controller. Pixel positions come from the current observation table (FrameObs.uv); the drawing uses
matplotlib's Agg backend (no window). Each label goes above its object if that spot is free, else below,
right or left (greedy, `place`), so labels neither cover the object nor each other.
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


CHAR_W, LINE_H, GAP = 3.7, 8.0, 6.0  # px per character / line at fontsize 6, dpi 100; distance from the dot


def place(u: float, v: float, label: str, taken: list, w: int, h: int) -> tuple[float, float]:
    """Top-left (x, y) of a label box near (u, v) that overlaps no earlier box and stays in the image: above,
    below, right, left, then the same at 2x and 3x the distance (the caller draws a leader line)."""
    lw = CHAR_W * max(len(x) for x in label.split("\n")) + 4
    lh = LINE_H * (label.count("\n") + 1) + 3
    for k in (1, 2, 3, 4):
        g = GAP * k + (k - 1) * lh / 2
        for x0, y0 in ((u - lw / 2, v - g - lh), (u - lw / 2, v + g), (u + g, v - lh / 2), (u - g - lw, v - lh / 2)):
            box = (x0, y0, x0 + lw, y0 + lh)
            inside = box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h
            if inside and not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in taken):
                taken.append(box)
                return x0, y0
    taken.append((u - lw / 2, v - GAP - lh, u + lw / 2, v - GAP))
    return u - lw / 2, v - GAP - lh  # no free spot: above (may overlap)


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
    hyp, taken = hypotheses(kb), []
    for oid, o in frame.objects.items():
        if not o.visible or o.uv is None:
            continue
        u, v = float(o.uv[0]), float(o.uv[1])
        label = "\n".join([oid] + hyp.get(oid, []))
        ax.plot([u], [v], marker="o", markersize=3, color="yellow")
        x, y = place(u, v, label, taken, w, h)
        b = taken[-1]
        ax.plot([u, min(max(u, b[0]), b[2])], [v, min(max(v, b[1]), b[3])], color="yellow", linewidth=0.6)
        ax.text(x + 2, y + 1.5, label, fontsize=6, color="black", ha="left", va="top",
                bbox={"boxstyle": "round,pad=0.15", "fc": "yellow", "ec": "none", "alpha": 0.8})
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return out
