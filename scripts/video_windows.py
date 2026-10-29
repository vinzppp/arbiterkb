"""Video windows around the robot's own actions in RoboMME VideoRepick / VideoPlaceOrder (front videos on disk),
shared by the image tests (describe_test.py; review_test.py at commit 8d0787f used the same windows). The windows
come from robot-state actions only (actions.py), so they do not depend on the container motion detector.

  python scripts/video_windows.py   # free: writes data/robomme_lerobot/review_windows.json for tools/extract_frames.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import load_params  # noqa: E402
from arbiter_kb.sources import lerobot as L  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402

TASKS, N_EP, PER_EP, HALF, STEP = ("VideoRepick", "VideoPlaceOrder"), 10, 3, 12, 4
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


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0, 0.0]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 3), round(c - h, 3), round(c + h, 3)]


def main() -> None:
    frames: dict = {}
    for w in windows(load_params()):
        frames.setdefault(w["key"], set()).update(w["frames"])
    (DATA / "review_windows.json").write_text(json.dumps({k: sorted(v) for k, v in frames.items()}))
    print(len(frames), "episodes ->", DATA / "review_windows.json")


if __name__ == "__main__":
    main()
