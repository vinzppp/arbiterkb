"""RoboMME LeRobot dataset episodes (data/robomme_lerobot/npz, written by tools/convert_robomme_lerobot.py) as
observation tables with **robot state only**: no objects (the dataset has no object poses and we run no vision
here), the TCP from forward kinematics (fk.py) and the gripper width (2 x finger position).

Ground truth for scoring is the dataset authors' subgoal text (`simple`, `grounded`), used only by evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..fk import tcp_path
from ..types import EpisodeMeta, FrameObs

ROOT = Path(__file__).resolve().parents[2] / "data" / "robomme_lerobot" / "npz"


def episode_files(task: str, n: int | None = None) -> list[Path]:
    return sorted(ROOT.glob(f"{task}_*.npz"))[:n]


def load(path: Path) -> dict:
    """{meta, tcp [T,3], width [T], demo [T], simple [T], grounded [T]}."""
    d = np.load(path)
    st = d["state"].astype(float)
    return {"meta": json.loads(str(d["meta"])), "tcp": tcp_path(st), "width": 2 * st[:, 7], "demo": d["is_demo"],
            "simple": [str(s) for s in d["simple"]], "grounded": [str(s) for s in d["grounded"]]}


def meta(ep: dict) -> EpisodeMeta:
    return EpisodeMeta("robomme_lerobot", ep["meta"]["episode"], "train", ep["meta"]["task"], [], None, None)


def frames(ep: dict, upto: int):
    """FrameObs for t < upto: no objects, hand and gripper only."""
    for t in range(upto):
        yield FrameObs(t, {}, ep["tcp"][t], float(ep["width"][t]), bool(ep["demo"][t]))


def segments(labels: list[str]) -> list[tuple[int, int, str]]:
    """[(start, end, text)] of consecutive equal subgoal labels (empty labels skipped)."""
    out, start = [], 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[start]:
            if labels[start]:
                out.append((start, t, labels[start]))
            start = t
    return out
