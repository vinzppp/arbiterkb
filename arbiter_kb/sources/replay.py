"""RoboMME replays (cached by scene-graph-robot) as FrameObs, via the simulator-truth oracle graph.

Layout: $ARBITER_REPLAY_DIR/{train,test}/ep<NNNN>_replay.npz (+ test/motion.json).
Copy them from the pod with scripts/copy_replays.sh. With no files, episodes() is empty and the
caller prints SKIPPED.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from .. import sg_compat as sg
from ..types import EpisodeMeta, FrameObs, GroundTruth, ObjObs

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "replays"


def replay_dir() -> Path:
    return Path(os.environ.get("ARBITER_REPLAY_DIR", DEFAULT_DIR))


class OracleReplaySource:
    name = "oracle"

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else replay_dir()
        self._cache: dict[int, tuple] = {}

    def episodes(self, split: str, n: int | None = None) -> list[int]:
        files = sorted((self.root / split).glob("ep*_replay.npz"))
        eps = [int(re.search(r"ep(\d+)_replay", f.name).group(1)) + (1000 if split == "test" else 0) for f in files]
        return eps[:n] if n else eps

    def path(self, ep: int) -> Path:
        split, idx = ("test", ep - 1000) if ep >= 1000 else ("train", ep)
        return self.root / split / f"ep{idx:04d}_replay.npz"

    def _load(self, ep: int):
        if ep not in self._cache:
            rep = sg.load_replay(self.path(ep))
            graph, full = sg.oracle_graph(rep)
            names = {}
            for o in rep.scene["objects"]:
                names[o["node"]] = f"bin_{o['node']}" if o["label_id"] == 0 else f"cube_{o['color']}"
            self._cache = {ep: (rep, graph, full, names)}  # keep one episode in memory (~0.5 GB)
        return self._cache[ep]

    def meta(self, ep: int) -> EpisodeMeta:
        rep, _, _, names = self._load(ep)
        objs = [(names[o["node"]], "container" if o["label_id"] == 0 else "cube",
                 {"color": o["color"]} if o.get("color") else {}) for o in rep.scene["objects"]]
        return EpisodeMeta("oracle", ep, "test" if ep >= 1000 else "train", rep.task_goal, objs,
                           rep.K_front, rep.E_front)

    def frames(self, ep: int):
        rep, g, _, names = self._load(ep)
        for t in range(len(g)):
            objs = {}
            for n, oid in names.items():
                row = g[t, n]
                vis = row[sg.F_HIDDEN] == 0 and np.all(np.isfinite(row[sg.F_X:sg.F_Z + 1]))
                cls = "container" if row[sg.F_LABEL] == 0 else "cube"
                objs[oid] = ObjObs(oid, cls, bool(vis), row[sg.F_X:sg.F_Z + 1].astype(float) if vis else None,
                                   row[sg.F_U:sg.F_V + 1].astype(float) if vis else None, 1.0)
            yield FrameObs(t, objs, rep.ee_xyz[t].astype(float), float(rep.gripper_state[t].sum()),
                           bool(rep.is_video[t]))

    def ground_truth(self, ep: int) -> GroundTruth:
        rep, g, full, names = self._load(ep)
        targets = sg.target_cube_nodes(rep.scene, rep.task_goal)
        active = sg.active_targets(rep, targets)
        t0 = sg.decision_frame(rep)
        t_star = {names[targets[0]]: t0}
        for tg in targets[1:]:  # later targets: the first frame they become active
            idx = np.nonzero(active == tg)[0]
            if len(idx):
                t_star[names[tg]] = int(idx[0])
        c2b = {names[c]: names[b] for c, b in rep.scene["cube_to_bin"].items()}
        node = {v: k for k, v in names.items()}
        bxyz = {c: full[t, node[c2b[c]]].astype(float) for c, t in t_star.items()}
        moved = self._moved(ep, c2b, full, names, t_star)
        return GroundTruth([names[t] for t in targets if names[t] in t_star], c2b, t_star, bxyz, moved)

    def _moved(self, ep, c2b, full, names, t_star) -> dict[str, bool]:
        mpath = self.root / "test" / "motion.json"
        if ep >= 1000 and mpath.exists():
            m = json.loads(mpath.read_text()).get(str(ep - 1000), {})
            if "moved" in m:
                return {c: bool(m["moved"]) for c in t_star}
        node = {v: k for k, v in names.items()}
        out = {}
        for c, t in t_star.items():
            xy = full[:, node[c2b[c]], :2]
            first = next((i for i in range(len(xy)) if np.all(np.isfinite(xy[i]))), t)
            out[c] = bool(np.linalg.norm(xy[t] - xy[first]) > 0.03)
        return out
