"""Forward rules. Each rule is a pure function (kb, new_evidence, frame, params) -> list[Op]; only
KB.apply mutates (DEVELOPMENT.md §1, §3.5). M0 implements R1, R2, R6 (minimal) and wires R7.
"""
from __future__ import annotations

import numpy as np

from .beliefs import Distribution
from .kb import KB, AddEvent, FlattenDist, MixDist, SetDist, SetPose, SetVis
from .observe import obs_id
from .params import RuleParams
from .types import FrameObs


def containers(kb: KB) -> list[str]:
    return [o.id for o in kb.objects.values() if o.cls == "container"]


def r1_cover(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """A cube vanishes while containers are within r_cover of where it was (in the same frame, which
    covers RoboMME's teleport) -> Inside(cube) ∝ exp(-d²/2σ²) over those containers."""
    ops = []
    for ev in new_ev:
        if ev.kind != "disappear" or kb.objects[ev.args[0]].cls != "cube":
            continue
        cube, last = ev.args[0], np.asarray(ev.data["last_xyz"])
        cand, sup = {}, {}
        for c in containers(kb):
            o = frame.objects[c]
            if o.visible and (d := float(np.linalg.norm(o.xyz[:2] - last[:2]))) < p.r_cover:
                cand[c], sup[c] = d, [ev.id, obs_id(frame.t, c)]
        dist = Distribution.from_distances(cand, p.sigma_cover, frame.t, sup, p.eps_none, [ev.id])
        best, pm = dist.argmax()
        ops.append(SetDist("Inside", cube, dist))
        ops.append(SetVis(cube, "hidden" if cand else "unknown", pm, frame.t, [ev.id]))
        if best is not None:
            ops.append(AddEvent(frame.t, "cover", (best, cube), pm, [ev.id]))
    return ops


def r6_swap(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """Two containers passed close to each other: with probability w their tracks may have exchanged
    identity, so every HIDDEN cube with mass on either gets w of its mass moved across.
    M0: w = w_mix · (1 - closeness), closeness = min distance / threshold."""
    ops = []
    for ev in new_ev:
        if ev.kind != "near_pass":
            continue
        a, b = ev.args
        w = p.w_mix * max(0.0, 1.0 - ev.data["closeness"])
        ops.append(AddEvent(frame.t, "swap", (a, b), 1.0, [ev.id]))  # the motion itself is certain
        for (pred, cube), d in kb.relations.items():
            hidden = kb.objects[cube].visibility.value == "hidden"
            if pred == "Inside" and hidden and (d.probs.get(a, 0) > 0 or d.probs.get(b, 0) > 0) and w > 0:
                ops.append(MixDist("Inside", cube, a, b, w, frame.t, ev.id))
    return ops


def r7_decay(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """Unconfirmed beliefs drift toward uniform over containers: p <- λ^Δt·p + (1-λ^Δt)·u.
    Rules run every frame, so Δt = frames since this distribution last changed."""
    if p.lam >= 1.0:
        return []
    bins = containers(kb)
    return [FlattenDist("Inside", cube, p.lam ** max(1, frame.t - d.t), bins, frame.t)
            for (pred, cube), d in kb.relations.items() if pred == "Inside" and bins]


def r2_follow(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """Hidden cube display pose = probability-weighted container pose (decisions use the distribution)."""
    ops = []
    for (pred, cube), d in kb.relations.items():
        if pred != "Inside" or kb.objects[cube].visibility.value != "hidden":
            continue
        pts = [(q, kb.objects[c].pose.value) for c, q in d.probs.items() if c is not None]
        pts = [(q, x) for q, x in pts if x is not None]
        if pts and d.support:
            qs = np.array([q for q, _ in pts])
            xyz = (qs[:, None] * np.stack([x for _, x in pts])).sum(0) / qs.sum()
            ops.append(SetPose(cube, xyz, d.max_prob(), frame.t, "inferred", list(d.support)))
    return ops


RULES = [("R1", r1_cover), ("R6", r6_swap), ("R7", r7_decay), ("R2", r2_follow)]
