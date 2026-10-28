"""Forward rules. Each rule is a pure function (kb, new_evidence, frame, params) -> list[Op]; only
KB.apply mutates (DEVELOPMENT.md §1, §3.5). M0: R1, R2, R6, R7 (wired). M1: R5, R6 with 3D distance.
"""
from __future__ import annotations

import numpy as np

from .beliefs import Distribution, joint_exclusive
from .kb import KB, AddEvent, AddEvidence, FlattenDist, MixDist, SetDist, SetPose, SetVis
from .observe import obs_id
from .params import RuleParams
from .types import FrameObs


def containers(kb: KB) -> list[str]:
    return [o.id for o in kb.objects.values() if o.cls == "container"]


def r1_cover(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """A cube vanishes while containers are within r_cover of where it was (in the same frame, which
    covers RoboMME's teleport) -> Inside(cube) ∝ exp(-d²/2σ²) over those containers.
    No container within r_cover (e.g. a noisy last sighting): the cover was not seen, so every visible
    container is a candidate and 'elsewhere' gets eps_far (no hypothesis above even odds). Only with no
    visible container at all is the belief {None: 1} and the cube 'unknown' (M2 infra finding)."""
    ops = []
    for ev in new_ev:
        if ev.kind != "disappear" or kb.objects[ev.args[0]].cls != "cube":
            continue
        cube, last = ev.args[0], np.asarray(ev.data["last_xyz"])
        seen = {c: float(np.linalg.norm(frame.objects[c].xyz[:2] - last[:2])) for c in containers(kb)
                if frame.objects[c].visible}
        cand = {c: d for c, d in seen.items() if d < p.r_cover}
        eps = p.eps_none if cand else p.eps_far
        cand = cand or seen
        sup = {c: [ev.id, obs_id(frame.t, c)] for c in cand}
        dist = Distribution.from_distances(cand, p.sigma_cover, frame.t, sup, eps, [ev.id])
        best, pm = dist.argmax()
        ops.append(SetDist("Inside", cube, dist))
        ops.append(SetVis(cube, "hidden" if cand else "unknown", pm, frame.t, [ev.id]))
        if best is not None:
            ops.append(AddEvent(frame.t, "cover", (best, cube), pm, [ev.id]))
    return ops


def r6_swap(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """Two containers passed close to each other: with probability w their tracks may have exchanged
    identity, so every HIDDEN cube with mass on either gets w of its mass moved across.
    M1: w = w_max·exp(-d/d0), d = the pass's closest 3D centre distance (the detector always records it)."""
    ops = []
    for ev in new_ev:
        if ev.kind != "near_pass":
            continue
        a, b = ev.args
        w = p.w_max * float(np.exp(-ev.data["min_3d"] / p.d0))
        if w > 0:  # the motion is certain; the event marks a pass that may have exchanged identities
            ops.append(AddEvent(frame.t, "swap", (a, b), 1.0, [ev.id]))
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
    return [FlattenDist("Inside", cube, p.lam ** (frame.t - d.t), bins, frame.t)
            for (pred, cube), d in kb.relations.items() if pred == "Inside" and bins and frame.t > d.t]


def r5_exclusive(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """At most one cube per container, among the cubes covered IN THIS FRAME: their R1 beliefs are
    independent evidence, so the joint over them is exact. Cubes covered earlier are left out: their
    beliefs may already be R5 or R6 outputs (correlated), and recombining them would count the constraint
    twice (M1.1 code review). Covers on different frames are therefore not linked (DEVELOPMENT §3.5).
    No consistent assignment -> an `exclusive_conflict` evidence (for M3's R8), beliefs unchanged."""
    now = {e.args[0] for e in new_ev if e.kind == "disappear" and kb.objects[e.args[0]].cls == "cube"}
    fresh = {cube: d for (pred, cube), d in kb.relations.items()
             if pred == "Inside" and cube in now and kb.objects[cube].visibility.value == "hidden"}
    if len(fresh) < 2:
        return []
    out = joint_exclusive(fresh, frame.t)
    if out is None:
        sup = sorted({i for d in fresh.values() for i in d.support})
        return [AddEvidence(frame.t, "exclusive_conflict", tuple(sorted(fresh)), {"obs": [], "support": sup})]
    return [SetDist("Inside", c, out[c]) for c in fresh
            if any(abs(out[c].probs.get(k, 0) - fresh[c].probs.get(k, 0)) > 1e-9 for k in set(out[c].probs) | set(fresh[c].probs))]


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


RULES = [("R1", r1_cover), ("R6", r6_swap), ("R7", r7_decay), ("R5", r5_exclusive), ("R2", r2_follow)]
