"""Forward rules. Each rule is a pure function (kb, new_evidence, frame, params) -> list[Op]; only
KB.apply mutates (DEVELOPMENT.md §1, §3.5). M0: R1, R2, R6. M1: R5 (frozen since v1.6: it never fires on
stock layouts), R6 with 3D distance. R7 (decay) was removed in v1.6: it was never switched on.
M3: R3 grasp / place (gripper evidence), and R2 also moves a held object with the end effector.
"""
from __future__ import annotations

import numpy as np

from .beliefs import Distribution, joint_exclusive
from .kb import KB, AddEvent, AddEvidence, MixDist, SetDist, SetPose, SetVis
from .observe import obs_id
from .params import RuleParams
from .types import FrameObs


def containers(kb: KB) -> list[str]:
    return [o.id for o in kb.objects.values() if o.cls == "container"]


def r1_cover(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """A cube vanishes while containers are within r_cover of where it was (in the same frame, which
    covers RoboMME's teleport) -> Inside(cube) ∝ exp(-d²/2σ²) over those containers.
    No container within r_cover (e.g. a noisy last sighting): the cover was not seen, so the nearest
    visible container is the only candidate and 'elsewhere' gets eps_far = 0.5 (even odds; the nearest is
    listed first, so it stays the answer). Only with no visible container at all is the belief {None: 1}
    and the cube 'unknown'. Before, any miss was a confident {None: 1} (M2 infra finding)."""
    ops = []
    for ev in new_ev:
        if ev.kind != "disappear" or kb.objects[ev.args[0]].cls != "cube" \
                or ev.args[0] in held(kb):  # hidden by the fingers, not covered
            continue
        cube, last = ev.args[0], np.asarray(ev.data["last_xyz"])
        seen = {c: float(np.linalg.norm(frame.objects[c].xyz[:2] - last[:2])) for c in containers(kb)
                if frame.objects[c].visible}
        cand = {c: d for c, d in seen.items() if d < p.r_cover}
        eps = p.eps_none if cand else p.eps_far
        cand = cand or dict([min(seen.items(), key=lambda kv: kv[1])] if seen else [])
        sup = {c: [ev.id, obs_id(frame.t, c)] for c in cand}
        dist = Distribution.from_distances(cand, p.sigma_cover, frame.t, sup, eps, [ev.id])
        best, pm = dist.argmax()
        ops.append(SetDist("Inside", cube, dist))
        ops.append(SetVis(cube, "hidden" if cand else "unknown", pm, frame.t, [ev.id]))
        if best is not None:
            ops.append(AddEvent(frame.t, "cover", (best, cube), pm, [ev.id]))
    return ops


def r3_grasp(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """grip_close: the object nearest the end effector, within r_grasp, is held (event `grasp`).
    grip_open: each held object is released (event `place`, onto the object below it within r_on, else
    the table); it is `visible` again if seen, else `unknown` until the next sighting.
    disappear of a held object: it is `held` (hidden by the fingers), not visible (M3a review)."""
    ops = []
    for ev in new_ev:
        if ev.kind == "disappear" and ev.args[0] in held(kb):
            ops.append(SetVis(ev.args[0], "held", 1.0, frame.t, [ev.id]))
        ee = np.asarray(ev.data.get("ee", frame.ee_xyz), float)
        if ev.kind == "grip_close":
            near = [(float(np.linalg.norm(o.pose.value - ee)), o.id) for o in kb.objects.values()
                    if o.pose.value is not None and o.visibility.value != "hidden"]
            near = [dn for dn in near if dn[0] < p.r_grasp]
            if near:
                obj = min(near)[1]
                ops += [SetVis(obj, "held", 1.0, frame.t, [ev.id]), AddEvent(frame.t, "grasp", (obj,), 1.0, [ev.id])]
        elif ev.kind == "grip_open":
            for o in [kb.objects[k] for k in held(kb)]:
                vis = frame.objects[o.id].visible
                ops += [SetVis(o.id, "visible" if vis else "unknown", 1.0, frame.t, [ev.id]),
                        AddEvent(frame.t, "place", (o.id, below(kb, o.id, ee, p) or "table"), 1.0, [ev.id])]
    return ops


def held(kb: KB) -> list[str]:
    """Objects grasped and not placed since (from the event log; observe() may mark them visible)."""
    out: dict[str, bool] = {}
    for e in kb.events:
        if e.type in ("grasp", "place"):
            out[e.args[0]] = e.type == "grasp"
    return [k for k, v in out.items() if v]


def below(kb: KB, obj: str, xyz: np.ndarray, p: RuleParams) -> str | None:
    """The object right under xyz (xy within r_on, lower centre), the nearest one in xy. Seen objects only:
    a hidden cube's pose is its container's (R2), so a container put back would land "on" its own cube."""
    cands = [(float(np.linalg.norm(o.pose.value[:2] - xyz[:2])), o.id) for o in kb.objects.values()
             if o.id != obj and o.pose.value is not None and o.pose.value[2] < xyz[2]
             and o.visibility.value in ("visible", "held")]
    cands = [c for c in cands if c[0] < p.r_on]
    return min(cands)[1] if cands else None


def r6_swap(kb: KB, new_ev: list, frame: FrameObs, p: RuleParams) -> list:
    """Two containers passed close to each other: with probability w their tracks may have exchanged
    identity, so every HIDDEN cube with mass on either gets w of its mass moved across.
    M1: w = w_max·exp(-d/d0), d = the pass's closest 3D centre distance (the detector always records it).
    With r6_feature = "min_q": the pass's lowest track_q (the re-linking's margin), scale q0."""
    ops = []
    for ev in new_ev:
        if ev.kind != "near_pass":
            continue
        a, b = ev.args
        scale = p.q0 if p.r6_feature == "min_q" else p.d0
        w = p.w_max * float(np.exp(-ev.data[p.r6_feature] / scale))
        if w > 0:  # the motion is certain; the event marks a pass that may have exchanged identities
            ops.append(AddEvent(frame.t, "swap", (a, b), 1.0, [ev.id]))
        for (pred, cube), d in kb.relations.items():
            hidden = kb.objects[cube].visibility.value == "hidden"
            if pred == "Inside" and hidden and (d.probs.get(a, 0) > 0 or d.probs.get(b, 0) > 0) and w > 0:
                ops.append(MixDist("Inside", cube, a, b, w, frame.t, ev.id))
    return ops


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
    """Hidden cube display pose = probability-weighted container pose (decisions use the distribution).
    A held object that is not seen follows the end effector."""
    ops = [SetPose(o.id, np.asarray(frame.ee_xyz, float), o.visibility.conf, frame.t, "inferred",
                   list(o.visibility.support)) for o in kb.objects.values()
           if o.id in held(kb) and not frame.objects[o.id].visible]
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


RULES = [("R3", r3_grasp), ("R1", r1_cover), ("R6", r6_swap), ("R5", r5_exclusive), ("R2", r2_follow)]
