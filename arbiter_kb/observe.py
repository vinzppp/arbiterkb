"""Observations -> L0 poses and L1 visibility (DEVELOPMENT.md §3.4; the full state machine is M1).

Observation ids look like 'o<t>:<object id>'; the trace stores every observation, so any support
chain can end at the raw numbers.
"""
from __future__ import annotations

from .kb import SetPose, SetVis
from .types import FrameObs


def obs_id(t: int, oid: str) -> str:
    return f"o{t}:{oid}"


def observe(frame: FrameObs) -> list:
    ops = []
    for oid, o in frame.objects.items():
        if o.visible:
            sup = [obs_id(frame.t, oid)]
            ops += [SetPose(oid, o.xyz, o.track_q, frame.t, "observed", sup),
                    SetVis(oid, "visible", o.track_q, frame.t, sup)]
    return ops
