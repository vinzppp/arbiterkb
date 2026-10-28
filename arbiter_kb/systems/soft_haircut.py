"""SoftHaircut(feature): R6's own formula with no KB (DEVELOPMENT §6 M2 item 6).

Start from Bound's answer (a point mass). Each near-pass seen while the cube is hidden scores
w = w_max·exp(-feature/scale) with R6's parameters (RuleParams.w_max, .d0) and moves w of the mass
across, exactly as R6's swap_mix does: after one pass on Bound's container the confidence is
1 - w_max·exp(-feature/scale). After several passes another container can end up ahead (R6 does the
same), so the answer is Bound's except in those rows. A sighting of the cube clears the passes.
No cover kernel, no R5, no 'elsewhere' floor: what ArbiterKB adds beyond this is the KB.
"""
from __future__ import annotations

import math

from ..beliefs import Distribution
from ..detectors import Detectors
from ..params import Params
from ..planner import plan_pick
from ..types import EpisodeMeta, FrameObs, PlanStep
from .baselines import TrackerSystem, obs_rows


class SoftHaircut:
    def __init__(self, params: Params | None = None, feature: str = "min_3d"):
        self.params, self.feature = params or Params(), feature
        self.name = "soft_haircut" if feature == "min_3d" else f"soft_haircut_{feature}"
        self.inner = TrackerSystem("bound", self.params)

    def reset(self, meta: EpisodeMeta) -> None:
        self.inner.reset(meta)
        self.det, self.passes, self.rows, self.t = Detectors(self.params.detect), {}, [], -1
        self.cubes = [c for c in self.inner.order if self.inner.cls[c] == "cube"]

    def step(self, frame: FrameObs) -> None:
        self.t = frame.t
        self.inner.step(frame)
        rp = self.params.rule
        evs = [e for e in self.det.step(frame) if e.kind == "near_pass"]
        for cube in self.cubes:
            if frame.objects[cube].visible:
                self.passes[cube] = []
                continue
            for e in evs:
                w = rp.w_max * math.exp(-e.data[self.feature] / rp.d0)
                self.passes.setdefault(cube, []).append((*e.args, round(w, 6), frame.t))
        ans = {c: self.believe_inside(c).argmax() for c in self.cubes}
        last = {c: [p[-1][0], p[-1][1], p[-1][3], p[-1][2]] for c, p in self.passes.items() if p}  # a, b, t, w
        self.rows.append({"t": frame.t, "pred": {c: a[0] for c, a in ans.items()},
                          "p_max": {c: round(a[1], 4) for c, a in ans.items()}, "doubt": last,
                          "obs": obs_rows(frame, self.cubes + [a[0] for a in ans.values() if a[0]])})

    def believe_inside(self, cube: str) -> Distribution:
        """Bound's answer as a point mass, then R6's swap_mix for each pass since the cube was last seen,
        in order (a pass between two containers without mass changes nothing)."""
        d = self.inner.believe_inside(cube)
        if d.argmax()[0] is None:
            return d
        for a, b, w, t in self.passes.get(cube, []):
            d = d.swap_mix(a, b, w, t, f"pass@{t}")
        return d

    def pose(self, oid: str):
        return self.inner.xyz(oid)

    def plan(self, cube: str) -> PlanStep:
        d = self.believe_inside(cube)
        return plan_pick(d, self.inner.xyz, self.params.monitor, d.support)

    def trace_rows(self) -> list[dict]:
        return self.rows
