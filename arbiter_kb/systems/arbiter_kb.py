"""ArbiterKB system: the fast loop of DEVELOPMENT.md §3.1 around the KB.

Per frame, in fixed order: observe -> detect (evidence) -> rules R1, R6, R7, R2 -> [monitor: M3]
-> trigger -> arbiter backend (M0: RuleOnly, never changes anything). Every change is logged.
"""
from __future__ import annotations

from ..arbiter import RuleOnlyBackend, Trigger
from ..beliefs import Distribution
from ..detectors import Detectors
from ..kb import KB
from ..observe import observe
from ..params import Params
from ..planner import plan_pick
from ..rules import RULES
from ..types import EpisodeMeta, FrameObs, PlanStep
from ..views import text_view


def _ev_line(e) -> str:
    return f"{e.id} t={e.t} {e.kind}({', '.join(e.args)})"


class ArbiterKBSystem:
    def __init__(self, params: Params | None = None, backend=None, name: str = "arbiterkb"):
        self.name, self.params = name, params or Params()
        self.backend = backend or RuleOnlyBackend()

    def reset(self, meta: EpisodeMeta) -> None:
        self.kb = KB(meta.objects)
        self.kb.task["goal"] = meta.task_goal  # L4
        self.det = Detectors(self.params.detect)
        self.trigger = Trigger(self.params.arbiter)
        self.rows: list[dict] = []
        self.t = -1

    def step(self, frame: FrameObs) -> None:
        kb, rp, self.t = self.kb, self.params.rule, frame.t
        kb.apply(observe(frame), "observe")
        n0 = len(kb.evidence)
        kb.apply(self.det.step(frame), "detect")
        new_ev = kb.evidence[n0:]
        for name, rule in RULES:
            kb.apply(rule(kb, new_ev, frame, rp), name)
        arb = None
        reason = self.trigger.fires(kb, new_ev, frame.t)
        if reason:
            patch = self.backend.review(text_view(kb), [_ev_line(e) for e in new_ev])
            arb = {"reason": reason, "backend": self.backend.name, "patch": patch}
        self.rows.append({
            "t": frame.t,
            "obs": {k: [o.visible, None if o.xyz is None else [round(float(v), 4) for v in o.xyz], o.track_q]
                    for k, o in frame.objects.items()},
            "ops": kb.pop_log(),
            "arbiter": arb,
        })

    def believe_inside(self, cube: str) -> Distribution:
        return self.kb.inside(cube) or Distribution.point(None, self.t, [])

    def pose(self, oid: str):
        return self.kb.objects[oid].pose.value

    def plan(self, cube: str) -> PlanStep:
        d = self.believe_inside(cube)
        return plan_pick(d, self.pose, self.params.monitor, d.support)

    def trace_rows(self) -> list[dict]:
        return self.rows
