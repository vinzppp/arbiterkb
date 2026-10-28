"""Monitor (M0 stub): the decision rule only. R8 consistency checks and risk flags arrive in M3."""
from __future__ import annotations

from .params import MonitorParams


def decide(p_max: float, mp: MonitorParams) -> str:
    """act if confident, verify if not, otherwise act on the most likely hypothesis (flagged)."""
    if p_max >= mp.tau_hi:
        return "act"
    if p_max < mp.tau_lo:
        return "verify"
    return "act_on_argmax"
