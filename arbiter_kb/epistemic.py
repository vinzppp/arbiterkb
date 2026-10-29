"""How the memory knows, and what else could be true (NOTES "A — explicit alternatives"): per hidden cube, the most
likely container, how that is known (seen covered, then followed with the container's track) and the alternative
containers, each with the reason it is one (a close pass after the cover, where the tracker's ids may have been mixed
up). Alternatives propagate back from the end: a pass joins its partner when either container is already in the set.
It reports what the memory holds; it suggests no action.
"""
from __future__ import annotations

import re

from .kb import KB
from .narrative import inside, name

ALT_MAX_3D = 0.10  # m: passes closer than this make the partner an alternative (chosen on G train, NOTES)


def alternatives(kb: KB, cube: str, top: str, max_3d: float = ALT_MAX_3D) -> list[tuple[str, str]]:
    """[(container, reason)] other than `top`, most recent pass first."""
    covered = kb.objects[cube].t_last_seen
    passes = sorted(((e.t, e.args, e.data["min_3d"]) for e in kb.evidence if e.kind == "near_pass"), reverse=True)
    found, out = {top}, []
    for t, (a, b), d in passes:
        if t <= covered or d >= max_3d or not ({a, b} & found):
            continue
        for new, old in ((a, b), (b, a)):
            if new not in found and old in found:
                found.add(new)
                out.append((new, f"{old} and {new} passed {100 * d:.0f} cm apart at t={t}; the tracker may have "
                                  "mixed them up"))
    return out


def epistemic_lines(kb: KB) -> list[str]:
    ins, lines = inside(kb), []
    covers = {v.args[1]: (v.args[0], v.t) for v in kb.events if v.type == "cover"}
    for c, o in kb.objects.items():
        if o.cls != "cube" or o.visibility.value != "hidden" or not ins.get(c) or ins[c][0][0] is None:
            continue
        top = ins[c][0][0]
        how = (f"seen covered by {covers[c][0]} at t={covers[c][1]}" if c in covers else
               f"last seen at t={o.t_last_seen} next to {top}") + \
            f"; followed with {top}'s track since (the tracker's ids can be wrong)"
        alts = alternatives(kb, c, top)
        rest = "; ".join(f"could instead be under {k}: {why}" for k, why in alts) or "no close pass since the cover"
        lines.append(f"{name(kb, c)}: most likely under {top}. How known: {how}. {rest[0].upper() + rest[1:]}.")
    return lines


def epistemic_view(view: str, kb: KB) -> str:
    """The controller view plus the section (arm KA); the view alone when no cube is hidden."""
    lines = epistemic_lines(kb)
    return view + ("\nHow the memory knows (per hidden cube):\n" + "\n".join("  " + x for x in lines) if lines else "")


PROB = re.compile(r" \([01]\.\d{2}\)"            # "may hold the red cube (0.98)"
                  r"|(?<= \()[01]\.\d{2}; "           # "under bin_1 (0.98; bin_1 has moved ...)"
                  r"|(?<=\w) [01]\.\d{2}(?= \| |  \[)"  # Relations: "[Inside(cube_red)] bin_1 0.98 | bin_0 0.02  [e3]"
                  r"|(?<=\]) [01]\.\d{2}(?= \()")       # other relations: "[Near(a,b)] 0.90 (rule)"


def strip_probs(text: str) -> str:
    """The view with every probability (0.00-1.00) removed (arm KN, NOTES "A2"); times and distances stay."""
    return PROB.sub("", text)
