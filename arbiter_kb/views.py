"""Views of the KB: predicate facts with course-style unify / ask, and the text view the arbiter reads.

Predicates are generated on demand from the typed tables (DEVELOPMENT.md §3.2); they are never stored.
ask() returns answers WITH probabilities instead of True / False.
"""
from __future__ import annotations

from .kb import KB


def is_var(x) -> bool:
    return isinstance(x, str) and x.startswith("?")


def unify(pattern: tuple, fact: tuple, bindings: dict | None = None) -> dict | None:
    """Course Module 7 unify for flat tuples: variables are strings starting with '?'."""
    b = dict(bindings or {})
    if len(pattern) != len(fact):
        return None
    for p, f in zip(pattern, fact):
        if is_var(p):
            if p in b and b[p] != f:
                return None
            b[p] = f
        elif p != f:
            return None
    return b


def facts(kb: KB) -> list[tuple[tuple, float]]:
    """All facts the KB currently supports, each with its probability."""
    out = []
    for (pred, subj), d in kb.relations.items():
        out += [((pred, subj, k), p) for k, p in d.probs.items() if k is not None]
    for o in kb.objects.values():
        vis = o.visibility.value
        if vis == "hidden":
            out.append((("Hidden", o.id), o.visibility.conf))
        elif vis == "visible":
            out.append((("Visible", o.id), o.visibility.conf))
    return out


def ask(kb: KB, pattern: tuple) -> list[tuple[dict, float]]:
    """All bindings that satisfy `pattern`, most probable first."""
    answers = [(b, p) for f, p in facts(kb) if (b := unify(pattern, f)) is not None]
    return sorted(answers, key=lambda bp: (-bp[1], str(bp[0])))


def text_view(kb: KB) -> str:
    """One line per object, e.g.
    [cube_green] cube, hidden since t=32; inside bin_1 (.55) | bin_2 (.45); support e12,e31"""
    lines = []
    for o in kb.objects.values():
        vis = o.visibility.value
        line = f"[{o.id}] {o.description}, {vis}"
        if vis == "hidden":
            line += f" since t={o.t_last_seen + 1}"
        d = kb.inside(o.id)
        if d is not None:
            hyps = sorted(d.probs.items(), key=lambda kv: -kv[1])
            line += "; inside " + " | ".join(f"{k or 'elsewhere'} ({p:.2f})" for k, p in hyps)
            line += "; support " + ",".join(d.support[-4:])
        elif o.pose.value is not None and vis == "visible":
            x, y, _ = o.pose.value
            line += f"; at x={x:.2f} y={y:.2f}"
        lines.append(line)
    return "\n".join(lines)
