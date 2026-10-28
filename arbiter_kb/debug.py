"""Debugging from traces (E2).

  python -m arbiter_kb.debug explain  --run RUN_DIR --condition id_swap --system arbiterkb --ep 1001 --cube cube_green [--t 100]
  python -m arbiter_kb.debug timeline --run RUN_DIR --condition id_swap --system arbiterkb --ep 1001 --cube cube_green
(trace files are RUN_DIR/traces/<condition>_<system>_<ep>.jsonl)

explain walks the support ids of the cube's Inside distribution back to the raw observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .trace import read_trace


def index(rows: list[dict]) -> tuple[dict, dict, dict]:
    """evidence id -> (t, after), event id -> (t, after), t -> obs."""
    ev, evt, obs = {}, {}, {}
    for r in rows:
        obs[r["t"]] = r.get("obs", {})
        for op in r.get("ops", []):
            if op["op"] == "AddEvidence":
                ev[op["target"]] = (r["t"], op["after"])
            elif op["op"] == "AddEvent":
                evt[op["target"]] = (r["t"], op["after"])
    return ev, evt, obs


def timeline(rows: list[dict], cube: str) -> list[tuple[int, dict]]:
    """[(t, probs)] each frame, carrying the last distribution forward."""
    out, cur = [], {}
    for r in rows:
        for op in r.get("ops", []):
            if op["target"] == f"Inside({cube})" and isinstance(op["after"], dict):
                cur = op["after"].get("probs", op["after"])
        out.append((r["t"], dict(cur)))
    return out


def explain(rows: list[dict], cube: str, t: int | None = None) -> str:
    ev, evt, obs = index(rows)
    last = None
    for r in rows:
        if t is not None and r["t"] > t:
            break
        for op in r.get("ops", []):
            if op["target"] == f"Inside({cube})" and isinstance(op["after"], dict) and "support" in op["after"]:
                last = (r["t"], op)
    if last is None:
        return _explain_tracker(rows, cube, t)
    lines = [f"Inside({cube}) at t={last[0]} via {last[1]['op']} [{last[1]['rule']}]: {last[1]['after']['probs']}"]
    for ident in last[1]["after"]["support"]:
        lines += _walk(ident, ev, evt, obs, depth=1, seen=set())
    return "\n".join(lines)


def _explain_tracker(rows: list[dict], cube: str, t: int | None) -> str:
    """Deterministic baselines keep no beliefs: report the one-hot answer and the observation it rests on."""
    rows = [r for r in rows if "pred" in r and (t is None or r["t"] <= t)]
    if not rows:
        return f"no Inside({cube}) belief up to t={t}"
    r = rows[-1]
    pred, conf, doubt = r["pred"].get(cube), r.get("p_max", {}).get(cube, 1), r.get("doubt", {}).get(cube)
    why = f"; doubt from trigger on ({doubt[0]}, {doubt[1]}) at t={doubt[2]}" if doubt else ""
    return (f"{cube} -> {pred} with confidence {conf} at t={r['t']} (tracker: nearest container to the tracked "
            f"cube position; rests on observation o{r['t']}:{pred}, no evidence kept{why})")


def _walk(ident: str, ev: dict, evt: dict, obs: dict, depth: int, seen: set) -> list[str]:
    pad = "  " * depth
    if ident in seen:
        return [f"{pad}{ident} (see above)"]
    seen.add(ident)
    if ident.startswith("o"):
        t, oid = ident[1:].split(":", 1)
        return [f"{pad}{ident}: observation {oid} at t={t} -> {obs.get(int(t), {}).get(oid)}"]
    if ident in ev:
        t, a = ev[ident]
        out = [f"{pad}{ident}: evidence {a['kind']}({', '.join(a['args'])}) at t={t} {_brief(a['data'])}"]
        for o in a["data"].get("obs", []):
            out += _walk(o, ev, evt, obs, depth + 1, seen)
        return out
    if ident in evt:
        t, a = evt[ident]
        out = [f"{pad}{ident}: event {a['type']}({', '.join(a['args'])}) at t={t} conf={a['conf']}"]
        for s in a["support"]:
            out += _walk(s, ev, evt, obs, depth + 1, seen)
        return out
    return [f"{pad}{ident}: NOT FOUND in trace"]


def _brief(data: dict) -> str:
    return "{" + ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in data.items()
                           if k != "obs") + "}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["explain", "timeline"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--condition", default="none")
    ap.add_argument("--system", default="arbiterkb")
    ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--cube", required=True)
    ap.add_argument("--t", type=int)
    a = ap.parse_args()
    path = Path(a.run) / "traces" / f"{a.condition}_{a.system}_{a.ep}.jsonl"
    if not path.exists():
        found = sorted(p.stem for p in (Path(a.run) / "traces").glob("*.jsonl"))[:10]
        raise SystemExit(f"no trace {path.name}; available (first 10): {found}")
    rows = read_trace(path)
    if a.cmd == "explain":
        print(explain(rows, a.cube, a.t))
    else:
        prev = None
        for t, p in timeline(rows, a.cube):
            if p != prev:
                print(t, p)
                prev = p


if __name__ == "__main__":
    main()
