"""Astra's VideoUnmaskSwap failures after the first move (NOTES "Astra's 7 VideoUnmaskSwap failures"). Free, on the Mac:
reads result.json files copied from the pod and the stage-1 listing's objects; labels every failed episode by the
first rule that applies (arbiter_kb/callout.label), with the successes as the lift reference.

  python scripts/read_failures.py --results data/cl_pod/astra_pilot --objects data/cl_pod/callout/vus_objects.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arbiter_kb.callout import holds, label, objects, right_picks, right_sequence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/cl_pod/astra_pilot")
    ap.add_argument("--objects", default="data/cl_pod/callout/vus_objects.json")
    ap.add_argument("--task", default="VideoUnmaskSwap")
    a = ap.parse_args()
    scenes = json.loads(Path(a.objects).read_text())["episodes"]
    eps = []
    for f in sorted(Path(a.results).glob(f"{a.task}_ep*_*/result.json")):
        row = json.loads(f.read_text())
        sc = scenes[str(row["episode"])]
        objs = objects({k: v + [0.0] for k, v in sc["objects"].items()})
        seq = right_sequence(sc["goal"], objs)
        eps.append((row, holds(row.get("steps") or [], objs), seq))
    min_lift = {}
    for row, hs, seq in eps:
        if row.get("success"):
            for k, h in right_picks(hs, seq)[0]:
                min_lift[k] = min(min_lift.get(k, 9.0), h["lift"])
    print("successes:", sum(bool(r.get("success")) for r, _, _ in eps), "of", len(eps), "; lowest lift per right pick:",
          {k: round(v, 4) for k, v in sorted(min_lift.items())})
    for row, hs, seq in eps:
        if row.get("success"):
            continue
        lab, why = label(row, hs, seq, min_lift)
        print(json.dumps({"episode": row["episode"], "arm": row["arm"], "end": row.get("end_reason"),
                          "calls": row.get("model_calls"), "sequence": seq, "label": lab, "why": why,
                          "holds": [{k: h[k] for k in ("step", "object", "d", "flagged", "close_z", "lift")}
                                    for h in hs]}))


if __name__ == "__main__":
    main()
