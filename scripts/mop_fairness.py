"""Fairness diff of the nine "memory or policy?" arms (main NOTES 7327cf1, "Memory payloads hold no policy").

Input: stress/mop_probe.py outputs (one scripted trajectory per task, every arm's real requests).
Checks, at every decision of every task:
  P1  the policy section (after TASK POLICY) is byte-identical across the four memories, per policy,
      and equals the generic text or the task's split policy_matched.txt;
  M1  the memory side (system text before TASK POLICY, and the memory payload: the MEMORY block for
      ld / mm / kb, the whole conversation for fh) is byte-identical across the two policies, per memory;
  S1  every factorial arm's system starts with MEMI's FullHist scaffold (fh_g's text before TASK POLICY);
  D1  the decision turn before MEMORY is byte-identical across ld / mm / kb;
  R1  no deterministic memory payload (ld, kb) contains a recommendation word; hits in mm (MEMI's own
      field descriptions and the model's values) are counted and reported, not failed.
Free; reads files only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "stress")]

_SPEC = importlib.util.spec_from_file_location("mop_policy", ROOT / "stress/mop_policy.py")
_MOP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOP)
MEMORIES, MEMORY_HEAD, POLICIES, POLICY_HEAD, policy_text = (
    _MOP.MEMORIES, _MOP.MEMORY_HEAD, _MOP.POLICIES, _MOP.POLICY_HEAD, _MOP.policy_text)

RECOMMEND = re.compile(r"\b(choose|choosing|select|selecting|recommend\w*|should|next (?:action|step|option)|"
                       r"you must|do not|don't)\b", re.IGNORECASE)
ANSWER = 'Answer with {"choice"'


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def parts(arm: str, call: dict) -> dict:
    system = call["system"]
    cut = system.index(POLICY_HEAD)
    if arm.startswith("fh"):
        return {"pre_policy": system[:cut], "policy": system[cut + len(POLICY_HEAD):],
                "payload": json.dumps(call["messages"]), "turn": None}
    user = call["messages"][-1]["content"]
    at = user.index(MEMORY_HEAD, user.rindex(ANSWER))
    return {"pre_policy": system[:cut], "policy": system[cut + len(POLICY_HEAD):],
            "payload": user[at + len(MEMORY_HEAD):], "turn": user[:at]}


def audit_task(probe: dict) -> dict:
    task, prompts = probe["task"], probe["prompts"]
    n = min(len(v) for v in prompts.values())
    fails, hits, mm_hits = [], {}, 0
    for i in range(n):
        p = {arm: parts(arm, prompts[arm][i]) for arm in prompts if arm != "memi"}
        scaffold = p["fh_g"]["pre_policy"]
        for pol in POLICIES:
            want = policy_text(task, pol)
            for mem in MEMORIES:
                if p[f"{mem}_{pol}"]["policy"] != want:
                    fails.append(f"P1 {task} decision {i} {mem}_{pol}: policy text differs")
        for mem in MEMORIES:
            g, m = p[f"{mem}_g"], p[f"{mem}_m"]
            for key in ("pre_policy", "payload"):
                if g[key] != m[key]:
                    fails.append(f"M1 {task} decision {i} {mem}: {key} differs between policies")
        for arm, x in p.items():
            if not x["pre_policy"].startswith(scaffold):
                fails.append(f"S1 {task} decision {i} {arm}: system does not start with MEMI's scaffold")
        turns = {x["turn"] for arm, x in p.items() if not arm.startswith("fh")}
        if len(turns) != 1:
            fails.append(f"D1 {task} decision {i}: decision turns differ across fresh-call arms")
        for mem in ("ld", "kb"):
            found = RECOMMEND.findall(p[f"{mem}_g"]["payload"])
            if found:
                hits.setdefault(mem, []).extend(found)
                fails.append(f"R1 {task} decision {i} {mem}: recommendation words {sorted(set(found))}")
        mm_hits += len(RECOMMEND.findall(p["mm_g"]["payload"]))
    return {"task": task, "decisions": n, "fails": fails, "recommendation_hits": hits,
            "mm_payload_recommendation_words": mm_hits,
            "policy_sha": {pol: sha(policy_text(task, pol)) for pol in POLICIES},
            "memory_note_sha": {mem: sha(parts(f"{mem}_g", prompts[f"{mem}_g"][0])["pre_policy"]) for mem in MEMORIES}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("probes", type=Path, help="directory of mop_probe.py outputs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [audit_task(json.loads(p.read_text())) for p in sorted(a.probes.glob("*.json"))]
    report = {"passed": bool(rows) and not any(r["fails"] for r in rows), "tasks": rows}
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"passed": report["passed"], **{r["task"]: len(r["fails"]) for r in rows}}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
