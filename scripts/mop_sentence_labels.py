"""Descriptive sentence labels of MEMI's subgoal text (main NOTES 7327cf1 split rule 4, as changed by the
orchestrator on 2026-11-01: script + manual check, three classes, reported as shares per task).

The script computes, per sentence, whether it names a MEMI field literally (a reference only, never a
label); the labels themselves come from manual_labels.json.  Writes the filled sheet and per-task
shares.  Fails if a sentence has no manual label or a literal field-name match is not flagged.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def names_field(text: str, fields: list[str]) -> list[str]:
    return [n for n in fields if re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", text)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--g0", type=Path, default=Path("results/memory_or_policy/g0"))
    a = ap.parse_args()
    sheet = a.g0 / "labeling/memi_subgoal_sentence_labels.csv"
    manual = json.loads((a.g0 / "labeling/manual_labels.json").read_text())["labels"]
    rows = list(csv.DictReader(sheet.open()))
    out, shares, problems = [], {}, []
    for row in rows:
        fields = [f["name"] for f in json.loads((a.g0 / f"memi_split/{row['task']}/memory_fields.json").read_text())]
        literal = names_field(row["text"], fields)
        if row["sentence_id"] not in manual:
            problems.append(f"{row['sentence_id']}: no manual label")
            continue
        cls, ref, note = manual[row["sentence_id"]]
        if literal and not ref:
            problems.append(f"{row['sentence_id']}: names {literal} but is not flagged as a memory reference")
        out.append({"task": row["task"], "sentence_id": row["sentence_id"], "text": row["text"],
                    "script_field_names": ";".join(literal), "class": cls, "memory_reference": ref, "note": note})
    for task in dict.fromkeys(r["task"] for r in out):
        mine = [r for r in out if r["task"] == task]
        n, c = len(mine), Counter(r["class"] for r in mine)
        shares[task] = {"sentences": n,
                        "choice_rule": round(c["choice_rule"] / n, 3), "memory_update": round(c["memory_update"] / n, 3),
                        "memory_reference": round(sum(r["memory_reference"] for r in mine) / n, 3),
                        "script_literal_field_name": round(sum(bool(r["script_field_names"]) for r in mine) / n, 3)}
    with (a.g0 / "labeling/memi_subgoal_sentence_labels_filled.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    (a.g0 / "labeling/label_shares.json").write_text(json.dumps({"shares": shares, "problems": problems}, indent=1) + "\n")
    print(json.dumps(shares, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
