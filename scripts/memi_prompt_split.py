"""Split MEMI's optimized Luna prompts into memory (M-MEMI) and policy (P-matched).

Rules 1-2 of main NOTES "Memory or policy?" (7327cf1), "MEMI prompt split":
1. memory = every ``fields[]`` entry (name + description), unchanged; policy = the whole
   ``policy_prompt_for_subgoal`` text, no sentence moved;
2. every identifier from ``fields[].name`` in the policy text becomes "the " + the name
   with underscores as spaces.

Two implementations (a regex and a character scan) are run and must agree byte for byte;
that is a self-check of the script, not the hand split of rule 5.  The script also writes
the rule-4 labelling sheet (one row per subgoal sentence, label columns empty) and a
blank worksheet for the rule-5 hand split.  Free: reads one JSON file, calls nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

TASKS = ("RouteStick", "SwingXtimes", "VideoUnmaskSwap", "StopCube", "VideoRepick")
IDENT = re.compile(r"[A-Za-z0-9_]")


def spoken(name: str) -> str:
    return "the " + name.replace("_", " ")


def substitute_regex(text: str, names: list[str]) -> str:
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", spoken(name), text)
    return text


def substitute_scan(text: str, names: list[str]) -> str:
    """Independent second implementation: walk identifier tokens, replace whole matches."""
    wanted, out, i = set(names), [], 0
    while i < len(text):
        if IDENT.match(text[i]):
            j = i
            while j < len(text) and IDENT.match(text[j]):
                j += 1
            token = text[i:j]
            out.append(spoken(token) if token in wanted else token)
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def normalize(text: str) -> str:
    return " ".join(text.split())


def sentences(text: str) -> list[str]:
    """Sentence boundaries for the labelling sheet only (period/question/exclamation + space + capital or quote)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z'\"(\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_task(entry: dict) -> dict:
    names = [field["name"] for field in entry["fields"]]
    policy_a = substitute_regex(entry["policy_prompt_for_subgoal"], names)
    policy_b = substitute_scan(entry["policy_prompt_for_subgoal"], names)
    memory = [{"name": f["name"], "description": f["description"]} for f in entry["fields"]]
    doubled = len(re.findall(r"\bthe the\b", policy_a, flags=re.IGNORECASE)) - len(
        re.findall(r"\bthe the\b", entry["policy_prompt_for_subgoal"], flags=re.IGNORECASE))
    return {
        "field_names": names,
        "memory_fields": memory,
        "policy_matched": policy_a,
        "identical_bytes": policy_a == policy_b,
        "identical_normalized": normalize(policy_a) == normalize(policy_b),
        "substitutions": {n: len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])",
                                             entry["policy_prompt_for_subgoal"])) for n in names},
        "new_the_the": doubled,
        "names_left_in_policy": [n for n in names if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", policy_a)],
    }


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_task(task_dir: Path, entry: dict) -> dict:
    """Write one task's split; return its summary row (hashes, checks, counts)."""
    row = split_task(entry)
    task_dir.mkdir(exist_ok=True)
    (task_dir / "policy_matched.txt").write_text(row["policy_matched"])
    memory_text = json.dumps(row["memory_fields"], indent=1, ensure_ascii=False) + "\n"
    (task_dir / "memory_fields.json").write_text(memory_text)
    return {
        **{k: v for k, v in row.items() if k not in ("memory_fields", "policy_matched")},
        "policy_matched_sha256": sha(row["policy_matched"].encode()),
        "policy_matched_normalized_sha256": sha(normalize(row["policy_matched"]).encode()),
        "memory_fields_sha256": sha(memory_text.encode()),
        "subgoal_sentences": len(sentences(entry["policy_prompt_for_subgoal"])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts", type=Path, help="optimized_luna.json from MEMI bb6c37c")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True, help="directory for the rule-4/5 hand-work files")
    a = ap.parse_args()
    raw = a.prompts.read_bytes()
    prompts = json.loads(raw)
    a.out.mkdir(parents=True, exist_ok=True)
    a.labels.mkdir(parents=True, exist_ok=True)
    summary = {"source_sha256": sha(raw), "rules": "NOTES 7327cf1 MEMI prompt split rules 1-2", "tasks": {}}
    label_rows, worksheet = [], []
    for task in TASKS:
        summary["tasks"][task] = write_task(a.out / task, prompts[task])
        row = summary["tasks"][task]
        for i, text in enumerate(sentences(prompts[task]["policy_prompt_for_subgoal"]), 1):
            label_rows.append({"task": task, "sentence_id": f"{task}-{i:02d}", "text": text,
                               "coder_1": "", "coder_2": "", "adjudicated": "", "note": ""})
        worksheet.append(f"## {task}\n\nfields[].name: {', '.join(row['field_names'])}\n\n"
                         f"policy_prompt_for_subgoal (original):\n\n{prompts[task]['policy_prompt_for_subgoal']}\n\n"
                         "Hand split, P-matched (write the substituted text here):\n\n\n")
    with (a.labels / "memi_subgoal_sentence_labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)
    (a.labels / "memi_hand_split_worksheet.md").write_text(
        "# Rule-5 hand split worksheet (fill without opening the script's output)\n\n"
        "Rule 1: memory = fields[] unchanged; policy = the whole subgoal text. Rule 2: every fields[].name\n"
        "identifier in the policy text becomes \"the \" + the name with underscores as spaces.\n\n"
        + "\n".join(worksheet))
    (a.out / "split_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({t: {k: v[k] for k in ("identical_bytes", "substitutions", "new_the_the")}
                      for t, v in summary["tasks"].items()}, indent=1))


if __name__ == "__main__":
    main()
