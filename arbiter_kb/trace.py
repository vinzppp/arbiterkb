"""Per-episode traces (DEVELOPMENT.md §3.9): one JSON line per frame with the observations, every KB
change (op, target, before, after, rule) and any arbiter call."""
from __future__ import annotations

import json
from pathlib import Path


def write_trace(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float, sort_keys=True) + "\n")


def read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
