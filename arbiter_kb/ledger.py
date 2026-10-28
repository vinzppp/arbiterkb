"""Append-only test-read ledger (E5c): the test split may only be read with a parameter set whose
FREEZE entry precedes the read. Lines: '<UTC time>\t<FREEZE|READ>\t<params hash>\t<detail>'.
Location: $ARBITER_TEST_LOG or results/test_reads.log (committed)."""
from __future__ import annotations

import os
import time
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "results" / "test_reads.log"


def path() -> Path:
    return Path(os.environ.get("ARBITER_TEST_LOG", DEFAULT))


def record(kind: str, params_hash: str, detail: str) -> None:
    assert kind in ("FREEZE", "READ")
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{kind}\t{params_hash}\t{detail}\n")


def was_frozen(params_hash: str) -> bool:
    p = path()
    if not p.exists():
        return False
    return any(line.split("\t")[1:3] == ["FREEZE", params_hash] for line in p.read_text().splitlines() if line)
