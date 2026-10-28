"""M4 dry run (scripts/dry_run.py) with a fake transport: a cache replay writes the same dry.csv, and the
budget stops the run with the paid rows kept (review G_M4)."""
import argparse
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dry_run  # noqa: E402

from arbiter_kb.llm import LLM  # noqa: E402

ANSWER = '{"waypoint": [0.1, 0.0, 0.1, -3.1416, 0, 0], "gripper": 1, "target": "bin_0", "cites": ["o99999:bin_0"]}'


def args(**kw):
    a = dict(source="synthetic", arms="K,R", n=2, first=0, effort="default", workers=2, model="m",
             max_usd=1.0, max_tokens=100, print=False, call=True)
    return argparse.Namespace(**{**a, **kw})


def client(tmp_path, monkeypatch, cost=0.01, cap=1.0):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    calls = []

    def transport(body, key):
        calls.append(body)
        return {"choices": [{"message": {"content": ANSWER}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "cost": cost}}
    return LLM(cache_dir=tmp_path / "cache", max_usd=cap, transport=transport), calls


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def test_cache_replay_writes_the_same_dry_csv(tmp_path, monkeypatch):
    llm, calls = client(tmp_path, monkeypatch)
    n1, _ = dry_run.run(args(), tmp_path / "a", llm)
    llm2, _ = client(tmp_path, monkeypatch)
    n2, usage = dry_run.run(args(), tmp_path / "b", llm2)
    first, again = read(tmp_path / "a" / "dry.csv"), read(tmp_path / "b" / "dry.csv")
    assert n1 == n2 == 4 and len(calls) == 4 and usage["hits"] == 4 and usage["usd"] == 0
    assert [r["cache_hit"] for r in first] == ["0"] * 4 and [r["cache_hit"] for r in again] == ["1"] * 4
    strip = lambda rows: [{k: v for k, v in r.items() if k != "cache_hit"} for r in rows]  # noqa: E731
    assert strip(first) == strip(again) and first[0]["prompt_tokens"] == "1000"
    assert all("unknown cite 'o99999:bin_0'" in r["problems"] for r in first)  # an invented frame


def test_budget_stops_the_run_and_keeps_the_paid_rows(tmp_path, monkeypatch):
    llm, calls = client(tmp_path, monkeypatch, cost=0.10, cap=0.25)
    n, usage = dry_run.run(args(workers=1, n=4), tmp_path / "a", llm)
    rows = read(tmp_path / "a" / "dry.csv")
    assert len(calls) == n == len(rows) == 2 and usage["refused"] == 6
    assert sum(float(r["usd"]) for r in rows) == pytest.approx(0.20)
