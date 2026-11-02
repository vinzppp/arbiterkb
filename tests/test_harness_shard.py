import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS_SPEC = importlib.util.spec_from_file_location(
    "shard_common_harness", Path(__file__).resolve().parents[1] / "stress/common_harness_pod.py")
HARNESS = importlib.util.module_from_spec(HARNESS_SPEC)
HARNESS_SPEC.loader.exec_module(HARNESS)

ARMS = ["fh_g", "fh_m", "memi"]
SPEC = {"arms": ARMS, "tasks": ["A", "B", "C"], "watchdog": {"x": 1}}
STAGE = {"backend": "api", "episodes": [0, 1, 2, 3], "observations": [{"mode": "clean", "seed": 1}]}


def calls_for(tmp_path, monkeypatch, shard):
    seen = []
    monkeypatch.setattr(HARNESS, "run_supervised_one",
                        lambda a, spec, stage, sources, task, episode, arm, cfg: seen.append((task, episode, arm)) or 0)
    a = SimpleNamespace(out=tmp_path, stage="confirmation", shard=shard)
    assert HARNESS.run_parent(a, SPEC, STAGE, {}) == 0
    return seen


def test_parse_shard():
    assert HARNESS.parse_shard(None) == (0, 1)
    assert HARNESS.parse_shard("2/6") == (2, 6)
    for bad in ("6/6", "-1/3", "a/b", "1", "0/0"):
        with pytest.raises(SystemExit):
            HARNESS.parse_shard(bad)


def test_shards_partition_pairs_and_keep_arms_together(tmp_path, monkeypatch):
    whole = calls_for(tmp_path, monkeypatch, None)
    assert len(whole) == 3 * 4 * len(ARMS)
    parts = [calls_for(tmp_path / str(k), monkeypatch, f"{k}/3") for k in range(3)]
    assert sorted(sum(parts, [])) == sorted(whole)
    pairs = [{(t, e) for t, e, _ in part} for part in parts]
    assert not (pairs[0] & pairs[1] or pairs[0] & pairs[2] or pairs[1] & pairs[2])
    for part in parts:
        for t, e in {(t, e) for t, e, _ in part}:
            assert [arm for tt, ee, arm in part if (tt, ee) == (t, e)] == ARMS


def source_repo(tmp_path, monkeypatch):
    subprocess.run(("git", "init", str(tmp_path)), check=True, capture_output=True)
    subprocess.run(("git", "-C", str(tmp_path), "config", "user.name", "Harness Test"), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "config", "user.email", "harness@example.invalid"), check=True)
    ledger = tmp_path / "results/test_reads.log"
    ledger.parent.mkdir()
    ledger.write_text("2026-11-02T00:00:00Z\tFREEZE\tabc123\tfrozen\n")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(("git", "-C", str(tmp_path), "add", "."), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "commit", "-m", "baseline"), check=True, capture_output=True)
    monkeypatch.setattr(HARNESS, "ROOT", tmp_path)
    return ledger, tracked


def test_source_dirty_allows_only_appended_read_lines(tmp_path, monkeypatch):
    ledger, _ = source_repo(tmp_path, monkeypatch)
    with ledger.open("a") as stream:
        stream.write("2026-11-02T00:01:00Z\tREAD\tabc123\tconfirmation\n")
        stream.write("2026-11-02T00:02:00Z\tREAD\tabc123\tworker\n")
    assert not HARNESS.source_dirty()


def test_source_dirty_rejects_appended_non_read_line(tmp_path, monkeypatch):
    ledger, _ = source_repo(tmp_path, monkeypatch)
    with ledger.open("a") as stream:
        stream.write("2026-11-02T00:01:00Z\tFREEZE\tabc123\tnot allowed\n")
    assert HARNESS.source_dirty()


def test_source_dirty_rejects_deleted_ledger_line(tmp_path, monkeypatch):
    ledger, _ = source_repo(tmp_path, monkeypatch)
    ledger.write_text("")
    assert HARNESS.source_dirty()


def test_source_dirty_rejects_other_tracked_change(tmp_path, monkeypatch):
    ledger, tracked = source_repo(tmp_path, monkeypatch)
    with ledger.open("a") as stream:
        stream.write("2026-11-02T00:01:00Z\tREAD\tabc123\tconfirmation\n")
    tracked.write_text("dirty\n")
    assert HARNESS.source_dirty()
