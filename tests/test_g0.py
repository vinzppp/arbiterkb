"""G0 helpers: MEMI prompt split (rules 1-2) and the G2 snapping check."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G2, SPLIT = _load("g2_snapping"), _load("memi_prompt_split")
snap, train_files = G2.snap, G2.train_files
split_task, substitute_regex, substitute_scan = SPLIT.split_task, SPLIT.substitute_regex, SPLIT.substitute_scan


def test_substitution_replaces_whole_identifiers_only():
    text = "Update completed_cycles; completed_cycles_extra and xcompleted_cycles stay. (completed_cycles)"
    want = ("Update the completed cycles; completed_cycles_extra and xcompleted_cycles stay. "
            "(the completed cycles)")
    assert substitute_regex(text, ["completed_cycles"]) == want
    assert substitute_scan(text, ["completed_cycles"]) == want


def test_split_keeps_fields_unchanged_and_moves_no_sentence():
    entry = {"fields": [{"name": "target_anchor", "description": "Store the cube. Choose A."}],
             "policy_prompt_for_subgoal": "Select target_anchor. Then B."}
    row = split_task(entry)
    assert row["memory_fields"] == entry["fields"]
    assert row["policy_matched"] == "Select the target anchor. Then B."
    assert row["identical_bytes"] and row["names_left_in_policy"] == []


def test_split_counts_new_doubled_article():
    entry = {"fields": [{"name": "completed_cycles", "description": ""}],
             "policy_prompt_for_subgoal": "in the completed_cycles prompt"}
    assert split_task(entry)["new_the_the"] == 1


def test_snap_picks_nearest_container_and_reports_margin():
    oracle = np.full((3, 8), np.nan)
    oracle[0, 2:4], oracle[1, 2:4], oracle[2, 2:4] = (10, 10), (40, 10), (100, 100)
    got, dist, margin = snap(np.array([18.0, 10.0]), oracle, [0, 1, 2])
    assert got == 0 and dist == pytest.approx(8.0) and margin == pytest.approx(14.0)


def test_g2_reads_train_suffixes_only(tmp_path):
    for name in ("stock_ep000.npz", "stock_ep009.npz", "stock_ep010.npz", "stock_ep019.npz", "fast_ep001.npz"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in train_files(tmp_path, ["stock"])] == ["stock_ep000.npz", "stock_ep009.npz"]
