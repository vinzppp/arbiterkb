import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from paper_compare import memi_records, memi_summary  # noqa: E402


def test_memi_plan_reader_and_summary(tmp_path):
    base = tmp_path / "results/astra_plan/memi"
    base.mkdir(parents=True)
    rows = [{"task": "VideoUnmaskSwap", "episode": 0, "outcome": "success",
             "policy": {"usage": {"input_tokens": 10}}},
            {"task": "Other", "episode": 1, "outcome": "fail",
             "policy": {"usage": {"input_tokens": 20}}}]
    with gzip.open(base / "episodes.jsonl.gz", "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    got = memi_records(tmp_path, "astra_plan", "memi")
    assert got == rows
    summary = memi_summary(got)
    assert summary["success"][-2:] == [1, 2]
    assert summary["input_tokens_median"] == 15
