import json

from scripts.common_harness_report import audit, group_summary, load_rows, paired_tests, reset_digest


def test_load_rows_recovers_missing_arm_from_result_directory(tmp_path):
    path = tmp_path / "clean" / "Task_ep0_memi" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"arm":"unknown","task":"Task","episode":0}')
    row = load_rows(tmp_path)[0]
    assert row["arm"] == "memi"
    assert row["arm_recorded"] == "unknown"


def row(arm, success, digest="same", episode=0):
    return {"cell": "clean", "task": "VideoUnmaskSwap", "episode": episode, "arm": arm,
            "outcome": "success" if success else "fail", "success": success, "reset_digest": digest,
            "seconds": 1, "policy": {"usage": {"input_tokens": 100, "output_tokens": 10},
                                      "log": [{}], "statuses": {"valid": 1}},
            "provenance": {"episode": {"seed": 1}, "arbiter_dirty": False}, "_path": arm}


def test_report_pairs_outcomes_and_applies_holm():
    rows = [row("fullhist", False), row("memi", True), row("soft", False), row("arbiter", True)]
    out = paired_tests(rows, [["memi", "fullhist"], ["arbiter", "soft"]])
    assert out["clean/ALL"]["memi_vs_fullhist"]["n_paired"] == 1
    assert out["clean/ALL"]["memi_vs_fullhist"]["risk_difference"] == [1.0, 1.0, 1.0]
    assert out["clean/ALL"]["memi_vs_fullhist"]["p_holm"] == 1.0


def test_report_cost_and_leakage_audit():
    rows = [row(a, a == "arbiter") for a in ("fullhist", "memi", "soft", "arbiter")]
    prices = {"fresh_input": .1, "cached_input": .01, "cache_write": .125, "output": .5}
    assert group_summary(rows, prices)["episodes"] == 4
    assert audit(rows, ["fullhist", "memi", "soft", "arbiter"])["passed"]
    rows[-1]["reset_digest"] = "different"
    assert not audit(rows, ["fullhist", "memi", "soft", "arbiter"])["passed"]


def test_result_rows_are_json_serializable():
    assert json.dumps(row("fullhist", True))


def test_scripted_preflight_is_not_reported_as_policy_success():
    one = row("fullhist", False)
    one["policy"]["model"] = "none"
    summary = group_summary([one], {})
    assert summary["diagnostic_only"]
    assert "success" not in summary


def test_timeout_reset_digest_is_recovered_from_durable_events():
    timeout = row("soft", False)
    timeout.pop("reset_digest")
    timeout["partial_events"] = [
        {"event": "phase_end", "phase": "reset", "reset_digest": "same"},
        {"event": "phase_start", "phase": "llm"},
    ]
    rows = [row("fullhist", False), row("memi", False), timeout, row("arbiter", False)]
    assert reset_digest(timeout) == "same"
    assert audit(rows, ["fullhist", "memi", "soft", "arbiter"])["passed"]
