from scripts.paper_error_analysis import paired_diagnostics


def _row(ep, cube, pred, correct, brier):
    return {"ep": ep, "cube": cube, "pred": pred, "correct": correct, "brier": brier,
            "p_max": 0.8, "truth": ["box"], "probs": {pred: 0.8}, "audit": {}}


def test_paired_diagnostics_separates_flips_from_calibration():
    full = [_row("e1", "a", "x", True, 0.1), _row("e2", "b", "x", False, 0.6)]
    soft = [_row("e1", "a", "x", True, 0.2), _row("e2", "b", "y", True, 0.1)]
    got = paired_diagnostics(full, soft)
    assert got["counts"]["prediction_agrees"] == 1
    assert got["counts"]["prediction_disagrees"] == 1
    assert got["counts"]["soft_only_correct"] == 1
    assert got["full_brier_worse_n"] == 1
