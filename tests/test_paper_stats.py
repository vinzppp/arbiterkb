import pytest

from arbiter_kb.paper_stats import mcnemar_exact, paired_bootstrap, rate_interval


def test_mcnemar_exact_balanced_and_one_sided():
    assert mcnemar_exact([True, False], [False, True])["p_exact"] == 1.0
    got = mcnemar_exact([True] * 5, [False] * 5)
    assert got == {"a_only": 5, "b_only": 0, "discordant": 5, "p_exact": 0.0625}


def test_rate_interval_keeps_counts():
    rate, lo, hi, k, n = rate_interval([True, False, True])
    assert (rate, k, n) == (pytest.approx(2 / 3), 2, 3)
    assert lo < rate < hi


def test_paired_bootstrap_resamples_episodes_not_rows():
    a = [{"ep": "a", "cube": "x", "correct": True, "p_max": 1.0, "brier": 0.0},
         {"ep": "a", "cube": "y", "correct": True, "p_max": 1.0, "brier": 0.0},
         {"ep": "b", "cube": "x", "correct": False, "p_max": 1.0, "brier": 1.0}]
    b = [{**r, "correct": False, "brier": 1.0} for r in a]
    got = paired_bootstrap(a, b, "accuracy", n_boot=100, seed=4)
    assert got[0] == pytest.approx(2 / 3)
    assert got[1] <= got[0] <= got[2]
