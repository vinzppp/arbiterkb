import pytest

from arbiter_kb.paper_stats import holm_adjust, mcnemar_exact, paired_bootstrap, paired_randomization, rate_interval


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


def test_randomization_and_holm_are_deterministic():
    a = [{"ep": f"e{i}", "cube": "c", "correct": True, "p_max": 0.9, "brier": 0.01}
         for i in range(8)]
    b = [{**r, "correct": False, "p_max": 0.9, "brier": 1.0} for r in a]
    assert paired_randomization(a, b, "brier", n_perm=199, seed=4) < 0.05
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.5}) == {"a": 0.03, "b": 0.08, "c": 0.5}
