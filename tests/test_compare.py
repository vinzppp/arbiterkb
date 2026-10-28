"""compare.py: the array AURC equals metrics.aurc, bootstraps are seeded and paired, matched coverage."""
import numpy as np
import pytest

from arbiter_kb import compare as C
from arbiter_kb.metrics import aurc


def rows(ps, correct, eps=None):
    eps = eps or list(range(len(ps)))
    return [{"p_max": p, "correct": c, "episode": e, "cube": "c", "condition": "none", "system": "s", "pred": "a"}
            for p, c, e in zip(ps, correct, eps)]


def test_aurc_np_matches_metrics_with_ties():
    rng = np.random.default_rng(1)
    rs = rows(list(rng.choice([0.5, 0.8, 0.9, 1.0], 200)), list(rng.random(200) < 0.8))
    p = np.array([r["p_max"] for r in rs])
    err = np.array([0.0 if r["correct"] else 1.0 for r in rs])
    assert C.aurc_np(p, err) == pytest.approx(aurc(rs), abs=1e-12)


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    rs = rows(list(np.linspace(0.5, 1, 100)), [i % 7 != 0 for i in range(100)])
    a, b = C.bootstrap_aurc(rs, n_boot=300), C.bootstrap_aurc(rs, n_boot=300)
    assert a == b and a[1] <= a[0] <= a[2]


def test_paired_diff_of_a_system_with_itself_is_exactly_zero():
    rs = rows(list(np.linspace(0.5, 1, 50)), [i % 5 != 0 for i in range(50)])
    assert C.paired_aurc_diff(rs, rs, n_boot=200) == (0.0, 0.0, 0.0)


def test_paired_diff_sign_and_ci_excludes_zero_for_a_clear_winner():
    correct = [i % 4 != 0 for i in range(80)]
    good = rows([0.6 if not c else 1.0 for c in correct], correct)  # wrong rows ranked last
    bad = rows([1.0 if not c else 0.6 for c in correct], correct)  # wrong rows ranked first
    d, lo, hi = C.paired_aurc_diff(good, bad, n_boot=300)
    assert d < 0 and hi < 0


def test_threshold_for_coverage_and_silent_at():
    rs = rows([1.0] * 6 + [0.9] * 2 + [0.5] * 2, [True] * 5 + [False] + [True, False] + [True, True])
    th = C.threshold_for_coverage(rs, 0.8)
    assert th == 0.9 and C.coverage(rs, th) == 0.8
    assert C.threshold_for_coverage(rs, 0.7) == 0.9  # 0.6 and 0.8 tie in distance -> the higher coverage
    s, lo, hi = C.silent_at(rs, th)
    assert s == pytest.approx(0.2) and lo < 0.2 < hi
