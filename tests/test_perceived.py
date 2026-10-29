import numpy as np

from arbiter_kb.sg_compat import F_HIDDEN, F_U, F_X, F_Z
from arbiter_kb.sources.perceived import occluded


def _rows(T: int = 30) -> np.ndarray:
    """One container (slot 0) and one cube (slot 1); the cube's track climbs onto the container at t = 10."""
    p = np.zeros((T, 2, 8), np.float32)
    p[:, :, F_U:F_U + 2] = 100.0
    p[:, 0, F_X:F_Z + 1] = [0.0, 0.0, 0.05]
    p[:, 1, F_X:F_Z + 1] = [0.10, 0.0, 0.017]
    p[10:, 1, F_X:F_Z + 1] = [0.01, 0.0, 0.05]
    p[:5, 0, F_HIDDEN] = 1  # the container appears late, as in G
    return p


def test_cube_on_container_top_counts_as_covered():
    occ = occluded(_rows(), 1, r=0.05, dz=0.01)
    assert not occ[:10, 1].any() and occ[10:, 1].all() and not occ[:, 0].any()


def test_back_at_table_height_is_visible_again():
    p = _rows()
    p[20:, 1, F_X:F_Z + 1] = [0.2, 0.0, 0.017]
    occ = occluded(p, 1, r=0.05, dz=0.01)
    assert occ[10:20, 1].all() and not occ[20:, 1].any()


def test_far_from_any_container_is_not_covered():
    p = _rows()
    p[10:, 1, F_X:F_Z + 1] = [0.3, 0.0, 0.05]  # high, but not over a container
    assert not occluded(p, 1, r=0.05, dz=0.01)[:, 1].any()


def test_relink_follows_motion_through_a_label_swap():
    from arbiter_kb.relink import relink
    T = 40
    p = np.zeros((T, 2, 8), np.float32)
    p[:, :, F_U:F_U + 2] = 100.0
    x = np.linspace(-0.1, 0.1, T)
    p[:, 0, F_X:F_Z + 1] = np.stack([x, np.zeros(T), np.full(T, 0.05)], 1)   # moves +x
    p[:, 1, F_X:F_Z + 1] = np.stack([-x, np.zeros(T), np.full(T, 0.05)], 1)  # moves -x
    p[20:, [0, 1]] = p[20:, [1, 0]]  # the tracker swaps the two labels at the crossing
    out = relink(p, 2)
    assert np.all(np.diff(out[:, 0, F_X]) > 0) and np.all(np.diff(out[:, 1, F_X]) < 0)


def test_strip_probs_removes_probabilities_only():
    from arbiter_kb.epistemic import strip_probs
    text = ("  [Inside(cube_red)] bin_1 0.98 | bin_0 0.02  [e3]\n  [e7] t=32 cube_red covered by bin_0 at (0.01, 0.13) "
            "(conf 0.99) [e5]\n  event swap(bin_0, bin_1) conf 0.50\n  under bin_1 (0.98; bin_1 has moved 3 times since)"
            "\n  may hold the red cube (0.97); moved\n  [e9] t=40 bin_0 moved (0.02, 0.05) -> (0.03, 0.06)")
    out = strip_probs(text)
    assert "0.98" not in out and "0.99" not in out and "0.50" not in out and "0.97" not in out
    assert "(0.01, 0.13)" in out and "(0.02, 0.05) -> (0.03, 0.06)" in out and "under bin_1 (bin_1 has moved" in out


def test_q6_truth_uses_the_container_once_it_is_there():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from perceived_qa import truth
    import pytest
    from arbiter_kb.sources.perceived import PerceivedSource
    src = PerceivedSource()
    if not src.files:
        pytest.skip("no G perceived episodes in data/g_perceived")
    answers = {truth(src.load(e))["q6"] for e in src.episodes("train")[:40]}
    assert answers == {"yes", "no"}  # before the fix every episode read "no" (NaN container position at the cover)
