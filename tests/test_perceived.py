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
