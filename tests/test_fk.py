"""Panda forward kinematics: the home pose and a table-level grasp height (validated on the dataset in NOTES)."""
import numpy as np

from arbiter_kb.fk import BASE, tcp, tcp_path


def test_home_pose_is_above_the_table_in_front_of_the_base():
    p = tcp(np.array([0, 0, 0, -1.571, 0, 1.571, 0.785]))
    assert 0.45 < p[2] < 0.6 and abs(p[1]) < 1e-3 and p[0] > BASE[0]


def test_all_zero_joints_reach_straight_up():
    p = tcp(np.zeros(7))  # arm stretched up: x = base + 0.088 (a7), z = 0.333 + 0.316 + 0.384 - (flange + TCP)
    assert np.allclose(p, BASE + [0.088, 0, 0.333 + 0.316 + 0.384 - 0.107 - 0.1034], atol=1e-3)
    assert tcp_path(np.zeros((3, 8))).shape == (3, 3)
