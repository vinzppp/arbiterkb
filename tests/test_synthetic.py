import numpy as np

from arbiter_kb.sources.synthetic import COVER_T, SWAP_T0, SyntheticSource, swap_path


def test_teleport_cover_and_ground_truth():
    s = SyntheticSource()
    ep = s.episodes("train", 1)[0]
    frames, gt = list(s.frames(ep)), s.ground_truth(ep)
    f31, f32 = frames[COVER_T - 1], frames[COVER_T]
    cube = gt.targets[0]
    assert f31.objects[cube].visible and not f32.objects[cube].visible
    b = gt.cube_to_bin[cube]
    assert not f31.objects[b].visible and f32.objects[b].visible
    assert np.allclose(f32.objects[b].xyz[:2], f31.objects[cube].xyz[:2])  # appears exactly over it
    assert gt.t_star[cube] == len(frames) - 1


def test_swap_path_endpoints_and_bulge():
    A, B = np.array([0.0, 0.0]), np.array([0.1, 0.0])
    a0, b0 = swap_path(A, B, 0.0, 0.07)
    a1, b1 = swap_path(A, B, 1.0, 0.07)
    am, bm = swap_path(A, B, 0.5, 0.07)
    assert np.allclose(a0, A) and np.allclose(b1, A) and np.allclose(a1, B)
    assert abs(np.linalg.norm(am - bm) - 0.14) < 1e-9  # opposite lanes at mid-swap


def test_deterministic_and_covers_moved_and_back():
    s1, s2 = SyntheticSource(), SyntheticSource()
    eps = s1.episodes("test", 30)
    a = [f.objects for f in s1.frames(eps[0])]
    b = [f.objects for f in s2.frames(eps[0])]
    assert all(np.array_equal(x["bin_0"].xyz, y["bin_0"].xyz) for x, y in zip(a, b) if x["bin_0"].visible)
    moved = [list(s1.ground_truth(e).moved.values())[0] for e in eps]
    assert any(moved) and not all(moved)


def test_containers_move_only_during_swaps():
    s = SyntheticSource(n_swaps=1)
    ep = s.episodes("train", 1)[0]
    fr = list(s.frames(ep))
    for b in [k for k, o in fr[-1].objects.items() if o.cls == "container"]:
        assert np.allclose(fr[COVER_T].objects[b].xyz, fr[SWAP_T0].objects[b].xyz)
