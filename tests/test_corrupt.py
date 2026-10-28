from arbiter_kb.sources.corrupt import CorruptedSource, chosen
from arbiter_kb.sources.synthetic import SyntheticSource


def test_nested_selection():
    eps = range(1000, 1200)
    s25 = {e for e in eps if chosen(e, 0.25)}
    s50 = {e for e in eps if chosen(e, 0.5)}
    assert s25 <= s50 and 20 < len(s25) < 80


def test_id_swap_exchanges_labels_not_positions():
    base = SyntheticSource()
    src = CorruptedSource(base, p=1.0)
    for ep in base.episodes("test", 10):
        clean, bad = list(base.frames(ep)), list(src.frames(ep))
        log = src.log[ep]
        if not log["corrupted"]:
            continue
        a, b = log["ids"]
        t = log["t"]
        assert (clean[t - 1].objects[a].xyz == bad[t - 1].objects[a].xyz).all()
        assert (clean[t].objects[a].xyz == bad[t].objects[b].xyz).all()
        assert bad[t].objects[b].id == b  # the label lies; the observation is the other container's
        return
    raise AssertionError("no corrupted episode in the first 10")


def test_overlap_log_is_written_before_scoring():
    """Scoring at t* (the last frame) reads log[ep]['corrupted'] while the generator is still open."""
    from arbiter_kb.sources.realistic import OverlapSwapSource

    src = OverlapSwapSource(SyntheticSource(lane_offset=0.0), 1.0)
    ep = src.episodes("train", 1)[0]
    it = src.frames(ep)
    next(it)
    assert src.log[ep]["corrupted"] is True


def _sig(b: float, dt=(2,), drop_sw=(3,), drop_clean=(0,)):
    """A hand-made G signature: p_swap = sigmoid(b) for every pass."""
    return {"w": [0.0, 0.0], "b": b, "min_3d_range": [0.05, 0.25], "swap_dt": list(dt),
            "dropout_swapped": list(drop_sw), "dropout_clean": list(drop_clean)}


def test_measured_swap_never_swapping_changes_nothing():
    from arbiter_kb.sources.realistic import MeasuredSwapSource

    base = SyntheticSource(lane_offset=0.02)
    src = MeasuredSwapSource(base, _sig(-50.0))
    for ep in base.episodes("train", 5):
        clean, bad = list(base.frames(ep)), list(src.frames(ep))
        assert not src.log[ep]["corrupted"] and src.log[ep]["hidden_frames"] == 0
        assert all(c.objects[k].visible == b.objects[k].visible for c, b in zip(clean, bad) for k in c.objects)


def test_measured_swap_follows_a_dropout_and_moves_the_label():
    """p = 1: every pass swaps at t_min + 2, after one of its two containers was hidden for 3 frames."""
    from arbiter_kb.sources.realistic import MeasuredSwapSource, passes

    base = SyntheticSource(lane_offset=0.04)
    src = MeasuredSwapSource(base, _sig(50.0))
    ep = base.episodes("train", 1)[0]
    clean, bad = list(base.frames(ep)), list(src.frames(ep))
    log, ps = src.log[ep], passes(clean, src.v_move)
    assert log["corrupted"] and len(log["swaps"]) == len(ps)
    t_sw, a, b = log["swaps"][0][:3]
    assert t_sw == ps[0]["t_min"] + 2
    hidden = [k for k in (a, b) for t in range(t_sw - 3, t_sw) if not bad[t].objects[k].visible]
    assert len(hidden) == 3 and all(bad[t_sw - 4].objects[k].visible for k in (a, b))
    assert (bad[t_sw].objects[a].xyz == clean[t_sw].objects[b].xyz).all()  # the label now reports b
    assert log["label_jumps_m"] and min(log["label_jumps_m"]) > 0.03  # lane 0.04: centres ~8 cm apart


def test_measured_swap_is_seeded_and_clamps_distance():
    from arbiter_kb.sources.realistic import MeasuredSwapSource

    sig = {**_sig(0.0), "w": [-80.0, 0.0], "b": 4.0}
    src = MeasuredSwapSource(SyntheticSource(lane_offset=0.0), sig)
    assert src.p_swap(0.0, 0.01) == src.p_swap(0.05, 0.01) > src.p_swap(0.1, 0.01)
    ep = src.episodes("train", 1)[0]
    first = [[o.visible for o in f.objects.values()] for f in src.frames(ep)]
    again = [[o.visible for o in f.objects.values()] for f in src.frames(ep)]
    assert first == again and src.log[ep]["n_clamped"] == src.log[ep]["n_passes"]


def _first_swap(kind: str):
    from arbiter_kb.sources.realistic import MeasuredSwapSource

    base = SyntheticSource(lane_offset=0.04)
    src = MeasuredSwapSource(base, {**_sig(50.0, drop_sw=(0,)), "kind_counts": {kind: 1}})
    ep = base.episodes("train", 1)[0]
    clean, bad = list(base.frames(ep)), list(src.frames(ep))
    t_sw, a, b = src.log[ep]["swaps"][0][:3]
    assert src.log[ep]["swaps"][0][6] == kind
    return clean, bad, t_sw, a, b


def test_measured_merge_puts_two_labels_on_one_container():
    """G: 30 of 130 swaps leave both tracks on one container; the other container is then unreported."""
    clean, bad, t, a, b = _first_swap("merge")
    reported = [k for k in (a, b) if (bad[t].objects[k].xyz == clean[t].objects[a].xyz).all()]
    other = [k for k in (a, b) if (bad[t].objects[k].xyz == clean[t].objects[b].xyz).all()]
    assert sorted([len(reported), len(other)]) == [0, 2]  # both labels report one physical container
    assert all(bad[t - 1].objects[k].xyz is not None for k in (a, b))


def test_measured_lost_moves_one_label_and_hides_the_other():
    """G: 23 of 130 swaps take one track onto the other container while that track is lost."""
    clean, bad, t, a, b = _first_swap("lost")
    vis = [k for k in (a, b) if bad[t].objects[k].visible]
    assert len(vis) == 1
    k = vis[0]
    other = b if k == a else a
    assert (bad[t].objects[k].xyz == clean[t].objects[other].xyz).all()  # it reports the other container
    assert not any(f.objects[other].visible for f in bad[t:])  # and the lost label stays lost
