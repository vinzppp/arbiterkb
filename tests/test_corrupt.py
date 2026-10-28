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
