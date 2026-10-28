import hashlib

import pytest

from arbiter_kb.evaluate import run, save_run
from arbiter_kb.params import Params
from arbiter_kb.sources.synthetic import SyntheticSource
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem
from arbiter_kb.systems.baselines import GuardedBound, TrackerSystem
from arbiter_kb.systems.soft_haircut import SoftHaircut


class PoisonedSource:
    """Raises if frame t+1 is requested before every system finished step(t) (E5a)."""

    def __init__(self, inner, systems):
        self.inner, self.systems = inner, systems

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def frames(self, ep):
        for f in self.inner.frames(ep):
            assert all(getattr(s, "t", f.t - 1) == f.t - 1 for s in self.systems), "future frame requested early"
            yield f


def systems():
    return [TrackerSystem("frozen"), TrackerSystem("bound"), GuardedBound("haircut"), GuardedBound("overlap_haircut"),
            GuardedBound("jump"), SoftHaircut(), ArbiterKBSystem()]


def test_causality_and_no_ground_truth_access():
    sy = systems()
    src = PoisonedSource(SyntheticSource(), sy)
    rows, _, _ = run(sy, src, src.episodes("train", 2), Params(), "none")
    assert len(rows) == 2 * len(sy)
    for s in sy:
        assert not any(hasattr(s, a) for a in ("gt", "ground_truth")), "systems must never hold GroundTruth"


def test_test_split_refuses_in_code_params():
    src = SyntheticSource()
    for p in (Params(), Params(frozen=True)):  # built in code: no origin, so never allowed on test
        with pytest.raises(RuntimeError):
            run(systems(), src, src.episodes("test", 1), p, "none")


def test_test_split_needs_freeze_before_read(tmp_path):
    from arbiter_kb import ledger
    src = SyntheticSource()
    Params(frozen=True, note="t").save(tmp_path / "params.json")
    p = Params.load(tmp_path / "params.json")
    with pytest.raises(RuntimeError):  # loaded and frozen, but never logged as FREEZE
        run(systems(), src, src.episodes("test", 1), p, "none")
    ledger.record("FREEZE", p.hash(), "test")
    rows, _, _ = run(systems(), src, src.episodes("test", 1), p, "none")
    assert rows and f"READ\t{p.hash()}" in ledger.path().read_text()


def test_hand_edited_params_rejected(tmp_path):
    Params(frozen=True).save(tmp_path / "params.json")
    text = (tmp_path / "params.json").read_text().replace('"w_max": 0.5', '"w_max": 0.9')
    (tmp_path / "params.json").write_text(text)
    with pytest.raises(ValueError):
        Params.load(tmp_path / "params.json")


def test_reproducible_summary(tmp_path):
    src = SyntheticSource()
    hashes = []
    for i in range(2):
        rows, tr, _ = run(systems(), src, src.episodes("train", 4), Params(), "none")
        save_run(tmp_path / str(i), rows, Params(), {}, tr)
        hashes.append(hashlib.sha256((tmp_path / str(i) / "summary.csv").read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


def test_every_arbiterkb_row_has_support():
    src = SyntheticSource()
    rows, _, _ = run(systems(), src, src.episodes("train", 3), Params(), "none")
    assert all(r["n_support"] > 0 for r in rows if r["system"] == "arbiterkb")


def test_aurc_and_coverage():
    from arbiter_kb.metrics import aurc, summarize
    r = lambda p, ok: {"p_max": p, "correct": ok}  # noqa: E731
    assert aurc([r(0.9, True), r(0.6, False)]) == pytest.approx((0 + 0.5) / 2)  # the error answered last
    assert aurc([r(0.9, False), r(0.6, True)]) == pytest.approx((1 + 0.5) / 2)  # the error answered first
    assert aurc([r(0.5, False), r(0.5, True)]) == pytest.approx(0.5)  # ties enter together
    rows = [{**r(p, ok), "condition": "c", "system": "s", "brier": 0, "moved": False, "corrupted": False,
             "grasp_err_m": 0.0} for p, ok in [(0.9, True), (0.7, True), (0.95, False)]]
    s = summarize(rows, 0.8)[0]
    assert s["coverage"] == pytest.approx(2 / 3, abs=1e-4) and s["silent"] == pytest.approx(1 / 3, abs=1e-4)


def test_coverage_counts_p_exactly_tau_hi():
    from arbiter_kb.metrics import summarize
    rows = [{"p_max": 0.8, "correct": True, "condition": "c", "system": "s", "brier": 0, "moved": False,
             "corrupted": False, "grasp_err_m": 0.0}]
    assert summarize(rows, 0.8)[0]["coverage"] == 1.0
