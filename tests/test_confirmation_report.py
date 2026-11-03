import pytest

from scripts.confirmation_report import (classify_h1, classify_h3, find_pair_drops,
                                         sign_flip, stratified_ci)


def test_sign_flip_is_seeded_and_two_sided():
    values = [1.0] * 20
    assert sign_flip(values) == sign_flip(values)
    assert sign_flip(values) < .001
    assert sign_flip([-value for value in values]) == sign_flip(values)


def test_stratified_ci_weights_tasks_equally():
    values = {("a", i): 1.0 for i in range(2)} | {("b", i): 0.0 for i in range(8)}
    point, low, high = stratified_ci(values, n_boot=1000)
    assert point == pytest.approx(.5)
    assert low == pytest.approx(.5)
    assert high == pytest.approx(.5)


def test_preregistered_readings():
    assert classify_h1([.12, .10, .14], .01) == "supported"
    assert classify_h1([.05, .01, .09], .5) == "falsified"
    assert classify_h1([.09, .05, .13], .01) == "inconclusive"
    assert classify_h3([.03, .01, .05], .01) == "supported"
    assert classify_h3([-.04, -.08, -.01], .2) == "falsified"


def test_pair_drop_audit_uses_preregistered_directory(tmp_path):
    drop = tmp_path / "_pair_drops" / "StopCube_ep7.json"
    drop.parent.mkdir()
    drop.write_text("{}\n")
    (tmp_path / "unrelated" / "pair_drop.json").parent.mkdir()
    (tmp_path / "unrelated" / "pair_drop.json").write_text("{}\n")
    assert find_pair_drops(tmp_path) == [drop]
