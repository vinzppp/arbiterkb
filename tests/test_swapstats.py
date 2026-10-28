"""Phase G measurement on a fake perceived episode built from synthetic data: the tracker slots copy
the oracle containers, and in one episode two slots exchange containers at a pass."""
import numpy as np
import pytest

from arbiter_kb import sg_compat as sg
from arbiter_kb.sources.synthetic import SyntheticSource
from arbiter_kb.swapstats import auc, episode_passes, logistic_cv_auc, oracle_passes, separability


def fake_episode(ep: int, swap_at_pass: int | None, lane=0.0):
    """oracle from synthetic frames; perceived = the container rows (slots 0..nb-1, then cubes), with the
    two containers of pass `swap_at_pass` exchanged from its closest approach on."""
    src = SyntheticSource(lane_offset=lane)
    frames = list(src.frames(ep))
    ids = [oid for oid, _, _ in src.meta(ep).objects]
    T, N = len(frames), len(ids)
    orc = np.full((T, N, 8), np.nan, np.float32)
    for t, f in enumerate(frames):
        for n, oid in enumerate(ids):
            o = f.objects[oid]
            orc[t, n, sg.F_ID], orc[t, n, sg.F_LABEL] = n, 0 if o.cls == "container" else 1
            orc[t, n, sg.F_HIDDEN] = 0 if o.visible else 1
            if o.visible:
                orc[t, n, sg.F_U:sg.F_V + 1], orc[t, n, sg.F_X:sg.F_Z + 1] = o.uv, o.xyz
    per = orc.copy()
    nodes = [n for n in range(N) if orc[0, n, sg.F_LABEL] == 0]
    if swap_at_pass is not None:
        p = oracle_passes(orc, nodes)[swap_at_pass]
        a, b, t = p["a"], p["b"], p["t_min"]
        per[t:, [a, b]] = per[t:, [b, a]]
        per[:, [a, b], sg.F_ID] = [a, b]
    box = np.zeros((T, N, 4), np.float32)
    ok = np.isfinite(per[..., sg.F_U]) & (per[..., sg.F_HIDDEN] == 0)
    u, v = per[..., sg.F_U], per[..., sg.F_V]
    box[ok] = np.stack([(v - 12) / 256, (v + 12) / 256, (u - 12) / 256, (u + 12) / 256], -1)[ok]
    return {"perceived": per, "oracle": orc, "bbox": box}


def test_clean_episode_has_passes_and_no_swaps():
    rows, summary = episode_passes(fake_episode(3, None))
    assert summary["n_passes"] >= 1 and summary["n_swaps"] == 0
    assert summary["identity_accuracy"] > 0.99  # lane 0: at contact both slots sit on one point (a tie)


def test_injected_exchange_is_detected_at_that_pass_only():
    ep = fake_episode(3, 0)
    rows, summary = episode_passes(ep)
    assert rows[0]["swapped"] == 1 and sum(r["swapped"] for r in rows[1:]) == 0
    assert summary["identity_accuracy"] < 1.0
    assert rows[0]["min_3d"] < 0.03 and rows[0]["max_iou"] > 0.5  # lane 0: the containers meet


def test_auc_and_logistic_rule():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert auc(np.array([1, 2, 3, 4, 5, 6.0]), y) == 1.0
    assert auc(np.array([6, 5, 4, 3, 2, 1.0]), y) == 0.0
    assert auc(np.array([1, 1, 1, 1, 1, 1.0]), y) == 0.5
    assert np.isnan(auc(np.ones(3), np.zeros(3)))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + 0.1 * rng.normal(size=200) > 0).astype(int)
    assert logistic_cv_auc(X, y) > 0.95


def test_separability_reports_every_feature_and_a_best_rule():
    rows = []
    for ep in range(6):
        rows += episode_passes(fake_episode(ep, 0 if ep % 2 else None))[0]
    out = separability(rows)
    assert set(out["single"]) >= {"min_3d", "max_jump"} and out["best_rule"] is not None


def test_measure_swaps_end_to_end(tmp_path, monkeypatch):
    """The CLI on fake pod outputs: CSV, summary JSON and a gate line."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    d = tmp_path / "perceived"
    d.mkdir()
    for ep in range(6):
        e = fake_episode(ep, 0 if ep % 2 else None)
        np.savez_compressed(d / f"close_ep{ep}.npz", meta=json.dumps({"setting": "close", "seed": ep}), **e)
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, str(root / "scripts/measure_swaps.py"), "--dir", str(d), "--out", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert "GATE: SWAPS EXIST" in out
    summary = json.loads((tmp_path / "g_summary.json").read_text())
    assert summary["settings"][0]["swaps"] == 3 and (tmp_path / "g_passes.csv").exists()


def test_signature_records_swap_timing_and_a_fit():
    from arbiter_kb.swapstats import logistic_fit, signature

    rows = []
    for ep in range(8):
        rows += episode_passes(fake_episode(ep, 0 if ep % 2 else None, lane=0.02))[0]
    sig = signature(rows)
    assert sig["n_swaps"] == 4 and sig["swap_dt"] == [0, 0, 0, 0]  # injected at closest approach
    assert sig["dropout_swapped"] == [0, 0, 0, 0] and len(sig["w"]) == 2
    X = np.linspace(0, 1, 100)[:, None]
    w, b = logistic_fit(X, (X[:, 0] > 0.5).astype(float))
    assert w[0] > 0 and 1 / (1 + np.exp(-(0.9 * w[0] + b))) > 0.9
