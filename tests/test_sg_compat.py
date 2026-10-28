"""sg_compat is a verbatim copy of sg_pilot functions; when the original is present, outputs must match."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from arbiter_kb import sg_compat as sg
from arbiter_kb.sources.synthetic import E_FRONT, K_FRONT

SG = Path(os.environ.get("SG_PILOT_PATH", Path.home() / "Code/active/scene-graph-robot/notebooks/pilot/sg_pilot.py"))


def original():
    if not SG.exists():
        pytest.skip(f"original not found at {SG}")
    spec = importlib.util.spec_from_file_location("sg_pilot", SG)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sg_pilot"] = mod  # dataclasses need the module registered while it executes
    spec.loader.exec_module(mod)
    return mod


def synthetic_graph(T=40, N=3):
    rng = np.random.default_rng(0)
    g = np.full((T, N, 8), np.nan, np.float32)
    g[:, :, sg.F_ID] = np.arange(N)
    g[:, :, sg.F_LABEL] = [0, 0, 1]
    g[:, :, sg.F_X:sg.F_Z + 1] = rng.uniform(-0.1, 0.1, (T, N, 3))
    g[:, :, sg.F_HIDDEN] = 0
    g[20:, 2, sg.F_HIDDEN] = 1
    g[20:, 2, sg.F_X:sg.F_Z + 1] = np.nan
    return g


def test_copy_matches_original():
    orig = original()
    g, E = synthetic_graph(), np.repeat(E_FRONT[None], 40, 0)
    for rule in ("frozen", "bound"):
        a, _ = sg.apply_rule(g, rule, np.full(40, 0.08), np.zeros((40, 3)), K_FRONT, E, sg.RuleParams(r_bind=0.5))
        b, _ = orig.apply_rule(g, rule, np.full(40, 0.08), np.zeros((40, 3)), K_FRONT, E, orig.RuleParams(r_bind=0.5))
        assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))
    assert sg.wilson(3, 10) == orig.wilson(3, 10)


def test_projection_roundtrip():
    xyz = np.array([[0.05, -0.02, 0.03]])
    uv = sg.project_points(xyz, K_FRONT, E_FRONT)
    assert np.isfinite(uv).all() and (0 < uv).all() and (uv < 256).all()


def test_assign_slots_seeded_matches_original():
    orig = original()
    g = synthetic_graph()
    g[:, :, sg.F_U:sg.F_V + 1] = np.random.default_rng(1).uniform(0, 256, (40, 3, 2))
    g[:5, 1, sg.F_HIDDEN] = 1
    per = g.copy()
    per[:, :, sg.F_U:sg.F_V + 1] += 3.0
    assert sg.assign_slots_seeded(per, g) == orig.assign_slots_seeded(per, g)
