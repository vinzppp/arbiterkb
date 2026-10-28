from arbiter_kb.arbiter import Trigger, validate
from arbiter_kb.kb import KB, AddEvidence
from arbiter_kb.params import ArbiterParams

OBJS = [("bin_1", "container", {}), ("bin_2", "container", {}), ("cube_green", "cube", {"color": "green"})]


def kb_with_evidence():
    kb = KB(OBJS)
    kb.apply([AddEvidence(1, "near_pass", ("bin_1", "bin_2"), {"obs": []}),
              AddEvidence(2, "disappear", ("cube_green",), {"obs": []})])
    return kb


def test_valid_patch():
    patch = {"events": [{"type": "swap", "args": ["bin_1", "bin_2"], "strength": "ambiguous", "cites": ["e1"]}],
             "hypotheses": [{"cube": "cube_green", "keep": ["bin_1", "bin_2"], "cites": ["e1", "e2"]}]}
    assert validate(patch, kb_with_evidence()) == []


def test_invented_evidence_rejected():
    patch = {"events": [{"type": "swap", "args": ["bin_1", "bin_2"], "strength": "clear", "cites": ["e42"]}]}
    assert any("unknown evidence id e42" in e for e in validate(patch, kb_with_evidence()))


def test_coordinates_rejected():
    patch = {"hypotheses": [{"cube": "cube_green", "keep": ["bin_1"], "cites": ["e1"], "xyz": [0.1, 0.2, 0.0]}]}
    errs = validate(patch, kb_with_evidence())
    assert any("coordinate-like key" in e for e in errs) and any("numeric value" in e for e in errs)


def test_trigger_budget():
    kb = kb_with_evidence()
    tr = Trigger(ArbiterParams(every_k=1, max_calls=2))
    fired = [tr.fires(kb, kb.evidence[:1], t) for t in range(5)]
    assert fired.count("near_pass") == 2 and fired[2:] == [None, None, None]
