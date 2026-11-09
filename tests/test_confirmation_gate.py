import copy
import importlib.util
from pathlib import Path


GATE_SPEC = importlib.util.spec_from_file_location(
    "confirmation_gate", Path(__file__).resolve().parents[1] / "scripts/confirmation_gate.py")
GATE = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(GATE)


def spec():
    return {
        "arms": list(GATE.ARMS),
        "tasks": list(GATE.TASKS),
        "params_sha": "params",
        "sources": {"memi": "m", "robomme": "r", "maniskill": "s"},
        "memory_policy": {"memi_prompts_sha256": "prompt-digest-full"},
        "stages": {"executor_preflight": {
            "split": "val", "backend": "scripted", "episodes": [],
            "episodes_by_task": copy.deepcopy(GATE.PREFLIGHT_EPISODES),
        }},
    }


def rows(value):
    result = []
    for task, episode, split, arm in GATE.expected_preflight_keys(value):
        result.append({
            "task": task, "episode": episode, "split": split, "arm": arm,
            "reset_digest": f"reset-{task}",
            "policy": {"model": "none", "effort": "none", "usage": {}, "log": []},
            "provenance": {
                "manifest_sha": "manifest", "arbiter_commit": "commit",
                "arbiter_dirty": False, "sources": value["sources"],
                "params_sha": "params", "prompt_sha": "prompt-digest-fu",
            },
            "terminal": {"status": "task_failure"},
        })
    return result


def issues(value, evidence):
    return GATE.preflight_provenance_issues(value, "manifest", "commit", evidence)


def test_exact_clean_five_row_preflight_is_accepted():
    value = spec()
    assert issues(value, rows(value)) == []


def test_dirty_stale_or_mismatched_preflight_provenance_is_rejected():
    value = spec()
    evidence = rows(value)
    evidence[0]["provenance"].update({
        "manifest_sha": "old-manifest", "arbiter_commit": "old-commit", "arbiter_dirty": True,
        "sources": {"memi": "wrong"}, "params_sha": "wrong", "prompt_sha": "wrong",
    })
    evidence[0]["policy"]["log"] = [{"model_returned": "paid/model"}]
    evidence[0]["reset_digest"] = ""
    found = "\n".join(issues(value, evidence))
    for label in ("manifest SHA", "Arbiter commit", "clean flag", "pinned sources",
                  "params hash", "prompt hash", "zero model calls", "nonempty reset digest"):
        assert label in found


def test_extra_and_duplicate_preflight_rows_are_rejected():
    value = spec()
    evidence = rows(value)
    evidence.append(copy.deepcopy(evidence[0]))
    found = "\n".join(issues(value, evidence))
    assert "exact scheduled five" in found
    assert "duplicate" in found
