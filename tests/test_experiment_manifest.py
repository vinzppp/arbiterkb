import json

import pytest

from arbiter_kb.experiment_manifest import digest, load_manifest


def test_manifest_inheritance_is_hash_checked_and_deep_merged(tmp_path):
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"watchdog": {"reset": 1, "episode": 2}, "model": "fixed"}))
    child = tmp_path / "child.json"
    child.write_text(json.dumps({"extends": "parent.json", "parent_manifest_sha": digest(parent),
                                 "overrides": {"watchdog": {"reset": 3}}, "name": "child"}))
    resolved, child_sha = load_manifest(child)
    assert resolved["watchdog"] == {"reset": 3, "episode": 2}
    assert resolved["model"] == "fixed"
    assert resolved["name"] == "child"
    assert child_sha == digest(child)


def test_manifest_rejects_wrong_parent_hash(tmp_path):
    parent = tmp_path / "parent.json"
    parent.write_text("{}")
    child = tmp_path / "child.json"
    child.write_text(json.dumps({"extends": "parent.json", "parent_manifest_sha": "wrong"}))
    with pytest.raises(ValueError, match="parent manifest"):
        load_manifest(child)
