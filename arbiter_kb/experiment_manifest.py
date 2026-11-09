"""Load immutable experiment manifests with hash-checked inheritance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _merge(parent: dict, child: dict) -> dict:
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_manifest(path: Path) -> tuple[dict, str]:
    raw = json.loads(path.read_text())
    manifest_sha = digest(path)
    inherited = raw.get("extends")
    if not inherited:
        return raw, manifest_sha
    parent_path = path.parent / str(inherited)
    parent, parent_sha = load_manifest(parent_path)
    if raw.get("parent_manifest_sha") != parent_sha:
        raise ValueError(f"parent manifest is {parent_sha}, expected {raw.get('parent_manifest_sha')}")
    metadata = {key: value for key, value in raw.items()
                if key not in {"extends", "overrides", "parent_manifest_sha"}}
    resolved = _merge(parent, raw.get("overrides") or {})
    resolved.update(metadata)
    resolved["resolved_from"] = [str(parent_path), str(path)]
    resolved["parent_manifest_sha"] = parent_sha
    return resolved, manifest_sha
