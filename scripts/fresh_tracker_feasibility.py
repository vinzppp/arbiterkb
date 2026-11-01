"""Fail-closed Phase-0 audit for the fresh tracker-error replay study.

The audit reads metadata and filenames only.  In particular, it never opens a
G held-out npz or a RoboMME episode/video payload.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "robomme_common" / "fresh_tracker_error_replay_v1"
G_NAME = re.compile(r"_ep(\d{3})\.npz$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for p in path.rglob(pattern) if p.is_file()) if path.exists() else 0


def g_inventory(data: Path) -> dict:
    names = sorted(p.name for p in (data / "g_perceived").glob("*.npz"))
    suffixes = [int(m.group(1)) for n in names if (m := G_NAME.search(n))]
    return {
        "total_filename_inventory": len(names),
        "development_0_9": sum(i < 10 for i in suffixes),
        "heldout_10_19": sum(i >= 10 for i in suffixes),
        "payloads_opened": 0,
        "eligibility": "ineligible: development data used for rule work; heldout forbidden",
    }


def robomme_inventory(data: Path) -> dict:
    base = data / "robomme_lerobot"
    source_path, info_path = base / "SOURCE.json", base / "meta" / "info.json"
    source = json.loads(source_path.read_text())
    info = json.loads(info_path.read_text())
    families = {}
    for path in (base / "npz").glob("*.npz"):
        family = path.stem.rsplit("_", 1)[0]
        families[family] = families.get(family, 0) + 1
    return {
        "source": source,
        "source_metadata_sha256": {"SOURCE.json": sha256(source_path), "meta/info.json": sha256(info_path)},
        "declared_splits": info.get("splits"),
        "declared_total_episodes": info.get("total_episodes"),
        "declared_features": sorted(info.get("features", {})),
        "local_episode_npz_filenames": sum(families.values()),
        "local_episode_families": dict(sorted(families.items())),
        "local_extracted_frame_npz_filenames": count_files(base / "frames", "*.npz"),
        "local_mp4_files": count_files(base / "videos", "*.mp4"),
        "payloads_opened": 0,
        "eligibility": "ineligible: public train split and previously used in repository evaluations",
    }


def runtime_inventory() -> dict:
    modules = ["torch", "sg_pilot", "groundingdino", "segment_anything", "cutie", "cv2", "av"]
    found = {name: importlib.util.find_spec(name) is not None for name in modules}
    cuda = False
    if found["torch"]:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(),
        "host_modules": found,
        "host_cuda_available": cuda,
        "workspace_sg_exists": Path("/workspace/sg").is_dir(),
        "existing_cpu_image_manual_audit": {
            "image": "arbiterkb-robomme-cpu:bb6c37c",
            "platform_warning": "linux/amd64 image on linux/arm64 host",
            "cuda": False,
            "sg_pilot": False,
            "groundingdino": False,
            "segment_anything": False,
            "cutie": False,
            "cv2": True,
        },
        "frozen_perception_runnable_here": False,
    }


def precision() -> dict:
    z = 1.959963984540054
    out = {}
    for discordance in (0.3, 1.0):
        n = 30
        half = z * math.sqrt(discordance / n)
        needed = math.ceil(z * z * discordance / 0.1**2)
        out[str(discordance)] = {"n": n, "half_width_approx": half, "n_for_half_width_0_10": needed}
    return out


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def phase0_audit(data: Path, branch: str, head: str, main_ref: str) -> dict:
    return {
        "schema": 1,
        "phase": "phase0_feasibility",
        "decision": "INCONCLUSIVE_STOP_BEFORE_SCORING",
        "branch": branch,
        "source_head": head,
        "main_ref": main_ref,
        "main_ref_required": "069beb7688fe1b853dd1fcc09ba6e3e481372f87",
        "data_root": str(data),
        "g_perceived": g_inventory(data),
        "robomme_lerobot": robomme_inventory(data),
        "prior_closed_loop": {
            "result_files": count_files(data / "cl_pod", "result.json"),
            "raw_common_observation_traces_available": False,
            "eligibility": "ineligible: previously evaluated and not a fresh paired raw-trace batch",
        },
        "runtime": runtime_inventory(),
        "gates": {
            "fresh_visual_material": False,
            "actual_frozen_tracker_runnable": False,
            "legitimate_nonleaking_ground_truth": False,
            "zero_prior_overlap": False,
            "enough_natural_errors": False,
        },
        "heldout_records_opened": False,
        "heldout_filename_inventory_only": True,
        "comparative_outcomes_examined": False,
    }


def write_blocked_outputs(out: Path) -> None:
    blocked = {"schema": 1, "status": "not_run_phase0_blocked", "episodes": 0, "decision_points": 0}
    write_json(out / "perception_tracking_error_stats.json", {
        **blocked,
        "detector_failures": 0,
        "tracker_association_failures": 0,
        "occlusion_ambiguities": 0,
        "genuine_identity_ambiguities": 0,
        "memory_update_errors": 0,
        "note": "zeros mean no eligible observations were processed, not zero error prevalence",
    })
    write_json(out / "paired_memory_action_comparison.json", {
        **blocked,
        "primary_effect": None,
        "ledger": None,
        "arbiterkb": None,
        "paired_disagreements": None,
        "reason": "no eligible fresh common observation stream",
    })
    write_json(out / "episode_cluster_uncertainty.json", {
        **blocked,
        "estimate": None,
        "confidence_interval": None,
        "precision_planning": precision(),
        "sampling_unit": "episode",
        "provisional_30_episode_batch_is_confirmatory": False,
    })
    write_json(out / "runtime_cost.json", {
        "schema": 1,
        "new_simulator_rollouts": 0,
        "new_scored_episodes": 0,
        "new_llm_calls": 0,
        "new_api_cost_usd": 0.0,
        "paid_gpu_hours": 0.0,
        "heldout_records_opened": False,
        "observation_degradation_run": False,
    })


def write_repro(out: Path, branch: str, head: str, main_ref: str) -> None:
    sources = [
        "experiments/fresh_tracker_error_acquisition_v1.json",
        "scripts/fresh_tracker_feasibility.py",
        "stress/perceive.py",
        "arbiter_kb/relink.py",
        "scripts/canonical_identifiability.py",
        "params.json",
        "paper/FRESH_TRACKER_ERROR_REPLAY_PROTOCOL_ZH.md",
        "paper/FRESH_TRACKER_ERROR_REPLAY_REPORT_ZH.md",
        "paper/ROBOMME_CANONICAL_MEMORY_IDENTIFIABILITY_ZH.md",
        "paper/ROBOMME_CANONICAL_MEMORY_PROTOCOL_ZH.md",
        "paper/ROBOMME_V2_1_FAILURE_FORENSICS_ZH.md",
        "paper/ROBOMME_MEMORY_POLICY_FACTORIAL_PILOT_ZH.md",
    ]
    outputs = [
        "phase0_feasibility_audit.json",
        "perception_tracking_error_stats.json",
        "paired_memory_action_comparison.json",
        "episode_cluster_uncertainty.json",
        "runtime_cost.json",
        "validation.json",
    ]
    write_json(out / "reproducibility_manifest.json", {
        "schema": 1,
        "protocol": "fresh-tracker-error-replay-v1-phase0",
        "source_head": head,
        "branch": branch,
        "main_ref": main_ref,
        "artifact_sha256": {p: sha256(ROOT / p) for p in sources},
        "output_sha256": {p: sha256(out / p) for p in outputs if (out / p).exists()},
        "data_payload_hashes": {},
        "data_payload_hash_note": "heldout and ineligible episode payloads were not opened or hashed",
        "execution": {
            "heldout_records_opened": False,
            "new_perception_runs": 0,
            "new_memory_replays": 0,
            "new_api_cost_usd": 0.0,
        },
        "reproduce": "PYTHONPATH=. .venv/bin/python scripts/fresh_tracker_feasibility.py",
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = (ROOT / "data").resolve()
    branch = git("branch", "--show-current")
    head, main_ref = git("rev-parse", "HEAD"), git("rev-parse", "main")
    audit = phase0_audit(data, branch, head, main_ref)
    write_json(args.out / "phase0_feasibility_audit.json", audit)
    write_blocked_outputs(args.out)
    write_repro(args.out, branch, head, main_ref)
    if branch != "research/perception-belief-robomme" or main_ref != audit["main_ref_required"]:
        raise SystemExit("fail closed: branch or main ref mismatch")
    print(json.dumps({"decision": audit["decision"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
