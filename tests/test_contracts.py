"""E9: contract field names are frozen; changing them requires bumping CONTRACT_VERSION + a NOTES entry."""
from dataclasses import fields

from arbiter_kb import CONTRACT_VERSION, types

SNAPSHOT = {
    "ObjObs": ["id", "cls", "visible", "xyz", "uv", "track_q"],
    "FrameObs": ["t", "objects", "ee_xyz", "gripper_width", "is_video", "rgb"],
    "EpisodeMeta": ["source", "episode", "split", "task_goal", "objects", "K", "E"],
    "GroundTruth": ["targets", "cube_to_bin", "t_star", "bin_xyz_at_t_star", "moved"],
    "PlanStep": ["skill", "target", "conf", "decision", "support", "waypoints"],
}


def test_contracts_unchanged():
    assert CONTRACT_VERSION == 1
    for name, names in SNAPSHOT.items():
        assert [f.name for f in fields(getattr(types, name))] == names, name


def test_memory_system_protocol():
    from arbiter_kb.systems import MemorySystem
    assert {"reset", "step", "believe_inside", "pose", "plan", "trace_rows"} <= set(dir(MemorySystem))
