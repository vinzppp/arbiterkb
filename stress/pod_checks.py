"""The five free checks before B0 (docs/TEST_PLAN_COMPONENTS.md §6). ON THE POD, sg-main venv; no LLM, no perception.

  python stress/pod_checks.py --out /workspace/cl/pod_checks.json

1. actors: every attribute of env.unwrapped with a pose (name, position, yaw about z), and the scene's actor list;
2. grasp: whether the agent has is_grasping(actor), and what it returns at reset;
3. state: get_state_dict / set_state_dict round trip (truth after restore = truth before the step; the same action
   from the restored state gives the same truth), and the replay fallback (fresh env + the same action);
4. pick: arbiter_kb.skills.pick from true poses on VideoUnmaskSwap val ep 0, every container, fresh env each: status per
   waypoint, frames, grasp, lift;
5. solvers: functions named solve* in the installed robomme package (file:line), the dataset generator's skills.
Every check is wrapped: one failure is reported and the rest still run. Simulator truth here is never shown to a model.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "stress")]

from arbiter_kb.skills import pick  # noqa: E402

TASKS = ("VideoUnmaskSwap", "VideoPlaceOrder")


def make_env(task: str, episode: int, dataset: str = "val"):
    """closed_loop_pod's env (waypoint, 1300 steps), without perception's extras."""
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    return BenchmarkEnvBuilder(env_id=task, dataset=dataset, action_space="waypoint", gui_render=False,
                               max_steps=1300).make_env_for_episode(episode)


def _np(x):
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return x[0] if x.ndim == 2 else x


def yaw(q) -> float:
    """Yaw about world z from a ManiSkill quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in _np(q))
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def actors(u) -> dict:
    """{name: {p, yaw, attr}} for every attribute of the unwrapped env that has a pose (the agent excluded), keyed by
    the actor's own name (cube_red_0, target_0: the colour is there, not in the attribute; layer-1 listing on the
    pod); one actor reached through two attributes is kept once."""
    out = {}
    for name, v in vars(u).items():
        if name.startswith("_") or name == "agent":
            continue
        for k, a in ([(name, v)] if hasattr(v, "pose") else
                     [(f"{name}[{i}]", a) for i, a in enumerate(v)] if isinstance(v, (list, tuple)) else []):
            key = getattr(a, "name", None) or k
            if key in out:
                continue
            try:
                out[key] = {"p": np.round(_np(a.pose.p), 4).tolist(), "yaw": round(yaw(a.pose.q), 3),
                            "type": type(a).__name__, "attr": k}
            except Exception:  # noqa: BLE001 (an attribute with a non-actor pose)
                pass
    return out


def resolve(u, attr: str):
    """The actor behind an `actors` entry's attr ("spawned_bins[2]" -> u.spawned_bins[2])."""
    base, _, idx = attr.partition("[")
    v = getattr(u, base)
    return v[int(idx.rstrip("]"))] if idx else v


def truth(u) -> dict:
    return {k: v["p"] + [v["yaw"]] for k, v in actors(u).items()}


def gap(a: dict, b: dict) -> float:
    """Largest |difference| over the shared actors' (x, y, z, yaw)."""
    keys = a.keys() & b.keys()
    return max((float(np.max(np.abs(np.subtract(a[k], b[k])))) for k in keys), default=float("nan"))


def check_actors(out: dict) -> None:
    for task in TASKS:
        env = make_env(task, 0)
        _, info = env.reset()
        u = env.unwrapped
        scene = getattr(getattr(u, "scene", None), "actors", None)
        out[task] = {"goal": info.get("task_goal"), "actors": actors(u),
                     "scene_actors": sorted(scene) if isinstance(scene, dict) else repr(type(scene)),
                     "info_keys": sorted(info), "wrappers": _wrappers(env)}
        env.close()


def _wrappers(env) -> list:
    names, e = [], env
    while e is not None:
        names.append(type(e).__name__)
        e = getattr(e, "env", None)
    return names


def check_grasp(out: dict) -> None:
    env = make_env("VideoUnmaskSwap", 0)
    env.reset()
    u = env.unwrapped
    agent = getattr(u, "agent", None)
    out["has_is_grasping"] = hasattr(agent, "is_grasping")
    if out["has_is_grasping"]:
        out["signature"] = str(inspect.signature(agent.is_grasping))
        out["at_reset"] = {}
        for name, e in actors(u).items():
            a = resolve(u, e["attr"])
            if a is not None:
                try:
                    out["at_reset"][name] = bool(_np(agent.is_grasping(a)).reshape(-1)[0])
                except Exception as ex:  # noqa: BLE001
                    out["at_reset"][name] = repr(ex)
    env.close()


def hover(obs) -> np.ndarray:
    """A small move: 5 cm above the current TCP, same orientation, gripper kept."""
    tcp = np.asarray(_np(obs["eef_state_list"][-1]), float).reshape(-1)[:6]
    return np.array([*tcp[:2], tcp[2] + 0.05, *tcp[3:6], 0.0])


def check_state(out: dict) -> None:
    env = make_env("VideoUnmaskSwap", 0)
    obs, _ = env.reset()
    u = env.unwrapped
    out["has_get_state_dict"] = hasattr(u, "get_state_dict")
    a = hover(obs)
    t0 = truth(u)
    if out["has_get_state_dict"]:
        s = u.get_state_dict()
        o1 = env.step(a)[0]
        t1 = truth(u)
        u.set_state_dict(s)
        out["restore_gap_to_before"] = gap(t0, truth(u))
        o2 = env.step(a)[0]
        out["same_action_gap_after_restore"] = gap(t1, truth(u))
        out["frames_first_vs_restored"] = [len(o1.get("front_rgb_list") or []), len(o2.get("front_rgb_list") or [])]
        out["moved_by_action"] = gap(t0, t1)
    env.close()
    env = make_env("VideoUnmaskSwap", 0)  # replay fallback: a fresh env, the same action
    env.reset()
    env.step(a)
    out["replay_gap"] = gap(t1, truth(env.unwrapped)) if out["has_get_state_dict"] else "no reference"
    env.close()


def check_pick(out: dict) -> None:
    probe = make_env("VideoUnmaskSwap", 0)
    probe.reset()
    names = [k for k in actors(probe.unwrapped) if re.search(r"bin|container", k, re.I)]
    probe.close()
    out["containers"] = names
    for name in names:
        env = make_env("VideoUnmaskSwap", 0)  # only the first reset gives the episode's scene
        env.reset()
        u = env.unwrapped
        x, y, z, yw = truth(u)[name]
        agent, a = getattr(u, "agent", None), resolve(u, actors(u)[name]["attr"])
        rows = []
        for wp in pick(np.array([x, y, z]), yw):
            o, _, term, trunc, info = env.step(wp)
            row = {"wp": np.round(wp, 3).tolist(), "frames": len((o or {}).get("front_rgb_list") or []),
                   "status": (info or {}).get("status"), "error": (info or {}).get("error_message"),
                   "obj_z": round(float(_np(a.pose.p)[2]), 4)}
            if hasattr(agent, "is_grasping"):
                row["grasping"] = bool(_np(agent.is_grasping(a)).reshape(-1)[0])
            rows.append(row)
            if o is None or row["status"] == "error":
                break
        out[name] = {"start_z": round(z, 4), "yaw": round(yw, 3), "steps": rows}
        env.close()


def check_solvers(out: dict) -> None:
    import robomme
    root = Path(robomme.__file__).parent
    out["robomme_root"] = str(root)
    out["solvers"] = [f"{p.relative_to(root)}:{i}: {line.strip()}" for p in sorted(root.rglob("*.py"))
                      for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1)
                      if re.match(r"\s*def solve\w*\(", line)]
    out["multi_choice_files"] = sorted({str(p.relative_to(root)) for p in root.rglob("*.py")
                                        if "multi_choice" in p.read_text(errors="ignore")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/cl/pod_checks.json")
    ap.add_argument("--only", default="actors,grasp,state,pick,solvers")
    a = ap.parse_args()
    checks = {"actors": check_actors, "grasp": check_grasp, "state": check_state, "pick": check_pick,
              "solvers": check_solvers}
    report = {}
    for name in a.only.split(","):
        report[name] = {}
        try:
            checks[name](report[name])
        except Exception:  # noqa: BLE001 (report it; the other checks still run)
            report[name]["failed"] = traceback.format_exc()
        print(name, json.dumps(report[name], default=str)[:2000], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
