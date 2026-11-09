"""No-LLM reset preflight for the x86/CPU RoboMME fallback."""
from __future__ import annotations

import json
import platform
import time

TASKS = ("VideoUnmaskSwap", "ButtonUnmaskSwap", "PickHighlight", "VideoRepick")


def main():
    import robomme.robomme_env  # noqa: F401
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    rows = []
    for task in TASKS:
        started = time.time()
        builder = BenchmarkEnvBuilder(env_id=task, dataset="val", action_space="multi_choice", gui_render=False)
        n = int(builder.get_episode_num())
        env = builder.make_env_for_episode(0)
        obs, info = env.reset()
        rows.append({"task": task, "episodes": n, "reset_s": round(time.time() - started, 2),
                     "goal": str(info.get("task_goal", ""))[:200], "obs_keys": sorted(obs)})
        env.close()
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"platform": platform.platform(), "tasks": rows}, indent=2))


if __name__ == "__main__":
    main()
