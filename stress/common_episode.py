"""MEMI Plan episode loop with a hook for deterministic structured memories.

The executor, capture, text renderer, menus and failure taxonomy are imported
from MEMI's pinned release.  Only the memory-policy hook and audit fields differ.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from common_scene import SceneStream, degraded


def _episode_imports():
    from memi.plan.rollout import episode as ep
    from memi.plan.rollout.capture import NO_RECORD, Capture
    from memi.plan.rollout.video import VideoWriter

    return ep, NO_RECORD, Capture, VideoWriter


def _start(task, episode, split, policy, video_path):
    import robomme.robomme_env  # noqa: F401
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    ep, no_record, capture_cls, video_cls = _episode_imports()
    log = ep.EpisodeLog(task=task, episode=episode, split=split)
    env = BenchmarkEnvBuilder(env_id=task, dataset=split, action_space="multi_choice",
                              gui_render=False).make_env_for_episode(episode)
    cap, film = capture_cls(task, env), video_cls(video_path)
    return ep, no_record, log, env, cap, film


def _reset(ep, no_record, log, env, cap, film, task, episode, split, policy, cfg):
    if not ep.follow_wrapper_rule(env):
        cap._note("no planner wrapper found; need_parameter may disagree")
    cap.install()
    obs, info = env.reset()
    film.add(obs)
    goal = getattr(env.unwrapped, "language_goal", "") or info.get("task_goal", "") or ""
    log.goal = str(goal[0] if isinstance(goal, (list, tuple)) and goal else goal)
    log.menu = ep._menu(info)
    demo = [r for r in cap.rows if r.get("subgoal_online") != no_record] or list(cap.rows)
    n_reset = ep._batch(obs)
    video, first = (demo[:-1], demo[-1:]) if n_reset > 1 and demo else ([], demo[-1:])
    if not first:
        first = [cap.record(env.unwrapped.get_obs(), info, 0, False)]
    raw = video + first
    rows = degraded(raw, cfg["mode"], cfg["seed"], cfg["dropout"], cfg["jitter_px"])
    stream = SceneStream(task, rows, log.goal, episode, split)
    policy.start(task=task, goal=log.goal, menu=log.menu, has_demo=bool(video))
    if hasattr(policy, "reset_memory"):
        policy.reset_memory(stream.meta)
    frames = stream.feed(rows)
    if hasattr(policy, "observe"):
        policy.observe(frames)
    cap.start_execution()
    log.demo_rows, log.reset_batch = len(demo), n_reset
    return obs, info, video, first, rows, stream


def _timeline(ep, task, rows, n_video):
    timeline = ep.EpisodeTimeline(task)
    demo_seg, demo_n = "", 0
    if n_video:
        a, b = timeline.extend(rows[:n_video], demo=True)
        demo_seg, demo_n = timeline.text(a, b)
    start, end = timeline.extend(rows[n_video:], demo=False)
    return timeline, start, end, demo_seg, demo_n


def _classify(ep, d, obs, term, trunc, info, menu):
    info = info or {}
    if info.get("status") == "error":
        msg = str(info.get("error_message", ""))
        d.error = f"{info.get('exception_type', '')}: {msg}"[:400]
        d.result = "api_error" if any(x in msg for x in ep.REJECTED) else (
            "planner_error" if any(x in msg for x in ep.PLANNER) else "harness_error")
        return info
    d.batch = ep._batch(obs)
    if d.batch == 0 and not ep.on_menu(d.command, ep._menu(info) or menu):
        d.result = "api_error"
    elif d.batch == 0:
        d.result = "no_op"
    else:
        d.result, d.terminated = "executed", bool(ep._last(term)) or bool(ep._last(trunc))
        d.status = str(info.get("status", ""))
    return info


def _decision(ep, i, timeline, bounds, demo, log, policy, env, cap, film, stream, cfg):
    start, end, demo_seg, demo_n = bounds
    seg, n = timeline.text(start, end) if end > start else timeline.text(end - 1, end)
    shown = demo_n if i == 0 else 0
    if shown:
        from memi.plan.rollout import segments
        seg = segments.with_demo(demo_seg, seg)
    d = ep.Decision(index=i, t_from=start, t_to=end - 1, steps=n, segment=seg, demo_steps=shown)
    ts, before = time.time(), len(cap.exec_rows)
    d.command = policy.decide(seg, log.menu)
    print(json.dumps({"event": "executor_command", "decision": i, "command": d.command},
                     default=str), flush=True)
    memory = getattr(policy, "memory", None)
    if callable(getattr(memory, "beliefs", None)) and policy.calls:
        policy.calls[-1]["scored_beliefs"] = stream.score_beliefs(memory.beliefs())
    obs, _, term, trunc, info = env.step(d.command)
    film.add(obs)
    info = _classify(ep, d, obs, term, trunc, info, log.menu)
    raw = cap.exec_rows[before:]
    rows = degraded(raw, cfg["mode"], cfg["seed"] + i + 1, cfg["dropout"], cfg["jitter_px"])
    frames = stream.feed(rows)
    if hasattr(policy, "observe"):
        policy.observe(frames)
    d.captured = len(raw)
    start, end = timeline.extend(rows, demo=False)
    d.seconds = round(time.time() - ts, 2)
    log.menu = ep._menu(info) or log.menu
    return d, (start, end, demo_seg, demo_n)


def run_episode(task: str, episode: int, split: str, policy, cfg: dict,
                video_path: Path | None = None):
    ep, no_record, log, env, cap, film = _start(task, episode, split, policy, video_path)
    t0, stream = time.time(), None
    try:
        reset = _reset(ep, no_record, log, env, cap, film, task, episode, split, policy, cfg)
        _, _, video, _, rows, stream = reset
        timeline, start, end, demo_seg, demo_n = _timeline(ep, task, rows, len(video))
        log.demo = {"steps": demo_n, "chars": len(demo_seg)}
        bounds = (start, end, demo_seg, demo_n)
        decision_budget = getattr(policy, "max_decisions", ep.MAX_DECISIONS)
        for i in range(decision_budget):
            d, bounds = _decision(ep, i, timeline, bounds, video, log, policy, env, cap, film, stream, cfg)
            log.decisions.append(d.__dict__)
            if d.result in ("api_error", "planner_error", "harness_error"):
                log.outcome = d.result
                break
            if d.terminated:
                log.outcome = d.status or "terminated"
                break
        else:
            log.outcome = "preflight_complete" if decision_budget < ep.MAX_DECISIONS else "decision_budget"
    except Exception as exc:
        from memi.llm import CallFailed
        if isinstance(exc, CallFailed):
            raise
        log.error, log.outcome = traceback.format_exc()[-3000:], "harness_error"
    finally:
        log.capture_errors, log.policy = list(cap.errors), policy.summary()
        log.seconds, log.observation = round(time.time() - t0, 1), dict(cfg)
        log.arm, log.reset_digest = getattr(policy, "arm", "unknown"), getattr(stream, "digest", "")
        log.success = log.outcome == "success"
        film.close()
        try:
            cap.remove()
            env.close()
        except Exception:
            pass
    return log
