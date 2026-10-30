"""Free simulation of the provider's prompt cache for the closed loop's memory arm K (NOTES "K's request layout and the
provider's prompt cache"). No LLM call and no pod: the requests are rebuilt on the Mac.

Requests: G's perceived VideoUnmaskSwap opening videos (data/g_perceived, real perception) fed to ArbiterKBSystem, then
a scripted two-pick robot continuation (above bin A, close, lift, put down and open, above bin B, close, lift, done),
eight calls per episode. During the continuation every visible object gets N(0, 3 mm) noise per frame (perception is
not still), a held bin follows the hand, and the cube under a lifted bin becomes visible. Before each call the K
memory text is built exactly as closed_loop.Memory.text() builds it, and after it the decision is recorded as
closed_loop.Episode.execute() does. The two current frames are stand-ins: they change every call, so they never
match a cached prefix, and their token count is a parameter (--img-tokens; not known for GPT-6-Astra).

Cache model (OpenAI, GPT-5.6 and later; developers.openai.com/api/docs/guides/prompt-caching and
openrouter.ai/docs/features/prompt-caching, read 2026-10-30): a prefix is cached only at a breakpoint, which is the
end of a message (implicit mode) or a marked content block (explicit mode); reuse needs the whole rendered prefix up
to a breakpoint to match; the shortest cacheable prefix is 1,024 tokens; images are part of the prefix. Price as a
multiple of the uncached input rate: a cached read 0.1, a write 1.25 (the 30-minute TTL), anything else 1.0. In
implicit mode every request writes up to its last message end; in explicit mode only up to its last marked block.
Assumed, not documented: a later request reads the longest earlier breakpoint prefix that it starts with, and every
call of an episode lands on a machine holding that episode's earlier writes (all calls go to one provider; an episode's
calls are under a minute apart, well inside the TTL). Tokens: o200k_base (tiktoken), a proxy for the model's own
tokenizer; chat-template tokens per message are left out. tiktoken is not in the project's dependency whitelist, hence
tools/ (outside scripts/quality.py), run with a Python that has it and the project's requirements:

  PYTHONPATH=. python tools/cache_sim.py --episodes 10 --img-tokens 66
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter_kb.closed_loop import Memory  # noqa: E402
from arbiter_kb.controller import system_prompt, user_content  # noqa: E402
from arbiter_kb.params import Params  # noqa: E402
from arbiter_kb.self_memory import record  # noqa: E402
from arbiter_kb.sources.perceived import PerceivedSource  # noqa: E402
from arbiter_kb.systems.arbiter_kb import ArbiterKBSystem  # noqa: E402
from arbiter_kb.types import ObjObs  # noqa: E402
from arbiter_kb.self_memory import changes  # noqa: E402
from arbiter_kb.views import event_log  # noqa: E402

MIN_PREFIX, READ, WRITE = 1024, 0.1, 1.25
OPEN, CLOSED = 0.08, 0.01  # gripper width, m
# The camera calibration as the state carries it (fixed for an episode); the numbers are stand-ins of the same shape.
CALIB = {"wrist_camera_K": [[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]],
         "front_camera_K": [[221.7, 0.0, 128.0], [0.0, 221.7, 128.0], [0.0, 0.0, 1.0]],
         "front_camera_extrinsic_world_to_camera": [[0.0, 1.0, 0.0, 0.0], [0.5547, 0.0, -0.8321, 0.2219],
                                                    [-0.8321, 0.0, -0.5547, 1.2481]]}


def tokens(s: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(s, disallowed_special=()))


class Robot:
    """The scripted continuation after G's opening video: frames of each motion, every third fed to the KB."""

    def __init__(self, last, rng, reveal: dict):
        self.objs = {k: o for k, o in last.objects.items()}
        self.pos = {k: (o.xyz.copy() if o.xyz is not None else None) for k, o in self.objs.items()}
        self.ee, self.width, self.held, self.t = np.array([0.0, 0.0, 0.45]), OPEN, None, last.t
        self.rng, self.reveal = rng, reveal  # reveal: {bin: (cube id, xyz)}

    def frames(self, to, gripper: int, n: int = 30):
        """Move the hand to `to` in n frames; then apply the gripper command (the command acts at the end)."""
        from arbiter_kb.types import FrameObs
        start, out = self.ee.copy(), []
        for k in range(1, n + 1):
            self.ee = start + (np.asarray(to, float) - start) * k / n
            if k == n and gripper:
                self.width = CLOSED if gripper < 0 else OPEN
                if gripper < 0:
                    self.held = min((b for b in self.pos if b.startswith("bin_") and self.pos[b] is not None),
                                    key=lambda b: np.linalg.norm(self.pos[b][:2] - self.ee[:2]))
                else:
                    self.held = None
            if self.held:
                self.pos[self.held] = self.ee.copy()
                if self.ee[2] > 0.08 and self.held in self.reveal:  # the cube under a lifted bin is seen
                    c, p = self.reveal.pop(self.held)
                    self.pos[c] = p
            self.t += 1
            objs = {}
            for oid, o in self.objs.items():
                p = self.pos[oid]
                vis = p is not None and (o.cls != "cube" or oid in self.revealed())
                xyz = p + self.rng.normal(0, 0.003, 3) if vis else None
                objs[oid] = ObjObs(oid, o.cls, vis, xyz, None, 1.0)
            if k % 3 == 0 or k == n:
                out.append(FrameObs(self.t, objs, self.ee.copy(), self.width, False))
        return out

    def revealed(self) -> set:
        return {c for c in self.pos if c.startswith("cube_") and self.pos[c] is not None}


def episode(src: PerceivedSource, ep: int, rng) -> list[dict]:
    """Eight calls of one episode: per call the K request's parts as text, and the decision taken."""
    d = src.load(ep)
    meta = src.meta(ep)
    colors = d["meta"]["colors"][:2]
    goal = f"pick up the container that hides the {colors[0]} cube, then the container that hides the {colors[1]} cube"
    meta.task_goal = goal
    names = src.ids(d)
    slot = {n[0]: s for s, n in enumerate(names)}
    c2b = {f"cube_{names[int(c)][0][5:]}": f"bin_{b}" for c, b in d["meta"]["cube_to_bin"].items()}
    system = ArbiterKBSystem(Params())
    frames = list(src.frames(ep))
    mem = Memory("K", system, lambda obs: [])
    system.reset(meta)
    for f in frames:
        mem.frames.append(f)
        system.step(f)
    last = frames[-1]
    ora = d["oracle"][-1]
    bins = {b: ora[slot[b], 4:7].astype(float) for b in set(c2b.values())}
    robot = Robot(last, rng, {b: (c, np.array([*bins[b][:2], 0.017])) for c, b in c2b.items()})
    A, B = c2b[f"cube_{colors[0]}"], c2b[f"cube_{colors[1]}"]
    pa, pb = bins[A], bins[B]
    side = np.array([pa[0] - 0.05, pa[1] + (0.15 if pa[1] < 0.1 else -0.15), 0.035])
    plan = [(A, [*pa[:2], 0.17], 1), (A, [*pa[:2], 0.035], -1), (A, [*pa[:2], 0.22], 0), (A, list(side), 1),
            (B, [*pb[:2], 0.17], 0), (B, [*pb[:2], 0.035], -1), (B, [*pb[:2], 0.22], 0), (None, None, 0)]
    calls = []
    for target, wp, g in plan:
        mem_text = mem.text()
        state = {"tcp_pose_xyz_rpy": [round(float(v), 4) for v in robot.ee] + [-3.1416, 0.0, 0.0],
                 "finger_positions_m": [robot.width / 2] * 2, "elapsed_control_steps": int(robot.t * 3),
                 "wrist_camera_K": CALIB["wrist_camera_K"],
                 "wrist_camera_extrinsic_world_to_camera": [[0.0, -1.0, 0.0, round(float(robot.ee[1]), 4)],
                                                            [-1.0, 0.0, 0.0, round(float(robot.ee[0]), 4)],
                                                            [0.0, 0.0, -1.0, round(float(robot.ee[2]) + 0.1, 4)]],
                 "front_camera_K": CALIB["front_camera_K"],
                 "front_camera_extrinsic_world_to_camera": CALIB["front_camera_extrinsic_world_to_camera"]}
        ds = system.kb.task.get("decisions", [])
        calls.append({"goal": goal, "memory": mem_text, "state": state, "events": event_section(mem_text),
                      "full_log": event_log(system.kb, last=10 ** 9, low_level=False),
                      "changes": changes(system.kb, ds[-1]) if ds else []})
        if wp is None:
            break
        record(system.kb, mem.frames[-1].t, np.array(wp + [-3.1416, 0.0, 0.0]), g < 0, target, ())
        for f in robot.frames(wp, g):
            mem.frames.append(f)
            system.step(f)
    return calls


def event_section(view: str) -> str:
    """The event log part of the K view (from its heading to the next section or the end)."""
    lines, out, on = view.split("\n"), [], False
    for x in lines:
        if x.startswith("What the robot has done") or x.startswith("Event log"):
            on = True
        elif on and x and not x.startswith(" "):
            on = False
        if on:
            out.append(x)
    return "\n".join(out)


# ---- request layouts: a segment = (key, tokens); equal keys = byte-identical content ----------------------------------

def layouts(calls: list[dict], img: int, answer: int) -> dict:
    """Each layout -> list of requests; a request = list of (segments, marked[, ends a message]): implicit mode breaks
    at every message end, explicit mode only at marked blocks (here a marked block is a whole message unless the
    third field is False)."""
    sysk = system_prompt("K")
    S = ("sys", tokens(sysk))
    out = {k: [] for k in ("current", "a_reorder", "fixed_first", "b_append", "c_conversation")}
    convo, seen_lines, appended = [], set(), []
    for i, c in enumerate(calls):
        mem, goal = c["memory"], c["goal"]
        st_all = "Robot state: " + json.dumps(c["state"])
        dyn = {k: v for k, v in c["state"].items() if k not in CALIB}
        st_dyn = "Robot state: " + json.dumps(dyn)
        calib = user_content("K", goal, "", [], c["state"], "fixed_first")[1]["text"]  # the runner's fixed block
        goal_t = f"Task goal: {goal}"
        T = {"mem": tokens(mem), "goal": tokens(goal_t), "state": tokens(st_all), "state_dyn": tokens(st_dyn),
             "calib": tokens(calib), "events": tokens(c["events"]), "labels": tokens("Current front camera:")
             + tokens("Current wrist camera:")}
        frames = (f"frames{i}", 2 * img + T["labels"])
        mem_seg = (f"mem{i}", T["mem"])
        # current: system, then one user message: goal (when the view does not start with it), view, frames, state
        head = [] if mem.startswith("Goal:") else [(f"goal{i}", T["goal"])]  # its text changes position, so unique
        out["current"].append([([S], False), (head + [mem_seg, frames, (f"state{i}", T["state"])], False)])
        # a_reorder: the same text, reordered only: the goal first (the only part that is the same at every call)
        out["a_reorder"].append([([S], True), ([("goal", T["goal"])], True, False),
                                 ([mem_seg, frames, (f"state{i}", T["state"])], False)])
        # fixed_first (controller.user_content): as a_reorder, and the state's camera calibration joins the goal
        out["fixed_first"].append([([S], True), ([("goal", T["goal"]), ("calib", T["calib"])], True, False),
                                   ([mem_seg, frames, (f"sdyn{i}", T["state_dyn"])], False)])
        # b_append: fixed message; one appended message per call with the event-log lines new since the last call
        # (the whole log, never truncated: an append-only log cannot drop its oldest lines) and the belief changes
        # since the last decision (as corrections); then the rewritten view without its event log, the frames and
        # the state
        lines = [x for x in c["full_log"] if x not in seen_lines] + c["changes"]
        seen_lines |= set(c["full_log"])
        appended.append(([(f"delta{i}", tokens("\n".join(lines)))], True))
        cur = tokens(mem) - T["events"]
        out["b_append"].append([([S], True), ([("goal", T["goal"]), ("calib", T["calib"])], True), *appended,
                                ([(f"cur{i}", cur), frames, (f"sdyn{i}", T["state_dyn"])], False)])
        # c_conversation: H's way: every turn's full K content stays, with the model's answers between them
        # (every message end is a breakpoint in both modes)
        turn = (head + [mem_seg, frames, (f"state{i}", T["state"])], True)
        convo = convo + ([([(f"ans{i - 1}", answer)], True)] if i else []) + [turn]
        out["c_conversation"].append([([S], True), *convo])
    return out


def price(requests: list, explicit: bool) -> dict:
    """Tokens read, written and plain per layout, by the cache model in the docstring."""
    cache, tot = set(), {"prompt": 0, "read": 0, "write": 0, "plain": 0}
    for req in requests:
        segs, bps = [], []
        for msg, mark, *end in req:  # end False: a block inside a message (a breakpoint only when marked, explicit)
            segs += msg
            if mark and explicit or not explicit and (end or [True])[0]:
                bps.append(len(segs))
        cum = np.cumsum([0] + [n for _, n in segs])
        key = lambda k: tuple(s for s, _ in segs[:k])  # noqa: E731
        hit = max((cum[k] for k in bps if key(k) in cache and cum[k] >= MIN_PREFIX), default=0)
        last = max((cum[k] for k in bps if cum[k] >= MIN_PREFIX), default=0)
        n = int(cum[-1])
        tot["prompt"] += n
        tot["read"] += int(hit)
        tot["write"] += int(max(0, last - hit))
        tot["plain"] += int(n - max(hit, last))
        cache |= {key(k) for k in bps if cum[k] >= MIN_PREFIX}
    tot["cost"] = READ * tot["read"] + WRITE * tot["write"] + tot["plain"]
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--img-tokens", type=int, default=66)
    ap.add_argument("--answer-tokens", type=int, default=110)
    ap.add_argument("--show", action="store_true", help="print the first episode's K memory texts")
    a = ap.parse_args()
    src, rng = PerceivedSource(), np.random.default_rng(0)
    eps = src.episodes("train")[:a.episodes]
    rows = {}
    parts = {"sys": [], "mem": [], "events": [], "state": [], "calib": [], "goal": []}
    for ep in eps:
        calls = episode(src, ep, rng)
        if a.show and ep == eps[0]:
            for c in calls:
                print(c["memory"], "\n" + "-" * 80)
        for c in calls:
            parts["mem"].append(tokens(c["memory"]))
            parts["events"].append(tokens(c["events"]))
            parts["state"].append(tokens("Robot state: " + json.dumps(c["state"])))
        parts["sys"].append(tokens(system_prompt("K")))
        parts["calib"].append(tokens(user_content("K", "", "", [], calls[0]["state"], "fixed_first")[1]["text"]))
        parts["goal"].append(tokens("Task goal: " + calls[0]["goal"]))
        for name, reqs in layouts(calls, a.img_tokens, a.answer_tokens).items():
            for explicit in (False, True):
                t = price(reqs, explicit)
                r = rows.setdefault((name, explicit), dict.fromkeys(t, 0))
                for k, v in t.items():
                    r[k] += v
    print(f"episodes {len(eps)}, calls {len(parts['mem'])}, image tokens {a.img_tokens} each, two per call")
    print("tokens per call, median [min, max]: " + "; ".join(
        f"{k} {int(np.median(v))} [{min(v)}, {max(v)}]" for k, v in parts.items()))
    base = rows[("current", False)]["cost"]
    print("| layout | breakpoints | prompt tok | read | write | plain | read share | cost (x uncached rate) | vs current |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (name, explicit), t in rows.items():
        print(f"| {name} | {'explicit' if explicit else 'implicit'} | {t['prompt']} | {t['read']} | {t['write']} | "
              f"{t['plain']} | {t['read'] / t['prompt']:.2f} | {t['cost']:.0f} | {t['cost'] / base:.2f} |")


if __name__ == "__main__":
    main()
