"""Extract the front-camera frames the object-identity tests need from the RoboMME dataset videos
(data/robomme_lerobot/videos/image, AV1 in MP4; decoded with PyAV, which the system Python has and the project does
not depend on, hence tools/). Per episode: every frame where the gripper closes (what was grasped), the first frame
(the scene before anything moved) and the first execution frame (the decision point).
With --windows FILE (JSON {"<task>_<episode>": [frame indices]}, written by scripts/review_test.py --plan), exactly
those frames go to data/robomme_lerobot/review_frames/ instead.

  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -I tools/extract_frames.py --tasks VideoRepick,VideoPlaceOrder

Writes data/robomme_lerobot/frames/<task>_<episode>.npz: idx [K] (frame indices), rgb [K, 256, 256, 3].
"""
import argparse
import glob
import json
import os

import av
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "robomme_lerobot")
FPS = 10


def wanted(npz: str) -> list[int]:
    d = np.load(npz)
    finger, demo = d["state"][:, 7], d["is_demo"]
    closes = [t for t in range(1, len(finger)) if finger[t - 1] >= 0.035 > finger[t]]
    exec_start = int(np.argmin(demo)) if demo.any() and not demo.all() else 0
    return sorted({0, exec_start, *closes, *[min(t + 3, len(finger) - 1) for t in closes]})


def episodes_meta() -> dict:
    f = glob.glob(os.path.join(ROOT, "meta/episodes/**/*.parquet"), recursive=True)[0]
    return {r["episode_index"]: r for r in pq.read_table(f).to_pylist()}


def decode(path: str, t0: float, idx: list[int]) -> dict:
    """{frame index: rgb} for frames at t0 + idx / FPS (nearest decoded frame)."""
    want = {round(t0 * FPS) + i: i for i in idx}
    out = {}
    with av.open(path) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 1.0, 0) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            k = round(float(fr.pts * s.time_base) * FPS)
            if k in want:
                out[want[k]] = fr.to_ndarray(format="rgb24")
            if k > max(want):
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="VideoRepick,VideoPlaceOrder")
    ap.add_argument("--windows", default="")
    a = ap.parse_args()
    windows = json.load(open(a.windows)) if a.windows else None
    meta, out = episodes_meta(), os.path.join(ROOT, "review_frames" if windows else "frames")
    os.makedirs(out, exist_ok=True)
    for task in a.tasks.split(","):
        n = 0
        for npz in sorted(glob.glob(os.path.join(ROOT, "npz", f"{task}_*.npz"))):
            ep = json.loads(str(np.load(npz)["meta"]))["episode"]
            key = f"{task}_{ep:04d}"
            if windows is not None and key not in windows:
                continue
            m = meta[ep]
            vid = os.path.join(ROOT, f"videos/image/chunk-{m['videos/image/chunk_index']:03d}/"
                                     f"file-{m['videos/image/file_index']:03d}.mp4")
            if not os.path.exists(vid):
                continue
            got = decode(vid, float(m["videos/image/from_timestamp"]), windows[key] if windows else wanted(npz))
            idx = sorted(got)
            np.savez_compressed(os.path.join(out, f"{key}.npz"), idx=np.array(idx),
                                rgb=np.stack([got[i] for i in idx]))
            n += 1
        print(task, n, "episodes ->", out, flush=True)


if __name__ == "__main__":
    main()
