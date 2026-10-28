"""VideoUnmaskSwapStress: RoboMME's VideoUnmaskSwap with the difficulty knobs exposed (phase G).

Runs ON THE POD in the sg-main venv (robomme @ 53673b3, mani_skill 3.0.0b21 fork). Read against
src/robomme/robomme_env/VideoUnmaskSwap.py (evidence-gaps-2026-10-28.md §7):
- swap_steps:  steps per swap (stock 50; _refresh_swap_schedule hard-codes it);
- lane_offset: sideways bulge of the two swap paths (stock 0.07; hard-coded in step()). Below about
  0.03 m the two bins pass through each other (kinematic), a real 3D overlap;
- n_swaps:     1..3 (stock draws it from the difficulty; the pair slots only go up to 3);
- cam_shift:   static (dx, dy, dz) added to the front camera eye and target (stock eye (0.3, 0, 0.4)).
It also logs every actor's pose after each step (`pose_log`), so ground truth positions exist for
every frame, which the stock replays lack during the video.
"""
from __future__ import annotations

import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from robomme.robomme_env.utils import (lift_and_drop_objectA_onto_objectB, lift_and_drop_objects_back_to_original,
                                       swap_flat_two_lane)
from robomme.robomme_env.VideoUnmaskSwap import VideoUnmaskSwap

SWAP_T0 = 64  # stock: swaps start at step 64, back to back


@register_env("VideoUnmaskSwapStress", max_episode_steps=2000)
class VideoUnmaskSwapStress(VideoUnmaskSwap):
    def __init__(self, *args, swap_steps: int = 50, lane_offset: float = 0.07, n_swaps: int | None = None,
                 cam_shift=(0.0, 0.0, 0.0), **kwargs):
        assert n_swaps is None or 1 <= n_swaps <= 3, "the stock env has pair slots for at most 3 swaps"
        self.swap_steps, self.lane_offset, self.n_swaps = int(swap_steps), float(lane_offset), n_swaps
        self.cam_shift = np.asarray(cam_shift, float)
        self.pose_log: list[np.ndarray] = []
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        eye = np.array([0.3, 0.0, 0.4]) + self.cam_shift
        target = np.array([0.0, 0.0, -0.2]) + self.cam_shift
        pose = sapien_utils.look_at(eye=eye.tolist(), target=target.tolist())
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    def _load_scene(self, options: dict):
        if self.n_swaps is not None:
            self.swap_times = self.n_swaps  # read by _refresh_swap_schedule inside the stock _load_scene
        self.pose_log = []
        super()._load_scene(options)

    def _refresh_swap_schedule(self):
        pairs = [(self.swap_pair1_idx1, self.swap_pair1_idx2), (self.swap_pair2_idx1, self.swap_pair2_idx2),
                 (self.swap_pair3_idx1, self.swap_pair3_idx2)][: self.swap_times]
        n = self.swap_steps
        self.swap_schedule = [(a, b, SWAP_T0 + k * n, SWAP_T0 + (k + 1) * n) for k, (a, b) in enumerate(pairs)]

    def _pick_partners(self, timestep: int) -> None:
        """Stock logic: when swap i starts, its partner is the bin nearest to its first bin."""
        for i, (_, _, start, end) in enumerate(self.swap_schedule):
            a, b = getattr(self, f"swap_pair{i + 1}_idx1"), getattr(self, f"swap_pair{i + 1}_idx2")
            if start <= timestep < end and b is None and a is not None:
                ref = self._get_actor_position(a)
                cands = [c for c in self.spawned_bins if c is not None and c is not a]
                best = min(cands, key=lambda c: np.linalg.norm(ref[:2] - self._get_actor_position(c)[:2]), default=None)
                if best is not None:
                    setattr(self, f"swap_pair{i + 1}_idx2", best)
                    self._refresh_swap_schedule()

    def step(self, action):
        """Copy of VideoUnmaskSwap.step with lane_offset and the swap timing as parameters."""
        timestep = self.elapsed_steps
        for bin_actor in getattr(self, "spawned_bins", []):
            lift_and_drop_objects_back_to_original(self, obj=bin_actor, start_step=0, end_step=32 * 2, cur_step=timestep)
        self._pick_partners(int(timestep))
        for a, b, start, end in self.swap_schedule:
            if a is None or b is None:
                continue
            swap_flat_two_lane(self, cube_a=a, cube_b=b, start_step=start, end_step=end, cur_step=timestep,
                               lane_offset=self.lane_offset, smooth=True, keep_upright=True,
                               other_cube=[x for x in self.spawned_bins if x not in (a, b)])
        for cube, bin_actor in getattr(self, "cube_bin_pairs", []):
            if cube is not None and bin_actor is not None:
                lift_and_drop_objectA_onto_objectB(self, obj_a=cube, obj_b=bin_actor, start_step=32 * 2,
                                                   end_step=self.swap_schedule[-1][3], cur_step=timestep)
        out = BaseEnv.step(self, action)  # skip the stock step: its swap block is what we replaced
        self.pose_log.append(np.stack([self._get_actor_position(x) for x in self.logged_actors()]))
        return out

    def logged_actors(self) -> list:
        """Bins first, then cubes, in the order sg_pilot.scene_objects numbers them."""
        return list(self.spawned_bins) + list(self.spawned_dynamic_cubes)
