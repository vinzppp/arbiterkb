"""Semantic actions from the robot's own state (L3 evidence; NOTES "Counting test results"): what the robot did,
in words a controller can count, without vision. Causal: each action is reported when it is complete.

  pick_up     the gripper closes and the finger opening settles above w_empty: something is between the fingers
              (NOTES "Grasps above z_low": the dataset's empty closes settle at 0.000, a cube at 0.0366; the hand's
              height does not decide it: RoboMME grasps a container at TCP 0.062);
  release     the gripper opens while holding; `how` = placed (hand low), set_down (mid), dropped (z > z_drop,
              e.g. into a bin from above);
  press       the gripper is closed but holds nothing and the hand dips below z_low and rises again by press_rise
              (RoboMME presses buttons with a closed gripper);
  hold_above  a hand pause while holding, with the hand above z_hold and off to one side (|y| > y_side);
              `side` = the robot's left (+y) or right (-y). A pause in the centre (e.g. just after a lift) is not one,
              nor a second pause on the same side before the hold moves to the other side.
Thresholds (DetectParams) were read off the dataset's subgoal segments; the recognised actions are checked
against those labels in NOTES before any arm uses them.
"""
from __future__ import annotations

import numpy as np

from .observe import obs_id
from .ops import AddEvidence
from .params import DetectParams
from .types import FrameObs


class Actions:
    def __init__(self, p: DetectParams):
        self.p, self.holding, self.dip = p, False, None  # dip: [min z, xyz] while a press is under way
        self.pending = None  # [last opening, xyz at the close] while a close has not settled
        self.side = None  # side of the last hold since the last pick-up / release (repeats are not new actions)

    def step(self, frame: FrameObs, closed: bool, new: list[AddEvidence]) -> list[AddEvidence]:
        """`closed`: the gripper state after this frame; `new`: this frame's grip / pause evidence."""
        ee = np.asarray(frame.ee_xyz, float)
        out = []
        for e in new:
            if e.kind == "grip_close":
                self.pending = [float(frame.gripper_width), ee.copy()]
            elif e.kind == "grip_open" and self.pending is not None:
                self.pending = None  # opened before settling: an empty close
            elif e.kind == "grip_open" and self.holding:
                self.holding, self.side = False, None
                how = "placed" if ee[2] < self.p.z_low else "dropped" if ee[2] > self.p.z_drop else "set_down"
                out.append(self._ev(frame, "release", ee, how=how))
            elif e.kind == "hand_pause" and self.holding and ee[2] > self.p.z_hold and abs(ee[1]) > self.p.y_side:
                side = "left" if ee[1] > 0 else "right"
                if side != self.side:
                    self.side = side
                    out.append(self._ev(frame, "hold_above", ee, side=side))
        return out + self._grasp(frame, closed, new) + self._press(frame, closed, ee)

    def _grasp(self, frame: FrameObs, closed: bool, new: list[AddEvidence]) -> list[AddEvidence]:
        """A pending close becomes a pick_up once the opening has settled with something between the fingers."""
        if self.pending is None or not closed or any(e.kind == "grip_close" for e in new):
            return []
        w, (last, at) = float(frame.gripper_width), self.pending
        if abs(w - last) >= self.p.w_settle:
            self.pending[0] = w
            return []
        self.pending = None
        if w <= self.p.w_empty:
            return []
        self.holding, self.side = True, None
        return [self._ev(frame, "pick_up", at, width=round(w, 4))]

    def _press(self, frame: FrameObs, closed: bool, ee: np.ndarray) -> list[AddEvidence]:
        if not closed or self.holding:
            self.dip = None
            return []
        if ee[2] < self.p.z_low and (self.dip is None or ee[2] < self.dip[0]):
            self.dip = [float(ee[2]), ee.copy()]
        if self.dip is not None and ee[2] > self.dip[0] + self.p.press_rise:
            at, self.dip = self.dip[1], None
            return [self._ev(frame, "press", at)]
        return []

    @staticmethod
    def _ev(frame: FrameObs, kind: str, xyz: np.ndarray, **extra) -> AddEvidence:
        data = {"ee": [round(float(v), 4) for v in xyz], **extra, "obs": [obs_id(frame.t, "ee")]}
        return AddEvidence(frame.t, kind, (), data)
