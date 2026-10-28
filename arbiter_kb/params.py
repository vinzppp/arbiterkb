"""All tunable parameters in one place (DEVELOPMENT.md §1: one source of parameters).

E5c: the test split may only be evaluated with a Params that (1) was loaded from a params.json whose
stored hash matches its content, (2) has frozen = true, and (3) has a FREEZE entry in the test-read
ledger (ledger.py) written before the read. A Params built in code has no origin and is refused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RuleParams:
    r_cover: float = 0.06  # m: a container within this xy distance of a vanishing cube can cover it
    sigma_cover: float = 0.02  # m: width of the distance kernel over cover candidates (R1)
    w_max: float = 0.5  # R6 (M1): w = w_max·exp(-d/d0), d = closest 3D centre distance of the pass
    d0: float = 0.02  # m
    eps_none: float = 0.01  # floor for the "elsewhere" hypothesis in R1
    r_grasp: float = 0.05  # m: R3, an object within this 3D distance of the end effector when the gripper closes is held
    r_on: float = 0.04  # m: On(x, y) / place target: xy distance at most this, and x above y
    r_near: float = 0.08  # m: Near(x, y): centre distance below this
    left_margin: float = 0.02  # m: LeftOf(x, y): x is this much further to the robot's left (+y)
    on_dz_min: float = 0.005  # m: On(x, y): x's centre at least this much above y's ...
    on_dz_max: float = 0.08  # m: ... and at most this much (extent is not estimated yet)
    eps_far: float = 0.5  # R1, no container within r_cover: nearest vs "elsewhere" at even odds (the cover was not seen)


@dataclass(frozen=True)
class DetectParams:
    d_px: float = 30.0  # px: two moving containers closer than this in the image -> near-pass
    d_3d: float = 0.10  # m: same, used when there is no camera
    v_move: float = 0.001  # m/frame (2 cm/s at 20 Hz): xy speed above which a container counts as moving
    g_close: float = 0.07  # m: gripper width below this -> closed (RoboMME: 0.04 per finger = fully open)
    g_open: float = 0.075  # m: above this -> open again (hysteresis)
    v_pause: float = 0.002  # m/frame: the hand counts as still below this speed (RoboMME dataset: 10 fps)
    n_pause: int = 4  # frames still, after moving, before a `hand_pause` evidence
    z_low: float = 0.05  # m: hand below this = at table / object level (actions.py; dataset: pick 0.020, place 0.015)
    z_drop: float = 0.12  # m: a release above this is a drop from above (dataset: into the bin at 0.20)
    z_hold: float = 0.07  # m: a pause while holding above this = holding something up (dataset: 0.10)
    press_rise: float = 0.03  # m: a press is complete when the hand rises this far above its lowest point
    y_side: float = 0.10  # m: |y| beyond this = the robot's left (+y) / right (-y) side (dataset: 0.21)


@dataclass(frozen=True)
class MonitorParams:
    tau_hi: float = 0.8  # p_max at or above -> "act"; wrong with p_max >= tau_hi is a silent failure
    tau_lo: float = 0.5  # p_max below -> "verify"


@dataclass(frozen=True)
class ArbiterParams:
    every_k: int = 50  # frames between periodic triggers while something is hidden
    max_calls: int = 6  # per episode
    p_clear: float = 0.9  # annotator verdict -> belief probability (fixed table, measured against truth in M4)
    p_ambiguous: float = 0.5
    p_unlikely: float = 0.1
    vlm_model: str = "qwen/qwen3.5-9b"  # clue-object lister and annotator (NOTES "VLM choice": 4/4 frames, 6 providers)
    vlm_provider: str = "Parasail,DeepInfra"  # only these, in order; both bf16 (DeepInfra was rate-limited, 34-65 s)
    vlm_thinking: bool = False  # the 9B is a thinking model; off for these labelling calls


@dataclass(frozen=True)
class BaselineParams:
    c: float = 0.5  # haircut confidence (= p_swap 0.5, the Bayes-optimal value under id_swap_overlap, on purpose)
    d_ov: float = 0.03  # m: OverlapHaircut reacts only to passes closer than this in 3D
    j_max: float = 0.03  # m: JumpDetector flags a label that moves further than this in one frame


@dataclass(frozen=True)
class Params:
    rule: RuleParams = field(default_factory=RuleParams)
    detect: DetectParams = field(default_factory=DetectParams)
    monitor: MonitorParams = field(default_factory=MonitorParams)
    arbiter: ArbiterParams = field(default_factory=ArbiterParams)
    baseline: BaselineParams = field(default_factory=BaselineParams)
    frozen: bool = False  # only a params.json loaded from disk may be frozen (E5c)
    note: str = ""
    origin: str = field(default="", compare=False)  # file it was loaded from; not part of the hash

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("origin")
        return d

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self.to_dict(), "hash": self.hash()}, indent=2, sort_keys=True))

    @staticmethod
    def load(path: Path) -> "Params":
        d = json.loads(Path(path).read_text())
        p = Params(
            rule=RuleParams(**d["rule"]),
            detect=DetectParams(**d["detect"]),
            monitor=MonitorParams(**d["monitor"]),
            arbiter=ArbiterParams(**d["arbiter"]),
            baseline=BaselineParams(**d.get("baseline", {})),
            frozen=d["frozen"],
            note=d.get("note", ""),
            origin=str(path),
        )
        if d.get("hash") != p.hash():
            raise ValueError(f"{path}: stored hash {d.get('hash')} != content hash {p.hash()} (edited by hand?)")
        return p
