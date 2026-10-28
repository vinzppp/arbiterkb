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
    lam: float = 1.0  # per-frame decay toward uniform for unconfirmed beliefs (R7); 1 = off
    eps_none: float = 0.01  # floor for the "elsewhere" hypothesis in R1
    eps_far: float = 0.5  # R1, no container within r_cover: "elsewhere" mass (the cover was not seen; even odds)


@dataclass(frozen=True)
class DetectParams:
    d_px: float = 30.0  # px: two moving containers closer than this in the image -> near-pass
    d_3d: float = 0.10  # m: same, used when there is no camera
    v_move: float = 0.001  # m/frame (2 cm/s at 20 Hz): xy speed above which a container counts as moving


@dataclass(frozen=True)
class MonitorParams:
    tau_hi: float = 0.8  # p_max at or above -> "act"; wrong with p_max >= tau_hi is a silent failure
    tau_lo: float = 0.5  # p_max below -> "verify"


@dataclass(frozen=True)
class ArbiterParams:
    every_k: int = 50  # frames between periodic triggers while something is hidden
    max_calls: int = 6  # per episode


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
