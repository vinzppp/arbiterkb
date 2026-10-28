"""All tunable parameters in one place (DEVELOPMENT.md §1: one source of parameters).

M0 values are untuned defaults. `frozen` marks a parameter set that may be evaluated on the test
split (E5c); in M0 the defaults are frozen as-is because nothing was tuned.
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
    w_mix: float = 0.5  # max probability that two passing tracks exchanged identity (R6)
    lam: float = 1.0  # per-frame decay toward uniform for unconfirmed beliefs (R7); 1 = off
    eps_none: float = 0.0  # floor for the "elsewhere" hypothesis (M1 sets > 0)


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
class Params:
    rule: RuleParams = field(default_factory=RuleParams)
    detect: DetectParams = field(default_factory=DetectParams)
    monitor: MonitorParams = field(default_factory=MonitorParams)
    arbiter: ArbiterParams = field(default_factory=ArbiterParams)
    frozen: bool = True  # M0: untuned defaults, frozen before any test read
    note: str = "M0 defaults (untuned)"

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self.to_dict(), "hash": self.hash()}, indent=2, sort_keys=True))

    @staticmethod
    def load(path: Path) -> "Params":
        d = json.loads(Path(path).read_text())
        return Params(
            rule=RuleParams(**d["rule"]),
            detect=DetectParams(**d["detect"]),
            monitor=MonitorParams(**d["monitor"]),
            arbiter=ArbiterParams(**d["arbiter"]),
            frozen=d["frozen"],
            note=d.get("note", ""),
        )
