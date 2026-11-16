"""Aerodynamics: Barrowman center of pressure and tabulated drag coefficients.

Barrowman's method treats each part of the rocket as contributing a normal-force slope
``CN_alpha`` (per radian of angle of attack, referenced to the body cross-section) acting
at a known point. The center of pressure is their weighted average:

    x_cp = sum(CN_alpha_i * x_i) / sum(CN_alpha_i)

It assumes small angles of attack, subsonic flow, a slender rigid body and thin fins. A
constant-diameter body tube contributes no normal force at small angles.

Reference: J. S. Barrowman, "The Practical Calculation of the Aerodynamic Characteristics
of Slender Finned Vehicles", 1967.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .components import Airframe

# Center of pressure of a nose cone as a fraction of its length from the tip.
NOSE_CP_FRACTION = {
    "conical": 2.0 / 3.0,
    "ogive": 0.466,
    "parabolic": 0.5,
    "elliptical": 1.0 / 3.0,
}
NOSE_CN_ALPHA = 2.0


@dataclass(frozen=True)
class CenterOfPressure:
    """Barrowman result. Positions are aft of the nose tip [m]; slopes are per radian."""

    position: float
    cn_alpha: float
    nose_cn_alpha: float
    nose_position: float
    fin_cn_alpha: float
    fin_position: float


def barrowman(airframe: Airframe) -> CenterOfPressure:
    """Center of pressure of a nose + body tube + trapezoidal fin set."""
    nose, fins = airframe.nose, airframe.fins
    d = airframe.diameter
    r = d / 2.0

    x_nose = NOSE_CP_FRACTION[nose.shape] * nose.length

    cr, ct, s, xs = fins.root_chord, fins.tip_chord, fins.span, fins.sweep
    mid_chord = math.hypot(s, xs + ct / 2.0 - cr / 2.0)  # length of the mid-chord line
    interference = 1.0 + r / (s + r)  # fin-body interference factor
    cn_fins = interference * (
        4.0
        * fins.count
        * (s / d) ** 2
        / (1.0 + math.sqrt(1.0 + (2.0 * mid_chord / (cr + ct)) ** 2))
    )
    x_fins = (
        airframe.fin_position
        + xs / 3.0 * (cr + 2.0 * ct) / (cr + ct)
        + (cr + ct - cr * ct / (cr + ct)) / 6.0
    )

    cn_total = NOSE_CN_ALPHA + cn_fins
    x_cp = (NOSE_CN_ALPHA * x_nose + cn_fins * x_fins) / cn_total
    return CenterOfPressure(x_cp, cn_total, NOSE_CN_ALPHA, x_nose, cn_fins, x_fins)


@dataclass
class CdTable:
    """Drag coefficient interpolated from a table of (Mach, Cd) points.

    Outside the table's Mach range the end values are held. Instances are callable, so
    one can be passed anywhere a ``cd(mach)`` function is expected.
    """

    mach: np.ndarray
    cd: np.ndarray

    def __post_init__(self) -> None:
        self.mach = np.asarray(self.mach, dtype=float)
        self.cd = np.asarray(self.cd, dtype=float)
        if self.mach.ndim != 1 or self.mach.shape != self.cd.shape or self.mach.size < 1:
            raise ValueError("mach and cd must be 1D arrays of equal, non-zero length")
        if np.any(np.diff(self.mach) <= 0):
            raise ValueError("Mach numbers must be strictly increasing")
        if np.any(self.cd < 0):
            raise ValueError("drag coefficients must be non-negative")

    def __call__(self, mach: float) -> float:
        return float(np.interp(mach, self.mach, self.cd))

    @classmethod
    def from_csv(cls, path: str | Path) -> CdTable:
        """Read a two-column ``mach,cd`` CSV.

        Blank lines, lines starting with ``#`` and a header row are skipped. Rows are
        sorted by Mach number and rows repeating a Mach number are dropped, so a
        Mach/Cd export of a simulated flight can be used directly.
        """
        rows: list[tuple[float, float]] = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(";", ",").split(",")
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue  # header or malformed row
        if not rows:
            raise ValueError(f"{path}: no (mach, cd) rows found")
        rows.sort()
        mach, cd = [rows[0][0]], [rows[0][1]]
        for m, c in rows[1:]:
            if m > mach[-1]:
                mach.append(m)
                cd.append(c)
        return cls(np.array(mach), np.array(cd))
