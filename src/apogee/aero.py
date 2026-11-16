"""Aerodynamics: Barrowman center of pressure and drag coefficients.

Drag can be a table of Cd against Mach number (:class:`CdTable`) or an estimate built up
from the geometry (:class:`DragBuildup`).

Barrowman's method treats each part of the rocket as contributing a normal-force slope
``CN_alpha`` (per radian of angle of attack, referenced to the body cross-section) acting
at a known point. The center of pressure is their weighted average:

    x_cp = sum(CN_alpha_i * x_i) / sum(CN_alpha_i)

It assumes small angles of attack, subsonic flow, a slender rigid body and thin fins. A
constant-diameter body tube contributes no normal force at small angles.

References: J. S. Barrowman, "The Practical Calculation of the Aerodynamic Characteristics
of Slender Finned Vehicles", 1967. S. Niskanen, "OpenRocket technical documentation", 2013,
section 3.4 (drag).
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


# Drag build-up
#
# The drag of a rocket flying straight is the sum of a few separate effects, each with
# its own simple formula. All coefficients are referenced to the body cross-section.

SUBSONIC_LIMIT = 0.8  # Mach number above which the drag estimate is too low
SEA_LEVEL_SOUND_SPEED = 340.294  # m/s
SEA_LEVEL_VISCOSITY = 1.4607e-5  # kinematic, m^2/s


def _blend(mach: float, subsonic: float, supersonic: float) -> float:
    """Subsonic value below Mach 0.9, supersonic above 1.1, a straight line between."""
    if mach <= 0.9:
        return subsonic
    if mach >= 1.1:
        return supersonic
    return subsonic + (supersonic - subsonic) * (mach - 0.9) / 0.2


def skin_friction_coefficient(reynolds: float, relative_roughness: float, mach: float) -> float:
    """Skin friction coefficient of a fully turbulent boundary layer.

    ``reynolds`` is based on the rocket's length and ``relative_roughness`` is the surface
    roughness height divided by that length. A smooth surface follows the turbulent
    flat-plate law; past a critical Reynolds number the roughness sets the friction and
    it stops falling. Both are then corrected for compressibility.
    """
    smooth = 1.48e-2 if reynolds < 1.0e4 else 1.0 / (1.50 * math.log(reynolds) - 5.6) ** 2
    smooth *= _blend(mach, 1.0 - 0.1 * mach**2, 1.0 / (1.0 + 0.15 * mach**2) ** 0.58)
    if relative_roughness <= 0.0:
        return smooth
    rough = 0.032 * relative_roughness**0.2 / (1.0 + 0.18 * mach**2)
    return max(smooth, rough)


def base_drag_coefficient(mach: float) -> float:
    """Drag of the low-pressure wake behind a blunt base, per unit base area."""
    return 0.12 + 0.13 * mach**2 if mach < 1.0 else 0.25 / mach


def stagnation_drag_coefficient(mach: float) -> float:
    """Drag of a blunt face square to the flow, per unit frontal area."""
    if mach < 1.0:
        ratio = 1.0 + mach**2 / 4.0 + mach**4 / 40.0
    else:
        ratio = 1.84 - 0.76 / mach**2 + 0.166 / mach**4 + 0.035 / mach**6
    return 0.85 * ratio


def rounded_edge_drag_coefficient(mach: float) -> float:
    """Drag of a rounded leading edge square to the flow, per unit frontal area."""
    if mach < 0.9:
        return (1.0 - mach**2) ** -0.417 - 1.0
    if mach < 1.0:
        return 1.0 - 1.785 * (mach - 0.9)
    return 1.214 - 0.502 / mach**2 + 0.1095 / mach**4


@dataclass(frozen=True)
class DragBreakdown:
    """Drag coefficient by source, each referenced to the body cross-section."""

    body_friction: float
    fin_friction: float
    nose_pressure: float
    base: float
    fin_pressure: float

    @property
    def total(self) -> float:
        return (
            self.body_friction
            + self.fin_friction
            + self.nose_pressure
            + self.base
            + self.fin_pressure
        )


class DragBuildup:
    """Zero-angle-of-attack drag coefficient estimated from the airframe's geometry.

    The parts, following the method OpenRocket documents:

    - **skin friction** on the wetted area of the body and fins, from a turbulent
      boundary layer on a surface of the given ``roughness`` [m], corrected for the
      body's fineness and the fins' thickness
    - **nose pressure drag**, zero for shapes that blend smoothly into the body
    - **base drag** from the wake behind the flat aft end
    - **fin pressure drag** on the leading and trailing edges, depending on the profile

    It is subsonic: there is no wave drag, so it underpredicts above about Mach 0.8
    (``SUBSONIC_LIMIT``).
    Launch lugs and rail buttons are not included, and the base drag is not reduced
    while the motor is burning.

    Skin friction depends on the Reynolds number, which depends on speed and altitude and
    not only on Mach number, so the flight model calls :meth:`at`. Calling the object with
    a Mach number alone evaluates it at sea level, which is enough for a plot or a table.

    The default roughness of 60 micrometres is a regular painted surface; 20 is smooth
    paint and 150 is unfinished.
    """

    def __init__(self, airframe: Airframe, roughness: float = 60e-6) -> None:
        if roughness < 0:
            raise ValueError("roughness must be non-negative")
        fins = airframe.fins
        if fins.thickness is None:
            raise ValueError("the drag estimate needs the fin thickness (fins.thickness_m)")
        self.airframe = airframe
        self.roughness = roughness
        d = airframe.diameter
        self.length = airframe.length
        self._reference = math.pi * d * d / 4.0
        self._relative_roughness = roughness / self.length

        body_wetted = airframe.nose.wetted_area(d / 2.0) + math.pi * d * airframe.body.length
        fin_wetted = 2.0 * fins.count * fins.planform_area
        fineness = self.length / d
        self._body_factor = (1.0 + 1.0 / (2.0 * fineness)) * body_wetted / self._reference
        self._fin_factor = (
            (1.0 + 2.0 * fins.thickness / fins.mean_aerodynamic_chord) * fin_wetted
        ) / self._reference
        self._nose_pressure = 0.8 * math.sin(airframe.nose.joint_angle(d / 2.0)) ** 2
        self._fin_frontal = fins.count * fins.thickness * fins.span / self._reference
        self._sweep_factor = math.cos(fins.leading_edge_sweep) ** 2
        self._profile = fins.profile

    def breakdown(self, mach: float, unit_reynolds: float) -> DragBreakdown:
        """Drag by source at a Mach number and Reynolds number per metre (speed / viscosity)."""
        friction = skin_friction_coefficient(
            unit_reynolds * self.length, self._relative_roughness, mach
        )
        base = base_drag_coefficient(mach)
        if self._profile == "square":
            leading, trailing = stagnation_drag_coefficient(mach), base
        elif self._profile == "rounded":
            leading, trailing = rounded_edge_drag_coefficient(mach), base / 2.0
        else:  # airfoil
            leading, trailing = rounded_edge_drag_coefficient(mach), 0.0
        return DragBreakdown(
            body_friction=friction * self._body_factor,
            fin_friction=friction * self._fin_factor,
            nose_pressure=self._nose_pressure,
            base=base,
            fin_pressure=(leading * self._sweep_factor + trailing) * self._fin_frontal,
        )

    def at(self, mach: float, unit_reynolds: float) -> float:
        """Total drag coefficient at a Mach number and Reynolds number per metre."""
        return self.breakdown(mach, unit_reynolds).total

    def __call__(self, mach: float) -> float:
        """Total drag coefficient at a Mach number, at sea level."""
        return self.at(mach, mach * SEA_LEVEL_SOUND_SPEED / SEA_LEVEL_VISCOSITY)
