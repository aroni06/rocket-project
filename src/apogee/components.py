"""Airframe geometry and mass properties.

An :class:`Airframe` is a nose cone, one constant-diameter body tube, one set of
trapezoidal fins and any number of point masses (recovery gear, avionics, ballast).
All positions are distances aft of the nose tip, in metres.

The airframe answers two questions the flight model can't: where the dry center of
gravity is, and (through :mod:`apogee.aero`) where the center of pressure is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

NOSE_SHAPES = ("conical", "ogive", "parabolic", "elliptical")


@dataclass
class NoseCone:
    """Nose cone. ``cg`` is measured from the tip.

    If ``cg`` is not given it defaults to 2/3 of the length, which is exact for a thin
    conical shell and a reasonable guess for the other shapes. Weigh and balance the real
    part when it matters.
    """

    length: float
    shape: str = "ogive"
    mass: float = 0.0
    cg: float | None = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("nose length must be positive")
        if self.shape not in NOSE_SHAPES:
            raise ValueError(f"nose shape must be one of {NOSE_SHAPES}")
        if self.mass < 0:
            raise ValueError("nose mass must be non-negative")
        if self.cg is None:
            self.cg = 2.0 * self.length / 3.0


@dataclass
class BodyTube:
    """Constant-diameter body tube, starting at the base of the nose cone."""

    length: float
    diameter: float
    mass: float = 0.0

    def __post_init__(self) -> None:
        if self.length <= 0 or self.diameter <= 0:
            raise ValueError("body length and diameter must be positive")
        if self.mass < 0:
            raise ValueError("body mass must be non-negative")


@dataclass
class FinSet:
    """A set of identical trapezoidal fins.

    ``sweep`` is how far aft the tip's leading edge sits from the root's leading edge.
    ``position`` is the root leading edge's distance from the nose tip; left as ``None``
    the root trailing edge is flush with the aft end of the body.

    Give ``mass`` for the whole set, or ``thickness`` and ``density`` to compute it from
    the planform (useful when a design sweep changes the fin size).
    """

    count: int
    root_chord: float
    tip_chord: float
    span: float
    sweep: float = 0.0
    position: float | None = None
    mass: float | None = None
    thickness: float | None = None
    density: float | None = None

    def __post_init__(self) -> None:
        if self.count < 3:
            raise ValueError("need at least 3 fins")
        if self.root_chord <= 0 or self.span <= 0:
            raise ValueError("fin root chord and span must be positive")
        if self.tip_chord < 0 or self.sweep < 0:
            raise ValueError("fin tip chord and sweep must be non-negative")
        if self.mass is None:
            if self.thickness is not None and self.density is not None:
                self.mass = self.count * self.planform_area * self.thickness * self.density
            else:
                self.mass = 0.0
        if self.mass < 0:
            raise ValueError("fin mass must be non-negative")

    @property
    def planform_area(self) -> float:
        """Area of one fin [m^2]."""
        return 0.5 * (self.root_chord + self.tip_chord) * self.span

    @property
    def centroid(self) -> float:
        """Chordwise centroid of the planform, aft of the root leading edge [m]."""
        cr, ct, xs = self.root_chord, self.tip_chord, self.sweep
        return (xs * (cr + 2 * ct) + cr * cr + cr * ct + ct * ct) / (3 * (cr + ct))


@dataclass
class PointMass:
    """Anything with mass and no aerodynamic effect: recovery gear, avionics, ballast."""

    name: str
    mass: float
    position: float

    def __post_init__(self) -> None:
        if self.mass < 0:
            raise ValueError(f"mass {self.name!r} must be non-negative")


@dataclass
class Airframe:
    """Everything except the motor.

    ``mass_override`` and ``cg_override`` replace the values built up from the components
    with measured ones (a scale and a balance point beat any estimate).
    """

    nose: NoseCone
    body: BodyTube
    fins: FinSet
    masses: list[PointMass] = field(default_factory=list)
    mass_override: float | None = None
    cg_override: float | None = None

    def __post_init__(self) -> None:
        if self.fin_position < self.nose.length:
            raise ValueError("fins must sit on the body tube, aft of the nose cone")
        if self.fin_position + self.fins.root_chord > self.length + 1e-9:
            raise ValueError("fin root extends past the aft end of the body")
        if self.mass_override is None and self._component_mass <= 0:
            raise ValueError("airframe has no mass: give component masses or a dry mass")

    @property
    def diameter(self) -> float:
        return self.body.diameter

    @property
    def length(self) -> float:
        return self.nose.length + self.body.length

    @property
    def fin_position(self) -> float:
        """Root leading edge of the fins, aft of the nose tip [m]."""
        if self.fins.position is not None:
            return self.fins.position
        return self.length - self.fins.root_chord

    def _items(self) -> list[tuple[float, float]]:
        """(mass, cg position) for every component."""
        items = [
            (self.nose.mass, self.nose.cg),
            (self.body.mass, self.nose.length + self.body.length / 2.0),
            (self.fins.mass, self.fin_position + self.fins.centroid),
        ]
        items += [(m.mass, m.position) for m in self.masses]
        return items

    @property
    def _component_mass(self) -> float:
        return math.fsum(m for m, _ in self._items())

    @property
    def mass(self) -> float:
        """Dry mass without the motor [kg]."""
        return self.mass_override if self.mass_override is not None else self._component_mass

    @property
    def cg(self) -> float:
        """Dry center of gravity without the motor, aft of the nose tip [m]."""
        if self.cg_override is not None:
            return self.cg_override
        total = self._component_mass
        if total <= 0:
            raise ValueError("give component masses or a measured dry CG to locate the CG")
        return math.fsum(m * x for m, x in self._items()) / total
