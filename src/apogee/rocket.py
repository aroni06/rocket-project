"""Rocket definition.

Milestone 2 treats the airframe as a single body with a fixed dry mass and a drag
coefficient that is either constant or a function of Mach number. Component geometry,
center of gravity and center of pressure arrive with the stability model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .motor import Motor


@dataclass
class Rocket:
    """A single-stage rocket.

    ``dry_mass`` is everything except the motor [kg]. ``diameter`` is the body tube
    outer diameter used as the drag reference [m]. ``cd`` is either a constant or a
    callable ``cd(mach)``. ``motor`` may be ``None`` for unpowered test cases.
    """

    name: str
    dry_mass: float
    diameter: float
    motor: Motor | None
    cd: float | Callable[[float], float] = 0.5

    def __post_init__(self) -> None:
        if self.dry_mass <= 0:
            raise ValueError("dry_mass must be positive")
        if self.diameter <= 0:
            raise ValueError("diameter must be positive")

    @property
    def reference_area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    @property
    def liftoff_mass(self) -> float:
        return self.mass_at(0.0)

    def mass_at(self, t: float) -> float:
        motor_mass = self.motor.mass_at(t) if self.motor else 0.0
        return self.dry_mass + motor_mass

    def thrust_at(self, t: float) -> float:
        return self.motor.thrust_at(t) if self.motor else 0.0

    def drag_coefficient(self, mach: float) -> float:
        return float(self.cd(mach)) if callable(self.cd) else float(self.cd)
