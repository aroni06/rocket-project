"""Rocket definition.

For the trajectory the rocket is a point mass with a fixed dry mass and a drag coefficient
that is constant, a function of Mach number or estimated from the geometry, plus any number
of parachutes.

Give it an :class:`~apogee.components.Airframe` as well and it can also report stability:
the center of pressure (Barrowman), the center of gravity as propellant burns, and the
static margin between them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from .aero import CenterOfPressure, DragBuildup, barrowman
from .components import Airframe
from .motor import Motor
from .recovery import Parachute


@dataclass
class Rocket:
    """A single-stage rocket.

    ``dry_mass`` is everything except the motor [kg]. ``diameter`` is the body tube
    outer diameter used as the drag reference [m]. ``cd`` is a constant, a callable
    ``cd(mach)`` or a :class:`~apogee.aero.DragBuildup`. ``motor`` may be ``None`` for
    unpowered test cases.
    ``parachutes`` deploy in flight according to their triggers.

    ``airframe`` is optional geometry for stability. ``motor_position`` is the distance
    from the nose tip to the motor's forward end [m]; left as ``None`` the motor's aft
    end is flush with the aft end of the body.
    """

    name: str
    dry_mass: float
    diameter: float
    motor: Motor | None
    cd: float | Callable[[float], float] = 0.5
    parachutes: list[Parachute] = field(default_factory=list)
    airframe: Airframe | None = None
    motor_position: float | None = None

    @classmethod
    def from_airframe(
        cls,
        name: str,
        airframe: Airframe,
        motor: Motor | None,
        cd: float | Callable[[float], float] = 0.5,
        parachutes: list[Parachute] | None = None,
        motor_position: float | None = None,
    ) -> Rocket:
        """Build a rocket whose dry mass and diameter come from its airframe."""
        return cls(
            name=name,
            dry_mass=airframe.mass,
            diameter=airframe.diameter,
            motor=motor,
            cd=cd,
            parachutes=parachutes or [],
            airframe=airframe,
            motor_position=motor_position,
        )

    def __post_init__(self) -> None:
        if self.dry_mass <= 0:
            raise ValueError("dry_mass must be positive")
        if self.diameter <= 0:
            raise ValueError("diameter must be positive")
        names = [p.name for p in self.parachutes]
        if len(set(names)) != len(names):
            raise ValueError("parachute names must be unique")

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

    def drag_coefficient(self, mach: float, unit_reynolds: float | None = None) -> float:
        """Drag coefficient at a Mach number.

        ``unit_reynolds`` is the Reynolds number per metre (speed / kinematic viscosity).
        Only a drag estimate built from the geometry uses it; without it that estimate
        assumes sea level.
        """
        if isinstance(self.cd, DragBuildup) and unit_reynolds is not None:
            return self.cd.at(mach, unit_reynolds)
        return float(self.cd(mach)) if callable(self.cd) else float(self.cd)

    # Stability. These need an airframe; without one they raise ValueError.

    def _require_airframe(self) -> Airframe:
        if self.airframe is None:
            raise ValueError(f"{self.name} has no airframe geometry, so stability is unknown")
        return self.airframe

    @property
    def center_of_pressure(self) -> CenterOfPressure:
        return barrowman(self._require_airframe())

    @property
    def motor_cg(self) -> float:
        """Motor center of gravity aft of the nose tip [m], taken as its mid-length."""
        airframe = self._require_airframe()
        length = self.motor.length if self.motor else 0.0
        front = self.motor_position
        if front is None:
            front = airframe.length - length
        return front + length / 2.0

    def cg_at(self, t: float) -> float:
        """Center of gravity aft of the nose tip at time ``t`` [m].

        Propellant burns away at the motor's mid-length, so the CG of an aft-mounted
        motor moves forward during the burn.
        """
        airframe = self._require_airframe()
        motor_mass = self.motor.mass_at(t) if self.motor else 0.0
        return (self.dry_mass * airframe.cg + motor_mass * self.motor_cg) / (
            self.dry_mass + motor_mass
        )

    def static_margin_at(self, t: float) -> float:
        """Static margin in calibers: (x_cp - x_cg) / diameter. Positive is stable."""
        return (self.center_of_pressure.position - self.cg_at(t)) / self.diameter
