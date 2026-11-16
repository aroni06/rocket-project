"""Parachute recovery.

Each parachute adds drag ``1/2 rho v^2 Cd A`` against the air-relative velocity once it
deploys. Under a canopy the rocket quickly settles at the terminal velocity where chute
drag balances weight:

    v_t = sqrt(2 m g / (rho Cd A))

A parachute deploys when its trigger fires plus an optional delay:

- ``"apogee"``: when vertical velocity reaches zero (an altimeter or flight computer)
- ``"altitude"``: when the rocket descends through ``altitude`` metres above the pad
  (a dual-deploy main). If apogee is below that altitude it deploys at apogee instead.
- ``"burnout"``: a motor ejection charge, which fires ``delay`` seconds after burnout
  whatever the rocket is doing

Inflation is treated as instant and deployed chutes stay deployed, so a drogue keeps
adding drag after the main opens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TRIGGERS = ("apogee", "altitude", "burnout")


@dataclass
class Parachute:
    """A parachute or streamer. ``diameter`` is the nominal canopy diameter [m]."""

    name: str
    diameter: float
    cd: float = 1.5
    trigger: str = "apogee"
    altitude: float | None = None  # m above pad, for trigger = "altitude"
    delay: float = 0.0  # s after the trigger

    def __post_init__(self) -> None:
        if self.diameter <= 0:
            raise ValueError(f"parachute {self.name!r}: diameter must be positive")
        if self.cd <= 0:
            raise ValueError(f"parachute {self.name!r}: cd must be positive")
        if self.trigger not in TRIGGERS:
            raise ValueError(f"parachute {self.name!r}: trigger must be one of {TRIGGERS}")
        if self.trigger == "altitude" and (self.altitude is None or self.altitude <= 0):
            raise ValueError(f"parachute {self.name!r}: altitude trigger needs altitude > 0")
        if self.delay < 0:
            raise ValueError(f"parachute {self.name!r}: delay must be non-negative")

    @property
    def area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    @property
    def drag_area(self) -> float:
        """Cd times canopy area [m^2]."""
        return self.cd * self.area


def terminal_velocity(mass: float, drag_area: float, density: float, g: float) -> float:
    """Steady descent speed where drag balances weight [m/s]."""
    return math.sqrt(2.0 * mass * g / (density * drag_area))
