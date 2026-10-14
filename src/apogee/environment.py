"""Launch site environment: atmosphere, gravity and wind."""

from __future__ import annotations

from dataclasses import dataclass, field

from .atmosphere import G0, Atmosphere, StandardAtmosphere

R_EARTH_MEAN = 6_371_000.0  # m


@dataclass
class Environment:
    """Conditions at the launch site.

    ``site_elevation`` is the pad height above sea level [m]; flight altitudes are
    measured above the pad. ``wind_speed`` is a constant horizontal wind [m/s] blowing
    toward +x (downrange). ``inverse_square_gravity`` switches between
    g = g0 (R / (R + h_msl))^2 and a constant g0.
    """

    atmosphere: Atmosphere = field(default_factory=StandardAtmosphere)
    site_elevation: float = 0.0
    wind_speed: float = 0.0
    inverse_square_gravity: bool = True
    g0: float = G0

    def gravity(self, altitude_agl: float) -> float:
        if not self.inverse_square_gravity:
            return self.g0
        r = R_EARTH_MEAN + self.site_elevation + altitude_agl
        return self.g0 * (R_EARTH_MEAN / r) ** 2
