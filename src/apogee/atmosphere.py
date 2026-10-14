"""Atmosphere models.

The default is the 1976 U.S. Standard / ISA atmosphere for the first three layers
(0 to 32 km geopotential altitude), which covers every model and high-power rocket
flight. Altitudes passed in are geometric altitudes above mean sea level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Protocol

G0 = 9.80665  # standard gravity [m/s^2]
R_AIR = 287.05287  # specific gas constant of dry air [J/(kg K)]
GAMMA = 1.4  # ratio of specific heats for air
R_EARTH = 6_356_766.0  # radius used for geopotential altitude [m]

# (base geopotential altitude [m], base temperature [K], lapse rate [K/m], base pressure [Pa])
_ISA_LAYERS = (
    (0.0, 288.15, -0.0065, 101_325.0),
    (11_000.0, 216.65, 0.0, 22_632.06),
    (20_000.0, 216.65, 0.001, 5_474.889),
)


class AtmosphereState(NamedTuple):
    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m^3
    speed_of_sound: float  # m/s


class Atmosphere(Protocol):
    def at(self, altitude: float) -> AtmosphereState:
        """Return the atmosphere state at a geometric altitude above sea level [m]."""
        ...


@dataclass(frozen=True)
class StandardAtmosphere:
    """ISA troposphere and lower stratosphere, valid from about -1 km to 32 km.

    ``temperature_offset`` shifts the whole temperature profile, which is a simple
    way to model a hot or cold launch day.
    """

    temperature_offset: float = 0.0  # K

    def at(self, altitude: float) -> AtmosphereState:
        h = R_EARTH * altitude / (R_EARTH + altitude)  # geopotential altitude
        base_h, base_t, lapse, base_p = _ISA_LAYERS[0]
        for layer in _ISA_LAYERS[1:]:
            if h >= layer[0]:
                base_h, base_t, lapse, base_p = layer
        dh = h - base_h
        t_std = base_t + lapse * dh
        if lapse == 0.0:
            p = base_p * math.exp(-G0 * dh / (R_AIR * base_t))
        else:
            p = base_p * (t_std / base_t) ** (-G0 / (lapse * R_AIR))
        t = t_std + self.temperature_offset
        rho = p / (R_AIR * t)
        a = math.sqrt(GAMMA * R_AIR * t)
        return AtmosphereState(t, p, rho, a)


@dataclass(frozen=True)
class ConstantAtmosphere:
    """Uniform density and speed of sound. Used for analytic test cases."""

    density: float = 1.225
    speed_of_sound: float = 340.294
    temperature: float = 288.15
    pressure: float = 101_325.0

    def at(self, altitude: float) -> AtmosphereState:
        return AtmosphereState(self.temperature, self.pressure, self.density, self.speed_of_sound)
