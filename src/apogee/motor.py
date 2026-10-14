"""Solid rocket motor model built from a thrust curve.

Thrust is linearly interpolated between curve points. Propellant is consumed in
proportion to the impulse delivered so far, the same convention OpenRocket uses:

    m_prop(t) = m_prop,0 * (1 - I(t) / I_total)

Thrust curves are read from RASP ``.eng`` files, the format published on
https://www.thrustcurve.org.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Motor:
    """A motor defined by its thrust curve and masses.

    ``time`` starts at 0 and ``thrust`` is zero after the last point. Masses are in kg,
    lengths in m.
    """

    name: str
    time: np.ndarray
    thrust: np.ndarray
    propellant_mass: float
    loaded_mass: float
    diameter: float = 0.0
    length: float = 0.0
    manufacturer: str = ""
    _cumulative_impulse: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.thrust = np.asarray(self.thrust, dtype=float)
        if self.time.ndim != 1 or self.time.shape != self.thrust.shape or self.time.size < 2:
            raise ValueError("time and thrust must be 1D arrays of equal length >= 2")
        if self.time[0] != 0.0:
            raise ValueError("thrust curve must start at t = 0")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("thrust curve times must be strictly increasing")
        if np.any(self.thrust < 0):
            raise ValueError("thrust must be non-negative")
        if not 0 <= self.propellant_mass <= self.loaded_mass:
            raise ValueError("need 0 <= propellant_mass <= loaded_mass")
        segment_impulse = 0.5 * (self.thrust[1:] + self.thrust[:-1]) * np.diff(self.time)
        self._cumulative_impulse = np.concatenate([[0.0], np.cumsum(segment_impulse)])
        if self.total_impulse <= 0:
            raise ValueError("thrust curve has zero total impulse")

    @classmethod
    def constant(
        cls,
        thrust: float,
        burn_time: float,
        propellant_mass: float,
        loaded_mass: float | None = None,
        name: str = "constant",
    ) -> Motor:
        """A motor with constant thrust for ``burn_time`` seconds, then zero."""
        if loaded_mass is None:
            loaded_mass = propellant_mass
        return cls(
            name,
            np.array([0.0, burn_time]),
            np.array([thrust, thrust]),
            propellant_mass,
            loaded_mass,
        )

    @classmethod
    def from_eng(cls, path: str | Path) -> Motor:
        """Load the first motor in a RASP ``.eng`` file."""
        return parse_eng(Path(path).read_text())

    @property
    def burn_time(self) -> float:
        return float(self.time[-1])

    @property
    def total_impulse(self) -> float:
        return float(self._cumulative_impulse[-1])

    @property
    def average_thrust(self) -> float:
        return self.total_impulse / self.burn_time

    @property
    def max_thrust(self) -> float:
        return float(self.thrust.max())

    @property
    def impulse_class(self) -> str:
        """NAR/TRA impulse class letter (A = 1.26-2.5 N s, each letter doubles)."""
        i = self.total_impulse
        if i <= 0.3125:
            return "1/8A"
        if i <= 0.625:
            return "1/4A"
        if i <= 1.25:
            return "1/2A"
        return chr(ord("A") + max(0, math.ceil(math.log2(i / 2.5))))

    def thrust_at(self, t: float) -> float:
        if t < 0.0 or t > self.time[-1]:
            return 0.0
        return float(np.interp(t, self.time, self.thrust))

    def impulse_at(self, t: float) -> float:
        """Impulse delivered from ignition to time ``t`` [N s], exact for the linear curve."""
        if t <= 0.0:
            return 0.0
        if t >= self.time[-1]:
            return self.total_impulse
        k = int(np.searchsorted(self.time, t, side="right")) - 1
        f_t = self.thrust_at(t)
        return float(
            self._cumulative_impulse[k] + 0.5 * (self.thrust[k] + f_t) * (t - self.time[k])
        )

    def propellant_mass_at(self, t: float) -> float:
        return self.propellant_mass * (1.0 - self.impulse_at(t) / self.total_impulse)

    def mass_at(self, t: float) -> float:
        """Total motor mass (casing plus remaining propellant) [kg]."""
        return self.loaded_mass - self.propellant_mass + self.propellant_mass_at(t)


def parse_eng(text: str) -> Motor:
    """Parse the first motor in RASP ``.eng`` text.

    Header line: ``name diameter_mm length_mm delays propellant_kg total_kg manufacturer``,
    then ``time thrust`` pairs. Lines starting with ``;`` are comments.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(";")]
    if not lines:
        raise ValueError("empty .eng file")
    header = lines[0].split()
    if len(header) < 7:
        raise ValueError(f"malformed .eng header: {lines[0]!r}")
    name, dia_mm, len_mm, _delays, prop_kg, total_kg = header[:6]
    manufacturer = " ".join(header[6:])

    times: list[float] = []
    thrusts: list[float] = []
    for ln in lines[1:]:
        parts = ln.split()
        try:
            t, f = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            break  # next motor header or trailing junk
        if times and t <= times[-1]:
            break
        times.append(t)
        thrusts.append(f)
        if f == 0.0 and t > 0.0:
            break  # RASP curves end with a zero-thrust point
    if not times:
        raise ValueError("no thrust data in .eng file")
    if times[0] > 0.0:
        times.insert(0, 0.0)
        thrusts.insert(0, 0.0)

    return Motor(
        name=name,
        time=np.array(times),
        thrust=np.array(thrusts),
        propellant_mass=float(prop_kg),
        loaded_mass=float(total_kg),
        diameter=float(dia_mm) / 1000.0,
        length=float(len_mm) / 1000.0,
        manufacturer=manufacturer,
    )
