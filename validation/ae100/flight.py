"""What would have to be true for the simulator to reproduce the real flight?

The rocket in this folder flew to 212 ft on a windy day, against about 360 ft simulated.
The wind was not measured. This script finds the wind speed that makes the simulated climb
match the altimeter, then reads the ejection delay and the parachute's drag off the trace.

    python validation/ae100/flight.py [--plot flight.png]

Two things about a barometric altimeter matter here:

- It turns pressure into altitude assuming a standard atmosphere. On a hot day the air is
  thinner, a metre of climb drops the pressure less, and the altimeter reads low: by 6%
  at the 88 F it recorded. The simulated altitudes are converted to what the altimeter
  would have read before they are compared.
- It starts recording when it detects the launch, not at ignition, so its clock is offset
  from the simulation's by an unknown fraction of a second. The offset is fitted too.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from apogee import simulate
from apogee.config import build_config, read_config
from apogee.flight import FlightResult
from apogee.openrocket import read_simulation

HERE = Path(__file__).parent
FT = 0.3048
SITE_ELEVATION = 436 * FT  # from the altimeter's header [m]
STANDARD_TEMPERATURE = 288.15 - 0.0065 * SITE_ELEVATION  # at the site [K]
# The climb is fitted from when the altimeter's readings settle to just before ejection.
FIT_FROM, DESCENT_FROM, DESCENT_TO = 0.5, 5.6, 10.0  # altimeter clock [s]


def read_altimeter(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Time [s], altitude above the pad [m] and mean temperature [K]."""
    data = np.loadtxt(path, delimiter=",", comments="#", skiprows=5)
    return data[:, 0], data[:, 1] * FT, float((data[:, 3].mean() - 32.0) / 1.8 + 273.15)


class Flight:
    """The design flown at the real launch site, with the unknowns left adjustable."""

    def __init__(self, temperature: float) -> None:
        self.data = read_config(HERE / "config.toml")
        self.offset = temperature - STANDARD_TEMPERATURE
        # What the altimeter reads for each metre actually climbed.
        self.indicated = STANDARD_TEMPERATURE / temperature

    def fly(
        self, wind: float, extra_mass: float = 0.0, delay: float = 3.0, chute_cd: float = 0.8
    ) -> FlightResult:
        data = copy.deepcopy(self.data)
        data["environment"] = {
            "site_elevation_m": SITE_ELEVATION,
            "temperature_offset_K": self.offset,
            "wind_speed_mps": wind,
        }
        payload = next(m for m in data["mass"] if m["name"] == "payload")
        payload["mass_kg"] += extra_mass
        data["parachute"][0].update(cd=chute_cd, delay_s=delay)
        cfg = build_config(data, HERE)
        return simulate(cfg.rocket, cfg.environment, cfg.rail)


def misfit(result: FlightResult, scale: float, t: np.ndarray, h: np.ndarray) -> tuple[float, float]:
    """RMS altitude error [m] at the best clock offset [s] between altimeter and ignition."""
    best = (np.inf, 0.0)
    for offset in np.arange(-0.3, 1.5, 0.005):
        error = np.interp(t + offset, result.time, result.z) * scale - h
        rms = float(np.sqrt(np.mean(error**2)))
        if rms < best[0]:
            best = (rms, float(offset))
    return best


@dataclass
class Fit:
    """The real flight read through the simulator."""

    temperature: float  # air temperature from the altimeter [K]
    recorded_apogee: float  # as the altimeter read it [m]
    designed: FlightResult  # the design, at the real site and temperature
    wind: float  # wind that reproduces the climb [m/s]
    windy: FlightResult
    rms: float  # altitude error of that fit over the climb [m]
    clock_offset: float  # altimeter clock behind ignition [s]
    ejection_delay: float  # burnout to ejection [s]
    descent_rate: float  # measured, corrected for temperature [m/s]
    chute_cd: float  # parachute drag coefficient that gives that descent rate
    fitted: FlightResult  # wind, ejection delay and parachute drag all as measured
    extra_mass: float  # the other explanation: extra mass with no wind [kg]
    extra_mass_rms: float


def analyse() -> tuple[Fit, Flight, np.ndarray, np.ndarray]:
    t, h, temperature = read_altimeter(HERE / "altimeter.csv")
    ejection = t[np.argmin(h)]  # the ejection charge shows as one wildly low reading
    climb = (t >= FIT_FROM) & (t < ejection - 0.02)
    flight = Flight(temperature)

    def wind_error(wind: float) -> float:
        return misfit(flight.fly(wind), flight.indicated, t[climb], h[climb])[0]

    def mass_error(extra: float) -> float:
        return misfit(flight.fly(0.0, extra), flight.indicated, t[climb], h[climb])[0]

    wind = float(minimize_scalar(wind_error, bounds=(0.0, 15.0), method="bounded").x)
    extra = float(minimize_scalar(mass_error, bounds=(0.0, 0.2), method="bounded").x)
    windy = flight.fly(wind)
    rms, offset = misfit(windy, flight.indicated, t[climb], h[climb])

    delay = ejection + offset - windy.events["burnout"].time
    descent = (t >= DESCENT_FROM) & (t <= DESCENT_TO)
    measured_rate = float(-np.polyfit(t[descent], h[descent], 1)[0] / flight.indicated)
    chute_cd = 0.8 * (windy.landing_descent_rate / measured_rate) ** 2
    fit = Fit(
        temperature=temperature,
        recorded_apogee=float(h[t < ejection].max()),
        designed=flight.fly(2.0),
        wind=wind,
        windy=windy,
        rms=rms,
        clock_offset=offset,
        ejection_delay=delay,
        descent_rate=measured_rate,
        chute_cd=chute_cd,
        fitted=flight.fly(wind, delay=delay, chute_cd=chute_cd),
        extra_mass=extra,
        extra_mass_rms=mass_error(extra),
    )
    return fit, flight, t, h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", metavar="PNG", help="save the comparison plot")
    args = parser.parse_args()
    fit, flight, t, h = analyse()
    low = 1.0 - flight.indicated

    print(f"Air temperature from the altimeter: {fit.temperature - 273.15:.1f} C, so it reads")
    print(f"  {100 * low:.1f}% low. Recorded apogee {fit.recorded_apogee:.1f} m,")
    print(f"  about {fit.recorded_apogee / flight.indicated:.1f} m in fact.\n")
    print("As designed (2 m/s wind), at this site and temperature:")
    print(f"  apogee {fit.designed.apogee:.1f} m,", end=" ")
    print(f"descent {fit.designed.landing_descent_rate:.1f} m/s\n")
    print("Fitted to the altimeter's climb:")
    print(f"  wind                {fit.wind:5.1f} m/s  ({fit.wind * 2.23694:.0f} mph),", end=" ")
    print(f"{fit.rms:.1f} m RMS error")
    print(f"  apogee              {fit.windy.apogee:5.1f} m", end="    ")
    print(f"({fit.windy.apogee * flight.indicated:.1f} m on the altimeter)")
    print(f"  top vertical speed  {fit.windy.vz.max():5.1f} m/s  (altimeter's estimate: 26.5)")
    print(f"  ejection delay      {fit.ejection_delay:5.2f} s    (nominal 3)")
    print(f"  descent rate        {fit.descent_rate:5.1f} m/s", end="  ")
    print(f"(predicted {fit.windy.landing_descent_rate:.1f})")
    print(f"  parachute Cd        {fit.chute_cd:5.2f}      (design 0.8)\n")
    print("The climb alone cannot tell wind from weight. No wind and a rocket")
    print(f"  {1000 * fit.extra_mass:.0f} g heavier than designed fits as well", end=" ")
    print(f"({fit.extra_mass_rms:.1f} m RMS).")
    print("  The design's masses are the measured ones, which leaves the wind.")

    if args.plot:
        plot(args.plot, t, h, fit, flight)
        print(f"\nWrote {args.plot}")


def plot(path: str, t: np.ndarray, h: np.ndarray, fit: Fit, flight: Flight) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from apogee.plots import GRID, INK, INK_MUTED, SERIES

    sim = read_simulation(HERE / "ae100.ork")
    designed, fitted, offset = fit.designed, fit.fitted, fit.clock_offset
    ejection = t[np.argmin(h)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    ax.plot(
        sim.time, sim["Altitude"], color=SERIES[1], linewidth=3.2, alpha=0.55,
        label="OpenRocket, as designed (2 m/s wind)",
    )  # fmt: skip
    ax.plot(
        designed.time, designed.z, color=SERIES[0], linewidth=1.6,
        label="apogee, as designed (2 m/s wind)",
    )  # fmt: skip
    ax.plot(
        fitted.time, fitted.z * flight.indicated, color=SERIES[2], linewidth=1.8,
        label=f"apogee, fitted: {fit.wind:.0f} m/s wind, measured ejection and descent",
    )  # fmt: skip
    good = np.abs(t - ejection) > 0.02
    ax.plot(
        t[good] + offset, h[good], "o", color=INK_MUTED, markersize=3.2, markeredgewidth=0,
        label="Altimeter",
    )  # fmt: skip
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 120)
    ax.set_xlabel("Time since ignition [s]", color=INK)
    ax.set_ylabel("Altitude above pad [m]", color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(frameon=False, labelcolor=INK, loc="upper right", fontsize=9)
    ax.set_title("AE 100 rocket: simulated and flown", loc="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
