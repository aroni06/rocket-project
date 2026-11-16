"""Recover the Estes A8 and B4 motor files from the simulations in the design file.

    python validation/simple_model_rocket/make_motors.py

The comparison should fly the same motors OpenRocket flew. Estes_C6.eng is the published
NAR certification curve, and it is the one OpenRocket used. The A8 and B4 curves in
OpenRocket's database are not the ones at hand (the published A8 file burns for 0.53 s and
weighs 13.9 g; OpenRocket's burns for 0.73 s and weighs 16.3 g), so they are rebuilt from
what OpenRocket stored.

Its simulations hold the thrust over every time step, which is the thrust curve, and the
motor's mass rounded to the gram. The rounding hides the masses, but not the times at which
the rounded value changes: each change is the moment the mass passes a half gram. The
motor's mass falls in proportion to the impulse delivered, so a straight-line fit through
those moments gives the loaded and propellant masses to a few hundredths of a gram.

The method is checked on the C6, where the answer is known: rebuilt from the simulation it
comes out within 0.1% of the published file in impulse and 0.02 g in both masses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apogee import Motor
from apogee.openrocket import OpenRocketSimulation, read_simulations

HERE = Path(__file__).parent
# Motor name, the simulation that flew it, and its ejection delays.
MOTORS = [("A8", 0, "3"), ("B4", 1, "4")]
C6_SIMULATION = 4  # the longest C6 flight, for checking the method
TOLERANCE = 0.02  # N, how far the simplified curve may sit from OpenRocket's samples


def simplify(t: np.ndarray, f: np.ndarray, tolerance: float) -> list[int]:
    """Indices of the points to keep so that straight lines between them stay within
    ``tolerance`` of every dropped point (Ramer-Douglas-Peucker on the thrust axis)."""
    keep = [0, len(t) - 1]
    stack = [(0, len(t) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        line = f[a] + (f[b] - f[a]) * (t[a + 1 : b] - t[a]) / (t[b] - t[a])
        error = np.abs(f[a + 1 : b] - line)
        worst = int(np.argmax(error))
        if error[worst] > tolerance:
            mid = a + 1 + worst
            keep.append(mid)
            stack += [(a, mid), (mid, b)]
    return sorted(keep)


def masses(sim: OpenRocketSimulation, impulse_fraction: np.ndarray) -> tuple[float, float]:
    """Loaded and propellant mass [kg] from when the rounded motor mass changes."""
    mass = sim["Motor mass"]
    steps = np.nonzero(np.diff(mass) != 0)[0]
    fraction = 0.5 * (impulse_fraction[steps] + impulse_fraction[steps + 1])
    crossing = 0.5 * (mass[steps] + mass[steps + 1])
    slope, loaded = np.polyfit(fraction, crossing, 1)
    return float(loaded), float(-slope)


def rebuild(sim: OpenRocketSimulation) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Thrust curve (time, thrust), loaded mass and propellant mass from a simulation."""
    burnout = sim.events["burnout"]
    burning = sim.time <= burnout
    # OpenRocket stores the average thrust over the step that ends at each time, so a
    # sample belongs to the middle of its step.
    ends = sim.time[burning]
    middles = ends - 0.5 * np.diff(ends, prepend=0.0)
    t = np.concatenate([[0.0], middles, [burnout]])
    f = np.concatenate([[0.0], sim["Thrust"][burning], [0.0]])
    impulse = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])
    loaded, propellant = masses(sim, np.interp(sim.time, t, impulse / impulse[-1]))
    keep = simplify(t, f, TOLERANCE)
    return t[keep], f[keep], loaded, propellant


def main() -> None:
    sims = read_simulations(HERE / "simple_model_rocket.ork")
    for name, index, delays in MOTORS:
        t, f, loaded, propellant = rebuild(sims[index])
        lines = [
            f"; Estes {name}, as flown by OpenRocket 23.09 in simple_model_rocket.ork.",
            "; Thrust is the thrust column of the stored simulation, thinned to the points",
            f"; that reproduce it within {TOLERANCE} N. Masses are fitted to its motor-mass",
            "; column. Written by make_motors.py; see that file for the method.",
            f"{name} 18 70 {delays} {propellant:.5f} {loaded:.5f} Estes",
        ]
        lines += [f"{ti:.3f} {fi:.3f}" for ti, fi in zip(t[1:], f[1:], strict=True)]
        (HERE / f"Estes_{name}.eng").write_text("\n".join(lines) + "\n")
        impulse = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))
        print(
            f"{name}: {len(t) - 1} points, {impulse:.3f} N s, loaded {1000 * loaded:.2f} g, "
            f"propellant {1000 * propellant:.2f} g"
        )

    published = Motor.from_eng(HERE / "Estes_C6.eng")
    t, f, loaded, propellant = rebuild(sims[C6_SIMULATION])
    impulse = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))
    print(
        f"Check on the C6: rebuilt {impulse:.3f} N s, {1000 * loaded:.2f} g, "
        f"{1000 * propellant:.2f} g; published {published.total_impulse:.3f} N s, "
        f"{1000 * published.loaded_mass:.2f} g, {1000 * published.propellant_mass:.2f} g"
    )


if __name__ == "__main__":
    main()
