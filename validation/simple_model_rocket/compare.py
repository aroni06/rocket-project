"""Fly all five motor configurations of the simple model rocket against OpenRocket.

    python validation/simple_model_rocket/compare.py [--plot comparison.png]

The design file stores one simulation for each of its motor configurations. config.toml is
the default one (C6-5); the others differ only in the motor and its ejection delay.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from apogee import simulate
from apogee.config import build_config, read_config
from apogee.flight import FlightResult
from apogee.openrocket import Comparison, OpenRocketSimulation, compare, read_simulations
from apogee.rocket import Rocket

HERE = Path(__file__).parent
# Label, motor file and ejection delay [s], in the order of the file's simulations.
CONFIGURATIONS = [
    ("A8-3", "Estes_A8.eng", 3.0),
    ("B4-4", "Estes_B4.eng", 4.0),
    ("C6-3", "Estes_C6.eng", 3.0),
    ("C6-5", "Estes_C6.eng", 5.0),
    ("C6-7", "Estes_C6.eng", 7.0),
]
ROWS = [
    "Liftoff mass",
    "CG at liftoff",
    "Center of pressure",
    "Drag coefficient at top speed",
    "Rail exit speed",
    "Maximum speed",
    "Apogee",
    "Time to apogee",
    "Descent rate at landing",
    "Flight time",
]

Case = tuple[str, Rocket, FlightResult, OpenRocketSimulation]


def fly_all() -> list[Case]:
    """Each configuration flown here, with the OpenRocket simulation of the same one."""
    data = read_config(HERE / "config.toml")
    sims = read_simulations(HERE / "simple_model_rocket.ork")
    cases = []
    for (label, motor, delay), sim in zip(CONFIGURATIONS, sims, strict=True):
        variant = copy.deepcopy(data)
        variant["motor"]["file"] = motor
        variant["parachute"][0]["delay_s"] = delay
        cfg = build_config(variant, HERE)
        cases.append((label, cfg.rocket, simulate(cfg.rocket, cfg.environment, cfg.rail), sim))
    return cases


def table(cases: list[Case]) -> dict[str, dict[str, Comparison]]:
    """Comparison rows by configuration, then by quantity."""
    return {
        label: {row.label: row for row in compare(rocket, result, sim)}
        for label, rocket, result, sim in cases
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", metavar="PNG", help="save altitude against time for all five")
    args = parser.parse_args()
    cases = fly_all()
    rows = table(cases)

    print("Difference from OpenRocket [%]".ljust(34) + "".join(f"{c:>8}" for c in rows))
    for name in ROWS:
        print(name.ljust(34) + "".join(f"{rows[c][name].percent:+8.1f}" for c in rows))
    print("\nApogee [m]".ljust(35) + "".join(f"{c:>8}" for c in rows))
    print("  apogee".ljust(34) + "".join(f"{rows[c]['Apogee'].ours:8.1f}" for c in rows))
    print("  OpenRocket".ljust(34) + "".join(f"{rows[c]['Apogee'].openrocket:8.1f}" for c in rows))

    if args.plot:
        plot(args.plot, cases)
        print(f"\nWrote {args.plot}")


def plot(path: str, cases: list[Case]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from apogee.plots import GRID, INK, INK_MUTED, SERIES

    fig, axes = plt.subplots(1, len(cases), figsize=(13, 3.4), layout="constrained")
    for ax, (label, _, result, sim) in zip(axes, cases, strict=True):
        ax.plot(
            sim.time, sim["Altitude"], color=SERIES[1], linewidth=3.2, alpha=0.55,
            label="OpenRocket",
        )  # fmt: skip
        ax.plot(result.time, result.z, color=SERIES[0], linewidth=1.6, label="apogee")
        ax.set_title(label, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Time [s]", color=INK)
        ax.set_xlim(0, 1.6 * sim.events["apogee"])  # the climb and the start of the descent
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
        ax.tick_params(colors=INK_MUTED)
    axes[0].set_ylabel("Altitude above pad [m]", color=INK)
    axes[0].legend(frameon=False, labelcolor=INK, loc="lower center", fontsize=9)
    fig.suptitle(
        "Simple model rocket on five motors: this simulator against OpenRocket",
        color=INK,
        fontsize=12,
    )
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
