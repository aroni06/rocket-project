"""Command-line interface: ``apogee simulate config.toml``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apogee", description="Rocket flight simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="simulate one flight from a TOML config")
    sim.add_argument("config", help="path to a rocket config (.toml)")
    sim.add_argument("--plot", metavar="PNG", help="save altitude/speed/acceleration plot")
    sim.add_argument("--csv", metavar="CSV", help="save the full time history")

    args = parser.parse_args(argv)
    if args.command == "simulate":
        return _simulate(args)
    return 1


def _simulate(args: argparse.Namespace) -> int:
    from .config import load_config
    from .flight import simulate

    cfg = load_config(args.config)
    motor = cfg.rocket.motor
    if motor is not None:
        print(
            f"Motor: {motor.name} ({motor.impulse_class}, {motor.total_impulse:.1f} N s, "
            f"{motor.burn_time:.2f} s burn)"
        )
    try:
        result = simulate(cfg.rocket, cfg.environment, cfg.rail)
    except RuntimeError as exc:
        print(f"Simulation failed: {exc}", file=sys.stderr)
        return 1
    print(result)
    if args.csv:
        result.to_csv(args.csv)
        print(f"Wrote {args.csv}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        from .plots import plot_flight

        plot_flight(result, args.plot)
        print(f"Wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
