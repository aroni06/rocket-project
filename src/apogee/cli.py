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
    sim.add_argument("--trajectory", metavar="PNG", help="save the flight path plot")
    sim.add_argument("--drawing", metavar="PNG", help="save a side view with CG and CP marked")

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
    margin = result.static_margin_at("rail_exit")
    if margin is not None:
        print(f"Stability: {_stability_verdict(margin)}")
    if args.csv:
        result.to_csv(args.csv)
        print(f"Wrote {args.csv}")
    if args.plot or args.trajectory or args.drawing:
        import matplotlib

        matplotlib.use("Agg")
        from .plots import plot_flight, plot_rocket, plot_trajectory

        if args.plot:
            plot_flight(result, args.plot)
            print(f"Wrote {args.plot}")
        if args.trajectory:
            plot_trajectory(result, args.trajectory)
            print(f"Wrote {args.trajectory}")
        if args.drawing:
            if cfg.rocket.airframe is None:
                print("No drawing: the config has no [nose], [body] and [fins]", file=sys.stderr)
                return 1
            plot_rocket(cfg.rocket, args.drawing)
            print(f"Wrote {args.drawing}")
    return 0


def _stability_verdict(margin: float) -> str:
    """Rule-of-thumb reading of the static margin at rail exit, in calibers."""
    at = f"{margin:.2f} cal at rail exit"
    if margin < 0:
        return f"UNSTABLE, {at}: the center of pressure is ahead of the center of gravity"
    if margin < 1:
        return f"marginal, {at}: the usual target is 1 to 2 cal"
    if margin <= 2.5:
        return f"stable, {at}"
    return f"overstable, {at}: expect strong weathercocking in wind"


if __name__ == "__main__":
    raise SystemExit(main())
