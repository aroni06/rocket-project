"""Command-line interface: ``apogee simulate``, ``apogee sweep`` and ``apogee optimize``."""

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

    swp = sub.add_parser("sweep", help="fly every design on a study's grid")
    swp.add_argument("study", help="path to a design study (.toml)")
    swp.add_argument("--csv", metavar="CSV", help="save every design and its metrics")
    swp.add_argument("--plot", metavar="PNG", help="save a plot of the design space")

    opt = sub.add_parser("optimize", help="find the best design that meets a study's limits")
    opt.add_argument("study", help="path to a design study (.toml)")
    opt.add_argument(
        "--plot", metavar="PNG", help="also sweep the grid and save a plot with the optimum"
    )

    args = parser.parse_args(argv)
    commands = {"simulate": _simulate, "sweep": _sweep, "optimize": _optimize}
    return commands[args.command](args)


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


def _sweep(args: argparse.Namespace) -> int:
    from .study import Study

    result = Study.from_file(args.study).sweep()
    print(result)
    if args.csv:
        result.to_csv(args.csv)
        print(f"Wrote {args.csv}")
    if args.plot:
        _save_sweep_plot(result, args.plot)
    return 0 if result.best is not None else 1


def _optimize(args: argparse.Namespace) -> int:
    from .study import Study

    study = Study.from_file(args.study)
    optimum = study.optimize()
    print(optimum)
    if args.plot:
        _save_sweep_plot(study.sweep(), args.plot, optimum)
    return 0 if optimum.feasible else 1


def _save_sweep_plot(result, path: str, optimum=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from .plots import plot_sweep

    plot_sweep(result, path, optimum)
    print(f"Wrote {path}")


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
