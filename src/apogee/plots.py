"""Standard flight plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .atmosphere import G0
from .flight import FlightResult
from .openrocket import OpenRocketSimulation
from .rocket import Rocket
from .study import DesignPoint, Optimum, SweepResult, Variable

LINE = "#2a78d6"
ACCENT = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
HATCH = "#a9a8a2"
# Fixed order for series that are told apart by color (motors in a sweep).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
# One hue, light to dark, for a quantity shown as color over a 2D design space.
RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
UNITS = {"m": "m", "kg": "kg", "mps": "m/s", "deg": "deg", "s": "s", "g": "g", "cal": "cal"}


def _event_labels(result: FlightResult) -> dict[str, str]:
    """Events worth a marker line: burnout, apogee and each parachute deployment."""
    labels = {"burnout": "Burnout", "apogee": "Apogee"}
    for ev in result.deployments:
        labels[ev.name] = ev.name.removesuffix("_deploy").capitalize()
    return {k: v for k, v in labels.items() if k in result.events}


def plot_flight(result: FlightResult, path: str | Path | None = None) -> Figure:
    """Altitude, speed and vertical acceleration against time, with flight events marked."""
    event_labels = _event_labels(result)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True, layout="constrained")
    series = [
        (result.z, "Altitude above pad [m]"),
        (result.speed, "Speed [m/s]"),
        (result.az / G0, "Vertical acceleration [g]"),
    ]
    for ax, (values, label) in zip(axes, series, strict=True):
        ax.plot(result.time, values, color=LINE, linewidth=1.8)
        ax.set_ylabel(label, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
        ax.tick_params(colors=INK_MUTED)
        for key in event_labels:
            ax.axvline(result.events[key].time, color=INK_MUTED, linewidth=0.8, linestyle="--")

    # Events closer together than 3% of the flight share one label so the text doesn't collide.
    span = result.time[-1] - result.time[0]
    groups: list[tuple[float, list[str]]] = []
    for key, label in sorted(event_labels.items(), key=lambda kv: result.events[kv[0]].time):
        t_ev = result.events[key].time
        if groups and t_ev - groups[-1][0] < 0.03 * span:
            groups[-1][1].append(label)
        else:
            groups.append((t_ev, [label]))
    top = axes[0]
    for t_ev, names in groups:
        label = " / ".join(names)
        top.text(
            t_ev,
            0.45,
            f"{label} ",
            color=INK_MUTED,
            fontsize=9,
            rotation=90,
            va="top",
            ha="right",
            transform=top.get_xaxis_transform(),
        )
    apo = result.events["apogee"]
    top.annotate(
        f"{apo.altitude:.0f} m",
        (apo.time, apo.altitude),
        xytext=(8, -16),
        textcoords="offset points",
        color=INK,
        fontsize=10,
    )

    axes[-1].set_xlabel("Time since ignition [s]", color=INK)
    fig.suptitle(f"{result.rocket_name}: simulated flight", color=INK, fontsize=12)
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def trajectory_marks(result: FlightResult) -> list[tuple[str, str, bool]]:
    """Events to label on a flight path: (event key, label, label goes on the right)."""
    deploys = {k: f"{v} opens" for k, v in _event_labels(result).items() if k.endswith("_deploy")}
    marks = {"apogee": f"Apogee {result.apogee:.0f} m", **deploys}
    land = result.events["landing"]
    marks["landing"] = f"Landing, {land.downrange:.0f} m from pad"
    # Alternate label sides so neighbouring markers (apogee, drogue) don't overlap.
    return [
        (key, label, i % 2 == 0 or key == "landing") for i, (key, label) in enumerate(marks.items())
    ]


def plot_trajectory(result: FlightResult, path: str | Path | None = None) -> Figure:
    """Flight path in the vertical plane (downrange vs altitude), drawn to scale."""
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.plot(result.x, result.z, color=LINE, linewidth=1.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Downrange [m] (wind blows toward +x)", color=INK)
    ax.set_ylabel("Altitude above pad [m]", color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)

    for key, label, right in trajectory_marks(result):
        ev = result.events[key]
        ax.plot(ev.downrange, ev.altitude, "o", color=INK, markersize=4)
        ax.annotate(
            label,
            (ev.downrange, ev.altitude),
            xytext=(8, 6) if right else (-8, -12),
            textcoords="offset points",
            ha="left" if right else "right",
            color=INK,
            fontsize=9,
        )
    ax.plot(0, 0, "^", color=INK_MUTED, markersize=6)
    fig.suptitle(f"{result.rocket_name}: flight path", color=INK, fontsize=12)
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_rocket(rocket: Rocket, path: str | Path | None = None) -> Figure:
    """Side view of the rocket with its center of gravity and center of pressure.

    The fins are drawn as one pair in the plane of the page whatever the fin count.
    """
    airframe = rocket.airframe
    if airframe is None:
        raise ValueError(f"{rocket.name} has no airframe geometry to draw")
    r, length = airframe.body.diameter / 2, airframe.length
    r_nose, x_body = airframe.nose_diameter / 2, airframe.body_start
    fins, x_fin = airframe.fins, airframe.fin_position

    fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    xn, yn = airframe.nose.profile(r_nose)
    outline_x = np.concatenate([xn, [x_body, length, length, x_body], xn[::-1]])
    outline_y = np.concatenate([yn, [r, r, -r, -r], -yn[::-1]])
    ax.fill(outline_x, outline_y, facecolor="none", edgecolor=INK_MUTED, linewidth=1.4)
    ax.plot([airframe.nose.length] * 2, [-r_nose, r_nose], color=INK_MUTED, linewidth=0.8)
    if airframe.transition is not None:
        ax.plot([x_body] * 2, [-r, r], color=INK_MUTED, linewidth=0.8)
    fin_x = [
        x_fin,
        x_fin + fins.sweep,
        x_fin + fins.sweep + fins.tip_chord,
        x_fin + fins.root_chord,
    ]
    for sign in (1, -1):
        fin_y = [sign * r, sign * (r + fins.span), sign * (r + fins.span), sign * r]
        ax.fill(fin_x, fin_y, facecolor="none", edgecolor=INK_MUTED, linewidth=1.4)

    burn = rocket.motor.burn_time if rocket.motor else 0.0
    cp = rocket.center_of_pressure.position
    cg0, cg1 = rocket.cg_at(0.0), rocket.cg_at(burn)
    reach = max(r + fins.span, r_nose)
    ax.plot(cg0, 0, "o", color=LINE, markersize=9)
    ax.plot(cp, 0, "o", color=ACCENT, markersize=9)
    ax.annotate(
        f"CG at liftoff, {cg0:.3f} m",
        (cg0, 0),
        xytext=(cg0, 1.5 * reach),
        ha="right",
        color=INK,
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    ax.annotate(
        f"CP, {cp:.3f} m",
        (cp, 0),
        xytext=(cp, 1.5 * reach),
        ha="left",
        color=INK,
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    if rocket.motor and abs(cg1 - cg0) > 1e-6:
        ax.plot(cg1, 0, "o", markerfacecolor="none", markeredgecolor=LINE, markersize=9)
        ax.annotate(
            f"CG at burnout, {cg1:.3f} m",
            (cg1, 0),
            xytext=(cg1, -1.7 * reach),
            ha="right",
            color=INK,
            fontsize=10,
            arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
        )

    ax.set_aspect("equal")
    ax.set_xlim(-0.03 * length, 1.03 * length)
    ax.set_ylim(-2.3 * reach, 2.3 * reach)
    ax.set_xlabel("Distance from nose tip [m]", color=INK)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    margin0, margin1 = rocket.static_margin_at(0.0), rocket.static_margin_at(burn)
    fig.suptitle(
        f"{rocket.name}: static margin {margin0:.2f} cal at liftoff, {margin1:.2f} cal at burnout",
        color=INK,
        fontsize=12,
    )
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


# Design studies


def plot_comparison(
    rocket: Rocket,
    result: FlightResult,
    sim: OpenRocketSimulation,
    path: str | Path | None = None,
) -> Figure:
    """A flight and an OpenRocket simulation of the same rocket, plotted together.

    Altitude and speed against time, and the drag coefficient during the climb. The drag
    coefficients are both taken at the speed and altitude of OpenRocket's flight, so that
    panel compares the two drag models and not the two trajectories.
    """
    climb = sim.time <= sim.events.get("apogee", sim.time[-1])
    drag = "Drag coefficient" in sim.columns and rocket.airframe is not None
    fig, axes = plt.subplots(
        1, 3 if drag else 2, figsize=(12 if drag else 8.5, 3.9), layout="constrained"
    )
    panels = [
        (result.time, result.z, sim.time, sim["Altitude"], "Altitude above pad [m]"),
        (result.time, result.speed, sim.time, sim["Total velocity"], "Speed [m/s]"),
    ]
    if drag:
        ours = [
            rocket.drag_coefficient(mach, reynolds / rocket.airframe.length)
            for mach, reynolds in zip(
                sim["Mach number"][climb], sim["Reynolds number"][climb], strict=True
            )
        ]
        panels.append(
            (
                sim.time[climb],
                ours,
                sim.time[climb],
                sim["Drag coefficient"][climb],
                "Drag coefficient",
            )
        )
    for ax, (t_ours, ours, t_theirs, theirs, label) in zip(axes, panels, strict=True):
        ax.plot(t_theirs, theirs, color=SERIES[1], linewidth=3.2, alpha=0.55, label="OpenRocket")
        ax.plot(t_ours, ours, color=SERIES[0], linewidth=1.6, label="apogee")
        ax.set_xlabel("Time [s]", color=INK)
        ax.set_ylabel(label, color=INK)
        _style(ax)
    axes[-1].set_ylim(bottom=0)
    axes[0].legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.suptitle(f"{result.rocket_name}: this simulator against OpenRocket", color=INK, fontsize=12)
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_sweep(
    sweep: SweepResult, path: str | Path | None = None, optimum: Optimum | None = None
) -> Figure:
    """Plot a sweep's objective over its design space, showing which designs are allowed.

    The form follows the study's variables: a line for one range, bars for one list of
    choices, a line per choice for a list and a range, and a filled contour map for two
    ranges. Designs that break a constraint are drawn faded (lines, bars) or veiled (map).
    The best design on the grid is marked, or the optimizer's result if ``optimum`` is given.
    """
    variables = list(sweep.study.variables)
    numeric = [v for v in variables if v.is_numeric]
    named = [v for v in variables if not v.is_numeric]
    if len(variables) == 1 and named:
        fig = _sweep_bars(sweep)
    elif len(variables) == 1:
        fig = _sweep_lines(sweep, numeric[0], None, optimum)
    elif len(variables) == 2 and len(named) == 1 and len(named[0].grid) <= len(SERIES):
        fig = _sweep_lines(sweep, numeric[0], named[0], optimum)
    elif len(variables) == 2 and not named:
        fig = _sweep_map(sweep, optimum)
    else:
        raise ValueError(
            "a sweep plot needs one or two variables, at most one of them a list of names "
            f"with up to {len(SERIES)} entries; use the CSV for larger studies"
        )
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def _pretty(name: str) -> str:
    """'parachute.main.diameter_m' -> 'Parachute main diameter [m]'."""
    words = name.replace(".", "_").split("_")
    unit = UNITS.get(words[-1]) if len(words) > 1 else None
    if unit:
        words = words[:-1]
    text = " ".join(words).capitalize()
    return f"{text} [{unit}]" if unit else text


def _with_unit(metric: str, value: float) -> str:
    unit = UNITS.get(metric.rsplit("_", 1)[-1])
    return f"{value:.0f} {unit}" if unit else f"{value:.3g}"


def _worst_margin(sweep: SweepResult) -> np.ndarray:
    """Over the grid, the scaled distance inside the tightest limit; negative breaks it."""
    worst = np.full(sweep.shape, np.inf)
    for c in sweep.study.constraints:
        for margin in c.margins(sweep.metric(c.metric)):
            worst = np.minimum(worst, np.nan_to_num(margin, nan=-1.0))
    return worst


def _choice_label(value: object) -> str:
    text = str(value)
    return Path(text).stem if "/" in text or "\\" in text else text


def _style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)


def _title(ax: plt.Axes, sweep: SweepResult) -> None:
    study = sweep.study
    ax.set_title(study.name, loc="left", color=INK, fontsize=12, fontweight="bold", pad=24)
    limits = "; ".join(f"{_pretty(c.metric)} {c.describe()}" for c in study.constraints)
    subtitle = f"Limits: {limits}" if limits else "No constraints"
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_MUTED, fontsize=8.5, va="bottom")


def _marked_point(sweep: SweepResult, optimum: Optimum | None) -> tuple[DesignPoint | None, str]:
    if optimum is not None and optimum.feasible:
        return optimum.best, "Optimum"
    return sweep.best, "Best on the grid"


def _mark(ax: plt.Axes, x: float, y: float, text: str, color: str = ACCENT) -> None:
    ax.plot(
        x, y, "o", color=color, markersize=10, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5
    )
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(10, 10),
        color=INK,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": "none"},
    )


def _sweep_lines(
    sweep: SweepResult, xvar: Variable, series: Variable | None, optimum: Optimum | None
) -> Figure:
    """Objective against one numeric variable, with one line per named choice."""
    study = sweep.study
    metric = study.objective.metric
    variables = list(study.variables)
    x_axis = variables.index(xvar)
    values = np.moveaxis(sweep.metric(metric), x_axis, -1).reshape(-1, len(xvar.grid))
    margins = np.moveaxis(_worst_margin(sweep), x_axis, -1).reshape(-1, len(xvar.grid))
    x = np.array(xvar.grid, dtype=float)
    # Between grid points, interpolate so the bold part of a line ends where a limit is met.
    fine = np.linspace(x[0], x[-1], 400)
    names = [_choice_label(v) for v in series.grid] if series else [None]

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name, color, y, margin in zip(names, SERIES, values, margins, strict=False):
        ax.plot(x, y, color=color, linewidth=1.2, alpha=0.45)
        ok = np.interp(fine, x, margin) >= 0
        y_fine = np.where(ok, np.interp(fine, x, y), np.nan)
        ax.plot(fine, y_fine, color=color, linewidth=2.6, solid_capstyle="round")
        if name is not None:
            ax.plot([], [], color=color, linewidth=2.6, label=name)
            ax.annotate(
                name,
                (x[-1], y[-1]),
                textcoords="offset points",
                xytext=(6, 0),
                va="center",
                color=INK,
                fontsize=9,
            )
    ax.plot([], [], color=INK_MUTED, linewidth=2.6, label="Meets every limit")
    ax.plot([], [], color=INK_MUTED, linewidth=1.2, alpha=0.45, label="Breaks a limit")
    for c in study.constraints:
        if c.metric != metric:
            continue
        for bound in (c.lower, c.upper):
            if bound is not None:
                ax.axhline(bound, color=INK_MUTED, linewidth=1.0)
                ax.annotate(
                    f"limit {bound:g}",
                    (x[0], bound),
                    textcoords="offset points",
                    xytext=(2, 4),
                    color=INK_MUTED,
                    fontsize=8.5,
                )
    point, label = _marked_point(sweep, optimum)
    if point is not None:
        value = point.metrics[metric]
        text = f"{label}: {_with_unit(metric, value)}"
        _mark(ax, point.values[xvar.path], value, text, color=INK)
    ax.set_xlabel(_pretty(xvar.path), color=INK)
    ax.set_ylabel(_pretty(metric), color=INK)
    ax.margins(x=0.02)
    if series:
        ax.set_xlim(right=x[-1] + 0.22 * (x[-1] - x[0]))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="best")
    _style(ax)
    _title(ax, sweep)
    return fig


def _sweep_bars(sweep: SweepResult) -> Figure:
    """Objective for each entry of a single list of named choices."""
    study = sweep.study
    metric = study.objective.metric
    names = [_choice_label(v) for v in study.variables[0].grid]
    values = sweep.metric(metric)
    allowed = sweep.feasible
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.8), layout="constrained")
    rows = np.arange(len(names))
    ax.barh(rows[allowed], values[allowed], height=0.5, color=LINE, label="Meets every limit")
    ax.barh(
        rows[~allowed], values[~allowed], height=0.5, color=LINE, alpha=0.3, label="Breaks a limit"
    )
    for row, value in zip(rows, values, strict=True):
        if np.isfinite(value):
            ax.annotate(
                f"{value:.0f}",
                (value, row),
                textcoords="offset points",
                xytext=(5, 0),
                va="center",
                color=INK,
                fontsize=9,
            )
    ax.set_yticks(rows, names)
    ax.invert_yaxis()
    ax.set_xlabel(_pretty(metric), color=INK)
    ax.margins(x=0.1)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="lower right")
    _style(ax)
    ax.grid(False, axis="y")
    _title(ax, sweep)
    return fig


def _sweep_map(sweep: SweepResult, optimum: Optimum | None) -> Figure:
    """Objective as a filled contour map over two numeric variables."""
    from matplotlib.colors import LinearSegmentedColormap

    study = sweep.study
    metric = study.objective.metric
    xvar, yvar = study.variables
    x = np.array(xvar.grid, dtype=float)
    y = np.array(yvar.grid, dtype=float)
    z = sweep.metric(metric).T  # rows follow y, columns follow x

    fig, ax = plt.subplots(figsize=(8, 5.6), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    filled = ax.contourf(x, y, z, levels=10, cmap=cmap)
    bar = fig.colorbar(filled, ax=ax, pad=0.02)
    bar.set_label(_pretty(metric), color=INK)
    bar.ax.tick_params(colors=INK_MUTED)
    bar.outline.set_visible(False)

    # Veil everything that breaks a limit, then draw and name each limit's boundary.
    worst = _worst_margin(sweep)
    if study.constraints and worst.min() < 0:
        ax.contourf(x, y, worst.T, levels=[-np.inf, 0.0], colors=[SURFACE], alpha=0.8)
        hatch = ax.contourf(x, y, worst.T, levels=[-np.inf, 0.0], colors="none", hatches=["///"])
        hatch.set_edgecolor(HATCH)
        hatch.set_linewidth(0.0)
    for c in study.constraints:
        field = sweep.metric(c.metric).T
        for bound in (c.lower, c.upper):
            if bound is None or not np.nanmin(field) < bound < np.nanmax(field):
                continue
            edge = ax.contour(x, y, field, levels=[bound], colors=[INK], linewidths=1.6)
            name = _pretty(c.metric).split(" [")[0]
            ax.clabel(edge, fmt={bound: f"{name} = {bound:g}"}, fontsize=8.5, inline_spacing=6)

    point, label = _marked_point(sweep, optimum)
    if point is not None:
        value = point.metrics[metric]
        text = f"{label}: {_with_unit(metric, value)}"
        _mark(ax, point.values[xvar.path], point.values[yvar.path], text)
    ax.set_xlabel(_pretty(xvar.path), color=INK)
    ax.set_ylabel(_pretty(yvar.path), color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    _title(ax, sweep)
    if study.constraints and worst.min() < 0:
        ax.text(
            1.0,
            1.02,
            "Hatched: breaks a limit",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=INK_MUTED,
            fontsize=8.5,
        )
    return fig
