"""Standard flight plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .atmosphere import G0
from .flight import FlightResult
from .rocket import Rocket

LINE = "#2a78d6"
ACCENT = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


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

    deploys = {k: f"{v} opens" for k, v in _event_labels(result).items() if k.endswith("_deploy")}
    marks = {"apogee": f"Apogee {result.apogee:.0f} m", **deploys}
    land = result.events["landing"]
    marks["landing"] = f"Landing, {land.downrange:.0f} m from pad"
    for i, (key, label) in enumerate(marks.items()):
        ev = result.events[key]
        ax.plot(ev.downrange, ev.altitude, "o", color=INK, markersize=4)
        # Alternate label sides so neighbouring markers (apogee, drogue) don't overlap.
        right = i % 2 == 0 or key == "landing"
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


def _nose_profile(shape: str, length: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Radius of the nose cone against distance from the tip."""
    x = np.linspace(0.0, length, 60)
    u = x / length
    if shape == "conical":
        y = radius * u
    elif shape == "parabolic":
        y = radius * (2 * u - u**2)
    elif shape == "elliptical":
        y = radius * np.sqrt(1 - (1 - u) ** 2)
    else:  # tangent ogive
        rho = (radius**2 + length**2) / (2 * radius)
        y = np.sqrt(rho**2 - (length - x) ** 2) + radius - rho
    return x, y


def plot_rocket(rocket: Rocket, path: str | Path | None = None) -> Figure:
    """Side view of the rocket with its center of gravity and center of pressure.

    The fins are drawn as one pair in the plane of the page whatever the fin count.
    """
    airframe = rocket.airframe
    if airframe is None:
        raise ValueError(f"{rocket.name} has no airframe geometry to draw")
    r, length = airframe.diameter / 2, airframe.length
    fins, x_fin = airframe.fins, airframe.fin_position

    fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    xn, yn = _nose_profile(airframe.nose.shape, airframe.nose.length, r)
    outline_x = np.concatenate([xn, [length, length], xn[::-1]])
    outline_y = np.concatenate([yn, [r, -r], -yn[::-1]])
    ax.fill(outline_x, outline_y, facecolor="none", edgecolor=INK_MUTED, linewidth=1.4)
    ax.plot([airframe.nose.length] * 2, [-r, r], color=INK_MUTED, linewidth=0.8)
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
    reach = r + fins.span
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
