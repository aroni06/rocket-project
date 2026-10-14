"""Standard flight plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .atmosphere import G0
from .flight import FlightResult

LINE = "#2a78d6"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

_EVENT_LABELS = {"rail_exit": "Rail exit", "burnout": "Burnout", "apogee": "Apogee"}


def plot_flight(result: FlightResult, path: str | Path | None = None) -> Figure:
    """Altitude, speed and vertical acceleration against time, with flight events marked."""
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
        for key in _EVENT_LABELS:
            if key in result.events:
                ax.axvline(result.events[key].time, color=INK_MUTED, linewidth=0.8, linestyle="--")

    top = axes[0]
    for key, label in _EVENT_LABELS.items():
        if key in result.events:
            top.text(
                result.events[key].time,
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
