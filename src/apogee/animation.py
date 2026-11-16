"""Animated flight: the path drawn as the rocket flies it, beside a live readout.

The animation is a GIF so it plays anywhere, including a README. A real flight spends a
few seconds going up and a minute or two coming down, so playback is not uniform in
time: frames are spaced half by time and half by distance flown, which slows the ascent
down enough to watch and keeps the descent from taking the whole clip.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .atmosphere import G0
from .flight import FlightResult
from .plots import ACCENT, GRID, INK, INK_MUTED, LINE, SURFACE, trajectory_marks

PHASES = {"rail": "On the rail", "powered": "Powered ascent", "coast": "Coasting"}


def frame_times(result: FlightResult, frames: int) -> np.ndarray:
    """Flight times for ``frames`` evenly spaced frames, from liftoff to landing.

    Spacing is the average of "even in time" and "even in distance flown".
    """
    t = result.time
    distance = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(result.x), np.diff(result.z)))])
    progress = 0.5 * (t - t[0]) / (t[-1] - t[0]) + 0.5 * distance / distance[-1]
    return np.interp(np.linspace(0.0, 1.0, frames), progress, t)


def phase_at(result: FlightResult, t: float) -> str:
    """What the rocket is doing at time ``t``, in words."""
    opened = [ev for ev in result.deployments if ev.time <= t]
    if t >= result.events["landing"].time:
        return "Landed"
    if opened:
        return f"Under {opened[-1].name.removesuffix('_deploy')}"
    if t >= result.events["apogee"].time:
        return "Falling"
    k = min(int(np.searchsorted(result.time, t)), result.time.size - 1)
    return PHASES.get(str(result.phase[k]), "Coasting")


def _readout(fmt: str, value: float) -> str:
    """Format a live value, without the "-0 m" a value just below zero would give."""
    text = fmt.format(value)
    return fmt.format(0.0) if float(text.split()[0]) == 0 else text


def _milestones(result: FlightResult) -> list[tuple[float, str, str]]:
    """Headline numbers and the flight time at which each becomes known."""
    events = result.events
    landing = events["landing"]
    rows: list[tuple[float, str, str]] = []
    if (ev := events.get("rail_exit")) is not None:
        rows.append((ev.time, "Rail exit speed", f"{ev.speed:.1f} m/s"))
    ascent = result.time <= events["apogee"].time
    rows.append(
        (
            float(result.time[ascent][np.argmax(result.speed[ascent])]),
            "Max speed",
            f"{result.max_speed:.0f} m/s",
        )
    )
    if (ev := events.get("burnout")) is not None:
        rows.append((ev.time, "Max acceleration", f"{result.max_acceleration / G0:.1f} g"))
    rows.append((events["apogee"].time, "Apogee", f"{result.apogee:.0f} m"))
    rows.append((landing.time, "Landing descent rate", f"{result.landing_descent_rate:.1f} m/s"))
    rows.append((landing.time, "Landing distance", f"{abs(landing.downrange):.0f} m"))
    return sorted(rows, key=lambda row: row[0])


def animate_flight(
    result: FlightResult,
    path: str | Path,
    *,
    duration: float = 8.0,
    hold: float = 2.0,
    fps: int = 15,
    dpi: int = 100,
) -> Path:
    """Save an animated GIF of the flight to ``path``.

    The clip plays the flight in ``duration`` seconds and then holds the last frame for
    ``hold`` seconds so the final numbers can be read before it loops.
    """
    if duration <= 0 or hold < 0 or fps <= 0:
        raise ValueError("duration and fps must be positive and hold non-negative")
    path = Path(path)
    moving = max(2, round(duration * fps))
    times = frame_times(result, moving)
    times = np.concatenate([times, np.full(round(hold * fps), times[-1])])

    t = result.time
    x_at, z_at = np.interp(times, t, result.x), np.interp(times, t, result.z)
    readouts = [
        ("Altitude", np.interp(times, t, result.z), "{:.0f} m"),
        ("Speed", np.interp(times, t, result.speed), "{:.0f} m/s"),
        ("Vertical speed", np.interp(times, t, result.vz), "{:+.0f} m/s"),
        ("Vertical acceleration", np.interp(times, t, result.az) / G0, "{:+.1f} g"),
        ("Mach", np.interp(times, t, result.mach), "{:.2f}"),
        ("Downrange", x_at, "{:+.0f} m"),
    ]
    milestones = _milestones(result)
    marks = trajectory_marks(result)

    fig = plt.figure(figsize=(9, 5.2), facecolor=SURFACE)
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.5, 1], left=0.08, right=0.98, top=0.9, bottom=0.11
    )
    ax = fig.add_subplot(grid[0])
    panel = fig.add_subplot(grid[1])
    panel.axis("off")
    fig.suptitle(result.rocket_name, x=0.08, ha="left", color=INK, fontsize=12, fontweight="bold")

    # Flight path. The limits come from the whole flight so the view never jumps.
    ax.set_facecolor(SURFACE)
    ax.plot(result.x, result.z, alpha=0.0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)
    ax.set_xlabel("Downrange [m] (wind blows toward +x)", color=INK)
    ax.set_ylabel("Altitude above pad [m]", color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    ax.plot(0, 0, "^", color=INK_MUTED, markersize=6)
    (flown,) = ax.plot([], [], color=LINE, linewidth=2.0, solid_capstyle="round")
    (rocket,) = ax.plot(
        [], [], "o", color=ACCENT, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2
    )
    mark_artists = []
    for key, label, right in marks:
        ev = result.events[key]
        (dot,) = ax.plot(ev.downrange, ev.altitude, "o", color=INK, markersize=4, visible=False)
        note = ax.annotate(
            label,
            (ev.downrange, ev.altitude),
            xytext=(8, 6) if right else (-8, -12),
            textcoords="offset points",
            ha="left" if right else "right",
            color=INK,
            fontsize=9,
            visible=False,
        )
        mark_artists.append((ev.time, dot, note))

    # Readout: the clock and phase, live values, then headline numbers as they are reached.
    def label(y: float, text: str, **kwargs) -> plt.Text:
        return panel.text(0.0, y, text, transform=panel.transAxes, va="top", **kwargs)

    def value(y: float, **kwargs) -> plt.Text:
        return panel.text(1.0, y, "", transform=panel.transAxes, va="top", ha="right", **kwargs)

    clock = label(1.0, "", color=INK, fontsize=20, fontweight="bold")
    phase = label(0.89, "", color=INK_MUTED, fontsize=11)
    live = []
    for row, (name, _, _) in enumerate(readouts):
        y = 0.77 - 0.062 * row
        label(y, name, color=INK_MUTED, fontsize=10)
        live.append(value(y, color=INK, fontsize=10))
    top = 0.77 - 0.062 * len(readouts) - 0.03
    panel.plot([0, 1], [top, top], color=GRID, linewidth=1.0, transform=panel.transAxes)
    reached = []
    for row, (_, name, _) in enumerate(milestones):
        y = top - 0.035 - 0.062 * row
        label(y, name, color=INK_MUTED, fontsize=10)
        reached.append(value(y, color=INK, fontsize=10, fontweight="bold"))
    fig.text(
        0.98,
        0.02,
        "Playback slows the ascent and speeds up the descent",
        ha="right",
        color=INK_MUTED,
        fontsize=8,
    )

    def draw(i: int) -> list:
        now = times[i]
        past = t <= now
        flown.set_data(np.append(result.x[past], x_at[i]), np.append(result.z[past], z_at[i]))
        rocket.set_data([x_at[i]], [z_at[i]])
        for when, dot, note in mark_artists:
            dot.set_visible(now >= when)
            note.set_visible(now >= when)
        clock.set_text(f"T+ {now:.1f} s")
        phase.set_text(phase_at(result, now))
        for text, (_, series, fmt) in zip(live, readouts, strict=True):
            text.set_text(_readout(fmt, float(series[i])))
        for text, (when, _, shown) in zip(reached, milestones, strict=True):
            text.set_text(shown if now >= when else "")
        return [flown, rocket, clock, phase, *live, *reached]

    animation = FuncAnimation(fig, draw, frames=times.size, interval=1000 / fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
