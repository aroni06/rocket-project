"""Read OpenRocket simulations and compare a flight against one.

An OpenRocket design file (``.ork``) stores the results of its simulations next to the
design: every variable at every time step, the flight events and the launch conditions.
:func:`read_simulation` reads that, and :func:`compare` lines a flight of the same rocket
in this simulator up against it, quantity by quantity.

The design itself is not converted. Write the config by hand from the design, then check
it here: the first rows of the comparison (mass, center of gravity, center of pressure)
show whether the two programs are describing the same rocket before the flight numbers
are compared.

All OpenRocket data is in SI units, with angles in radians.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .flight import FlightResult
from .rocket import Rocket


@dataclass
class OpenRocketSimulation:
    """One simulation stored in an OpenRocket file.

    ``columns`` maps OpenRocket's variable names (``"Altitude"``, ``"Total velocity"``,
    ``"Drag coefficient"``...) to arrays over time; index the simulation directly to get
    one, ``sim["Altitude"]``. ``events`` maps event names (``"launchrod"``, ``"burnout"``,
    ``"apogee"``, ``"groundhit"``...) to times. ``summary`` holds the headline results
    OpenRocket stores (``"maxaltitude"``, ``"groundhitvelocity"``...) and ``conditions``
    the launch conditions (``"windaverage"``, ``"launchrodlength"``...).
    """

    name: str
    columns: dict[str, np.ndarray]
    events: dict[str, float] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)
    conditions: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, column: str) -> np.ndarray:
        try:
            return self.columns[column]
        except KeyError:
            raise KeyError(
                f"{column!r} is not in the simulation; it has {sorted(self.columns)}"
            ) from None

    @property
    def time(self) -> np.ndarray:
        return self["Time"]

    def at(self, column: str, time: float) -> float:
        """A variable at a time, interpolated, skipping steps where it is undefined."""
        values = self[column]
        ok = np.isfinite(values)
        return float(np.interp(time, self.time[ok], values[ok]))


def _floats(items: dict[str, str]) -> dict[str, float]:
    out = {}
    for key, value in items.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _root(path: Path) -> ET.Element:
    """Parse an .ork file, which is either plain XML or a zip archive holding one."""
    raw = path.read_bytes()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = [n for n in archive.namelist() if n.endswith(".ork")] or archive.namelist()
            raw = archive.read(names[0])
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"{path}: not an OpenRocket file ({exc})") from exc


def read_simulations(path: str | Path) -> list[OpenRocketSimulation]:
    """Every simulation in an OpenRocket file that has stored results."""
    path = Path(path)
    simulations = []
    for node in _root(path).iter("simulation"):
        flight = node.find("flightdata")
        branch = flight.find("databranch") if flight is not None else None
        if branch is None:
            continue  # never run, or saved without its data
        names = branch.attrib["types"].split(",")
        rows = [point.text.split(",") for point in branch.findall("datapoint")]
        if not rows:
            continue
        data = np.array(rows, dtype=float)
        events: dict[str, float] = {}
        for event in branch.findall("event"):
            events.setdefault(event.attrib["type"], float(event.attrib["time"]))
        conditions = node.find("conditions")
        simulations.append(
            OpenRocketSimulation(
                name=node.findtext("name", default="simulation").strip(),
                columns={name: data[:, i] for i, name in enumerate(names)},
                events=events,
                summary=_floats(dict(flight.attrib)),
                conditions=_floats(
                    {child.tag: child.text for child in conditions}
                    if conditions is not None
                    else {}
                ),
            )
        )
    return simulations


def read_simulation(path: str | Path, index: int = 0) -> OpenRocketSimulation:
    """One simulation from an OpenRocket file, the first unless ``index`` says otherwise."""
    simulations = read_simulations(path)
    if not simulations:
        raise ValueError(
            f"{path}: no simulation results stored. Run the simulation in OpenRocket and "
            "save the file with its simulation data."
        )
    try:
        return simulations[index]
    except IndexError:
        raise ValueError(f"{path}: has {len(simulations)} simulation(s), not {index + 1}") from None


@dataclass(frozen=True)
class Comparison:
    """One quantity from both programs."""

    label: str
    unit: str
    ours: float
    openrocket: float

    @property
    def difference(self) -> float:
        return self.ours - self.openrocket

    @property
    def percent(self) -> float:
        """Difference as a percentage of OpenRocket's value."""
        return 100.0 * self.difference / abs(self.openrocket) if self.openrocket else float("nan")


def compare(rocket: Rocket, result: FlightResult, sim: OpenRocketSimulation) -> list[Comparison]:
    """Line a flight up against an OpenRocket simulation of the same rocket.

    The rows run from what goes into the flight (mass, balance, drag) to what comes out
    (speeds, apogee, descent), so a disagreement can be traced to where it starts.
    Quantities either side doesn't have are left out.
    """
    rows: list[Comparison] = []

    def add(label: str, unit: str, ours: float | None, theirs: float | None) -> None:
        if ours is not None and theirs is not None and np.isfinite(ours) and np.isfinite(theirs):
            rows.append(Comparison(label, unit, float(ours), float(theirs)))

    t_burnout = sim.events.get("burnout")
    columns = sim.columns

    if "Mass" in columns:
        add("Liftoff mass", "kg", rocket.liftoff_mass, sim["Mass"][0])
        if t_burnout is not None:
            add("Burnout mass", "kg", rocket.mass_at(t_burnout), sim.at("Mass", t_burnout))
    if rocket.airframe is not None and "CG location" in columns:
        add("CG at liftoff", "m", rocket.cg_at(0.0), sim["CG location"][0])
        if t_burnout is not None:
            add("CG at burnout", "m", rocket.cg_at(t_burnout), sim.at("CG location", t_burnout))
    if rocket.airframe is not None and "Total velocity" in columns:
        fastest = int(np.nanargmax(sim["Total velocity"]))
        if "CP location" in columns:
            # OpenRocket's CP moves with angle of attack. Take it at top speed, where the
            # rocket is flying straight and Barrowman's small-angle value applies.
            add(
                "Center of pressure",
                "m",
                rocket.center_of_pressure.position,
                sim["CP location"][fastest],
            )
        if {"Drag coefficient", "Mach number", "Reynolds number"} <= columns.keys():
            add(
                "Drag coefficient at top speed",
                "",
                rocket.drag_coefficient(
                    sim["Mach number"][fastest],
                    sim["Reynolds number"][fastest] / rocket.airframe.length,
                ),
                sim["Drag coefficient"][fastest],
            )
    add("Rail exit speed", "m/s", result.rail_exit_speed, _rod_exit_speed(sim))
    add("Maximum speed", "m/s", result.max_speed, sim.summary.get("maxvelocity"))
    if t_burnout is not None and "burnout" in result.events:
        add(
            "Burnout altitude",
            "m",
            result.events["burnout"].altitude,
            sim.at("Altitude", t_burnout),
        )
    add("Apogee", "m", result.apogee, sim.summary.get("maxaltitude"))
    add("Time to apogee", "s", result.time_to_apogee, sim.summary.get("timetoapogee"))
    if "Lateral distance" in columns and "apogee" in sim.events:
        add(
            "Sideways distance at apogee",
            "m",
            abs(result.events["apogee"].downrange),
            sim.at("Lateral distance", sim.events["apogee"]),
        )
    if "Vertical velocity" in columns:
        add(
            "Descent rate at landing",
            "m/s",
            result.landing_descent_rate,
            -sim["Vertical velocity"][-1],
        )
    add("Flight time", "s", result.flight_time, sim.summary.get("flighttime"))
    return rows


def _rod_exit_speed(sim: OpenRocketSimulation) -> float | None:
    """OpenRocket's speed as the rocket passes the end of the rod.

    Its own figure is the speed at the first time step past the rod, which can be several
    percent further on while the motor is at peak thrust. Interpolate to the rod's length.
    """
    rod = sim.conditions.get("launchrodlength")
    if rod is None or "Total velocity" not in sim.columns:
        return None
    travelled = sim["Altitude"]
    if "Lateral distance" in sim.columns:
        travelled = np.hypot(travelled, sim["Lateral distance"])
    climbing = sim.time <= sim.events.get("apogee", sim.time[-1])
    if not np.any(travelled[climbing] >= rod):
        return None
    return float(np.interp(rod, travelled[climbing], sim["Total velocity"][climbing]))


def format_comparison(rows: list[Comparison]) -> str:
    """The comparison as a plain-text table."""
    width = max(len(row.label) + len(row.unit) + 3 for row in rows)
    lines = [f"{'':{width}}  {'apogee':>10}  {'OpenRocket':>10}  {'difference':>10}"]
    for row in rows:
        label = f"{row.label} [{row.unit}]" if row.unit else row.label
        lines.append(
            f"{label:{width}}  {row.ours:10.4g}  {row.openrocket:10.4g}  {row.percent:+9.1f}%"
        )
    return "\n".join(lines)
