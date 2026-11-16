"""Load a rocket, environment and launch rail from a TOML config file.

Example::

    [rocket]
    name = "Example L1"
    dry_mass_kg = 1.10
    diameter_m = 0.066
    cd = 0.45                      # or cd_file = "cd_vs_mach.csv" (columns: mach, cd),
                                   # or leave both out and add a [drag] table

    [motor]
    file = "../data/motors/Synthetic_H200.eng"   # relative to this config file
    position_m = 1.10              # optional: nose tip to motor front; default is aft-flush

    [environment]
    site_elevation_m = 0
    wind_speed_mps = 0

    [rail]
    length_m = 1.83
    angle_deg = 90

    # Optional geometry, needed for stability (center of pressure, static margin).
    # Positions are metres aft of the nose tip. With these sections, dry mass and CG
    # are summed from the parts unless [rocket] gives measured dry_mass_kg / dry_cg_m.
    [nose]
    shape = "ogive"                # conical, ogive, parabolic or elliptical
    length_m = 0.25
    mass_kg = 0.12

    [body]
    length_m = 1.0
    diameter_m = 0.066
    mass_kg = 0.50

    [fins]
    count = 3
    root_chord_m = 0.15
    tip_chord_m = 0.06
    span_m = 0.065
    sweep_m = 0.07
    mass_kg = 0.12                 # or thickness_m and density_kgm3
    profile = "rounded"            # edge shape for drag: square, rounded or airfoil

    # Optional: estimate the drag coefficient from the geometry above instead of giving cd.
    # Needs fins.thickness_m.
    [drag]
    model = "geometry"
    roughness_m = 60e-6            # surface roughness; 60 micrometres is regular paint

    [[mass]]                       # recovery gear, avionics, ballast...
    name = "recovery"
    mass_kg = 0.18
    position_m = 0.40

    [[parachute]]                  # repeat the block for each chute
    name = "main"
    diameter_m = 0.9
    cd = 1.5
    deploy = "apogee"              # "apogee", "altitude" or "burnout"
    altitude_m = 150               # only for deploy = "altitude"
    delay_s = 0                    # seconds after the trigger

A constant-thrust motor can be given instead of a file with ``thrust_N``,
``burn_time_s``, ``propellant_mass_kg`` and optionally ``loaded_mass_kg``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .aero import CdTable, DragBuildup
from .atmosphere import StandardAtmosphere
from .components import Airframe, BodyTube, FinSet, NoseCone, PointMass
from .environment import Environment
from .flight import LaunchRail
from .motor import Motor
from .recovery import Parachute
from .rocket import Rocket


@dataclass
class SimulationConfig:
    rocket: Rocket
    environment: Environment
    rail: LaunchRail


def read_config(path: str | Path) -> dict:
    """Read a config file into a plain dict, without building anything from it."""
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def load_config(path: str | Path) -> SimulationConfig:
    path = Path(path)
    return build_config(read_config(path), path.parent, source=str(path))


def build_config(data: dict, base: str | Path = ".", source: str = "config") -> SimulationConfig:
    """Build a rocket, environment and rail from config data.

    ``base`` is the directory that relative file paths in the data are resolved against.
    Design studies call this with a modified copy of a config's data for every design.
    """
    base = Path(base)
    try:
        return SimulationConfig(
            rocket=_rocket(
                data,
                base,
                _motor(data["motor"], base),
                [_parachute(p) for p in data.get("parachute", [])],
            ),
            environment=_environment(data.get("environment", {})),
            rail=_rail(data.get("rail", {})),
        )
    except KeyError as exc:
        raise ValueError(f"{source}: missing required key {exc}") from exc


def _motor(cfg: dict, base: Path) -> Motor:
    if "file" in cfg:
        return Motor.from_eng(base / cfg["file"])
    return Motor.constant(
        thrust=cfg["thrust_N"],
        burn_time=cfg["burn_time_s"],
        propellant_mass=cfg["propellant_mass_kg"],
        loaded_mass=cfg.get("loaded_mass_kg"),
        name=cfg.get("name", "constant"),
    )


def _rocket(data: dict, base: Path, motor: Motor, parachutes: list[Parachute]) -> Rocket:
    cfg = data["rocket"]
    name = cfg.get("name", "rocket")
    cd = CdTable.from_csv(base / cfg["cd_file"]) if "cd_file" in cfg else cfg.get("cd", 0.5)
    drag = data.get("drag")
    if drag is not None:
        if drag.get("model", "geometry") != "geometry":
            raise ValueError('[drag] model must be "geometry"')
        if "cd" in cfg or "cd_file" in cfg:
            raise ValueError("give either [drag] or cd / cd_file in [rocket], not both")
    geometry = [key for key in ("nose", "body", "fins") if key in data]
    if not geometry:
        if drag is not None:
            raise ValueError("[drag] needs the geometry: [nose], [body] and [fins]")
        return Rocket(name, cfg["dry_mass_kg"], cfg["diameter_m"], motor, cd, parachutes)
    if len(geometry) < 3:
        raise ValueError("stability needs all of [nose], [body] and [fins]")
    airframe = _airframe(data, cfg)
    if drag is not None:
        cd = DragBuildup(airframe, roughness=drag.get("roughness_m", 60e-6))
    return Rocket.from_airframe(
        name, airframe, motor, cd, parachutes, data["motor"].get("position_m")
    )


def _airframe(data: dict, rocket: dict) -> Airframe:
    nose, body, fins = data["nose"], data["body"], data["fins"]
    return Airframe(
        nose=NoseCone(
            length=nose["length_m"],
            shape=nose.get("shape", "ogive"),
            mass=nose.get("mass_kg", 0.0),
            cg=nose.get("cg_m"),
        ),
        body=BodyTube(
            length=body["length_m"],
            diameter=body.get("diameter_m", rocket.get("diameter_m")),
            mass=body.get("mass_kg", 0.0),
        ),
        fins=FinSet(
            count=fins["count"],
            root_chord=fins["root_chord_m"],
            tip_chord=fins["tip_chord_m"],
            span=fins["span_m"],
            sweep=fins.get("sweep_m", 0.0),
            position=fins.get("position_m"),
            mass=fins.get("mass_kg"),
            thickness=fins.get("thickness_m"),
            density=fins.get("density_kgm3"),
            profile=fins.get("profile", "rounded"),
        ),
        masses=[PointMass(m["name"], m["mass_kg"], m["position_m"]) for m in data.get("mass", [])],
        mass_override=rocket.get("dry_mass_kg"),
        cg_override=rocket.get("dry_cg_m"),
    )


def _parachute(cfg: dict) -> Parachute:
    return Parachute(
        name=cfg["name"],
        diameter=cfg["diameter_m"],
        cd=cfg.get("cd", 1.5),
        trigger=cfg.get("deploy", "apogee"),
        altitude=cfg.get("altitude_m"),
        delay=cfg.get("delay_s", 0.0),
    )


def _environment(cfg: dict) -> Environment:
    return Environment(
        atmosphere=StandardAtmosphere(temperature_offset=cfg.get("temperature_offset_K", 0.0)),
        site_elevation=cfg.get("site_elevation_m", 0.0),
        wind_speed=cfg.get("wind_speed_mps", 0.0),
    )


def _rail(cfg: dict) -> LaunchRail:
    return LaunchRail(length=cfg.get("length_m", 1.5), angle=cfg.get("angle_deg", 90.0))
