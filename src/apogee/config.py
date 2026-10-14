"""Load a rocket, environment and launch rail from a TOML config file.

Example::

    [rocket]
    name = "Example L1"
    dry_mass_kg = 1.10
    diameter_m = 0.066
    cd = 0.45

    [motor]
    file = "../data/motors/Synthetic_H200.eng"   # relative to this config file

    [environment]
    site_elevation_m = 0
    wind_speed_mps = 0

    [rail]
    length_m = 1.83
    angle_deg = 90

A constant-thrust motor can be given instead of a file with ``thrust_N``,
``burn_time_s``, ``propellant_mass_kg`` and optionally ``loaded_mass_kg``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .atmosphere import StandardAtmosphere
from .environment import Environment
from .flight import LaunchRail
from .motor import Motor
from .rocket import Rocket


@dataclass
class SimulationConfig:
    rocket: Rocket
    environment: Environment
    rail: LaunchRail


def load_config(path: str | Path) -> SimulationConfig:
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    try:
        return SimulationConfig(
            rocket=_rocket(data["rocket"], _motor(data["motor"], path.parent)),
            environment=_environment(data.get("environment", {})),
            rail=_rail(data.get("rail", {})),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc


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


def _rocket(cfg: dict, motor: Motor) -> Rocket:
    return Rocket(
        name=cfg.get("name", "rocket"),
        dry_mass=cfg["dry_mass_kg"],
        diameter=cfg["diameter_m"],
        motor=motor,
        cd=cfg.get("cd", 0.5),
    )


def _environment(cfg: dict) -> Environment:
    return Environment(
        atmosphere=StandardAtmosphere(temperature_offset=cfg.get("temperature_offset_K", 0.0)),
        site_elevation=cfg.get("site_elevation_m", 0.0),
        wind_speed=cfg.get("wind_speed_mps", 0.0),
    )


def _rail(cfg: dict) -> LaunchRail:
    return LaunchRail(length=cfg.get("length_m", 1.5), angle=cfg.get("angle_deg", 90.0))
