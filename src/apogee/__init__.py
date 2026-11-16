"""apogee: a 2D rocket flight simulator and design optimizer."""

from .aero import CdTable, barrowman
from .atmosphere import ConstantAtmosphere, StandardAtmosphere
from .components import Airframe, BodyTube, FinSet, NoseCone, PointMass
from .config import load_config
from .environment import Environment
from .flight import FlightEvent, FlightResult, LaunchRail, simulate
from .motor import Motor, parse_eng
from .recovery import Parachute
from .rocket import Rocket

__all__ = [
    "Airframe",
    "BodyTube",
    "CdTable",
    "ConstantAtmosphere",
    "Environment",
    "FinSet",
    "FlightEvent",
    "FlightResult",
    "LaunchRail",
    "Motor",
    "NoseCone",
    "Parachute",
    "PointMass",
    "Rocket",
    "StandardAtmosphere",
    "barrowman",
    "load_config",
    "parse_eng",
    "simulate",
]

__version__ = "0.1.0"
