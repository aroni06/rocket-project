"""apogee: a 2D rocket flight simulator and design optimizer."""

from .atmosphere import ConstantAtmosphere, StandardAtmosphere
from .config import load_config
from .environment import Environment
from .flight import FlightEvent, FlightResult, LaunchRail, simulate
from .motor import Motor, parse_eng
from .rocket import Rocket

__all__ = [
    "ConstantAtmosphere",
    "Environment",
    "FlightEvent",
    "FlightResult",
    "LaunchRail",
    "Motor",
    "Rocket",
    "StandardAtmosphere",
    "load_config",
    "parse_eng",
    "simulate",
]

__version__ = "0.1.0"
