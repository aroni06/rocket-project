"""apogee: a 2D rocket flight simulator and design optimizer."""

from .aero import CdTable, barrowman
from .atmosphere import ConstantAtmosphere, StandardAtmosphere
from .components import Airframe, BodyTube, FinSet, LaunchLug, NoseCone, PointMass, Transition
from .config import load_config
from .environment import Environment
from .flight import FlightEvent, FlightResult, LaunchRail, simulate
from .motor import Motor, parse_eng
from .recovery import Parachute
from .rocket import Rocket
from .study import Constraint, Objective, Study, Variable

__all__ = [
    "Airframe",
    "BodyTube",
    "CdTable",
    "ConstantAtmosphere",
    "Constraint",
    "Environment",
    "FinSet",
    "FlightEvent",
    "FlightResult",
    "LaunchLug",
    "LaunchRail",
    "Motor",
    "NoseCone",
    "Objective",
    "Parachute",
    "PointMass",
    "Rocket",
    "StandardAtmosphere",
    "Study",
    "Transition",
    "Variable",
    "barrowman",
    "load_config",
    "parse_eng",
    "simulate",
]

__version__ = "0.1.0"
