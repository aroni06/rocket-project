"""Two-dimensional point-mass flight simulation.

State vector ``y = [x, z, vx, vz]`` in a flat-Earth ground frame: x is downrange
distance and z is altitude above the pad, both in metres. Mass is not part of the state
because it is a known function of time from the motor model.

Equation of motion (thrust T, drag D, gravity g):

    m(t) dv/dt = T(t) u + D + m g

where ``u`` is the rail direction while on the rail and the direction of the air-relative
velocity afterwards (zero angle of attack, the usual 3-DOF assumption). Drag is

    D = -1/2 rho Cd(M) A |v_air| v_air,   v_air = v - v_wind.

The flight is integrated phase by phase so that discontinuities (rail exit, thrust curve
corners, burnout) always land on a step boundary:

    rail -> powered -> coast -> apogee -> descent -> landing
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .atmosphere import G0
from .environment import Environment
from .rocket import Rocket

EventFn = Callable[[float, np.ndarray], float]


@dataclass
class LaunchRail:
    """Launch rail or rod. ``angle`` is the elevation above horizontal in degrees."""

    length: float = 1.5
    angle: float = 90.0

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError("rail length must be non-negative")
        if not 0.0 < self.angle <= 90.0:
            raise ValueError("rail angle must be in (0, 90] degrees")

    @property
    def direction(self) -> tuple[float, float]:
        a = math.radians(self.angle)
        return math.cos(a), math.sin(a)


@dataclass(frozen=True)
class FlightEvent:
    name: str
    time: float  # s
    altitude: float  # m above pad
    downrange: float  # m
    speed: float  # m/s, ground-relative
    vertical_velocity: float  # m/s

    @classmethod
    def from_state(cls, name: str, t: float, y: Sequence[float]) -> FlightEvent:
        x, z, vx, vz = (float(v) for v in y)
        return cls(name, float(t), z, x, math.hypot(vx, vz), vz)


@dataclass
class FlightResult:
    """Time histories and key events of one simulated flight (SI units)."""

    rocket_name: str
    time: np.ndarray
    x: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vz: np.ndarray
    mass: np.ndarray
    thrust: np.ndarray
    drag: np.ndarray
    mach: np.ndarray
    dynamic_pressure: np.ndarray
    ax: np.ndarray  # horizontal acceleration [m/s^2]
    az: np.ndarray  # vertical acceleration [m/s^2]
    phase: np.ndarray
    events: dict[str, FlightEvent] = field(default_factory=dict)

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.vx, self.vz)

    @property
    def acceleration(self) -> np.ndarray:
        """Magnitude of the acceleration vector [m/s^2]."""
        return np.hypot(self.ax, self.az)

    @property
    def apogee(self) -> float:
        return self.events["apogee"].altitude

    @property
    def time_to_apogee(self) -> float:
        return self.events["apogee"].time

    @property
    def max_speed(self) -> float:
        return float(self.speed.max())

    @property
    def max_mach(self) -> float:
        return float(self.mach.max())

    @property
    def max_acceleration(self) -> float:
        return float(self.acceleration.max())

    @property
    def max_dynamic_pressure(self) -> float:
        return float(self.dynamic_pressure.max())

    @property
    def rail_exit_speed(self) -> float | None:
        ev = self.events.get("rail_exit")
        return ev.speed if ev else None

    @property
    def landing_speed(self) -> float:
        return self.events["landing"].speed

    @property
    def flight_time(self) -> float:
        return self.events["landing"].time

    def summary(self) -> dict[str, float]:
        """Key performance numbers, keyed by name with units."""
        out: dict[str, float] = {
            "apogee_m": self.apogee,
            "time_to_apogee_s": self.time_to_apogee,
            "max_speed_mps": self.max_speed,
            "max_mach": self.max_mach,
            "max_acceleration_g": self.max_acceleration / G0,
            "max_dynamic_pressure_Pa": self.max_dynamic_pressure,
        }
        if (ev := self.events.get("rail_exit")) is not None:
            out["rail_exit_speed_mps"] = ev.speed
        if (ev := self.events.get("burnout")) is not None:
            out["burnout_time_s"] = ev.time
            out["burnout_altitude_m"] = ev.altitude
            out["burnout_speed_mps"] = ev.speed
        out["landing_speed_mps"] = self.landing_speed
        out["landing_downrange_m"] = self.events["landing"].downrange
        out["flight_time_s"] = self.flight_time
        return out

    def to_csv(self, path: str | Path) -> None:
        cols = {
            "time_s": self.time,
            "x_m": self.x,
            "z_m": self.z,
            "vx_mps": self.vx,
            "vz_mps": self.vz,
            "mass_kg": self.mass,
            "thrust_N": self.thrust,
            "drag_N": self.drag,
            "mach": self.mach,
            "dynamic_pressure_Pa": self.dynamic_pressure,
            "ax_mps2": self.ax,
            "az_mps2": self.az,
        }
        data = np.column_stack(list(cols.values()))
        np.savetxt(path, data, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")

    def __str__(self) -> str:
        labels = {
            "apogee_m": ("Apogee", "m"),
            "time_to_apogee_s": ("Time to apogee", "s"),
            "max_speed_mps": ("Max speed", "m/s"),
            "max_mach": ("Max Mach", ""),
            "max_acceleration_g": ("Max acceleration", "g"),
            "max_dynamic_pressure_Pa": ("Max dynamic pressure", "Pa"),
            "rail_exit_speed_mps": ("Rail exit speed", "m/s"),
            "burnout_time_s": ("Burnout time", "s"),
            "burnout_altitude_m": ("Burnout altitude", "m"),
            "burnout_speed_mps": ("Burnout speed", "m/s"),
            "landing_speed_mps": ("Landing speed", "m/s"),
            "landing_downrange_m": ("Landing downrange", "m"),
            "flight_time_s": ("Flight time", "s"),
        }
        lines = [f"Flight summary: {self.rocket_name}"]
        for key, value in self.summary().items():
            label, unit = labels[key]
            lines.append(f"  {label:<22} {value:>10.2f} {unit}")
        return "\n".join(lines)


class _Dynamics:
    def __init__(self, rocket: Rocket, env: Environment, rail: LaunchRail) -> None:
        self.rocket = rocket
        self.env = env
        self.rail_dir = rail.direction
        self.area = rocket.reference_area

    def evaluate(self, t: float, y: np.ndarray, on_rail: bool) -> tuple[float, ...]:
        """Return (ax, az, thrust, drag, mass, mach, dynamic_pressure)."""
        _, z, vx, vz = y
        rocket, env = self.rocket, self.env
        m = rocket.mass_at(t)
        thrust = rocket.thrust_at(t)
        atm = env.atmosphere.at(env.site_elevation + z)

        vax, vaz = vx - env.wind_speed, vz
        v_air = math.hypot(vax, vaz)
        mach = v_air / atm.speed_of_sound
        q = 0.5 * atm.density * v_air * v_air
        drag = q * rocket.drag_coefficient(mach) * self.area

        if on_rail or v_air < 1e-9:
            ux, uz = self.rail_dir
        else:
            ux, uz = vax / v_air, vaz / v_air

        fx = thrust * ux
        fz = thrust * uz - m * env.gravity(z)
        if v_air > 0.0:
            fx -= drag * vax / v_air
            fz -= drag * vaz / v_air

        if on_rail:
            rx, rz = self.rail_dir
            f_par = fx * rx + fz * rz
            v_par = vx * rx + vz * rz
            if f_par <= 0.0 and v_par <= 0.0:
                ax = az = 0.0  # resting on the pad
            else:
                ax, az = f_par / m * rx, f_par / m * rz
        else:
            ax, az = fx / m, fz / m
        return ax, az, thrust, drag, m, mach, q

    def rhs(self, on_rail: bool) -> Callable[[float, np.ndarray], list[float]]:
        def f(t: float, y: np.ndarray) -> list[float]:
            ax, az, *_ = self.evaluate(t, y, on_rail)
            return [y[2], y[3], ax, az]

        return f


def _event(fn: EventFn, direction: float) -> EventFn:
    fn.terminal = True  # type: ignore[attr-defined]
    fn.direction = direction  # type: ignore[attr-defined]
    return fn


class _Integrator:
    def __init__(self, dyn: _Dynamics, max_step: float, rtol: float, atol: float) -> None:
        self.dyn = dyn
        self.opts = {"method": "DOP853", "max_step": max_step, "rtol": rtol, "atol": atol}
        self.t: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        self.phase: list[np.ndarray] = []

    def run(
        self,
        phase: str,
        t_start: float,
        y_start: np.ndarray,
        t_final: float,
        events: dict[str, EventFn],
        breakpoints: Sequence[float] = (),
    ) -> tuple[float, np.ndarray, str | None]:
        """Integrate across breakpoints until a terminal event fires or ``t_final``."""
        knots = sorted(b for b in breakpoints if t_start < b < t_final) + [t_final]
        on_rail = phase == "rail"
        rhs = self.dyn.rhs(on_rail)

        def speed_peak(t: float, y: np.ndarray) -> float:
            ax, az, *_ = self.dyn.evaluate(t, y, on_rail)
            return ax * y[2] + az * y[3]  # d|v|/dt * |v|

        speed_peak.terminal = False  # type: ignore[attr-defined]
        speed_peak.direction = -1.0  # type: ignore[attr-defined]

        names, fns = list(events), [*events.values(), speed_peak]
        t_a, y_a = t_start, np.asarray(y_start, dtype=float)
        for t_b in knots:
            sol = solve_ivp(rhs, (t_a, t_b), y_a, events=fns, **self.opts)
            if sol.status == -1:
                raise RuntimeError(f"integration failed in {phase} phase: {sol.message}")
            t_out, y_out = sol.t, sol.y
            if len(sol.t_events[-1]):
                # Keep exact speed maxima as samples so max speed is not limited by step size.
                t_out = np.concatenate([t_out, sol.t_events[-1]])
                y_out = np.concatenate([y_out, sol.y_events[-1].T], axis=1)
                order = np.argsort(t_out, kind="stable")
                t_out, y_out = t_out[order], y_out[:, order]
            self._append(t_out, y_out, phase)
            for name, t_ev in zip(names, sol.t_events, strict=False):
                if len(t_ev):
                    return float(sol.t[-1]), sol.y[:, -1], name
            t_a, y_a = float(sol.t[-1]), sol.y[:, -1]
        return t_a, y_a, None

    def _append(self, t: np.ndarray, y: np.ndarray, phase: str) -> None:
        if self.t and t.size and t[0] == self.t[-1][-1]:
            t, y = t[1:], y[:, 1:]
        self.t.append(t)
        self.y.append(y)
        self.phase.append(np.full(t.size, phase, dtype=object))


def simulate(
    rocket: Rocket,
    environment: Environment | None = None,
    rail: LaunchRail | None = None,
    *,
    max_time: float = 600.0,
    max_step: float = 0.05,
    rtol: float = 1e-9,
    atol: float = 1e-9,
    initial_speed: float = 0.0,
) -> FlightResult:
    """Simulate one flight from ignition to landing.

    ``initial_speed`` starts the rocket already moving along the rail direction. It is
    used for analytic test cases (e.g. an unpowered coast) and is 0 for a real launch.
    """
    env = environment or Environment()
    rail = rail or LaunchRail()
    dyn = _Dynamics(rocket, env, rail)
    integ = _Integrator(dyn, max_step, rtol, atol)
    motor = rocket.motor
    burn_time = motor.burn_time if motor else 0.0
    thrust_knots = list(motor.time[1:]) if motor else []

    ux, uz = rail.direction
    y = np.array([0.0, 0.0, initial_speed * ux, initial_speed * uz])
    t = 0.0
    events: dict[str, FlightEvent] = {}

    landing = _event(lambda t, y: y[1], -1.0)
    apogee = _event(lambda t, y: y[3], -1.0)

    # Rail: constrained along the rail until the rocket has travelled its full length.
    if motor and rail.length > 0:
        rail_exit = _event(lambda t, y: y[0] * ux + y[1] * uz - rail.length, 1.0)
        t, y, fired = integ.run("rail", t, y, burn_time, {"rail_exit": rail_exit}, thrust_knots)
        if fired is None:
            raise RuntimeError(
                f"{rocket.name} did not clear the {rail.length} m rail before burnout; "
                "thrust-to-weight ratio is too low"
            )
        events["rail_exit"] = FlightEvent.from_state("rail_exit", t, y)

    # Powered flight.
    if t < burn_time:
        t, y, fired = integ.run("powered", t, y, burn_time, {"landing": landing}, thrust_knots)
        if fired == "landing":
            raise RuntimeError(f"{rocket.name} hit the ground under power")
    if motor:
        events["burnout"] = FlightEvent.from_state("burnout", t, y)

    # Coast to apogee.
    if y[3] > 0.0:
        t, y, fired = integ.run("coast", t, y, max_time, {"apogee": apogee})
        if fired is None:
            raise RuntimeError(f"max_time = {max_time} s reached before apogee")
        events["apogee"] = FlightEvent.from_state("apogee", t, y)
    else:
        # Already descending at burnout (e.g. a very low-angle launch): take the highest
        # recorded point as apogee.
        times = np.concatenate(integ.t)
        states = np.concatenate(integ.y, axis=1)
        k = int(np.argmax(states[1]))
        events["apogee"] = FlightEvent.from_state("apogee", times[k], states[:, k])

    # Descent. Ballistic for now; parachutes arrive with the recovery model.
    t, y, fired = integ.run("descent", t, y, max_time, {"landing": landing})
    if fired is None:
        raise RuntimeError(f"max_time = {max_time} s reached before landing")
    events["landing"] = FlightEvent.from_state("landing", t, y)

    return _build_result(rocket, dyn, integ, events)


def _build_result(
    rocket: Rocket, dyn: _Dynamics, integ: _Integrator, events: dict[str, FlightEvent]
) -> FlightResult:
    time = np.concatenate(integ.t)
    states = np.concatenate(integ.y, axis=1)
    phase = np.concatenate(integ.phase)
    derived = np.array(
        [dyn.evaluate(ti, states[:, i], phase[i] == "rail") for i, ti in enumerate(time)]
    )
    ax, az, thrust, drag, mass, mach, q = derived.T
    return FlightResult(
        rocket_name=rocket.name,
        time=time,
        x=states[0],
        z=states[1],
        vx=states[2],
        vz=states[3],
        mass=mass,
        thrust=thrust,
        drag=drag,
        mach=mach,
        dynamic_pressure=q,
        ax=ax,
        az=az,
        phase=phase,
        events=events,
    )
