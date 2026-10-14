"""Simulator vs closed-form solutions.

Each case strips the model down to physics that can be solved by hand, so any
disagreement points at the integrator, event handling or bookkeeping.
"""

import math

import pytest
from scipy.optimize import brentq

from apogee import ConstantAtmosphere, Environment, LaunchRail, Motor, Rocket, simulate
from apogee.atmosphere import G0
from apogee.environment import R_EARTH_MEAN

FLAT = Environment(atmosphere=ConstantAtmosphere(), inverse_square_gravity=False)


def test_vacuum_constant_thrust_matches_rocket_equation():
    thrust, burn, prop, loaded, dry = 100.0, 2.0, 0.2, 0.3, 1.0
    motor = Motor.constant(thrust, burn, prop, loaded)
    rocket = Rocket("vacuum", dry_mass=dry, diameter=0.05, motor=motor, cd=0.0)
    rail = LaunchRail(length=1.0, angle=90.0)
    result = simulate(rocket, FLAT, rail)

    g, m0 = G0, dry + loaded
    mdot = prop / burn
    ve = thrust / mdot

    def m(t):
        return m0 - mdot * t

    def v(t):
        return -ve * math.log(m(t) / m0) - g * t

    def h(t):
        return ve / mdot * (m0 - m(t) + m(t) * math.log(m(t) / m0)) - 0.5 * g * t * t

    t_rail = brentq(lambda t: h(t) - rail.length, 1e-6, burn)
    assert result.events["rail_exit"].time == pytest.approx(t_rail, rel=1e-6)
    assert result.rail_exit_speed == pytest.approx(v(t_rail), rel=1e-6)

    burnout = result.events["burnout"]
    assert burnout.speed == pytest.approx(v(burn), rel=1e-7)
    assert burnout.altitude == pytest.approx(h(burn), rel=1e-7)

    apogee = h(burn) + v(burn) ** 2 / (2 * g)
    assert result.apogee == pytest.approx(apogee, rel=1e-7)
    assert result.time_to_apogee == pytest.approx(burn + v(burn) / g, rel=1e-7)


def test_vertical_coast_with_quadratic_drag():
    m, d, cd, v0 = 1.0, 0.1, 0.5, 100.0
    rocket = Rocket("coast", dry_mass=m, diameter=d, motor=None, cd=cd)
    result = simulate(rocket, FLAT, LaunchRail(length=0.0), initial_speed=v0)

    g = G0
    k = 0.5 * 1.225 * cd * math.pi * d**2 / 4
    h_max = m / (2 * k) * math.log(1 + k * v0**2 / (m * g))
    t_up = math.sqrt(m / (g * k)) * math.atan(v0 * math.sqrt(k / (m * g)))
    assert result.apogee == pytest.approx(h_max, rel=1e-7)
    assert result.time_to_apogee == pytest.approx(t_up, rel=1e-7)

    # Falling from rest through height h_max with quadratic drag
    v_land = math.sqrt(m * g / k * (1 - math.exp(-2 * k * h_max / m)))
    assert result.landing_speed == pytest.approx(v_land, rel=1e-6)


def test_drag_free_projectile_range():
    v0, angle = 50.0, 45.0
    rocket = Rocket("ball", dry_mass=1.0, diameter=0.05, motor=None, cd=0.0)
    result = simulate(rocket, FLAT, LaunchRail(length=0.0, angle=angle), initial_speed=v0)

    th = math.radians(angle)
    assert result.events["landing"].downrange == pytest.approx(
        v0**2 * math.sin(2 * th) / G0, rel=1e-7
    )
    assert result.apogee == pytest.approx((v0 * math.sin(th)) ** 2 / (2 * G0), rel=1e-7)
    assert result.landing_speed == pytest.approx(v0, rel=1e-7)


def test_energy_conserved_without_drag_or_thrust():
    env = Environment(atmosphere=ConstantAtmosphere(), inverse_square_gravity=True)
    rocket = Rocket("ballistic", dry_mass=1.0, diameter=0.05, motor=None, cd=0.0)
    result = simulate(rocket, env, LaunchRail(length=0.0, angle=70.0), initial_speed=300.0)

    mu = G0 * R_EARTH_MEAN**2
    energy = 0.5 * result.speed**2 - mu / (R_EARTH_MEAN + result.z)
    assert energy.max() - energy.min() < 1e-8 * abs(energy[0])


def test_apogee_converges_with_tolerance(example_config):
    from apogee import load_config

    cfg = load_config(example_config)
    loose = simulate(cfg.rocket, cfg.environment, cfg.rail, rtol=1e-6, atol=1e-6)
    tight = simulate(cfg.rocket, cfg.environment, cfg.rail, rtol=1e-11, atol=1e-11)
    assert loose.apogee == pytest.approx(tight.apogee, rel=1e-5)
    assert loose.max_speed == pytest.approx(tight.max_speed, rel=1e-5)
