import math

import pytest

from apogee import (
    ConstantAtmosphere,
    Environment,
    LaunchRail,
    Motor,
    Rocket,
    load_config,
    simulate,
)
from apogee.atmosphere import G0, StandardAtmosphere
from apogee.recovery import Parachute, terminal_velocity

FLAT = Environment(atmosphere=ConstantAtmosphere(), inverse_square_gravity=False)


def _ballistic(parachutes, wind=0.0, v0=100.0):
    env = Environment(
        atmosphere=ConstantAtmosphere(), inverse_square_gravity=False, wind_speed=wind
    )
    rocket = Rocket("lob", dry_mass=1.0, diameter=0.05, motor=None, cd=0.0, parachutes=parachutes)
    return simulate(rocket, env, LaunchRail(length=0.0), initial_speed=v0)


def test_fall_under_chute_matches_closed_form():
    # Chute opens at apogee; from rest with quadratic drag, v(h)^2 = v_t^2 (1 - exp(-2 g h / v_t^2))
    chute = Parachute("main", diameter=0.9, cd=1.5)
    r = _ballistic([chute])
    h = r.apogee
    assert h == pytest.approx(100.0**2 / (2 * G0), rel=1e-7)
    v_t = terminal_velocity(1.0, chute.drag_area, 1.225, G0)
    v_land = v_t * math.sqrt(1 - math.exp(-2 * G0 * h / v_t**2))
    assert r.landing_descent_rate == pytest.approx(v_land, rel=1e-6)
    assert r.events["main_deploy"].time == pytest.approx(r.time_to_apogee, rel=1e-9)


def test_terminal_velocity_formula():
    assert terminal_velocity(2.0, 1.0, 1.0, 10.0) == pytest.approx(math.sqrt(40.0))


def test_deploy_delay_after_apogee():
    r = _ballistic([Parachute("main", diameter=0.9, trigger="apogee", delay=2.0)])
    assert r.events["main_deploy"].time == pytest.approx(r.time_to_apogee + 2.0, rel=1e-9)


def test_motor_ejection_deploys_after_burnout_delay_even_before_apogee():
    motor = Motor.constant(thrust=100.0, burn_time=1.0, propellant_mass=0.05, loaded_mass=0.1)
    chute = Parachute("main", diameter=0.6, trigger="burnout", delay=1.5)
    rocket = Rocket("early", dry_mass=1.0, diameter=0.05, motor=motor, cd=0.4, parachutes=[chute])
    r = simulate(rocket, FLAT, LaunchRail(1.0))
    deploy = r.events["main_deploy"]
    assert deploy.time == pytest.approx(2.5, rel=1e-9)
    assert deploy.vertical_velocity > 0  # still climbing: an early deploy
    assert deploy.time < r.time_to_apogee


def test_dual_deploy_main_opens_at_set_altitude():
    drogue = Parachute("drogue", diameter=0.3, trigger="apogee")
    main = Parachute("main", diameter=1.2, trigger="altitude", altitude=150.0)
    r = _ballistic([drogue, main], v0=150.0)
    assert r.events["main_deploy"].altitude == pytest.approx(150.0, abs=1e-6)
    assert [ev.name for ev in r.deployments] == ["drogue_deploy", "main_deploy"]

    # Under the drogue alone the rocket should be close to the drogue's terminal velocity.
    v_drogue = terminal_velocity(1.0, drogue.drag_area, 1.225, G0)
    assert r.summary()["drogue_descent_rate_mps"] == pytest.approx(v_drogue, rel=1e-3)
    # The drogue stays attached, so both canopies set the final rate.
    v_both = terminal_velocity(1.0, drogue.drag_area + main.drag_area, 1.225, G0)
    assert r.landing_descent_rate == pytest.approx(v_both, rel=1e-3)


def test_main_altitude_above_apogee_deploys_at_apogee():
    main = Parachute("main", diameter=0.9, trigger="altitude", altitude=5_000.0)
    r = _ballistic([main])
    assert r.events["main_deploy"].time == pytest.approx(r.time_to_apogee, rel=1e-9)


def test_chute_drifts_with_the_wind():
    calm = _ballistic([Parachute("main", diameter=0.9)])
    windy = _ballistic([Parachute("main", diameter=0.9)], wind=5.0)
    assert abs(calm.events["landing"].downrange) < 1e-6
    # Under a canopy the rocket picks up the wind's speed, so it drifts downwind.
    assert windy.events["landing"].downrange > 0
    assert windy.vx[-1] == pytest.approx(5.0, rel=1e-3)


def test_bigger_chute_lands_softer_and_drifts_further():
    small = _ballistic([Parachute("main", diameter=0.6)], wind=5.0)
    big = _ballistic([Parachute("main", diameter=1.2)], wind=5.0)
    assert big.landing_descent_rate < small.landing_descent_rate
    assert big.events["landing"].downrange > small.events["landing"].downrange


def test_dual_deploy_drifts_less_than_single_deploy(example_config):
    root = example_config.parent
    single = load_config(root / "example_l1.toml")
    dual = load_config(root / "example_dual_deploy.toml")
    rs = simulate(single.rocket, single.environment, single.rail)
    rd = simulate(dual.rocket, dual.environment, dual.rail)
    drift = {
        name: r.events["landing"].downrange - r.events["apogee"].downrange
        for name, r in {"single": rs, "dual": rd}.items()
    }
    assert drift["dual"] < drift["single"]
    assert rd.landing_descent_rate < 6.0


def test_ascent_acceleration_excludes_opening_shock(example_config):
    cfg = load_config(example_config)
    r = simulate(cfg.rocket, cfg.environment, cfg.rail)
    assert r.opening_load("main") > 0
    assert r.max_acceleration == pytest.approx(
        r.acceleration[r.time < r.events["main_deploy"].time].max()
    )


def test_descent_rate_is_faster_in_thin_air():
    # Same chute, high-altitude site: terminal velocity scales with 1/sqrt(rho).
    rho_low = StandardAtmosphere().at(0.0).density
    rho_high = StandardAtmosphere().at(1_500.0).density
    assert terminal_velocity(1.0, 1.0, rho_high, G0) > terminal_velocity(1.0, 1.0, rho_low, G0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diameter": 0.0},
        {"diameter": 0.5, "cd": 0.0},
        {"diameter": 0.5, "trigger": "nosecone"},
        {"diameter": 0.5, "trigger": "altitude"},
        {"diameter": 0.5, "delay": -1.0},
    ],
)
def test_invalid_parachutes_raise(kwargs):
    with pytest.raises(ValueError):
        Parachute("bad", **kwargs)


def test_duplicate_parachute_names_raise():
    with pytest.raises(ValueError, match="unique"):
        Rocket("r", 1.0, 0.05, None, parachutes=[Parachute("a", 0.5), Parachute("a", 0.6)])
