import math

import pytest

from apogee import Environment, LaunchRail, Motor, Rocket, load_config, simulate
from apogee.cli import main


def test_example_flight_is_physically_ordered(example_config):
    cfg = load_config(example_config)
    r = simulate(cfg.rocket, cfg.environment, cfg.rail)
    ev = r.events
    assert 0 < ev["rail_exit"].time < ev["burnout"].time < ev["apogee"].time < ev["landing"].time
    assert ev["rail_exit"].altitude == pytest.approx(cfg.rail.length, rel=1e-6)
    assert r.apogee == pytest.approx(r.z.max(), rel=1e-6)
    assert abs(ev["apogee"].vertical_velocity) < 1e-6
    assert ev["landing"].altitude == pytest.approx(0.0, abs=1e-6)
    # Mass is the loaded mass at ignition and the burnt-out mass afterwards.
    motor = cfg.rocket.motor
    assert r.mass[0] == pytest.approx(cfg.rocket.dry_mass + motor.loaded_mass)
    assert r.mass[-1] == pytest.approx(
        cfg.rocket.dry_mass + motor.loaded_mass - motor.propellant_mass
    )
    assert set(r.summary()) >= {"apogee_m", "max_speed_mps", "landing_speed_mps"}


def test_drag_lowers_apogee(example_config):
    cfg = load_config(example_config)
    with_drag = simulate(cfg.rocket, cfg.environment, cfg.rail)
    cfg.rocket.cd = 0.0
    no_drag = simulate(cfg.rocket, cfg.environment, cfg.rail)
    assert no_drag.apogee > with_drag.apogee


def test_mach_dependent_drag_callable(example_config):
    cfg = load_config(example_config)
    cfg.rocket.cd = lambda mach: 0.45 + 0.2 * mach
    r = simulate(cfg.rocket, cfg.environment, cfg.rail)
    assert r.apogee > 0


def test_angled_launch_flies_downrange_and_wind_moves_landing():
    motor = Motor.constant(thrust=150.0, burn_time=1.5, propellant_mass=0.12, loaded_mass=0.2)
    rocket = Rocket("angled", dry_mass=1.0, diameter=0.054, motor=motor, cd=0.5)
    rail = LaunchRail(length=1.5, angle=80.0)
    calm = simulate(rocket, Environment(), rail)
    assert calm.events["landing"].downrange > 0
    assert calm.apogee < simulate(rocket, Environment(), LaunchRail(1.5, 90.0)).apogee

    windy = simulate(rocket, Environment(wind_speed=8.0), rail)
    assert not math.isclose(windy.events["landing"].downrange, calm.events["landing"].downrange)


def test_underpowered_rocket_never_leaves_rail():
    motor = Motor.constant(thrust=5.0, burn_time=1.0, propellant_mass=0.01, loaded_mass=0.02)
    rocket = Rocket("heavy", dry_mass=2.0, diameter=0.05, motor=motor)
    with pytest.raises(RuntimeError, match="did not clear"):
        simulate(rocket)


def test_cli_writes_outputs(example_config, tmp_path, capsys):
    csv, png = tmp_path / "flight.csv", tmp_path / "flight.png"
    assert main(["simulate", str(example_config), "--csv", str(csv), "--plot", str(png)]) == 0
    out = capsys.readouterr().out
    assert "Apogee" in out
    assert csv.read_text().startswith("time_s,")
    assert png.stat().st_size > 0
