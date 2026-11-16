"""Drag estimated from the geometry: each piece against a formula, then the whole."""

import math
from pathlib import Path

import numpy as np
import pytest

from apogee import (
    Airframe,
    BodyTube,
    Environment,
    FinSet,
    LaunchRail,
    Motor,
    NoseCone,
    Rocket,
    load_config,
    simulate,
)
from apogee.aero import (
    DragBuildup,
    base_drag_coefficient,
    rounded_edge_drag_coefficient,
    skin_friction_coefficient,
    stagnation_drag_coefficient,
)
from apogee.cli import main

ROOT = Path(__file__).resolve().parents[1]


def airframe(span=0.06, profile="rounded", nose="conical", body_length=0.8) -> Airframe:
    return Airframe(
        nose=NoseCone(length=0.2, shape=nose, mass=0.1),
        body=BodyTube(length=body_length, diameter=0.05, mass=0.4),
        fins=FinSet(
            count=4,
            root_chord=0.10,
            tip_chord=0.05,
            span=span,
            sweep=0.03,
            thickness=0.003,
            density=1800.0,
            profile=profile,
        ),
    )


# The pieces


@pytest.mark.parametrize("reynolds", [1e5, 1e6, 1e7, 1e8, 1e9])
def test_smooth_skin_friction_matches_the_prandtl_schlichting_law(reynolds):
    textbook = 0.455 / math.log10(reynolds) ** 2.58
    assert skin_friction_coefficient(reynolds, 0.0, 0.0) == pytest.approx(textbook, rel=0.04)


def test_roughness_stops_skin_friction_falling():
    relative = 60e-6 / 1.25
    floor = 0.032 * relative**0.2
    assert skin_friction_coefficient(1e5, relative, 0.0) > floor  # still on the smooth law
    assert skin_friction_coefficient(1e7, relative, 0.0) == pytest.approx(floor)
    assert skin_friction_coefficient(1e9, relative, 0.0) == pytest.approx(floor)
    # Continuous where the two laws cross.
    grid = np.geomspace(1e5, 1e7, 400)
    values = [skin_friction_coefficient(re, relative, 0.0) for re in grid]
    assert max(abs(np.diff(values))) < 5e-5


def test_compressibility_lowers_skin_friction_without_a_jump():
    assert skin_friction_coefficient(1e7, 0.0, 0.5) == pytest.approx(
        0.975 * skin_friction_coefficient(1e7, 0.0, 0.0)
    )
    mach = np.linspace(0.0, 2.0, 401)
    values = [skin_friction_coefficient(1e7, 0.0, m) for m in mach]
    assert np.all(np.diff(values) < 1e-5)
    assert max(abs(np.diff(values))) < 5e-5


def test_base_and_edge_drag_have_no_real_jump_at_mach_one():
    assert base_drag_coefficient(0.0) == pytest.approx(0.12)
    assert stagnation_drag_coefficient(0.0) == pytest.approx(0.85)
    assert rounded_edge_drag_coefficient(0.0) == pytest.approx(0.0)
    for fn in (base_drag_coefficient, stagnation_drag_coefficient, rounded_edge_drag_coefficient):
        for mach in (0.9, 1.0):
            # The published fits meet to within half a percent.
            assert fn(mach - 1e-9) == pytest.approx(fn(mach + 1e-9), abs=6e-3)


def test_nose_wetted_area_matches_closed_forms():
    r = 0.025
    cone = NoseCone(length=0.2, shape="conical")
    assert cone.wetted_area(r) == pytest.approx(math.pi * r * math.hypot(r, 0.2), rel=1e-9)
    hemisphere = NoseCone(length=r, shape="elliptical")
    assert hemisphere.wetted_area(r) == pytest.approx(2 * math.pi * r * r, rel=1e-4)
    # An ogive is fuller than a cone of the same length, so it has more skin.
    assert NoseCone(length=0.2, shape="ogive").wetted_area(r) > cone.wetted_area(r)
    assert cone.joint_angle(r) == pytest.approx(math.atan(r / 0.2))
    assert NoseCone(length=0.2, shape="ogive").joint_angle(r) == 0.0


# The whole build-up


def test_buildup_matches_a_hand_calculation():
    frame = airframe()
    mach, unit_reynolds = 0.3, 7.0e6
    parts = DragBuildup(frame, roughness=0.0).breakdown(mach, unit_reynolds)

    d, length = 0.05, 1.0
    ref = math.pi * d**2 / 4
    cf = (1 - 0.1 * mach**2) / (1.50 * math.log(unit_reynolds * length) - 5.6) ** 2
    body_wetted = math.pi * (d / 2) * math.hypot(d / 2, 0.2) + math.pi * d * 0.8
    assert parts.body_friction == pytest.approx(cf * (1 + d / (2 * length)) * body_wetted / ref)

    fin_wetted = 2 * 4 * 0.5 * (0.10 + 0.05) * 0.06
    mac = 2 * (0.10**2 + 0.10 * 0.05 + 0.05**2) / (3 * 0.15)
    assert parts.fin_friction == pytest.approx(cf * (1 + 2 * 0.003 / mac) * fin_wetted / ref)

    assert parts.nose_pressure == pytest.approx(0.8 * (d / 2) ** 2 / ((d / 2) ** 2 + 0.2**2))
    base = 0.12 + 0.13 * mach**2
    assert parts.base == pytest.approx(base)
    leading = ((1 - mach**2) ** -0.417 - 1) * 0.06**2 / (0.06**2 + 0.03**2)
    frontal = 4 * 0.003 * 0.06 / ref
    assert parts.fin_pressure == pytest.approx((leading + base / 2) * frontal)
    assert parts.total == pytest.approx(
        parts.body_friction
        + parts.fin_friction
        + parts.nose_pressure
        + parts.base
        + parts.fin_pressure
    )
    assert 0.3 < parts.total < 0.8  # a sensible number for a slender sport rocket


def test_drag_responds_to_the_geometry():
    def cd(**kwargs):
        roughness = kwargs.pop("roughness", 60e-6)
        return DragBuildup(airframe(**kwargs), roughness)(0.3)

    assert cd(span=0.09) > cd(span=0.06) > cd(span=0.03)
    assert cd(profile="square") > cd(profile="rounded") > cd(profile="airfoil")
    assert cd(body_length=1.2) > cd(body_length=0.8)
    assert cd(roughness=150e-6) > cd(roughness=60e-6) > cd(roughness=0.0)
    # Only a cone meets the body at an angle, which costs pressure drag at the shoulder.
    at = (0.3, 7.0e6)
    assert DragBuildup(airframe(nose="conical")).breakdown(*at).nose_pressure > 0.0
    assert DragBuildup(airframe(nose="ogive")).breakdown(*at).nose_pressure == 0.0


def test_buildup_needs_the_fin_thickness():
    frame = airframe()
    frame.fins.thickness = None
    with pytest.raises(ValueError, match="thickness"):
        DragBuildup(frame)
    with pytest.raises(ValueError, match="roughness"):
        DragBuildup(airframe(), roughness=-1.0)
    with pytest.raises(ValueError, match="profile"):
        airframe(profile="knife")


# In flight


def test_flight_uses_the_local_reynolds_number():
    frame = airframe()
    drag_model = DragBuildup(frame, roughness=0.0)
    motor = Motor.constant(thrust=80.0, burn_time=1.5, propellant_mass=0.06, loaded_mass=0.12)
    rocket = Rocket.from_airframe("smooth", frame, motor, cd=drag_model)
    env = Environment(site_elevation=1500.0)
    result = simulate(rocket, env, LaunchRail(length=1.5))

    # Halfway up the coast, the recorded drag is q * Cd * A with Cd at the local conditions.
    k = int(np.searchsorted(result.time, 0.5 * (1.5 + result.time_to_apogee)))
    state = env.atmosphere.at(env.site_elevation + result.z[k])
    speed = result.speed[k]
    cd_local = drag_model.at(result.mach[k], speed / state.kinematic_viscosity)
    expected = result.dynamic_pressure[k] * cd_local * rocket.reference_area
    assert result.drag[k] == pytest.approx(expected, rel=1e-9)
    # Thinner air means a lower Reynolds number and more friction than at sea level.
    assert cd_local > drag_model(result.mach[k])
    assert state.kinematic_viscosity > 1.4607e-5


def test_sea_level_viscosity_matches_the_standard_atmosphere():
    state = Environment().atmosphere.at(0.0)
    assert state.kinematic_viscosity == pytest.approx(1.4607e-5, rel=1e-3)


# Configs and the command line


def _config(tmp_path, old: str, new: str) -> Path:
    text = (ROOT / "configs" / "example_dual_deploy.toml").read_text()
    assert old in text
    text = text.replace(old, new).replace("../data/motors", (ROOT / "data/motors").as_posix())
    path = tmp_path / "rocket.toml"
    path.write_text(text)
    return path


def test_example_config_estimates_drag_from_its_geometry():
    cfg = load_config(ROOT / "configs" / "example_dual_deploy.toml")
    assert isinstance(cfg.rocket.cd, DragBuildup)
    assert cfg.rocket.cd.roughness == 60e-6
    assert 0.45 < cfg.rocket.drag_coefficient(0.3) < 0.60


def test_drag_table_cannot_be_combined_with_a_fixed_cd(tmp_path):
    path = _config(tmp_path, "[drag]", "cd = 0.5\n\n[drag]")
    with pytest.raises(ValueError, match="not both"):
        load_config(path)


def test_drag_table_needs_a_known_model_and_the_fin_thickness(tmp_path):
    with pytest.raises(ValueError, match="model"):
        load_config(_config(tmp_path, 'model = "geometry"', 'model = "magic"'))
    with pytest.raises(ValueError, match="thickness"):
        load_config(_config(tmp_path, "thickness_m = 0.0032", "mass_kg = 0.12"))


def test_cli_reports_the_drag_estimate_and_warns_when_it_is_out_of_range(tmp_path, capsys):
    assert main(["simulate", str(ROOT / "configs" / "example_dual_deploy.toml")]) == 0
    out = capsys.readouterr().out
    assert "Drag: estimated from geometry, Cd 0.5" in out
    assert "Warning" not in out

    hot = _config(
        tmp_path,
        'file = "../data/motors/Synthetic_H250.eng"',
        "thrust_N = 900.0\nburn_time_s = 1.2\npropellant_mass_kg = 0.5\nloaded_mass_kg = 0.8",
    )
    assert main(["simulate", str(hot)]) == 0
    out = capsys.readouterr().out
    assert "Warning: the flight reaches Mach" in out
    assert "no wave drag" in out
