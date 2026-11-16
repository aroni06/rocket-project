import math

import numpy as np
import pytest

from apogee import LaunchRail, Motor, Rocket, load_config, simulate
from apogee.aero import NOSE_CP_FRACTION, CdTable, barrowman
from apogee.components import Airframe, BodyTube, FinSet, NoseCone, PointMass


def _airframe(**fin_overrides) -> Airframe:
    fins = {
        "count": 3,
        "root_chord": 0.10,
        "tip_chord": 0.05,
        "span": 0.05,
        "sweep": 0.05,
        "position": 0.60,
        "mass": 0.03,
    }
    fins.update(fin_overrides)
    return Airframe(
        nose=NoseCone(length=0.20, shape="conical", mass=0.05),
        body=BodyTube(length=0.60, diameter=0.05, mass=0.20),
        fins=FinSet(**fins),
    )


def _with_absolute_motor_path(config):
    """Config text that still finds its motor file when copied to another folder."""
    motors = config.parent.parent / "data" / "motors"
    return config.read_text().replace("../data/motors", str(motors))


def test_barrowman_matches_hand_calculation():
    # Worked by hand from Barrowman's equations for the airframe above:
    #   interference K = 1 + 0.025 / 0.075                        = 1.333333
    #   mid-chord line = sqrt(0.05^2 + (0.05 + 0.025 - 0.05)^2)   = 0.0559017
    #   fins CN_a = K * 4*3*(0.05/0.05)^2 / (1 + sqrt(1 + (2*0.0559017/0.15)^2))
    #             = 1.333333 * 12 / 2.247219                      = 7.119911
    #   fins x    = 0.60 + (0.05/3)(0.10 + 0.10)/0.15 + (0.15 - 0.005/0.15)/6 = 0.641667
    #   nose CN_a = 2 at 2/3 * 0.20                               = 0.133333
    #   x_cp = (2 * 0.133333 + 7.119911 * 0.641667) / 9.119911    = 0.530189
    cp = barrowman(_airframe())
    assert cp.fin_cn_alpha == pytest.approx(7.119911, rel=1e-6)
    assert cp.fin_position == pytest.approx(0.641667, rel=1e-6)
    assert cp.nose_position == pytest.approx(0.133333, rel=1e-5)
    assert cp.cn_alpha == pytest.approx(9.119911, rel=1e-6)
    assert cp.position == pytest.approx(0.530189, rel=1e-5)


@pytest.mark.parametrize("shape", list(NOSE_CP_FRACTION))
def test_nose_shape_sets_nose_cp(shape):
    af = _airframe()
    af.nose.shape = shape
    assert barrowman(af).nose_position == pytest.approx(NOSE_CP_FRACTION[shape] * 0.20)


def test_fin_changes_move_cp_the_expected_way():
    base = barrowman(_airframe()).position
    assert barrowman(_airframe(span=0.07)).position > base  # bigger fins: CP moves aft
    assert barrowman(_airframe(count=4)).position > base  # more fins: CP moves aft
    assert barrowman(_airframe(position=0.50)).position < base  # fins forward: CP forward


def test_four_fins_have_four_thirds_the_normal_force_of_three():
    three = barrowman(_airframe()).fin_cn_alpha
    four = barrowman(_airframe(count=4)).fin_cn_alpha
    assert four == pytest.approx(three * 4 / 3)


def test_fin_centroid_for_simple_shapes():
    rect = FinSet(3, root_chord=0.1, tip_chord=0.1, span=0.05)
    tri = FinSet(3, root_chord=0.1, tip_chord=0.0, span=0.05)
    assert rect.centroid == pytest.approx(0.05)
    assert tri.centroid == pytest.approx(0.1 / 3)
    assert rect.planform_area == pytest.approx(0.005)


def test_fin_mass_from_thickness_and_density():
    fins = FinSet(3, 0.10, 0.05, 0.05, thickness=0.003, density=1800.0)
    assert fins.mass == pytest.approx(3 * 0.00375 * 0.003 * 1800.0)


def test_dry_mass_and_cg_sum_over_components():
    af = _airframe()
    af.masses.append(PointMass("ballast", 0.10, 0.25))
    assert af.mass == pytest.approx(0.05 + 0.20 + 0.03 + 0.10)
    fin_cg = 0.60 + af.fins.centroid
    expected = (0.05 * (2 / 3 * 0.20) + 0.20 * 0.50 + 0.03 * fin_cg + 0.10 * 0.25) / 0.38
    assert af.cg == pytest.approx(expected)


def test_measured_mass_and_cg_override_the_sum():
    af = _airframe()
    af.mass_override, af.cg_override = 0.5, 0.42
    assert (af.mass, af.cg) == (0.5, 0.42)


def test_fins_default_to_flush_with_the_aft_end():
    af = _airframe(position=None)
    assert af.fin_position == pytest.approx(0.80 - 0.10)


def test_nose_ballast_raises_static_margin():
    af = _airframe()
    rocket = Rocket.from_airframe("r", af, motor=None)
    before = rocket.static_margin_at(0.0)
    af.masses.append(PointMass("nose weight", 0.05, 0.10))
    heavier = Rocket.from_airframe("r", af, motor=None)
    assert heavier.static_margin_at(0.0) > before


def test_cg_moves_forward_and_margin_grows_as_propellant_burns():
    motor = Motor(
        "m", np.array([0.0, 1.0]), np.array([50.0, 50.0]), 0.06, 0.10, diameter=0.029, length=0.12
    )
    rocket = Rocket.from_airframe("r", _airframe(), motor)
    assert rocket.motor_cg == pytest.approx(0.80 - 0.06)  # aft-flush, mid-length
    assert rocket.cg_at(1.0) < rocket.cg_at(0.0)
    assert rocket.static_margin_at(1.0) > rocket.static_margin_at(0.0)
    # Liftoff CG: dry airframe plus the loaded motor at its mid-length.
    af = rocket.airframe
    expected = (af.mass * af.cg + 0.10 * 0.74) / (af.mass + 0.10)
    assert rocket.cg_at(0.0) == pytest.approx(expected)
    cp = rocket.center_of_pressure.position
    assert rocket.static_margin_at(0.0) == pytest.approx((cp - expected) / 0.05)

    result = simulate(rocket, rail=LaunchRail(1.0))
    assert result.cp == pytest.approx(cp)
    assert result.static_margin_at("rail_exit") > result.static_margin[0]
    assert result.min_static_margin == pytest.approx(result.static_margin[0])
    assert result.static_margin_at("burnout") == pytest.approx(rocket.static_margin_at(1.0))
    assert "liftoff_static_margin_cal" in result.summary()


def test_rocket_without_geometry_has_no_stability():
    rocket = Rocket("plain", dry_mass=1.0, diameter=0.05, motor=None)
    with pytest.raises(ValueError, match="no airframe"):
        rocket.static_margin_at(0.0)
    result = simulate(rocket, rail=LaunchRail(0.0), initial_speed=30.0)
    assert result.static_margin is None
    assert result.min_static_margin is None
    assert "liftoff_static_margin_cal" not in result.summary()


def test_example_config_is_stable(example_config):
    cfg = load_config(example_config)
    assert cfg.rocket.airframe is not None
    assert cfg.rocket.dry_mass == pytest.approx(1.10)
    assert cfg.rocket.diameter == pytest.approx(0.066)
    result = simulate(cfg.rocket, cfg.environment, cfg.rail)
    assert 1.0 < result.static_margin_at("rail_exit") < 2.5


def test_partial_geometry_in_config_is_rejected(example_config, tmp_path):
    text = _with_absolute_motor_path(example_config)
    start, end = text.index("[fins]"), text.index("[[mass]]")
    broken = tmp_path / "broken.toml"
    broken.write_text(text[:start] + text[end:])
    with pytest.raises(ValueError, match="needs all of"):
        load_config(broken)


@pytest.mark.parametrize(
    "build",
    [
        lambda: NoseCone(length=0.0),
        lambda: NoseCone(length=0.2, shape="pointy"),
        lambda: BodyTube(length=0.5, diameter=0.0),
        lambda: FinSet(2, 0.1, 0.05, 0.05),
        lambda: FinSet(3, 0.1, 0.05, 0.0),
        lambda: _airframe(position=0.10),  # on the nose cone
        lambda: _airframe(position=0.75),  # hangs off the back
    ],
)
def test_invalid_geometry_raises(build):
    with pytest.raises(ValueError):
        build()


def test_cd_table_interpolates_and_holds_ends():
    table = CdTable([0.0, 0.5, 1.0], [0.40, 0.44, 0.80])
    assert table(0.25) == pytest.approx(0.42)
    assert table(0.75) == pytest.approx(0.62)
    assert table(-1.0) == pytest.approx(0.40)
    assert table(3.0) == pytest.approx(0.80)


def test_cd_table_from_csv_skips_headers_sorts_and_dedupes(tmp_path):
    path = tmp_path / "cd.csv"
    path.write_text(
        "# exported\nMach number,Drag coefficient\n0.5,0.44\n0.0,0.40\n0.5,0.99\n1.0,0.8\n"
    )
    table = CdTable.from_csv(path)
    np.testing.assert_allclose(table.mach, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(table.cd, [0.40, 0.44, 0.80])


def test_cd_file_in_config_changes_the_flight(example_config, tmp_path):
    text = _with_absolute_motor_path(example_config)
    (tmp_path / "cd.csv").write_text("mach,cd\n0.0,0.45\n0.3,0.45\n0.6,0.90\n")
    lines = [
        'cd_file = "cd.csv"' if line.startswith("cd = ") else line for line in text.splitlines()
    ]
    cfg_path = tmp_path / "rocket.toml"
    cfg_path.write_text("\n".join(lines))
    with_table = load_config(cfg_path)
    assert isinstance(with_table.rocket.cd, CdTable)
    base = load_config(example_config)
    high = simulate(with_table.rocket, with_table.environment, with_table.rail)
    low = simulate(base.rocket, base.environment, base.rail)
    assert high.apogee < low.apogee  # more drag above Mach 0.3


@pytest.mark.parametrize("bad", [([0.0, 0.0], [0.4, 0.5]), ([0.0, 1.0], [0.4, -0.1])])
def test_invalid_cd_tables_raise(bad):
    with pytest.raises(ValueError):
        CdTable(*bad)


def test_static_margin_definition_is_cp_minus_cg_over_diameter():
    rocket = Rocket.from_airframe("r", _airframe(), motor=None)
    cp, cg = rocket.center_of_pressure.position, rocket.cg_at(0.0)
    assert rocket.static_margin_at(0.0) == pytest.approx((cp - cg) / 0.05)
    assert math.isclose(rocket.cg_at(0.0), rocket.airframe.cg)
