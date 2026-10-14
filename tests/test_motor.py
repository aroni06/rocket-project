import numpy as np
import pytest

from apogee.motor import Motor, parse_eng

TRIANGLE = """
; comment lines are ignored
Tri 29 100 0 0.05 0.10 Test
0.5 100
1.0 0
"""


def test_parse_header_and_curve(synthetic_eng):
    m = Motor.from_eng(synthetic_eng)
    assert m.name == "Synthetic_H250"
    assert m.diameter == pytest.approx(0.038)
    assert m.length == pytest.approx(0.152)
    assert m.propellant_mass == pytest.approx(0.135)
    assert m.loaded_mass == pytest.approx(0.260)
    assert m.burn_time == pytest.approx(1.32)
    assert m.impulse_class == "H"


def test_curve_without_zero_start_gets_origin_point():
    m = parse_eng(TRIANGLE)
    np.testing.assert_allclose(m.time, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(m.thrust, [0.0, 100.0, 0.0])
    assert m.total_impulse == pytest.approx(50.0)  # triangle area
    assert m.average_thrust == pytest.approx(50.0)
    assert m.max_thrust == pytest.approx(100.0)


def test_impulse_is_exact_for_linear_segments():
    m = parse_eng(TRIANGLE)
    # Area under the rising edge F = 200 t up to t = 0.25 s is 0.5 * 0.25 * 50
    assert m.impulse_at(0.25) == pytest.approx(6.25)
    assert m.impulse_at(0.5) == pytest.approx(25.0)
    assert m.impulse_at(2.0) == pytest.approx(50.0)


def test_mass_depletes_with_impulse():
    m = parse_eng(TRIANGLE)
    assert m.mass_at(0.0) == pytest.approx(0.10)
    assert m.mass_at(0.5) == pytest.approx(0.10 - 0.025)  # half the impulse delivered
    assert m.mass_at(1.0) == pytest.approx(0.05)
    assert m.mass_at(5.0) == pytest.approx(0.05)


def test_thrust_is_zero_outside_burn():
    m = Motor.constant(thrust=40.0, burn_time=2.0, propellant_mass=0.1)
    assert m.thrust_at(-0.1) == 0.0
    assert m.thrust_at(1.0) == 40.0
    assert m.thrust_at(2.1) == 0.0
    assert m.total_impulse == pytest.approx(80.0)


@pytest.mark.parametrize(
    ("impulse", "letter"),
    [(1.0, "1/2A"), (2.5, "A"), (2.6, "B"), (10.0, "C"), (160.0, "G"), (161.0, "H"), (5000, "L")],
)
def test_impulse_class(impulse, letter):
    m = Motor.constant(thrust=impulse, burn_time=1.0, propellant_mass=0.01)
    assert m.impulse_class == letter


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time": [0.0, 0.0], "thrust": [1.0, 1.0]},
        {"time": [0.1, 1.0], "thrust": [1.0, 1.0]},
        {"time": [0.0, 1.0], "thrust": [-1.0, 1.0]},
        {"time": [0.0, 1.0], "thrust": [0.0, 0.0]},
    ],
)
def test_invalid_curves_raise(kwargs):
    with pytest.raises(ValueError):
        Motor(name="bad", propellant_mass=0.1, loaded_mass=0.2, **kwargs)
