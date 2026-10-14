import pytest

from apogee.atmosphere import StandardAtmosphere


@pytest.mark.parametrize(
    ("altitude", "temperature", "pressure", "density"),
    [
        # 1976 U.S. Standard Atmosphere tables (geometric altitude)
        (0.0, 288.15, 101_325.0, 1.2250),
        (1_000.0, 281.651, 89_874.6, 1.11164),
        (5_000.0, 255.676, 54_048.3, 0.736429),
        (11_000.0, 216.774, 22_699.9, 0.364801),
        (20_000.0, 216.650, 5_529.31, 0.0889097),
    ],
)
def test_matches_standard_atmosphere_table(altitude, temperature, pressure, density):
    s = StandardAtmosphere().at(altitude)
    assert s.temperature == pytest.approx(temperature, rel=1e-4)
    assert s.pressure == pytest.approx(pressure, rel=1e-4)
    assert s.density == pytest.approx(density, rel=1e-4)


def test_sea_level_speed_of_sound():
    assert StandardAtmosphere().at(0.0).speed_of_sound == pytest.approx(340.294, rel=1e-5)


def test_temperature_offset_lowers_density():
    hot = StandardAtmosphere(temperature_offset=15.0).at(0.0)
    assert hot.temperature == pytest.approx(303.15)
    assert hot.density < 1.225
