"""Reading OpenRocket simulations, and the validation case that ships with the repo."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from apogee import load_config, simulate
from apogee.cli import main
from apogee.openrocket import (
    Comparison,
    compare,
    format_comparison,
    read_simulation,
    read_simulations,
)

CASE = Path(__file__).resolve().parents[1] / "validation" / "ae100"

MINIMAL = """<?xml version='1.0' encoding='utf-8'?>
<openrocket version="1.9" creator="OpenRocket 23.09">
  <rocket><name>test</name></rocket>
  <simulations>
    <simulation status="uptodate">
      <name>Calm day</name>
      <conditions><windaverage>0.0</windaverage><launchrodlength>1.5</launchrodlength>
        <atmosphere model="isa"/></conditions>
      <flightdata maxaltitude="20.0" timetoapogee="2.0" flighttime="4.0">
        <databranch name="Sustainer" types="Time,Altitude,Vertical velocity">
          <event time="0" type="ignition"/>
          <event time="2.0" type="apogee"/>
          <datapoint>0,0,20</datapoint>
          <datapoint>1,15,10</datapoint>
          <datapoint>2,20,NaN</datapoint>
          <datapoint>4,0,-20</datapoint>
        </databranch>
      </flightdata>
    </simulation>
    <simulation status="notsimulated"><name>Never run</name></simulation>
  </simulations>
</openrocket>
"""


@pytest.fixture(scope="module")
def case():
    cfg = load_config(CASE / "config.toml")
    result = simulate(cfg.rocket, cfg.environment, cfg.rail)
    sim = read_simulation(CASE / "ae100.ork")
    return cfg.rocket, result, sim


def test_reads_a_plain_xml_file_and_skips_simulations_without_data(tmp_path):
    path = tmp_path / "plain.ork"
    path.write_text(MINIMAL)
    sims = read_simulations(path)
    assert [s.name for s in sims] == ["Calm day"]
    sim = read_simulation(path)
    assert list(sim.columns) == ["Time", "Altitude", "Vertical velocity"]
    assert sim["Altitude"].tolist() == [0.0, 15.0, 20.0, 0.0]
    assert sim.events == {"ignition": 0.0, "apogee": 2.0}
    assert sim.summary == {"maxaltitude": 20.0, "timetoapogee": 2.0, "flighttime": 4.0}
    assert sim.conditions == {"windaverage": 0.0, "launchrodlength": 1.5}
    assert sim.at("Altitude", 0.5) == pytest.approx(7.5)
    # Undefined steps are skipped, not interpolated through.
    assert sim.at("Vertical velocity", 2.0) == pytest.approx(0.0)
    with pytest.raises(KeyError, match="Mach number"):
        sim["Mach number"]
    with pytest.raises(ValueError, match="has 1 simulation"):
        read_simulation(path, 1)


def test_files_without_results_or_that_are_not_openrocket_are_rejected(tmp_path):
    empty = tmp_path / "empty.ork"
    empty.write_text("<openrocket><simulations/></openrocket>")
    with pytest.raises(ValueError, match="no simulation results"):
        read_simulation(empty)
    junk = tmp_path / "junk.ork"
    junk.write_text("not xml at all")
    with pytest.raises(ValueError, match="not an OpenRocket file"):
        read_simulation(junk)


def test_reads_the_zipped_file_openrocket_saves(case):
    _, _, sim = case
    assert sim.name == "Simulation 1"
    assert len(sim.columns) == 54
    assert len(sim.time) == 181
    assert sim.events["launchrod"] == pytest.approx(0.308)
    assert sim.events["burnout"] == pytest.approx(1.65)
    assert sim.summary["maxaltitude"] == pytest.approx(110.166)
    assert sim.conditions["windaverage"] == 2.0
    assert sim.conditions["launchrodlength"] == 1.0
    assert np.nanmax(sim["Altitude"]) == pytest.approx(110.166, abs=0.05)


# How far each quantity may sit from OpenRocket's, in percent. The design goes in as
# OpenRocket describes it and nothing is tuned, so these are the validation result.
TOLERANCE = {
    "Liftoff mass": 0.5,
    "Burnout mass": 0.5,
    "CG at liftoff": 1.0,
    "CG at burnout": 1.0,
    "Center of pressure": 1.5,
    "Drag coefficient at top speed": 2.0,
    "Rail exit speed": 2.0,
    "Maximum speed": 1.0,
    "Burnout altitude": 2.0,
    "Apogee": 3.5,
    "Time to apogee": 3.5,
    "Descent rate at landing": 1.5,
    "Flight time": 2.5,
}


def test_ae100_rocket_matches_openrocket(case):
    rows = {row.label: row for row in compare(*case)}
    assert set(rows) == set(TOLERANCE) | {"Sideways distance at apogee"}
    for label, limit in TOLERANCE.items():
        assert abs(rows[label].percent) < limit, f"{label}: {rows[label].percent:+.2f}%"


def test_drag_estimate_tracks_openrocket_through_the_whole_climb(case):
    rocket, _, sim = case
    flying = (sim.time > sim.events["liftoff"]) & (sim.time < sim.events["ejectioncharge"])
    ours = np.array(
        [
            rocket.drag_coefficient(mach, reynolds / rocket.airframe.length)
            for mach, reynolds in zip(
                sim["Mach number"][flying], sim["Reynolds number"][flying], strict=True
            )
        ]
    )
    theirs = sim["Drag coefficient"][flying]
    assert np.max(np.abs(ours / theirs - 1.0)) < 0.02
    parts = rocket.cd.breakdown(0.1, 0.1 * 340.3 / 1.46e-5)
    # Base and pressure drag are constants here, and agree to OpenRocket's three decimals.
    assert parts.base == pytest.approx(0.060, abs=1e-3)
    assert parts.fin_pressure + parts.transition_pressure == pytest.approx(0.080, abs=2e-3)


def test_the_known_gap_is_the_turn_into_the_wind(case):
    # This model points the rocket into the relative wind the moment it leaves the rod.
    # OpenRocket's rocket has inertia and turns more slowly, so it flies less far upwind
    # and loses less altitude to the wind. With no wind this model flies higher than
    # OpenRocket's windy flight; with the wind, lower.
    cfg = load_config(CASE / "config.toml")
    cfg.environment.wind_speed = 0.0
    calm = simulate(cfg.rocket, cfg.environment, cfg.rail)
    _, windy, sim = case
    assert calm.apogee > sim.summary["maxaltitude"] > windy.apogee
    sideways = {row.label: row for row in compare(*case)}["Sideways distance at apogee"]
    assert sideways.ours > 1.5 * sideways.openrocket


def test_comparison_rows_and_table():
    row = Comparison("Apogee", "m", 98.0, 100.0)
    assert row.difference == pytest.approx(-2.0)
    assert row.percent == pytest.approx(-2.0)
    text = format_comparison([row, Comparison("Drag coefficient", "", 0.3, 0.3)])
    assert "OpenRocket" in text.splitlines()[0]
    assert "Apogee [m]" in text and "-2.0%" in text
    assert "Drag coefficient " in text and "+0.0%" in text


def test_compare_leaves_out_what_a_simulation_does_not_have(case, tmp_path):
    rocket, result, _ = case
    path = tmp_path / "plain.ork"
    path.write_text(MINIMAL)
    rows = compare(rocket, result, read_simulation(path))
    assert [row.label for row in rows] == [
        "Apogee",
        "Time to apogee",
        "Descent rate at landing",
        "Flight time",
    ]


def test_cli_compare_prints_the_table_and_saves_the_plot(tmp_path, capsys):
    plot = tmp_path / "comparison.png"
    code = main(
        ["compare", str(CASE / "config.toml"), str(CASE / "ae100.ork"), "--plot", str(plot)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "OpenRocket flew: 2 m/s wind, 1 m rod" in out
    assert "This config has: 2 m/s wind, 1 m rod" in out
    assert "Apogee [m]" in out
    assert plot.stat().st_size > 10_000

    empty = tmp_path / "empty.ork"
    empty.write_text("<openrocket/>")
    assert main(["compare", str(CASE / "config.toml"), str(empty)]) == 1
    assert "no simulation results" in capsys.readouterr().err


def test_real_flight_is_explained_by_a_strong_wind_or_a_heavier_rocket():
    spec = importlib.util.spec_from_file_location("ae100_flight", CASE / "flight.py")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fit, flight, _, _ = module.analyse()
    assert 7.5 < fit.wind < 9.0  # m/s
    assert fit.rms < 1.5  # metres, over a 65 m climb
    assert fit.windy.apogee * flight.indicated == pytest.approx(fit.recorded_apogee, abs=1.5)
    assert 2.2 < fit.ejection_delay < 3.0
    assert 0.35 < fit.chute_cd < 0.5
    assert 0.06 < fit.extra_mass < 0.10
    assert fit.fitted.landing_descent_rate == pytest.approx(fit.descent_rate, rel=0.02)
