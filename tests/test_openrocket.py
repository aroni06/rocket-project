"""Reading OpenRocket simulations, and the validation case that ships with the repo."""

import importlib.util
import sys
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

VALIDATION = Path(__file__).resolve().parents[1] / "validation"
CASE = VALIDATION / "ae100"
SIMPLE = VALIDATION / "simple_model_rocket"


def _load(path: Path):
    """Import a validation script, which lives outside the package."""
    name = f"validation_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    module = _load(CASE / "flight.py")
    fit, flight, _, _ = module.analyse()
    assert 7.5 < fit.wind < 9.0  # m/s
    assert fit.rms < 1.5  # metres, over a 65 m climb
    assert fit.windy.apogee * flight.indicated == pytest.approx(fit.recorded_apogee, abs=1.5)
    assert 2.2 < fit.ejection_delay < 3.0
    assert 0.35 < fit.chute_cd < 0.5
    assert 0.06 < fit.extra_mass < 0.10
    assert fit.fitted.landing_descent_rate == pytest.approx(fit.descent_rate, rel=0.02)


def test_rod_exit_speed_is_taken_at_the_end_of_the_rod_not_the_next_time_step(case):
    # OpenRocket reports the speed at the first step past the rod. For this rocket that is
    # 5.7 cm and 1% further on; at an A8's peak thrust it is 2.5%.
    _, result, sim = case
    rows = {row.label: row for row in compare(*case)}
    assert rows["Rail exit speed"].openrocket < sim.summary["launchrodvelocity"]
    assert abs(rows["Rail exit speed"].percent) < 0.3


# The simple model rocket: OpenRocket's own example design, on five motor configurations.
SIMPLE_TOLERANCE = {
    "Liftoff mass": 1.0,  # OpenRocket stores mass to the gram, 1.5% of this rocket
    "CG at liftoff": 0.3,
    "Center of pressure": 2.0,
    "Drag coefficient at top speed": 3.0,
    "Rail exit speed": 0.3,
    "Maximum speed": 0.6,
    "Apogee": 2.5,
    "Time to apogee": 2.0,
    "Descent rate at landing": 2.0,
    "Flight time": 2.5,
}


@pytest.fixture(scope="module")
def simple_cases():
    return _load(SIMPLE / "compare.py").fly_all()


def test_simple_model_rocket_matches_openrocket_on_all_five_motors(simple_cases):
    assert [label for label, *_ in simple_cases] == ["A8-3", "B4-4", "C6-3", "C6-5", "C6-7"]
    for label, rocket, result, sim in simple_cases:
        rows = {row.label: row for row in compare(rocket, result, sim)}
        for quantity, limit in SIMPLE_TOLERANCE.items():
            percent = rows[quantity].percent
            assert abs(percent) < limit, f"{label} {quantity}: {percent:+.2f}%"


def test_simple_model_rocket_apogee_is_low_by_a_similar_amount_on_every_motor(simple_cases):
    # The apogees span a factor of six. A difference that stays between 0 and 2.5% low on
    # all of them is a property of the model (the turn into the wind), not a coincidence.
    errors = [
        100 * (result.apogee / sim.summary["maxaltitude"] - 1) for _, _, result, sim in simple_cases
    ]
    assert all(-2.5 < error < 0 for error in errors)
    apogees = [sim.summary["maxaltitude"] for *_, sim in simple_cases]
    assert max(apogees) / min(apogees) > 6


def test_component_masses_add_up_to_openrockets_dry_mass(simple_cases):
    # OpenRocket's total minus its motor, both recovered to a few hundredths of a gram by
    # the method in make_motors.py, is 48.1 to 48.3 g on every configuration.
    _, rocket, _, _ = simple_cases[0]
    assert rocket.dry_mass == pytest.approx(0.04822, abs=2e-5)
    assert rocket.airframe.cg == pytest.approx(0.2058, abs=2e-4)


def test_motor_rebuild_recovers_the_published_c6():
    motors = _load(SIMPLE / "make_motors.py")
    sim = read_simulations(SIMPLE / "simple_model_rocket.ork")[motors.C6_SIMULATION]
    t, f, loaded, propellant = motors.rebuild(sim)
    published = motors.Motor.from_eng(SIMPLE / "Estes_C6.eng")
    impulse = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))
    assert impulse == pytest.approx(published.total_impulse, rel=2e-3)
    assert loaded == pytest.approx(published.loaded_mass, abs=5e-5)
    assert propellant == pytest.approx(published.propellant_mass, abs=5e-5)
    assert t[-1] == pytest.approx(published.burn_time)
    # The committed A8 and B4 files are what the script writes.
    for name, index, _ in motors.MOTORS:
        t, f, loaded, propellant = motors.rebuild(
            read_simulations(SIMPLE / "simple_model_rocket.ork")[index]
        )
        committed = motors.Motor.from_eng(SIMPLE / f"Estes_{name}.eng")
        assert committed.loaded_mass == pytest.approx(loaded, abs=1e-5)
        assert committed.total_impulse == pytest.approx(
            float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t))), rel=1e-3
        )
