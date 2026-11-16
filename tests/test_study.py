"""Design studies: sweeps and the optimizer against problems with a known answer."""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from apogee import (
    ConstantAtmosphere,
    Constraint,
    Environment,
    LaunchRail,
    Objective,
    Rocket,
    Study,
    Variable,
    load_config,
    simulate,
)
from apogee.atmosphere import G0
from apogee.cli import main
from apogee.study import ConfigEvaluator, UnknownMetricError, set_config_value

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "studies"
FLAT = Environment(atmosphere=ConstantAtmosphere(), inverse_square_gravity=False)


def bowl(values):
    """A smooth bowl with its minimum at (1, 2), outside the limit x + y <= 2."""
    x, y = values["x"], values["y"]
    return {"cost": (x - 1) ** 2 + (y - 2) ** 2, "sum": x + y}


def bowl_study(**kwargs) -> Study:
    variables = [Variable("x", -1.0, 3.0, steps=9), Variable("y", -1.0, 3.0, steps=9)]
    constraints = [Constraint("sum", upper=2.0)]
    return Study(bowl, variables, Objective("cost", "minimize"), constraints, **kwargs)


# The optimizer on problems with a closed-form answer


def test_optimizer_finds_constrained_minimum_of_a_bowl():
    # Closest point to (1, 2) on the line x + y = 2 is (0.5, 1.5), at distance^2 = 0.5.
    optimum = bowl_study().optimize()
    assert optimum.feasible
    assert optimum.values["x"] == pytest.approx(0.5, abs=1e-5)
    assert optimum.values["y"] == pytest.approx(1.5, abs=1e-5)
    assert optimum.metrics["cost"] == pytest.approx(0.5, rel=1e-6)
    assert optimum.active == ["sum at its maximum of 2"]


def test_optimizer_reports_a_variable_bound_that_limits_the_design():
    variables = [Variable("x", -1.0, 0.5), Variable("y", 0.0, 1.0)]
    optimum = Study(bowl, variables, Objective("cost", "minimize")).optimize()
    assert optimum.values == pytest.approx({"x": 0.5, "y": 1.0}, abs=1e-6)
    assert optimum.active == [
        "x at the top of its range, 0.5",
        "y at the top of its range, 1",
    ]


def test_target_objective_lands_on_the_target():
    study = Study(
        lambda v: {"area": v["side"] ** 2},
        [Variable("side", 0.0, 5.0)],
        Objective("area", "target", 4.0),
    )
    optimum = study.optimize()
    assert optimum.values["side"] == pytest.approx(2.0, abs=1e-4)


def test_choices_are_enumerated_and_ranked():
    # For each offset the best x is the offset itself, so cost is 0; "far" breaks the limit.
    def evaluate(values):
        offset = {"near": 0.2, "far": 0.9}[values["kind"]]
        return {"cost": (values["x"] - offset) ** 2, "offset": offset}

    study = Study(
        evaluate,
        [Variable("kind", values=("far", "near")), Variable("x", 0.0, 1.0)],
        Objective("cost", "minimize"),
        [Constraint("offset", upper=0.5)],
    )
    optimum = study.optimize()
    assert optimum.values["kind"] == "near"
    assert optimum.values["x"] == pytest.approx(0.2, abs=1e-5)
    assert [(p.values["kind"], p.feasible) for p in optimum.candidates] == [
        ("near", True),
        ("far", False),
    ]
    assert "breaks a constraint" in str(optimum)


def test_infeasible_study_says_so():
    variables = [Variable("x", 0.0, 1.0), Variable("y", 0.0, 1.0)]
    limits = [Constraint("sum", lower=10.0)]
    optimum = Study(bowl, variables, Objective("cost", "minimize"), limits).optimize()
    assert not optimum.feasible
    assert "NO FEASIBLE DESIGN" in str(optimum)
    # The closest design is the one that breaks the limit by the least: both at their top.
    assert optimum.values == pytest.approx({"x": 1.0, "y": 1.0}, abs=1e-6)


def test_optimizer_finds_the_mass_that_coasts_highest():
    # A fixed impulse J gives a rocket of mass m the speed J/m. With quadratic drag it then
    # coasts to h(m) = m/(2k) ln(1 + k (J/m)^2 / (m g)). Too light and drag stops it at
    # once; too heavy and it starts too slowly. The best mass is in between.
    impulse, d, cd = 20.0, 0.05, 0.5
    k = 0.5 * 1.225 * cd * math.pi * d**2 / 4

    def closed_form(m):
        return m / (2 * k) * math.log(1 + k * (impulse / m) ** 2 / (m * G0))

    def evaluate(values):
        m = values["mass_kg"]
        rocket = Rocket("coaster", dry_mass=m, diameter=d, motor=None, cd=cd)
        result = simulate(rocket, FLAT, LaunchRail(length=0.0), initial_speed=impulse / m)
        return {"apogee_m": result.apogee}

    best = minimize_scalar(lambda m: -closed_form(m), bounds=(0.02, 1.0), method="bounded")
    study = Study(evaluate, [Variable("mass_kg", 0.02, 1.0)], Objective("apogee_m"))
    optimum = study.optimize(starts=2, samples=6)
    assert 0.03 < best.x < 0.9  # the answer is not at either end of the range
    assert optimum.metrics["apogee_m"] == pytest.approx(closed_form(best.x), rel=1e-6)
    assert optimum.values["mass_kg"] == pytest.approx(best.x, rel=2e-3)


# The example studies


def test_recovery_study_matches_hand_sizing():
    # Under canopy, drag balances weight, so the landing speed limit fixes the total drag
    # area: CdA_total = 2 m g / (rho v^2). When the main opens at the drogue's descent
    # rate, the deceleration is g * CdA_main / (CdA_drogue + CdA_body), so the opening
    # load limit fixes the ratio of the two. Two limits, two diameters.
    study = Study.from_file(STUDIES / "recovery.toml")
    limits = {c.metric: c.upper for c in study.constraints}
    cfg = load_config(ROOT / "configs" / "example_dual_deploy.toml")
    rocket, env = cfg.rocket, cfg.environment
    drogue, main_chute = rocket.parachutes
    mass = rocket.mass_at(1e9)
    # The body's own drag at the landing speed; it barely changes with speed down here.
    v_land = limits["landing_descent_rate_mps"]
    state = env.atmosphere.at(env.site_elevation)
    body = rocket.reference_area * rocket.drag_coefficient(
        v_land / state.speed_of_sound, v_land / state.kinematic_viscosity
    )

    total = 2 * mass * env.gravity(0.0) / (state.density * v_land**2)
    main_cda = total * limits["main_opening_load_g"] / (limits["main_opening_load_g"] + 1)
    drogue_cda = total - main_cda - body

    def diameter(cda, chute):
        return math.sqrt(4 * cda / (math.pi * chute.cd))

    optimum = study.optimize()
    assert optimum.feasible
    assert optimum.values["parachute.main.diameter_m"] == pytest.approx(
        diameter(main_cda, main_chute), rel=2e-3
    )
    assert optimum.values["parachute.drogue.diameter_m"] == pytest.approx(
        diameter(drogue_cda, drogue), rel=5e-3
    )
    assert len(optimum.active) == 2

    # The grid cannot beat the optimizer, and its best design is close to the optimum.
    sweep = study.sweep()
    assert sweep.best.score >= optimum.best.score - 1e-6
    assert sweep.best.score == pytest.approx(optimum.best.score, rel=0.15)


def test_motor_study_picks_the_motor_that_fits_the_field():
    study = Study.from_file(STUDIES / "motor_and_fins.toml")
    optimum = study.optimize(starts=1, samples=4)
    assert optimum.feasible
    assert optimum.values["motor.file"].endswith("Synthetic_H250.eng")
    # Smaller fins are lighter, so the optimizer shrinks them until stability stops it.
    assert optimum.metrics["rail_exit_static_margin_cal"] == pytest.approx(1.0, abs=1e-4)
    assert optimum.active == ["rail_exit_static_margin_cal at its minimum of 1"]
    by_motor = {p.values["motor.file"].rsplit("_", 1)[-1]: p for p in optimum.candidates}
    assert by_motor["H120.eng"].feasible
    assert not by_motor["I300.eng"].feasible  # flies through the altitude ceiling


# Sweeps


def test_sweep_points_match_single_flights():
    data_path = ROOT / "configs" / "example_dual_deploy.toml"
    study = Study.from_file(STUDIES / "recovery.toml")
    coarse = [
        Variable("parachute.drogue.diameter_m", 0.3, 0.5, steps=2),
        Variable("rail.angle_deg", values=(85.0, 90.0)),
    ]
    sweep = Study(study.evaluate, coarse, study.objective, study.constraints).sweep()
    assert sweep.shape == (2, 2)

    # The baseline design is on the grid: 0.3 m drogue, vertical rail.
    cfg = load_config(data_path)
    reference = simulate(cfg.rocket, cfg.environment, cfg.rail)
    assert sweep.metric("apogee_m")[0, 1] == pytest.approx(reference.apogee, rel=1e-8)
    assert sweep.metric("landing_descent_rate_mps")[0, 1] == pytest.approx(
        reference.landing_descent_rate, rel=1e-8
    )
    # A bigger drogue drifts further, whatever the rail angle.
    drift = sweep.metric("descent_drift_m")
    assert np.all(drift[1] > drift[0])


def test_sweep_best_is_the_best_feasible_grid_point():
    sweep = bowl_study().sweep()
    assert sweep.shape == (9, 9)
    best = sweep.best
    assert best.values == {"x": 0.5, "y": 1.5}
    assert int(sweep.feasible.sum()) == 45  # x + y <= 2 on a 9 x 9 grid of half steps
    assert "45 meet every constraint" in str(sweep)


def test_sweep_keeps_going_past_designs_that_cannot_be_built(tmp_path):
    study = Study.from_file(STUDIES / "recovery.toml")
    variables = [Variable("fins.count", values=(2, 3))]
    sweep = Study(study.evaluate, variables, study.objective).sweep()
    assert list(sweep.errors) == [(0,)]
    assert "at least 3 fins" in sweep.errors[(0,)]
    assert sweep.feasible.tolist() == [False, True]
    assert sweep.best.values == {"fins.count": 3}

    out = tmp_path / "sweep.csv"
    sweep.to_csv(out)
    rows = out.read_text().splitlines()
    assert rows[0].startswith("fins.count,feasible,apogee_m,")
    assert rows[1].startswith("2,0,,")
    assert rows[2].startswith("3,1,")


# Config addressing and study files


def test_set_config_value_addresses_tables_and_named_entries():
    data = {
        "fins": {"span_m": 0.06},
        "parachute": [
            {"name": "drogue", "diameter_m": 0.3},
            {"name": "main", "diameter_m": 0.9},
        ],
    }
    set_config_value(data, "fins.span_m", 0.08)
    set_config_value(data, "parachute.main.diameter_m", 1.2)
    assert data["fins"]["span_m"] == 0.08
    assert [p["diameter_m"] for p in data["parachute"]] == [0.3, 1.2]
    with pytest.raises(KeyError, match="fins.spam_m"):
        set_config_value(data, "fins.spam_m", 1.0)
    with pytest.raises(KeyError, match="parachute.reserve"):
        set_config_value(data, "parachute.reserve.diameter_m", 1.0)


def test_config_evaluator_leaves_the_base_config_untouched():
    study = Study.from_file(STUDIES / "recovery.toml")
    evaluator = study.evaluate
    assert isinstance(evaluator, ConfigEvaluator)
    before = evaluator.config({}).rocket.parachutes[1].diameter
    changed = evaluator.config({"parachute.main.diameter_m": 1.3}).rocket.parachutes[1].diameter
    assert (before, changed) == (0.9, 1.3)
    assert evaluator.config({}).rocket.parachutes[1].diameter == 0.9


def _study_file(tmp_path, body: str):
    path = tmp_path / "study.toml"
    base = (ROOT / "configs" / "example_dual_deploy.toml").as_posix()
    path.write_text(f'base = "{base}"\n{body}')
    return path


def test_study_file_with_a_mistyped_variable_is_rejected(tmp_path):
    path = _study_file(
        tmp_path,
        '[objective]\nmetric = "apogee_m"\n[[variable]]\npath = "fins.spam_m"\nmin = 0\nmax = 1\n',
    )
    with pytest.raises(ValueError, match="fins.spam_m"):
        Study.from_file(path)


def test_study_file_without_an_objective_is_rejected(tmp_path):
    path = _study_file(tmp_path, '[[variable]]\npath = "fins.span_m"\nmin = 0.04\nmax = 0.1\n')
    with pytest.raises(ValueError, match="objective"):
        Study.from_file(path)


def test_unknown_metric_is_an_error_not_an_infeasible_design(tmp_path):
    path = _study_file(
        tmp_path,
        '[objective]\nmetric = "apogee_ft"\n'
        '[[variable]]\npath = "fins.span_m"\nmin = 0.05\nmax = 0.07\nsteps = 2\n',
    )
    study = Study.from_file(path)
    with pytest.raises(UnknownMetricError, match="apogee_ft.*apogee_m"):
        study.sweep()
    with pytest.raises(UnknownMetricError):
        study.optimize()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower": 1.0},
        {"lower": 2.0, "upper": 1.0},
        {"lower": 0.0, "upper": 1.0, "steps": 1},
        {"values": ()},
        {"values": (1, 2), "lower": 0.0},
    ],
)
def test_bad_variables_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Variable("x", **kwargs)


def test_bad_objectives_and_constraints_are_rejected():
    with pytest.raises(ValueError):
        Objective("apogee_m", "biggest")
    with pytest.raises(ValueError):
        Objective("apogee_m", "target")
    with pytest.raises(ValueError):
        Constraint("apogee_m")
    with pytest.raises(ValueError):
        Constraint("apogee_m", lower=2.0, upper=1.0)


# Command line and plots


def test_cli_sweep_and_optimize(tmp_path, capsys):
    study = STUDIES / "recovery.toml"
    csv, png = tmp_path / "sweep.csv", tmp_path / "sweep.png"
    assert main(["sweep", str(study), "--csv", str(csv), "--plot", str(png)]) == 0
    out = capsys.readouterr().out
    assert "81 designs" in out
    assert len(csv.read_text().splitlines()) == 82
    assert png.stat().st_size > 0

    assert main(["optimize", str(study)]) == 0
    out = capsys.readouterr().out
    assert "Limited by" in out
    assert "landing_descent_rate_mps at its maximum of 5" in out


def test_sweep_plots_cover_each_kind_of_design_space(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from apogee.plots import plot_sweep

    def evaluate(values):
        scale = {"a": 1.0, "b": 2.0}[values.get("kind", "a")]
        x, y = values.get("x", 0.0), values.get("y", 0.0)
        return {"height_m": scale * (4 - (x - 1) ** 2 - y**2), "sum": x + y}

    x, y = Variable("x", 0.0, 2.0, steps=5), Variable("y", 0.0, 2.0, steps=5)
    kind = Variable("kind", values=("a", "b"))
    limit = [Constraint("sum", upper=2.5), Constraint("height_m", lower=1.0)]
    for name, variables in {"line": [x], "bars": [kind], "lines": [kind, x], "map": [x, y]}.items():
        study = Study(evaluate, variables, Objective("height_m"), limit, name=name)
        optimum = study.optimize() if name != "bars" else None
        out = tmp_path / f"{name}.png"
        plot_sweep(study.sweep(), out, optimum)
        assert out.stat().st_size > 0
    with pytest.raises(ValueError, match="one or two variables"):
        plot_sweep(Study(evaluate, [kind, x, y], Objective("height_m")).sweep())
