"""Design studies: parameter sweeps and constrained optimization.

A study answers "which design is best, given these limits?". It has four parts:

- **variables**: the config values that are free to change. Each one is either a range
  (``min``, ``max``) or a list of choices (``values``), for example a set of motor files
  or the parachute sizes a vendor sells.
- **an objective**: one metric to maximize, minimize or bring to a target value.
- **constraints**: limits on other metrics, such as static margin between 1 and 2.5
  calibers or landing no faster than 5 m/s.
- **an evaluator**: a function that turns variable values into metrics. For a rocket
  config that means building the rocket and flying it.

:meth:`Study.sweep` flies every point on a grid, which shows the whole design space and
is easy to plot. :meth:`Study.optimize` searches the ranges continuously with a gradient
based optimizer (SLSQP) started from several points, once for every combination of the
list-valued variables. The two should agree, and the tests check that they do.

A variable is addressed by its dotted path in the config file: ``fins.span_m``,
``rail.angle_deg``, ``motor.file``. Entries of repeated tables are addressed by name:
``parachute.main.diameter_m``, ``mass.ballast.mass_kg``.

Metrics are the keys of :meth:`apogee.flight.FlightResult.summary` (``apogee_m``,
``rail_exit_speed_mps``, ``rail_exit_static_margin_cal``, ``landing_descent_rate_mps``,
...) plus the extras added by :func:`design_metrics`.
"""

from __future__ import annotations

import copy
import csv
import itertools
import math
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .config import SimulationConfig, build_config, read_config
from .flight import FlightResult, simulate

Values = Mapping[str, Any]
Evaluator = Callable[[Values], dict[str, float]]

GOALS = ("maximize", "minimize", "target")

# What building or flying an impossible design raises: bad geometry (ValueError), a
# rocket that never leaves the rail (RuntimeError), a division by zero.
DESIGN_FAILURES = (ValueError, RuntimeError, ArithmeticError)

# A constraint counts as met if it is violated by less than this fraction of its bound.
FEASIBILITY_TOL = 1e-6
# A constraint or variable bound counts as active (limiting the design) within this.
ACTIVE_TOL = 1e-3
# Step for finite-difference gradients, as a fraction of each variable's range. Large
# enough to sit well above the integrator's noise, small enough to stay local.
GRADIENT_STEP = 1e-4


class UnknownMetricError(ValueError):
    """An objective or constraint names a metric the evaluator does not produce."""


@dataclass(frozen=True)
class Variable:
    """A config value that the study may change.

    Give ``lower`` and ``upper`` for a continuous range, with ``steps`` grid points when
    swept. Or give ``values`` for a list of choices, which may be numbers or strings.
    """

    path: str
    lower: float | None = None
    upper: float | None = None
    steps: int = 5
    values: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if self.values is not None:
            if self.lower is not None or self.upper is not None:
                raise ValueError(f"variable {self.path!r}: give values or min/max, not both")
            if len(self.values) == 0:
                raise ValueError(f"variable {self.path!r}: values is empty")
            return
        if self.lower is None or self.upper is None:
            raise ValueError(f"variable {self.path!r}: needs min and max, or values")
        if not self.upper > self.lower:
            raise ValueError(f"variable {self.path!r}: max must be greater than min")
        if self.steps < 2:
            raise ValueError(f"variable {self.path!r}: steps must be at least 2")

    @property
    def is_choice(self) -> bool:
        return self.values is not None

    @property
    def is_numeric(self) -> bool:
        return self.values is None or all(_is_number(v) for v in self.values)

    @property
    def grid(self) -> tuple[Any, ...]:
        """The values this variable takes in a sweep."""
        if self.values is not None:
            return self.values
        return tuple(float(v) for v in np.linspace(self.lower, self.upper, self.steps))


@dataclass(frozen=True)
class Constraint:
    """Limits on one metric: ``lower <= metric <= upper``, either side optional."""

    metric: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError(f"constraint on {self.metric!r}: needs min, max or both")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"constraint on {self.metric!r}: min is greater than max")

    def margins(self, value: float) -> list[float]:
        """Scaled distance inside each limit; negative means that limit is violated."""
        out = []
        if self.lower is not None:
            out.append((value - self.lower) / _scale(self.lower))
        if self.upper is not None:
            out.append((self.upper - value) / _scale(self.upper))
        return out

    def describe(self) -> str:
        if self.lower is not None and self.upper is not None:
            return f"{self.lower:g} to {self.upper:g}"
        if self.lower is not None:
            return f"at least {self.lower:g}"
        return f"at most {self.upper:g}"


@dataclass(frozen=True)
class Objective:
    """What to optimize: maximize or minimize ``metric``, or bring it to ``value``."""

    metric: str
    goal: str = "maximize"
    value: float | None = None

    def __post_init__(self) -> None:
        if self.goal not in GOALS:
            raise ValueError(f"objective goal must be one of {GOALS}")
        if self.goal == "target" and self.value is None:
            raise ValueError("a target objective needs a value")

    def score(self, value: float) -> float:
        """A number that is smaller for better designs."""
        if self.goal == "maximize":
            return -value
        if self.goal == "minimize":
            return value
        return abs(value - self.value)

    def describe(self) -> str:
        if self.goal == "target":
            return f"{self.metric} as close as possible to {self.value:g}"
        return f"{self.goal} {self.metric}"


@dataclass
class DesignPoint:
    """One evaluated design: its variable values, metrics and whether it meets the limits.

    ``score`` is smaller for better designs. ``violation`` is how far outside the worst
    broken limit the design is, as a fraction of that limit; zero if it meets them all.
    """

    values: dict[str, Any]
    metrics: dict[str, float]
    feasible: bool
    score: float
    violation: float = 0.0


def design_metrics(config: SimulationConfig, result: FlightResult) -> dict[str, float]:
    """The flight summary plus a few numbers that are useful as objectives and limits.

    ``descent_drift_m`` is the distance drifted between apogee and landing and
    ``landing_distance_m`` is how far from the pad the rocket lands. Both are positive
    whichever way the wind blows.
    """
    out = result.summary()
    out["liftoff_mass_kg"] = config.rocket.liftoff_mass
    out["descent_drift_m"] = abs(out["landing_downrange_m"] - out["apogee_downrange_m"])
    out["landing_distance_m"] = abs(out["landing_downrange_m"])
    if (margin := result.min_static_margin) is not None:
        out["min_static_margin_cal"] = margin
    return out


def set_config_value(data: dict, path: str, value: Any) -> None:
    """Set the value at a dotted ``path`` in config data, in place.

    The path must already exist, so a typo raises instead of silently sweeping nothing.
    """
    node: Any = data
    keys = path.split(".")
    for depth, key in enumerate(keys[:-1]):
        node = _step(node, key, ".".join(keys[: depth + 1]))
    last = keys[-1]
    if not isinstance(node, dict) or last not in node:
        raise KeyError(f"{path!r} is not in the base config; add it there first")
    node[last] = value


def _step(node: Any, key: str, where: str) -> Any:
    if isinstance(node, dict) and key in node:
        return node[key]
    if isinstance(node, list):
        for entry in node:
            if isinstance(entry, dict) and entry.get("name") == key:
                return entry
    raise KeyError(f"{where!r} is not in the base config")


class ConfigEvaluator:
    """Turns variable values into metrics by flying a modified copy of a config.

    ``max_step`` caps the integrator's step. Events are still located exactly, so a
    coarser cap than a plotted flight needs changes the metrics by parts in 1e9 while
    making each flight several times faster.
    """

    def __init__(self, data: dict, base: str | Path = ".", max_step: float = 1.0) -> None:
        self.data = data
        self.base = Path(base)
        self.max_step = max_step

    def config(self, values: Values) -> SimulationConfig:
        """The rocket, environment and rail for one set of variable values."""
        data = copy.deepcopy(self.data)
        for path, value in values.items():
            set_config_value(data, path, value)
        return build_config(data, self.base)

    def fly(self, values: Values, **kwargs: Any) -> FlightResult:
        cfg = self.config(values)
        return simulate(cfg.rocket, cfg.environment, cfg.rail, **kwargs)

    def __call__(self, values: Values) -> dict[str, float]:
        cfg = self.config(values)
        result = simulate(cfg.rocket, cfg.environment, cfg.rail, max_step=self.max_step)
        return design_metrics(cfg, result)


@dataclass
class Study:
    """A design space, what to optimize in it and the limits a design must meet."""

    evaluate: Evaluator
    variables: Sequence[Variable]
    objective: Objective
    constraints: Sequence[Constraint] = ()
    name: str = "study"

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("a study needs at least one variable")
        paths = [v.path for v in self.variables]
        if len(set(paths)) != len(paths):
            raise ValueError("variable paths must be unique")

    @classmethod
    def from_file(cls, path: str | Path) -> Study:
        """Load a study from a TOML file. See ``studies/`` for examples::

        base = "../configs/example_dual_deploy.toml"

        [objective]
        metric = "apogee_m"
        goal = "maximize"            # "maximize", "minimize" or "target" (with value)

        [[variable]]
        path = "fins.span_m"
        min = 0.04
        max = 0.10
        steps = 7                    # grid points when swept

        [[variable]]
        path = "motor.file"          # file paths are relative to the study file
        values = ["../data/motors/A.eng", "../data/motors/B.eng"]

        [[constraint]]
        metric = "rail_exit_static_margin_cal"
        min = 1.0
        max = 2.5
        """
        path = Path(path)
        with path.open("rb") as fh:
            spec = tomllib.load(fh)
        try:
            base = (path.parent / spec["base"]).resolve()
            variables = [_variable(v, path.parent) for v in spec["variable"]]
            obj = spec["objective"]
            objective = Objective(obj["metric"], obj.get("goal", "maximize"), obj.get("value"))
            constraints = [
                Constraint(c["metric"], c.get("min"), c.get("max"))
                for c in spec.get("constraint", [])
            ]
        except KeyError as exc:
            raise ValueError(f"{path}: missing required key {exc}") from exc
        evaluator = ConfigEvaluator(read_config(base), base.parent)
        study = cls(evaluator, variables, objective, constraints, spec.get("name", path.stem))
        try:
            evaluator.config({v.path: v.grid[0] for v in variables})
        except KeyError as exc:
            raise ValueError(f"{path}: {exc.args[0]}") from exc
        return study

    # Scoring one design

    def point(self, values: Values) -> DesignPoint:
        """Evaluate one design and judge it against the objective and constraints."""
        values = dict(values)
        return self._judge(values, self.evaluate(values))

    def _judge(self, values: dict[str, Any], metrics: dict[str, float]) -> DesignPoint:
        violation = max([0.0, *(-m for m in self._margins(metrics))])
        score = self.objective.score(self._metric(metrics, self.objective.metric))
        return DesignPoint(values, metrics, violation <= FEASIBILITY_TOL, score, violation)

    def _metric(self, metrics: Mapping[str, float], name: str) -> float:
        try:
            return metrics[name]
        except KeyError:
            known = ", ".join(sorted(metrics))
            raise UnknownMetricError(
                f"metric {name!r} is not available. Available: {known}"
            ) from None

    def _margins(self, metrics: Mapping[str, float]) -> list[float]:
        out: list[float] = []
        for c in self.constraints:
            out.extend(c.margins(self._metric(metrics, c.metric)))
        return out

    # Sweep

    def sweep(self) -> SweepResult:
        """Evaluate every combination of the variables' grid values.

        A design that cannot be built or flown (for example it never leaves the rail) is
        recorded as infeasible with its error message rather than stopping the sweep.
        """
        grids = [v.grid for v in self.variables]
        shape = tuple(len(g) for g in grids)
        points = np.empty(shape, dtype=object)
        errors: dict[tuple[int, ...], str] = {}
        for index in np.ndindex(shape):
            values = {v.path: g[i] for v, g, i in zip(self.variables, grids, index, strict=True)}
            try:
                points[index] = self.point(values)
            except UnknownMetricError:
                raise
            except DESIGN_FAILURES as exc:
                points[index] = DesignPoint(values, {}, False, math.nan, math.inf)
                errors[index] = str(exc)
        return SweepResult(self, points, errors)

    # Optimize

    def optimize(self, starts: int = 3, samples: int | None = None, seed: int = 0) -> Optimum:
        """Find the best feasible design.

        Ranges are searched continuously and choices are enumerated: for each combination
        of the list-valued variables, the ranges are sampled at ``samples`` points (Latin
        hypercube), and SLSQP is run from the best ``starts`` of them. Variables and
        constraints are scaled to order one, and gradients are central differences taken
        inside the bounds, two extra flights per variable.

        SLSQP is a local method, so several starts guard against stopping at a local
        optimum; a sweep over the same study is the independent check.
        """
        ranges = [v for v in self.variables if not v.is_choice]
        choices = [v for v in self.variables if v.is_choice]
        candidates: list[DesignPoint] = []
        evaluations = 0
        for combo in itertools.product(*(v.values for v in choices)):
            fixed = {v.path: value for v, value in zip(choices, combo, strict=True)}
            problem = _Problem(self, ranges, fixed)
            best = problem.solve(starts, samples, seed) if ranges else problem.at(np.empty(0))
            evaluations += problem.evaluations
            if best is not None:
                candidates.append(best)
        if not candidates:
            raise RuntimeError(f"{self.name}: no design in the study could be built and flown")
        candidates.sort(key=_rank)
        return Optimum(self, candidates[0], candidates, evaluations)


def _rank(point: DesignPoint) -> tuple[float, float]:
    """Sort key: feasible designs by score, then the rest by how badly they break a limit."""
    return (0.0 if point.feasible else point.violation, point.score)


class _Problem:
    """The continuous part of a study, in variables scaled to the unit cube.

    Every flight is cached, so the objective, all constraints and their gradients at a
    point cost one flight plus two per variable, however many constraints there are.
    """

    def __init__(self, study: Study, ranges: Sequence[Variable], fixed: dict[str, Any]) -> None:
        self.study = study
        self.ranges = ranges
        self.fixed = fixed
        self.lower = np.array([v.lower for v in ranges], dtype=float)
        self.span = np.array([v.upper - v.lower for v in ranges], dtype=float)
        self.scale = 1.0
        self.evaluations = 0
        self._points: dict[bytes, DesignPoint | None] = {}
        self._jacobians: dict[bytes, np.ndarray] = {}

    def values(self, u: np.ndarray) -> dict[str, Any]:
        x = self.lower + np.clip(u, 0.0, 1.0) * self.span
        named = {v.path: float(xi) for v, xi in zip(self.ranges, x, strict=True)}
        return {**self.fixed, **named}

    def at(self, u: np.ndarray) -> DesignPoint | None:
        """The design at ``u``, or ``None`` if it cannot be built or flown."""
        key = np.asarray(u, dtype=float).tobytes()
        if key not in self._points:
            self.evaluations += 1
            try:
                self._points[key] = self.study.point(self.values(u))
            except UnknownMetricError:
                raise
            except DESIGN_FAILURES:
                self._points[key] = None
        return self._points[key]

    def _vector(self, u: np.ndarray) -> np.ndarray:
        """Scaled objective followed by every constraint margin."""
        point = self.at(u)
        if point is None:
            raise _FailedDesign
        objective = self.study.objective
        value = point.metrics[objective.metric]
        if objective.goal == "target":
            f = ((value - objective.value) / self.scale) ** 2
        else:
            f = objective.score(value) / self.scale
        return np.array([f, *self.study._margins(point.metrics)])

    def _jacobian(self, u: np.ndarray) -> np.ndarray:
        key = np.asarray(u, dtype=float).tobytes()
        if key not in self._jacobians:
            columns = []
            for i in range(u.size):
                # Central differences, shifted inward at a bound so every trial design
                # stays inside its range.
                lo, hi = np.array(u, dtype=float), np.array(u, dtype=float)
                lo[i] = max(u[i] - GRADIENT_STEP, 0.0)
                hi[i] = min(u[i] + GRADIENT_STEP, 1.0)
                columns.append((self._vector(hi) - self._vector(lo)) / (hi[i] - lo[i]))
            self._jacobians[key] = np.column_stack(columns)
        return self._jacobians[key]

    def solve(self, starts: int, samples: int | None, seed: int) -> DesignPoint | None:
        n = len(self.ranges)
        count = samples if samples is not None else max(8, 4 * n)
        sampler = qmc.LatinHypercube(d=n, seed=seed)
        seeds = [np.full(n, 0.5), *sampler.random(count)]
        ranked = sorted(
            ((point, u) for u in seeds if (point := self.at(u)) is not None),
            key=lambda pu: _rank(pu[0]),
        )
        if not ranked:
            return None
        objective = self.study.objective
        if objective.goal == "target":
            self.scale = _scale(objective.value)
        else:
            self.scale = _scale(ranked[0][0].metrics[objective.metric])

        n_margins = len(self.study._margins(ranked[0][0].metrics))
        constraints = [
            {
                "type": "ineq",
                "fun": lambda u, k=k: self._vector(u)[k + 1],
                "jac": lambda u, k=k: self._jacobian(u)[k + 1],
            }
            for k in range(n_margins)
        ]
        found = [point for point, _ in ranked[:1]]
        for _, u0 in ranked[:starts]:
            try:
                res = minimize(
                    lambda u: self._vector(u)[0],
                    u0,
                    jac=lambda u: self._jacobian(u)[0],
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * n,
                    constraints=constraints,
                    options={"ftol": 1e-10, "maxiter": 50},
                )
            except _FailedDesign:
                continue
            if (point := self.at(np.clip(res.x, 0.0, 1.0))) is not None:
                found.append(point)
        # SLSQP can stop a hair outside a limit; prefer the best design that is inside.
        found.extend(p for p in self._points.values() if p is not None and p.feasible)
        return min(found, key=_rank)


class _FailedDesign(Exception):
    """Raised inside the optimizer when a trial design cannot be built or flown."""


@dataclass
class SweepResult:
    """Every design on a study's grid. ``points`` has one axis per variable."""

    study: Study
    points: np.ndarray
    errors: dict[tuple[int, ...], str] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.points.shape

    def __iter__(self) -> Iterator[DesignPoint]:
        return iter(self.points.flat)

    def metric(self, name: str) -> np.ndarray:
        """One metric over the grid, NaN where a design failed or lacks it."""
        return np.array([p.metrics.get(name, math.nan) for p in self]).reshape(self.shape)

    @property
    def feasible(self) -> np.ndarray:
        return np.array([p.feasible for p in self], dtype=bool).reshape(self.shape)

    @property
    def best(self) -> DesignPoint | None:
        """The best feasible design on the grid, or ``None`` if there is none."""
        feasible = [p for p in self if p.feasible]
        return min(feasible, key=lambda p: p.score) if feasible else None

    def to_csv(self, path: str | Path) -> None:
        """One row per design: variable values, a feasible flag and every metric."""
        names: dict[str, None] = {}
        for p in self:
            names.update(dict.fromkeys(p.metrics))
        paths = [v.path for v in self.study.variables]
        with Path(path).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([*paths, "feasible", *names])
            for p in self:
                row = [p.values[k] for k in paths]
                row.append(int(p.feasible))
                row.extend(repr(p.metrics[m]) if m in p.metrics else "" for m in names)
                writer.writerow(row)

    def __str__(self) -> str:
        study = self.study
        n_feasible = int(self.feasible.sum())
        lines = [
            f"Sweep: {study.name}",
            f"  {self.points.size} designs, {n_feasible} meet every constraint"
            + (f", {len(self.errors)} could not be flown" if self.errors else ""),
        ]
        best = self.best
        if best is None:
            lines.append("  No design on the grid meets every constraint.")
        else:
            lines.append(f"Best on the grid ({study.objective.describe()})")
            lines.extend(_describe_point(study, best))
        return "\n".join(lines)


@dataclass
class Optimum:
    """Result of :meth:`Study.optimize`.

    ``candidates`` holds the best design found for each combination of the list-valued
    variables, best first, so it ranks the motors (or parachutes) against each other.
    """

    study: Study
    best: DesignPoint
    candidates: list[DesignPoint]
    evaluations: int

    @property
    def values(self) -> dict[str, Any]:
        return self.best.values

    @property
    def metrics(self) -> dict[str, float]:
        return self.best.metrics

    @property
    def feasible(self) -> bool:
        return self.best.feasible

    @property
    def active(self) -> list[str]:
        """What stops the design from being any better: limits and bounds it sits on."""
        out = []
        for c in self.study.constraints:
            value = self.best.metrics[c.metric]
            if c.lower is not None and abs(value - c.lower) <= ACTIVE_TOL * _scale(c.lower):
                out.append(f"{c.metric} at its minimum of {c.lower:g}")
            if c.upper is not None and abs(value - c.upper) <= ACTIVE_TOL * _scale(c.upper):
                out.append(f"{c.metric} at its maximum of {c.upper:g}")
        for v in self.study.variables:
            if v.is_choice:
                continue
            value, span = self.best.values[v.path], v.upper - v.lower
            if abs(value - v.lower) <= ACTIVE_TOL * span:
                out.append(f"{v.path} at the bottom of its range, {v.lower:g}")
            if abs(value - v.upper) <= ACTIVE_TOL * span:
                out.append(f"{v.path} at the top of its range, {v.upper:g}")
        return out

    def __str__(self) -> str:
        study = self.study
        lines = [f"Optimum: {study.name} ({study.objective.describe()})"]
        if not self.feasible:
            lines.append("  NO FEASIBLE DESIGN FOUND. Closest design:")
        lines.extend(_describe_point(study, self.best))
        if self.active:
            lines.append("Limited by")
            lines.extend(f"  {item}" for item in self.active)
        choices = [v for v in study.variables if v.is_choice]
        if choices and len(self.candidates) > 1:
            lines.append("Best design for each choice")
            for p in self.candidates:
                label = ", ".join(_format(p.values[v.path]) for v in choices)
                value = p.metrics[study.objective.metric]
                note = "" if p.feasible else "  (breaks a constraint)"
                lines.append(f"  {label:<38}{value:>10.2f}{note}")
        lines.append(f"Flights simulated: {self.evaluations}")
        return "\n".join(lines)


def _describe_point(study: Study, point: DesignPoint) -> list[str]:
    lines = [f"  {v.path:<38}{_format(point.values[v.path]):>10}" for v in study.variables]
    objective = study.objective.metric
    lines.append(f"  {objective:<38}{point.metrics[objective]:>10.2f}")
    for c in study.constraints:
        value = point.metrics[c.metric]
        ok = all(m >= -FEASIBILITY_TOL for m in c.margins(value))
        verdict = "" if ok else "  VIOLATED"
        lines.append(f"  {c.metric:<38}{value:>10.2f}  ({c.describe()}){verdict}")
    return lines


def _variable(spec: dict, base: Path) -> Variable:
    path = spec["path"]
    if "values" in spec:
        values = spec["values"]
        if path.endswith("file"):
            values = [str((base / v).resolve()) for v in values]
        return Variable(path, values=tuple(values))
    return Variable(path, spec["min"], spec["max"], spec.get("steps", 5))


def _scale(value: float) -> float:
    """A size to divide by so that quantities near ``value`` become order one."""
    return abs(value) if abs(value) > 1e-9 else 1.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format(value: Any) -> str:
    if _is_number(value):
        return f"{value:.4g}"
    text = str(value)
    return Path(text).stem if text.endswith((".eng", ".csv")) else text
