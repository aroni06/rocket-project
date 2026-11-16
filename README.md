# apogee

[![CI](https://github.com/aroni06/rocket-project/actions/workflows/ci.yml/badge.svg)](https://github.com/aroni06/rocket-project/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**A 2D rocket flight simulator for model and high-power rockets, written in Python.** From a
rocket's geometry, a motor thrust curve and its parachutes, it predicts apogee, max velocity,
acceleration and dynamic pressure, rail exit speed, static margin, parachute deployment
conditions, landing speed and wind drift.

> **Status: work in progress.** The flight model, parachute recovery and stability (milestones
> 1 to 4) are done and checked against closed-form solutions and hand calculations. OpenRocket
> validation and the design optimizer are next; see the [roadmap](#roadmap).

![Side view of the example rocket with CG and CP](docs/images/example_rocket.png)

![Simulated flight of the example rocket](docs/images/example_flight.png)

![Flight path of the example rocket](docs/images/example_trajectory.png)

*Example L1 rocket on a synthetic 38 mm H motor, flown dual deploy in a 4 m/s wind: a 0.3 m
drogue opens 1 s after apogee and a 0.9 m main opens at 150 m.*

## Quickstart

```bash
git clone https://github.com/aroni06/rocket-project.git
cd rocket-project
pip install -e ".[dev]"

apogee simulate configs/example_dual_deploy.toml \
    --plot flight.png --trajectory path.png --drawing rocket.png
pytest
```

```
Motor: Synthetic_H250 (H, 260.8 N s, 1.32 s burn)
Flight summary: Example L1 rocket, dual deploy
  Apogee                                   896.58 m
  Max speed                                170.48 m/s
  Max ascent acceleration                   17.83 g
  Rail exit speed                           24.24 m/s
  Rail exit static margin                    1.99 cal
  Drogue descent rate                       13.88 m/s
  Main deploy altitude                     150.00 m
  Main opening load (upper bound)            8.88 g
  Landing descent rate                       4.38 m/s
  Landing downrange                         88.62 m
  ...
Stability: stable, 1.99 cal at rail exit
```

Rockets are defined in TOML: a [single-deploy example](configs/example_l1.toml) using the
motor's ejection charge, and a [dual-deploy example](configs/example_dual_deploy.toml). Each
parachute opens on a trigger plus an optional delay: `apogee`, descending through a set
`altitude`, or `burnout` (a motor ejection delay).

The `[nose]`, `[body]`, `[fins]` and `[[mass]]` sections describe the airframe. Dry mass and
center of gravity are summed from the parts, or you can give measured values with `dry_mass_kg`
and `dry_cg_m`. Drag is a constant `cd` or a `cd_file` CSV of Mach number against drag
coefficient, for example one exported from OpenRocket. Motors are RASP `.eng`
files, the format published by [thrustcurve.org](https://www.thrustcurve.org). The bundled
`Synthetic_H250.eng` is a made-up curve so the example runs out of the box; download a real
`.eng` for real predictions.

From Python:

```python
from apogee import load_config, simulate

cfg = load_config("configs/example_l1.toml")
result = simulate(cfg.rocket, cfg.environment, cfg.rail)
print(result.apogee, result.rail_exit_speed, result.landing_descent_rate)
print(result.static_margin_at("rail_exit"), cfg.rocket.center_of_pressure.position)
```

## Model

State `y = [x, z, vx, vz]` in a flat-Earth ground frame (x downrange, z above the pad). Mass is a
known function of time from the motor, so it is not integrated.

$$
m(t)\,\dot{\mathbf v} = T(t)\,\hat{\mathbf u} \;-\; \tfrac12 \rho\, C_D(M)\, A\, |\mathbf v_{air}|\,\mathbf v_{air} \;+\; m\,\mathbf g(h),
\qquad \mathbf v_{air} = \mathbf v - \mathbf v_{wind}
$$

| Piece | Model |
| --- | --- |
| Thrust | RASP thrust curve, linear interpolation |
| Mass | Propellant burns in proportion to delivered impulse: m_p(t) = m_p0 (1 - I(t) / I_total) |
| Atmosphere | 1976 U.S. Standard Atmosphere, 0 to 32 km, launch-site elevation and temperature offset |
| Drag | Constant C_D, a C_D-vs-Mach table from a CSV file, or any function C_D(Mach) |
| Center of pressure | Barrowman's equations for the nose cone (conical, ogive, parabolic, elliptical) and trapezoidal fins with fin-body interference |
| Center of gravity | Mass-weighted average of nose, body tube, fins, point masses and the motor; moves forward as propellant burns |
| Static margin | (x_CP - x_CG) / d in calibers, reported at liftoff, rail exit and burnout |
| Gravity | Inverse-square, g = g0 (R / (R + h))^2 |
| Attitude | Along the rail until rail exit, then along the air-relative velocity (zero angle of attack) |
| Wind | Constant horizontal wind |
| Parachutes | Drag 1/2 rho v^2 Cd A per canopy against the air-relative velocity; triggered at apogee, at an altitude, or by motor ejection delay; drogues stay attached after the main opens |

The flight is integrated phase by phase with SciPy's `solve_ivp` (DOP853, adaptive step), and
every discontinuity lands on a step boundary: each thrust-curve corner, rail exit, burnout,
apogee and parachute deployment. Rail exit, apogee, deployment altitudes, landing and the speed
peak are located exactly with event functions rather than read off the nearest time step.

**Deliberately not modelled:** rotational (6-DOF) dynamics, weathercocking on the rail, wind
profiles and gusts, Earth rotation, multi-stage vehicles.

**Known limitations:**

- Barrowman's method assumes subsonic flow and small angles of attack, so the center of
  pressure does not shift with Mach number. One body diameter and one fin set are supported
  (no transitions or boat tails yet).
- The nose cone's own center of gravity defaults to 2/3 of its length from the tip. Weigh and
  balance real parts, or give the whole rocket's measured `dry_cg_m`.
- Parachutes inflate instantly, so the reported opening load is an upper bound on the real
  opening shock.
- The rocket turns into the relative wind the moment it leaves the rail. A real rocket has
  rotational inertia and turns more slowly, so the model overpredicts how far upwind it flies in
  a crosswind (220 m upwind at apogee in the example). OpenRocket comparisons will quantify this.

## Verification

The test suite (`pytest`, run in CI on every push) checks the simulator against cases with a
closed-form answer:

| Case | Compared quantity | Tolerance |
| --- | --- | --- |
| Constant thrust in vacuum | Rail exit time and speed, burnout speed and altitude, apogee (rocket equation with gravity) | 1e-7 relative |
| Vertical coast with quadratic drag | Apogee, time to apogee, landing speed | 1e-7 relative |
| Drag-free projectile at 45 degrees | Range, apogee, landing speed | 1e-7 relative |
| Ballistic arc, inverse-square gravity | Specific energy conserved | 1e-8 relative |
| Solver tolerance 1e-6 vs 1e-11 | Apogee and max speed | 1e-5 relative |
| Standard atmosphere | T, p, rho at 0, 1, 5, 11 and 20 km vs 1976 tables | 1e-4 relative |
| Fall under a parachute from apogee | Landing descent rate vs the closed-form quadratic-drag solution | 1e-6 relative |
| Drogue, then drogue plus main | Steady descent rates vs terminal velocity sqrt(2mg / rho Cd A) | 1e-3 relative |
| Deployment triggers | Apogee + delay, set altitude, burnout + ejection delay | exact event times |
| Descent in wind | Horizontal speed under canopy settles at wind speed | 1e-3 relative |
| Barrowman center of pressure | Fin normal-force slope, fin and nose CP, total CP vs a hand calculation | 1e-5 relative |
| Center of gravity | Component sum, motor at mid-length, forward shift during the burn | exact |

Comparison against OpenRocket on real rocket designs comes with milestone 5.

## Project layout

```
src/apogee/
  atmosphere.py    standard atmosphere
  motor.py         .eng parser, thrust(t), mass(t)
  rocket.py        rocket definition, CG through the burn, static margin
  environment.py   gravity, wind, launch site
  recovery.py      parachutes, deployment triggers, terminal velocity
  components.py    nose cone, body tube, fins, point masses, dry mass and CG
  aero.py          Barrowman center of pressure, Cd-vs-Mach tables
  flight.py        equations of motion, phases, events, FlightResult
  config.py        TOML loader
  plots.py         standard flight plots
  cli.py           `apogee simulate`
configs/           example rockets
data/motors/       thrust curves
tests/             unit and analytic verification tests
```

## Roadmap

- [x] 1. Repo skeleton, packaging, CI
- [x] 2. Core 2D flight model with analytic verification
- [x] 3. Recovery: single and dual deploy, landing speed, wind drift
- [x] 4. Aerodynamics and stability: C_D(Mach) import, Barrowman CP, CG, static margin
- [ ] 5. Validation against OpenRocket on 2 to 3 rockets
- [ ] 6. Parameter sweeps and constrained design optimizer
- [ ] 7. Flight animation, technical report and v1.0
- [ ] 8. After v1.0: liquid and hybrid engines with throttling

## License

MIT
