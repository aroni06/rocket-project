# Validation

The tests in `tests/` check the simulator against problems with a closed-form answer. This
folder checks it against two things that are not arithmetic: OpenRocket, the simulator most of
the hobby uses, and a real flight.

Each case is a folder holding an OpenRocket design saved with its simulation data (`.ork`), the
same rocket written by hand as an `apogee` config, and the motor file. Every number in the
config is read off the design. Nothing is adjusted to make the results agree.

```bash
apogee compare validation/ae100/config.toml validation/ae100/ae100.ork --plot comparison.png
```

## AE 100 rocket

A 46 cm, 300 g model rocket on an Estes D12-3: an elliptical nose cone wider than the body tube,
a short transition down to the tube, three small fins and a 30 cm parachute. It is a useful first
case because it is awkward. The static margin is about one caliber, it leaves a 1 m rod at only
12 m/s, and the nose-to-body transition takes a third of the normal force away from the nose.

![Side view of the AE 100 rocket](ae100/rocket.png)

### Against OpenRocket

OpenRocket's own simulation used a 1 m rod, 2 m/s wind and a standard atmosphere at sea level, so
the config does too.

| | apogee | OpenRocket | difference |
| --- | ---: | ---: | ---: |
| Liftoff mass [kg] | 0.2997 | 0.300 | -0.1% |
| Burnout mass [kg] | 0.2786 | 0.279 | -0.1% |
| CG at liftoff [m] | 0.1596 | 0.160 | -0.2% |
| CG at burnout [m] | 0.1395 | 0.139 | +0.3% |
| Center of pressure [m] | 0.2150 | 0.214 | +0.5% |
| Drag coefficient at top speed | 0.281 | 0.282 | -0.5% |
| Rail exit speed [m/s] | 11.92 | 12.03 | -0.9% |
| Maximum speed [m/s] | 40.60 | 40.50 | +0.2% |
| Burnout altitude [m] | 38.51 | 38.93 | -1.1% |
| **Apogee [m]** | **107.5** | **110.2** | **-2.4%** |
| Time to apogee [s] | 5.16 | 5.28 | -2.2% |
| Sideways distance at apogee [m] | 34.0 | 18.5 | +84% |
| Descent rate at landing [m/s] | 8.68 | 8.74 | -0.6% |
| Flight time [s] | 18.19 | 18.40 | -1.1% |

![Altitude, speed and drag coefficient from both simulators](ae100/openrocket.png)

**What agrees.** The inputs to the flight agree to better than 1%: the mass and its balance
point, the center of pressure from Barrowman's equations including the transition, and the drag
coefficient, which stays within 2% of OpenRocket's from liftoff to ejection. OpenRocket splits
its drag into friction, pressure and base drag; the last two match to three decimal places
(0.080 and 0.060). The flight follows: top speed within 0.2%, descent rate within 1%.

**What does not, and why.** The apogee is 2.4% low and the rocket ends up nearly twice as far
upwind. Both come from one simplification. This simulator has no rotation: the rocket points
into the relative wind the instant it leaves the rod. OpenRocket's rocket has inertia, turns
over a fraction of a second while it is still accelerating, and so tilts less. With the wind
set to zero this simulator reaches 112.1 m, so a 2 m/s wind costs it 4.6 m. If OpenRocket's
calm-air apogee is about the same, that wind costs OpenRocket about 2 m. The error grows with
the wind and with a slow rail exit, and 12 m/s is a slow rail exit.

### Against the real flight

The rocket flew with a PerfectFlite Pnut barometric altimeter on a hot, windy day. It reached
212 ft (65 m) where both simulators predict about 110 m, and came down at 12.5 m/s where they
predict 9. The wind was not measured.

```bash
python validation/ae100/flight.py --plot flight.png
```

![Altimeter data against the simulations](ae100/flight.png)

The script asks what would have to be true for the simulator to reproduce the altimeter, at the
real site elevation (436 ft) and temperature (88 F, read from the altimeter):

| | Fitted | Design value |
| --- | ---: | ---: |
| Wind | 8.2 m/s (18 mph) | 2 m/s |
| Ejection delay | 2.6 s | 3 s |
| Parachute drag coefficient | 0.41 | 0.8 |

With an 8.2 m/s wind the simulated climb follows the altimeter within 0.8 m RMS from half a
second after launch to ejection, and predicts a top vertical speed of 29 m/s against the
altimeter's own estimate of 26.5. A wind of that strength is consistent with the day: a rocket
leaving the rod at 12 m/s into an 8 m/s crosswind turns about 35 degrees off vertical at once,
and gravity keeps bending a tilted path further over for the rest of the burn.

How far to trust that number:

- **The climb alone cannot tell wind from weight.** No wind and a rocket 79 g (26%) heavier than
  the design fits the altimeter equally well. The wind figure assumes the rocket weighed what
  the design says. A weight measured on the day would settle it.
- **The instant turn overstates the effect of wind**, as the OpenRocket comparison shows, so the
  real wind was probably somewhat stronger than 8 m/s if wind was the whole story. On the other
  hand a rocket with one caliber of margin at 35 degrees angle of attack is far outside the
  small-angle aerodynamics both simulators assume, and may have lost more than either predicts.
- **The altimeter reads low in hot air.** It converts pressure to altitude with a standard
  atmosphere, and at 31 C a metre of climb changes the pressure 5.6% less. The 65 m it
  recorded was about 69 m. The script converts the simulation to what the altimeter would
  read before comparing.
- **The descent is a separate problem.** Whatever the wind, the rocket came down 40% faster
  than a fully open 30 cm parachute with a drag coefficient of 0.8 allows. Either it was
  heavier than the design or the parachute did not open fully.

The files: `ae100.ork` is the OpenRocket design with its simulation, `config.toml` the same
rocket for this simulator, `Estes_D12.eng` the motor (NAR certification data, from
thrustcurve.org), `altimeter.csv` the flight data with the altimeter's serial number removed.
