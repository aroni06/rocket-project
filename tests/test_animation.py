"""Flight animation: frame timing, the phase readout and the saved GIF."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image

from apogee import Parachute, Rocket, load_config, simulate
from apogee.animation import _readout, animate_flight, frame_times, phase_at
from apogee.cli import main


@pytest.fixture(scope="module")
def flight(request):
    root = request.config.rootpath
    cfg = load_config(root / "configs" / "example_dual_deploy.toml")
    return simulate(cfg.rocket, cfg.environment, cfg.rail, max_step=1.0)


def test_frames_run_from_liftoff_to_landing_and_slow_the_ascent(flight):
    times = frame_times(flight, 101)
    assert times[0] == pytest.approx(flight.time[0])
    assert times[-1] == pytest.approx(flight.events["landing"].time)
    assert np.all(np.diff(times) > 0)
    # The climb is an eighth of the flight time but about half of the distance flown,
    # so it gets well over an eighth of the frames without taking over the clip.
    share = np.mean(times <= flight.events["apogee"].time)
    assert flight.events["apogee"].time / flight.events["landing"].time < 0.15
    assert 0.25 < share < 0.45


def test_phase_readout_follows_the_flight(flight):
    events = flight.events
    assert phase_at(flight, 0.05) == "On the rail"
    assert phase_at(flight, events["burnout"].time - 0.1) == "Powered ascent"
    assert phase_at(flight, events["burnout"].time + 1.0) == "Coasting"
    assert phase_at(flight, events["apogee"].time + 0.5) == "Falling"
    assert phase_at(flight, events["drogue_deploy"].time + 1.0) == "Under drogue"
    assert phase_at(flight, events["main_deploy"].time + 1.0) == "Under main"
    assert phase_at(flight, events["landing"].time) == "Landed"


def test_readout_never_shows_negative_zero():
    assert _readout("{:.0f} m", -0.004) == "0 m"
    assert _readout("{:+.1f} g", -0.04) == "+0.0 g"
    assert _readout("{:+.0f} m/s", -14.2) == "-14 m/s"


def test_animation_is_saved_as_a_gif_that_holds_its_last_frame(flight, tmp_path):
    out = animate_flight(flight, tmp_path / "flight.gif", duration=1.0, hold=1.0, fps=6, dpi=50)
    with Image.open(out) as gif:
        assert gif.format == "GIF"
        assert gif.size == (450, 260)
        durations = []
        for k in range(gif.n_frames):
            gif.seek(k)
            durations.append(gif.info["duration"])
    # 6 moving frames, then the last one held: about two seconds in all.
    assert 6 <= len(durations) <= 12
    assert sum(durations) == pytest.approx(2000, abs=200)


def test_animation_works_without_parachutes_or_a_motor(tmp_path):
    rocket = Rocket("lawn dart", dry_mass=1.0, diameter=0.05, motor=None)
    result = simulate(rocket, initial_speed=60.0)
    out = animate_flight(result, tmp_path / "dart.gif", duration=0.5, hold=0.0, fps=6, dpi=40)
    assert out.stat().st_size > 0
    assert phase_at(result, result.events["apogee"].time + 1.0) == "Falling"

    rocket.parachutes.append(Parachute("main", 0.9))
    result = simulate(rocket, initial_speed=60.0)
    assert phase_at(result, result.events["landing"].time - 1.0) == "Under main"


def test_animation_rejects_a_clip_with_no_length(flight, tmp_path):
    with pytest.raises(ValueError):
        animate_flight(flight, tmp_path / "x.gif", duration=0.0)


def test_cli_writes_an_animation(tmp_path, capsys, request, monkeypatch):
    import apogee.animation as animation

    # The command line uses the full-length clip; shorten it so the test stays quick.
    full = animation.animate_flight
    monkeypatch.setattr(
        animation,
        "animate_flight",
        lambda result, path: full(result, path, duration=0.5, hold=0.0, fps=6, dpi=40),
    )
    config = request.config.rootpath / "configs" / "example_l1.toml"
    gif = tmp_path / "flight.gif"
    assert main(["simulate", str(config), "--animation", str(gif)]) == 0
    assert gif.stat().st_size > 0
    assert f"Wrote {gif}" in capsys.readouterr().out
