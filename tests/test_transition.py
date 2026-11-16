"""A nose cone wider or narrower than the body tube, joined by a conical transition."""

import math

import pytest

from apogee import Airframe, BodyTube, FinSet, NoseCone, Transition, load_config
from apogee.aero import DragBuildup, barrowman, base_drag_coefficient, skin_friction_coefficient
from apogee.config import build_config, read_config

MACH, UNIT_REYNOLDS = 0.3, 7.0e6


def airframe(nose_diameter=None, transition=None, body_diameter=0.05, **fins) -> Airframe:
    fin_set = {"count": 3, "root_chord": 0.10, "tip_chord": 0.05, "span": 0.05, "sweep": 0.05}
    fin_set.update(thickness=0.003, mass=0.03, **fins)
    return Airframe(
        nose=NoseCone(length=0.20, shape="conical", mass=0.05, diameter=nose_diameter),
        body=BodyTube(length=0.60, diameter=body_diameter, mass=0.20),
        fins=FinSet(**fin_set),
        transition=Transition(length=transition, mass=0.01) if transition else None,
    )


def test_a_change_of_diameter_needs_a_transition_and_a_transition_needs_a_change():
    with pytest.raises(ValueError, match="add a transition"):
        airframe(nose_diameter=0.07)
    with pytest.raises(ValueError, match="different from the body"):
        airframe(transition=0.05)
    with pytest.raises(ValueError, match="different from the body"):
        airframe(nose_diameter=0.05, transition=0.05)
    with pytest.raises(ValueError):
        Transition(length=0.0)
    with pytest.raises(ValueError):
        NoseCone(length=0.2, diameter=-0.05)


def test_the_transition_sits_between_the_nose_and_the_body():
    af = airframe(nose_diameter=0.07, transition=0.04)
    assert af.body_start == pytest.approx(0.24)
    assert af.length == pytest.approx(0.84)
    assert af.fin_position == pytest.approx(0.74)  # still flush with the aft end
    assert af.nose_diameter == 0.07
    assert af.diameter == 0.07  # the reference is the widest point
    assert airframe(nose_diameter=0.03, transition=0.04).diameter == 0.05
    assert airframe().body_start == pytest.approx(0.20)


def test_transition_mass_acts_at_the_centroid_of_its_surface():
    plain, stepped = airframe(), airframe(nose_diameter=0.07, transition=0.04)
    assert stepped.mass == pytest.approx(plain.mass + 0.01)
    # Conical shell from radius 0.035 to 0.025 over 0.04 m: centroid at
    # L (r1 + 2 r2) / (3 (r1 + r2)) = 0.04 * 0.085 / 0.18 = 0.018889 m from its front.
    centroid = Transition(0.04).shell_centroid(0.035, 0.025)
    assert centroid == pytest.approx(0.018889, rel=1e-4)
    # Everything aft of the nose moved back by the transition's length.
    moment = plain.mass * plain.cg + (plain.mass - 0.05) * 0.04 + 0.01 * (0.20 + centroid)
    assert stepped.cg == pytest.approx(moment / stepped.mass)
    # A full cone (front radius zero) has its surface centroid at 2/3 of its length.
    assert Transition(0.3).shell_centroid(0.0, 0.1) == pytest.approx(0.2)
    assert Transition(0.3).wetted_area(0.0, 0.1) == pytest.approx(
        math.pi * 0.1 * math.hypot(0.3, 0.1)
    )


def test_barrowman_with_a_boattail_matches_a_hand_calculation():
    # Nose 0.07 m wide (the reference), narrowing over 0.04 m to a 0.05 m body.
    #   nose        CN_a = 2 at 2/3 * 0.20                                  = 0.133333
    #   transition  CN_a = 2 [(0.05/0.07)^2 - 1]                            = -0.979592
    #               x = 0.20 + 0.04/3 [1 + (1 - 1.4)/(1 - 1.4^2)]           = 0.218889
    #   fins        K = 1 + 0.025/0.075 (the body radius)                   = 1.333333
    #               CN_a = K * 4*3*(0.05/0.07)^2 / 2.247219                 = 3.632608
    #               x = 0.74 + 0.041667                                     = 0.781667
    #   x_cp = (2*0.133333 - 0.979592*0.218889 + 3.632608*0.781667) / 4.653016 = 0.621479
    cp = barrowman(airframe(nose_diameter=0.07, transition=0.04))
    assert cp.nose_cn_alpha == pytest.approx(2.0)
    assert cp.transition_cn_alpha == pytest.approx(-0.979592, rel=1e-6)
    assert cp.transition_position == pytest.approx(0.218889, rel=1e-5)
    assert cp.fin_cn_alpha == pytest.approx(3.632608, rel=1e-6)
    assert cp.fin_position == pytest.approx(0.781667, rel=1e-6)
    assert cp.cn_alpha == pytest.approx(4.653016, rel=1e-6)
    assert cp.position == pytest.approx(0.621479, rel=1e-5)


def test_a_narrow_nose_and_shoulder_carry_the_normal_force_of_a_full_width_nose():
    # Slender-body theory: the normal force depends only on the change in cross-section,
    # so a nose that reaches full width in two steps adds up to the same 2 per radian.
    cp = barrowman(airframe(nose_diameter=0.03, transition=0.04))
    assert cp.nose_cn_alpha == pytest.approx(2.0 * 0.36)
    assert cp.transition_cn_alpha == pytest.approx(2.0 * 0.64)
    assert cp.nose_cn_alpha + cp.transition_cn_alpha == pytest.approx(2.0)
    assert cp.transition_position > 0.20 + 0.02  # nearer the wide end than the middle
    # The fins see the same body as on the plain airframe.
    assert cp.fin_cn_alpha == pytest.approx(barrowman(airframe()).fin_cn_alpha)


def test_a_boattail_moves_the_center_of_pressure_aft():
    # It takes normal force away from in front of the CG, which is stabilizing.
    plain = barrowman(airframe(body_diameter=0.07))
    stepped = barrowman(airframe(nose_diameter=0.07, transition=0.04))
    assert stepped.position - 0.04 > plain.position


def test_a_vanishing_step_gives_the_plain_airframe():
    plain = airframe()
    almost = Airframe(
        nose=NoseCone(length=0.20, shape="conical", mass=0.05, diameter=0.05 * (1 + 1e-7)),
        transition=Transition(length=0.10),
        body=BodyTube(length=0.50, diameter=0.05, mass=0.20),
        fins=plain.fins,
    )
    assert barrowman(almost).position == pytest.approx(barrowman(plain).position, rel=1e-6)
    ours, theirs = DragBuildup(almost).breakdown(MACH, UNIT_REYNOLDS), DragBuildup(plain)
    assert ours.total == pytest.approx(theirs.at(MACH, UNIT_REYNOLDS), rel=1e-5)


def test_drag_of_a_boattail_matches_a_hand_calculation():
    af = airframe(nose_diameter=0.07, transition=0.04)
    parts = DragBuildup(af).breakdown(MACH, UNIT_REYNOLDS)
    plain = DragBuildup(airframe(body_diameter=0.07)).breakdown(MACH, UNIT_REYNOLDS)
    base = base_drag_coefficient(MACH)
    share = (0.05 / 0.07) ** 2
    # Only the body tube's end is a blunt base: (0.05/0.07)^2 of the reference area.
    assert parts.base == pytest.approx(base * share)
    # The taper is 0.04 m over a 0.02 m change in diameter, a ratio of 2, so half of the
    # base drag on the area it closes off counts: (3 - 2) / 2.
    assert parts.transition_pressure == pytest.approx(base * (1 - share) * 0.5)
    assert plain.transition_pressure == 0.0
    # Friction acts on the nose, the transition and the narrower tube.
    reference = math.pi * 0.07**2 / 4
    wetted = (
        math.pi * 0.035 * math.hypot(0.20, 0.035)
        + math.pi * (0.035 + 0.025) * math.hypot(0.04, 0.01)
        + math.pi * 0.05 * 0.60
    )
    cf = skin_friction_coefficient(UNIT_REYNOLDS * 0.84, 60e-6 / 0.84, MACH)
    assert parts.body_friction == pytest.approx(
        cf * (1 + 1 / (2 * 0.84 / 0.07)) * wetted / reference, rel=1e-9
    )


@pytest.mark.parametrize(
    ("length", "factor"), [(0.01, 1.0), (0.02, 1.0), (0.04, 0.5), (0.06, 0.0), (0.2, 0.0)]
)
def test_a_long_boattail_has_no_pressure_drag_and_a_short_one_all_of_it(length, factor):
    parts = DragBuildup(airframe(nose_diameter=0.07, transition=length)).breakdown(MACH, 7e6)
    closed_off = 1 - (0.05 / 0.07) ** 2
    assert parts.transition_pressure == pytest.approx(
        base_drag_coefficient(MACH) * closed_off * factor, abs=1e-12
    )


def test_a_shoulder_has_the_pressure_drag_of_a_cone_on_the_area_it_adds():
    parts = DragBuildup(airframe(nose_diameter=0.03, transition=0.04)).breakdown(MACH, 7e6)
    half_angle = math.atan2(0.01, 0.04)
    assert parts.transition_pressure == pytest.approx(0.8 * math.sin(half_angle) ** 2 * 0.64)
    # The conical nose's own pressure drag acts on its smaller frontal area.
    nose_angle = math.atan2(0.015, 0.20)
    assert parts.nose_pressure == pytest.approx(0.8 * math.sin(nose_angle) ** 2 * 0.36)
    assert parts.base == pytest.approx(base_drag_coefficient(MACH))
    assert parts.total == pytest.approx(
        parts.body_friction
        + parts.fin_friction
        + parts.nose_pressure
        + parts.base
        + parts.fin_pressure
        + parts.transition_pressure
    )


def test_each_part_can_have_its_own_roughness():
    reynolds = 4.0e7  # fast enough that the roughness sets the friction
    default = DragBuildup(airframe(), roughness=60e-6).breakdown(MACH, reynolds)
    same = airframe()
    same.nose.roughness = same.body.roughness = same.fins.roughness = 60e-6
    assert DragBuildup(same, roughness=5e-6).breakdown(MACH, reynolds) == default

    rough_fins = airframe()
    rough_fins.fins.roughness = 500e-6
    parts = DragBuildup(rough_fins, roughness=60e-6).breakdown(MACH, reynolds)
    assert parts.fin_friction > 1.3 * default.fin_friction
    assert parts.body_friction == default.body_friction

    smooth_body = airframe()
    smooth_body.body.roughness = 0.0
    parts = DragBuildup(smooth_body, roughness=60e-6).breakdown(MACH, reynolds)
    assert parts.body_friction < default.body_friction
    assert parts.fin_friction == default.fin_friction

    rough_fins.fins.roughness = -1e-6
    with pytest.raises(ValueError, match="roughness"):
        DragBuildup(rough_fins)


def test_config_reads_the_transition_and_part_roughness(example_config, tmp_path):
    data = read_config(example_config)
    data["motor"]["file"] = str(example_config.parent / data["motor"]["file"])
    data["rocket"].pop("cd", None)
    data["nose"]["diameter_m"] = 0.08
    data["nose"]["roughness_m"] = 150e-6
    data["transition"] = {"length_m": 0.05, "mass_kg": 0.02, "roughness_m": 20e-6}
    data["fins"].setdefault("thickness_m", 0.003)
    data["drag"] = {"model": "geometry"}
    rocket = build_config(data, tmp_path).rocket
    plain = load_config(example_config).rocket
    assert rocket.airframe.transition.length == 0.05
    assert rocket.airframe.transition.roughness == 20e-6
    assert rocket.airframe.nose.roughness == 150e-6
    assert rocket.diameter == 0.08
    assert rocket.airframe.length == pytest.approx(plain.airframe.length + 0.05)
    assert rocket.dry_mass == pytest.approx(plain.dry_mass + 0.02)
    assert rocket.center_of_pressure.transition_cn_alpha < 0

    data["nose"].pop("diameter_m")
    with pytest.raises(ValueError, match="different from the body"):
        build_config(data, tmp_path)


def test_config_with_only_a_transition_is_rejected(example_config, tmp_path):
    data = read_config(example_config)
    data["motor"]["file"] = str(example_config.parent / data["motor"]["file"])
    for section in ("nose", "body", "fins"):
        del data[section]
    data["transition"] = {"length_m": 0.05}
    with pytest.raises(ValueError, match="needs all of"):
        build_config(data, tmp_path)
