"""Tests for the Calibrate step's camera solver and the offside tolerance."""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from var_offside.calibration import pitch_solver as ps
from var_offside.calibration.base import CalibrationResult
from var_offside.detection import PlayerDetection
from var_offside.offside.decision import OffsideDecision
from var_offside.offside.line_calculator import OffsideLine

W, H = 1920, 1080


def _camera(cx, cy, height, target, focal) -> ps.Camera:
    position = np.array([cx, cy, -height], dtype=np.float64)
    target_w = np.array([target[0], target[1], 0.0])
    return ps.Camera(float(focal), ps._look_at(position, target_w), position, W / 2.0, H / 2.0)


TRUE_CAMERAS = {
    "medium": _camera(8, 75, 22, (32, 2), 1900),
    "tight": _camera(8, 80, 28, (35, 2), 3600),
    "low": _camera(25, 58, 11, (35, 5), 1250),
}


def _px(camera: ps.Camera, field: Tuple[float, float]) -> np.ndarray:
    world = ps._field_to_world(np.array([field], dtype=np.float64))
    h = ps._project_h(camera.params(), camera, world)
    return h[0, :2] / h[0, 2]


def _point_pins(camera: ps.Camera, ids: List[str], noise=0.0, rng=None) -> List[ps.Pin]:
    pins = []
    for lid in ids:
        mark = ps._LANDMARK_BY_ID[lid]
        u, v = _px(camera, (mark.x, mark.y))
        if noise:
            u += rng.normal(0, noise)
            v += rng.normal(0, noise)
        pins.append(ps.Pin("point", lid, float(u), float(v)))
    return pins


def _line_pin(camera: ps.Camera, line_id: str, t: float) -> ps.Pin:
    line = ps._LINE_BY_ID[line_id]
    field = (line.a[0] + t * (line.b[0] - line.a[0]), line.a[1] + t * (line.b[1] - line.a[1]))
    u, v = _px(camera, field)
    return ps.Pin("line", line_id, float(u), float(v))


def _field_error(solved: ps.Camera, truth: ps.Camera) -> float:
    """Median field-x error at defender-like ground positions."""
    errors = []
    for field in ((84, 22), (84, 34), (84, 46), (91, 28), (91, 42)):
        pixel = _px(truth, field).reshape(1, 2)
        got = ps._pixels_to_field_x(solved.params(), solved, pixel)[0]
        errors.append(abs(got - field[0]))
    return float(np.median(errors))


SPREAD = ["r_box_front_far", "r_box_front_near", "r_six_front_far", "r_six_front_near"]
BUNCHED = ["r_box_front_far", "r_six_front_far", "r_box_goal_far", "r_six_goal_far"]


# ---------------------------------------------------------------------------
# From scratch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRUE_CAMERAS)
def test_scratch_recovers_a_known_camera(name):
    truth = TRUE_CAMERAS[name]
    solution = ps.solve(_point_pins(truth, SPREAD + ["r_spot"]), W, H)
    assert solution.camera is not None
    assert _field_error(solution.camera, truth) < 0.02
    assert solution.camera.focal == pytest.approx(truth.focal, rel=0.02)
    assert solution.fit.tone == "ok" and solution.fit.can_continue


def test_scratch_with_noise_stays_within_the_simulated_error():
    rng = np.random.default_rng(3)
    truth = TRUE_CAMERAS["medium"]
    errors = []
    for _ in range(12):
        pins = _point_pins(truth, SPREAD + ["r_box_goal_far", "r_box_goal_near"], 2.0, rng)
        errors.append(_field_error(ps.solve(pins, W, H).camera, truth))
    assert np.median(errors) < 0.2


def test_too_few_pins_asks_for_more():
    truth = TRUE_CAMERAS["medium"]
    solution = ps.solve(_point_pins(truth, SPREAD[:2]), W, H)
    assert solution.camera is None
    assert solution.fit.needed == 2
    assert solution.fit.label == "PLACE 2 MORE"
    assert not solution.fit.can_continue


def test_minimum_pins_ask_for_a_check():
    truth = TRUE_CAMERAS["medium"]
    solution = ps.solve(_point_pins(truth, SPREAD), W, H)
    assert solution.camera is not None
    assert solution.fit.label == "CHECK IT"
    assert solution.fit.can_continue


def test_bunched_points_are_refused():
    rng = np.random.default_rng(5)
    truth = TRUE_CAMERAS["medium"]
    refused = 0
    for _ in range(6):
        solution = ps.solve(_point_pins(truth, BUNCHED, 2.0, rng), W, H)
        refused += not solution.fit.can_continue
    assert refused == 6


def test_spread_points_score_better_than_bunched():
    truth = TRUE_CAMERAS["medium"]
    spread = ps.estimate_line_error(truth, _point_pins(truth, SPREAD), W, H)
    bunched = ps.estimate_line_error(truth, _point_pins(truth, BUNCHED), W, H)
    assert spread < ps.LINE_ERROR_WEAK_M
    assert bunched > ps.LINE_ERROR_BAD_M


def test_a_bad_point_is_named():
    truth = TRUE_CAMERAS["medium"]
    ids = SPREAD + ["r_spot", "r_box_goal_far", "r_box_goal_near"]
    pins = _point_pins(truth, ids)
    pins[4] = ps.Pin("point", "r_spot", pins[4].x + 60.0, pins[4].y - 35.0)
    fit = ps.solve(pins, W, H).fit
    assert fit.label == "POINTS DISAGREE"
    assert not fit.can_continue
    assert fit.worst == 4
    assert "penalty spot" in fit.message


def test_points_and_lines_solve_with_few_corners():
    truth = TRUE_CAMERAS["medium"]
    pins = _point_pins(truth, ["r_box_front_far", "r_box_front_near", "r_spot"])
    pins += [_line_pin(truth, "far_touchline", 0.7), _line_pin(truth, "r_box_far", 0.4)]
    solution = ps.solve(pins, W, H)
    assert solution.camera is not None
    assert _field_error(solution.camera, truth) < 0.05


def test_mostly_lines_solve_from_viewpoint_search():
    truth = TRUE_CAMERAS["medium"]
    pins = _point_pins(truth, ["r_box_front_far", "r_box_front_near"])
    for line, t in (
        ("r_box_far", 0.3),
        ("r_box_near", 0.6),
        ("far_touchline", 0.8),
        ("r_six_front", 0.2),
        ("r_six_front", 0.8),
    ):
        pins.append(_line_pin(truth, line, t))
    solution = ps.solve(pins, W, H)
    assert solution.camera is not None
    assert _field_error(solution.camera, truth) < 0.1


def test_warm_start_is_quick_and_stays_put():
    truth = TRUE_CAMERAS["medium"]
    pins = _point_pins(truth, SPREAD + ["r_spot", "r_box_goal_far"])
    first = ps.solve(pins, W, H).camera
    began = time.perf_counter()
    for _ in range(10):
        again = ps.solve(pins, W, H, start=first)
    per_solve = (time.perf_counter() - began) / 10
    assert again.camera.focal == pytest.approx(first.focal, rel=1e-3)
    assert per_solve < 0.05


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------


def _shifted(truth: ps.Camera) -> ps.Camera:
    """A slightly wrong automatic camera."""
    values = truth.params().copy()
    values[0] += 0.03
    values[1:4] += 0.004
    values[4:7] += np.array([0.7, -0.6, 0.4])
    return truth.with_params(values)


def test_edit_with_no_pins_keeps_the_base():
    base = _shifted(TRUE_CAMERAS["medium"])
    solution = ps.solve([], W, H, base=base)
    assert solution.camera is base
    assert solution.fit.can_continue


def test_edit_one_point_moves_it_there_and_barely_disturbs_the_rest():
    truth = TRUE_CAMERAS["medium"]
    base = _shifted(truth)
    mark = ps._LANDMARK_BY_ID["r_box_front_far"]
    goal = _px(truth, (mark.x, mark.y))
    solution = ps.solve([ps.Pin("point", mark.id, *map(float, goal))], W, H, base=base)
    assert np.linalg.norm(_px(solution.camera, (mark.x, mark.y)) - goal) < 2.0
    assert solution.fit.can_continue


def test_edit_many_points_reaches_the_truth():
    truth = TRUE_CAMERAS["medium"]
    base = _shifted(truth)
    before = _field_error(base, truth)
    ids = SPREAD + ["r_spot", "r_box_goal_far", "r_box_goal_near"]
    solution = ps.solve(_point_pins(truth, ids), W, H, base=base)
    after = _field_error(solution.camera, truth)
    assert after < 0.05 < before


def test_edit_a_line_pin_moves_that_line():
    truth = TRUE_CAMERAS["medium"]
    base = _shifted(truth)
    pin = _line_pin(truth, "r_box_front", 0.5)
    solution = ps.solve([pin], W, H, base=base)
    problem = ps._build_problem([pin])
    assert abs(ps._pin_residuals(solution.camera.params(), solution.camera, problem)[0]) < 2.0


def test_wrong_end_warns():
    truth = TRUE_CAMERAS["medium"]
    pins = _point_pins(truth, SPREAD + ["r_spot", "r_box_goal_far"])
    assert ps.solve(pins, W, H, attacking_side="left").fit.label == "CHECK THE END"
    assert ps.solve(pins, W, H, attacking_side="right").fit.label == "GOOD FIT"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_result_matches_the_pnlcalib_shape():
    truth = TRUE_CAMERAS["medium"]
    pins = _point_pins(truth, SPREAD + ["r_spot"])
    camera = ps.solve(pins, W, H).camera
    result = ps.to_calibration_result(camera, pins, W, H, "hand")

    assert isinstance(result, CalibrationResult)
    assert result.is_valid and result.has_3d_calibration
    assert result.method == "manual"
    assert result.debug_info["source"] == "hand"

    # Image to field through both code paths, and back.
    pixel = _px(truth, (90.0, 30.0))
    field = result.project_to_field(tuple(pixel))
    assert field == pytest.approx((90.0, 30.0), abs=0.05)
    back = result.project_to_image((90.0, 30.0))
    assert np.array(back) == pytest.approx(pixel, abs=1.0)

    # P must agree with the same point in PnLCalib's centred world.
    h = result.projection_matrix @ np.array([90.0 - 52.5, 30.0 - 34.0, 0.0, 1.0])
    assert h[:2] / h[2] == pytest.approx(pixel, abs=1.0)


def test_camera_round_trips_from_a_calibration_and_json():
    truth = TRUE_CAMERAS["tight"]
    result = ps.to_calibration_result(truth, [], W, H, "hand")
    camera = ps.Camera.from_calibration(result)
    assert camera.focal == pytest.approx(truth.focal)
    assert camera.position == pytest.approx(truth.position)
    again = ps.Camera.from_dict(camera.to_dict())
    assert again.rvec == pytest.approx(camera.rvec)
    assert ps.Camera.from_dict({"focal": "x"}) is None


def test_parse_pins_rejects_bad_input():
    assert ps.parse_pins([{"kind": "point", "ref": "c_spot", "x": 1, "y": 2}])[0].x == 1.0
    for bad in (
        {"kind": "point", "ref": "nope", "x": 1, "y": 2},
        {"kind": "line", "ref": "nope", "x": 1, "y": 2},
        {"kind": "blob", "ref": "c_spot", "x": 1, "y": 2},
        {"kind": "point", "ref": "c_spot", "x": "a", "y": 2},
    ):
        with pytest.raises(ValueError):
            ps.parse_pins([bad])


def test_pitch_model_lists_both_ends():
    model = ps.pitch_model()
    ids = {row["id"] for row in model["landmarks"]}
    assert {"r_spot", "l_spot", "c_spot"} <= ids
    assert {"r_goal", "l_goal", "halfway"} <= {row["id"] for row in model["lines"]}
    spot = next(row for row in model["landmarks"] if row["id"] == "r_spot")
    assert spot["x"] == pytest.approx(94.0)


# ---------------------------------------------------------------------------
# Offside tolerance
# ---------------------------------------------------------------------------


class _Calibration(CalibrationResult):
    """Maps a player's foot x directly to field metres."""

    def project_to_field(self, point):
        return (float(point[0]) / 10.0, 30.0)


def _decide(margin_m, attacking_side="right", calibrated=True, tolerance=None):
    # In the attacker's opponent half, whichever way they attack.
    line_px = 800.0 if attacking_side == "right" or not calibrated else 200.0
    line = OffsideLine(
        field_x=line_px / 10.0 if calibrated else None,
        image_x=line_px,
        reference_point=(line_px, 100.0),
        defining_player_idx=1,
        defining_body_part=None,
        ignored_player_idx=None,
        confidence=1.0,
    )
    sign = 1.0 if attacking_side == "right" else -1.0
    # The attacker's ground point is the box edge nearest the goal.
    foot = line_px + sign * margin_m * (10.0 if calibrated else 1.0)
    bbox = (foot - 40.0, 10.0, foot, 100.0) if attacking_side == "right" else (foot, 10.0, foot + 40.0, 100.0)
    players = [
        PlayerDetection(bbox=bbox, confidence=1.0, team_id=1),
        PlayerDetection(bbox=(line_px - 10.0, 10.0, line_px + 10.0, 100.0), confidence=1.0, team_id=2),
    ]
    calibration = None
    if calibrated:
        calibration = _Calibration(homography=np.eye(3), homography_inv=np.eye(3), method="pnl")
    decider = OffsideDecision() if tolerance is None else OffsideDecision(margin_tolerance_m=tolerance)
    return decider.evaluate(
        players,
        line,
        attacking_side=attacking_side,
        calibration=calibration,
        attacking_team=1,
        defending_team=2,
    )


@pytest.mark.parametrize("side", ["right", "left"])
def test_any_margin_past_the_line_is_offside_on_a_calibrated_frame(side):
    assert _decide(2.5, side).any_offside
    assert _decide(0.05, side).any_offside
    assert not _decide(-0.05, side).any_offside


def test_metre_tolerance_can_be_set():
    assert not _decide(0.05, tolerance=0.1).any_offside
    assert _decide(0.2, tolerance=0.1).any_offside


def test_pixel_tolerance_still_applies_without_calibration():
    assert not _decide(2.0, calibrated=False).any_offside
    assert _decide(4.0, calibrated=False).any_offside
