"""Regression tests for the reviewed offside rules.

The production implementations are real.  Only model inference, pipeline drawing,
and session infrastructure are replaced with lightweight test doubles.
"""

from __future__ import annotations

import types
from typing import Any, Dict, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from var_offside.calibration.base import CalibrationResult
from var_offside.config import get_settings
from var_offside.detection import GoalkeeperFilter, PlayerDetection, PlayerDetector
from var_offside.offside.decision import OffsideDecision
from var_offside.offside.line_calculator import (
    OffsideLine,
    OffsideLineCalculator,
    get_directional_ground_point,
)
from var_offside.offside.side_detector import SideDetector
from var_offside.pipeline import AnalysisResult
from var_offside.teams import TeamAssignmentResult
from var_offside.web import jobs, phases, sessions
from var_offside.web.calibration_stage import CalibrationState, auto_fit
from var_offside.web.server import app


# ---------------------------------------------------------------------------
# Shared fixtures and builders
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    store = sessions.SessionStore()
    monkeypatch.setattr(sessions, "store", store)
    monkeypatch.setattr(jobs, "manager", jobs.JobManager())
    monkeypatch.setattr(get_settings(), "output_dir", tmp_path, raising=False)
    monkeypatch.setattr(phases, "get_pipeline", lambda: object())
    yield
    store.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _person(
    x: float,
    team: Optional[int],
    *,
    side: str = "right",
    y: float = 400.0,
    keeper: bool = False,
    keypoints: Optional[np.ndarray] = None,
    width: float = 20.0,
) -> PlayerDetection:
    """Make a box whose direction-aligned ground point is exactly ``(x, y)``."""
    bbox = (x, 300.0, x + width, y) if side == "left" else (x - width, 300.0, x, y)
    return PlayerDetection(
        bbox=bbox,
        confidence=0.9,
        team_id=team,
        is_goalkeeper=keeper,
        keypoints=keypoints,
    )


def _pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.line_calculator = OffsideLineCalculator()
    pipeline.offside_decision = OffsideDecision()
    pipeline.side_detector = SideDetector()
    pipeline.goalkeeper_filter = GoalkeeperFilter()
    pipeline._draw_detections_debug.side_effect = (
        lambda image, detections, title="": image.copy()
    )
    pipeline._draw_offside_line_debug.side_effect = (
        lambda image, line, detections, calibration=None: image.copy()
    )
    pipeline.field_overlay.draw_calibration_result.side_effect = (
        lambda image, calibration: image.copy()
    )
    pipeline.field_overlay.draw_projected_field.side_effect = (
        lambda image, calibration: image.copy()
    )
    pipeline.visualizer.create_clean_result_view.side_effect = (
        lambda image, offside_result, calibration: image.copy()
    )
    return pipeline


def _analysis(
    detections: Sequence[PlayerDetection],
    *,
    width: int = 2000,
    height: int = 500,
    attacking_team: Optional[int] = 1,
    keeper_choice: Optional[str] = "not_in_frame",
    calibration: Optional[CalibrationResult] = None,
) -> AnalysisResult:
    result = AnalysisResult()
    result.image = np.zeros((height, width, 3), dtype=np.uint8)
    result.detections = list(detections)
    result.attacking_team = attacking_team
    result.defending_team = (
        1 if attacking_team == 2 else 2 if attacking_team == 1 else None
    )
    # Dataclass instances intentionally accept the forthcoming optional state.
    result.keeper_choice = keeper_choice
    result.calibration = calibration or CalibrationResult(method="auto_failed")
    return result


def _phase2(
    pipeline: MagicMock, result: AnalysisResult, side: str = "right"
) -> Dict[str, Any]:
    result, stages = phases.run_phase2(pipeline, result, side)
    return phases.serialize_verdict(result, list(stages))


def _manual_line(
    defender: PlayerDetection,
    defender_idx: int,
    *,
    side: str = "right",
    field_x: Optional[float] = None,
) -> OffsideLine:
    reference = (
        (defender.bbox[0] if side == "left" else defender.bbox[2]),
        defender.bbox[3],
    )
    return OffsideLine(
        field_x=field_x,
        image_x=float(reference[0]),
        reference_point=(float(reference[0]), float(reference[1])),
        defining_player_idx=defender_idx,
        defining_body_part=None,
    )


def _field_calibration(mirrored: bool = False, *, full_3d: bool = False) -> CalibrationResult:
    homography = (
        np.array([[-0.1, 0.0, 105.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
        if mirrored
        else np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])
    )
    return CalibrationResult(
        homography=homography,
        homography_inv=np.linalg.inv(homography),
        projection_matrix=np.eye(4) if full_3d else None,
        method="pnl",
        quality_score=0.9,
    )


def _rejected_camera() -> CalibrationResult:
    k = np.array([[20.0, 0.0, 640.0], [0.0, 20.0, 360.0], [0.0, 0.0, 1.0]])
    r = np.eye(3)
    position = np.array([0.0, -50.0, -5.0])
    return CalibrationResult(
        homography=np.eye(3),
        homography_inv=np.eye(3),
        camera_matrix=k,
        rotation_matrix=r,
        position=position,
        projection_matrix=k @ np.hstack([r, (-r @ position).reshape(3, 1)]),
        image_size=(720, 1280),
        method="pnl",
        quality_score=0.95,
    )


def _believable_camera(reverse_angle: bool = False) -> CalibrationResult:
    """A real broadcast camera on one touchline, or opposite it (pitch x runs leftward)."""
    tilt = np.radians(15.0)
    c, s = np.cos(tilt), np.sin(tilt)
    if reverse_angle:
        r = np.array([[-1.0, 0.0, 0.0], [0.0, -s, c], [0.0, c, s]])
        position = np.array([0.0, -60.0, -20.0])
    else:
        r = np.array([[1.0, 0.0, 0.0], [0.0, s, c], [0.0, -c, s]])
        position = np.array([0.0, 60.0, -20.0])
    k = np.array([[1500.0, 0.0, 640.0], [0.0, 1500.0, 360.0], [0.0, 0.0, 1.0]])
    return CalibrationResult(
        homography=np.eye(3),
        homography_inv=np.eye(3),
        camera_matrix=k,
        rotation_matrix=r,
        position=position,
        projection_matrix=k @ np.hstack([r, (-r @ position).reshape(3, 1)]),
        image_size=(720, 1280),
        method="pnl",
        quality_score=0.95,
    )


def _body_keypoints() -> np.ndarray:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    for index, x, y in (
        (0, 680.0, 320.0),
        (5, 685.0, 335.0),
        (6, 690.0, 335.0),
        (11, 684.0, 355.0),
        (12, 688.0, 355.0),
        (13, 682.0, 375.0),
        (14, 686.0, 375.0),
        (15, 680.0, 395.0),
        (16, 690.0, 395.0),
    ):
        keypoints[index] = (x, y, 0.95)
    # An outstretched wrist explains why the box extends to x=710.
    keypoints[9] = (710.0, 340.0, 0.99)
    return keypoints


# ---------------------------------------------------------------------------
# Review and Analyze choices
# ---------------------------------------------------------------------------


def test_bug01_phase1_leaves_keeper_team_and_side_choices_to_the_user():
    pipeline = _pipeline()
    detections = [
        _person(60, 1),
        _person(200, 1),
        _person(500, 2),
        _person(600, 2),
    ]
    pipeline.player_detector.detect.return_value = detections

    def assign_teams(image, players):
        for player, team in zip(players, (1, 1, 2, 2)):
            player.team_id = team
        return TeamAssignmentResult(
            detections=players,
            embeddings=np.zeros((len(players), 8), dtype=np.float32),
            clustering=None,
            team_colors={1: (255, 0, 0), 2: (0, 0, 255)},
        )

    pipeline.team_assigner.assign_teams.side_effect = assign_teams
    image = np.zeros((500, 1000, 3), dtype=np.uint8)

    result, stages = phases.run_phase1(pipeline, image)
    marked = sum(det.is_goalkeeper for det in result.detections)
    assert marked == 0, f"phase 1 auto-marked {marked} goalkeeper(s)"
    assert result.attacking_team is None
    assert result.defending_team is None
    assert result.side_result is None

    payload = phases.serialize_phase1(result, list(stages))
    assert "suggested_side" not in payload
    assert "side_reason" not in payload
    side_check = next(check for check in payload["checks"] if check["key"] == "side")
    assert (side_check["status"], side_check["detail"]) == (
        "check",
        "choose left or right",
    )


def test_bug02_analyze_never_restores_user_removed_keepers():
    detections = [_person(x, 1) for x in range(60, 421, 40)]
    detections += [_person(x, 2) for x in range(580, 941, 40)]
    assert len(detections) == 20
    result = _analysis(detections, width=1000, keeper_choice="not_in_frame")

    phases.run_phase2(_pipeline(), result, "right")

    marked = sum(det.is_goalkeeper for det in result.detections)
    assert marked == 0, f"Analyze restored {marked} goalkeeper role(s)"


def test_bug03_unmarked_real_keeper_requires_a_choice_before_a_call():
    pipeline = _pipeline()
    people = [_person(800, 1), _person(700, 2), _person(1000, 2)]
    result = _analysis(people, keeper_choice=None)

    payload = _phase2(pipeline, result)
    assert payload["verdict"] == "unknown", (
        "an unmarked real keeper was allowed to set a firm onside line"
    )
    assert payload.get("message")

    marked = _analysis(
        [_person(800, 1), _person(700, 2), _person(1000, 2, keeper=True)],
        keeper_choice="player",
    )
    phases.run_phase2(pipeline, marked, "right")
    marked_payload = phases.serialize_verdict(marked)
    assert marked.offside_line is not None
    assert marked.offside_line.image_x == pytest.approx(700.0)
    assert marked_payload["verdict"] == "offside"


def _png_bytes(width: int = 2000, height: int = 500) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.zeros((height, width, 3), dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def _assert_unknown_payload(
    payload: Dict[str, Any],
    *,
    exact_message: Optional[str] = None,
    message_parts: Sequence[str] = (),
) -> None:
    assert payload["verdict"] == "unknown", (
        f"expected an unknown verdict, got {payload['verdict']!r}"
    )
    message = payload.get("message")
    assert isinstance(message, str) and message
    if exact_message is not None:
        assert message == exact_message
    normalized = message.replace("_", " ").lower()
    for part in message_parts:
        assert part in normalized
    assert payload.get("estimate") is False
    assert payload["margin_m"] is None
    assert payload["line_field_x"] is None
    assert payload["line_image_points"] == []
    assert payload["last_defender"] == {"id": None, "part": None}
    assert payload["offside_player_ids"] == []
    assert payload["view3d_available"] is False
    assert not {"calibration", "offside", "birdseye", "result"}.intersection(
        payload["stages"]
    ), "an unknown answer retained stages from the previous firm verdict"


def test_bug03_api_unknown_choices_clear_stale_verdict_and_explain_why(
    client, monkeypatch
):
    pipeline = _pipeline()

    def fake_phase1(_pipeline, image):
        result = _analysis(
            [_person(800, 1), _person(700, 2), _person(650, 2)],
            width=image.shape[1],
            height=image.shape[0],
            keeper_choice=None,
        )
        result.image = image
        result.team_result = TeamAssignmentResult(
            detections=result.detections,
            embeddings=np.zeros((3, 8), dtype=np.float32),
            clustering=None,
            team_colors={1: (255, 0, 0), 2: (0, 0, 255)},
        )
        frame = image.copy()
        return result, {"original": frame, "detection": frame, "teams": frame}

    monkeypatch.setattr(phases, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(phases, "run_phase1", fake_phase1)
    monkeypatch.setattr(
        phases,
        "run_calibration",
        lambda _pipeline, _image: CalibrationResult(method="auto_failed"),
    )

    uploaded = client.post(
        "/api/sessions", files={"file": ("frame.png", _png_bytes(), "image/png")}
    )
    assert uploaded.status_code == 200, uploaded.text
    session_id = uploaded.json()["id"]
    detected = client.post(f"/api/sessions/{session_id}/detect")
    assert detected.status_code == 200, detected.text
    players = detected.json()["players"]

    # Exercise the real edit path before the first Analyze request.
    edited = client.put(
        f"/api/sessions/{session_id}/players", json={"players": players}
    )
    assert edited.status_code == 200, edited.text

    firm = client.post(
        f"/api/sessions/{session_id}/analyze",
        json={
            "attacking_side": "right",
            "attacking_team": 1,
            "keeper": "not_in_frame",
            "exclude_goalkeeper": True,
        },
    )
    assert firm.status_code == 200, firm.text
    assert firm.json()["verdict"] == "offside"
    assert "result" in firm.json()["stages"]

    missing_team = client.post(
        f"/api/sessions/{session_id}/analyze",
        json={"attacking_side": "right", "keeper": "not_in_frame"},
    )
    assert missing_team.status_code == 200, missing_team.text
    _assert_unknown_payload(
        missing_team.json(), exact_message="Choose the attacking team."
    )

    invalid_team = client.post(
        f"/api/sessions/{session_id}/analyze",
        json={
            "attacking_side": "right",
            "attacking_team": 3,
            "keeper": "not_in_frame",
        },
    )
    _assert_unknown_payload(
        invalid_team.json(), exact_message="Choose the attacking team."
    )

    for keeper in (None, "automatic"):
        body = {"attacking_side": "right", "attacking_team": 1}
        if keeper is not None:
            body["keeper"] = keeper
        response = client.post(f"/api/sessions/{session_id}/analyze", json=body)
        _assert_unknown_payload(
            response.json(), message_parts=("choose", "keeper", "not in frame")
        )

    no_marked_keeper = client.post(
        f"/api/sessions/{session_id}/analyze",
        json={"attacking_side": "right", "attacking_team": 1, "keeper": "player"},
    )
    _assert_unknown_payload(no_marked_keeper.json(), message_parts=("one", "keeper"))

    players[1]["role"] = "goalkeeper"
    assert client.put(
        f"/api/sessions/{session_id}/players", json={"players": players}
    ).status_code == 200
    conflict = client.post(
        f"/api/sessions/{session_id}/analyze",
        json={
            "attacking_side": "right",
            "attacking_team": 1,
            "keeper": "not_in_frame",
        },
    )
    _assert_unknown_payload(
        conflict.json(), message_parts=("not in frame", "keeper")
    )

    players[2]["role"] = "goalkeeper"
    assert client.put(
        f"/api/sessions/{session_id}/players", json={"players": players}
    ).status_code == 200
    two_keepers = client.post(
        f"/api/sessions/{session_id}/analyze",
        json={"attacking_side": "right", "attacking_team": 1, "keeper": "player"},
    )
    _assert_unknown_payload(two_keepers.json(), message_parts=("one", "keeper"))


# ---------------------------------------------------------------------------
# Defender ordering and team context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side", "positions", "keeper_index"),
    [
        ("right", (85.0, 92.0, 70.0, 80.0), 3),
        ("left", (15.0, 8.0, 30.0, 20.0), 3),
    ],
)
def test_bug04_marked_keeper_counts_in_second_defender_ranking(
    side, positions, keeper_index
):
    attacker_x, defender_x, other_defender_x, keeper_x = positions
    people = [
        _person(attacker_x, 1, side=side),
        _person(defender_x, 2, side=side),
        _person(other_defender_x, 2, side=side),
        _person(keeper_x, 2, side=side, keeper=True),
    ]
    calculator = OffsideLineCalculator()
    line = calculator.calculate_offside_line(
        people,
        attacking_side=side,
        exclude_goalkeeper=True,
        defending_team=2,
    )
    assert line is not None
    outcome = OffsideDecision().evaluate(
        people,
        line,
        attacking_side=side,
        exclude_goalkeeper=True,
        attacking_team=1,
        defending_team=2,
    )
    actual = (line.defining_player_idx, line.image_x, outcome.any_offside)
    assert actual == (keeper_index, keeper_x, True), (
        "the off-line keeper was skipped instead of ranked with all defenders"
    )


def test_bug05_keeper_cluster_does_not_override_the_requested_defending_team():
    result = _analysis(
        [
            _person(60, 1),
            _person(80, 1),
            _person(65, 2),
            _person(70, 2),
            _person(100, 1, keeper=True),
        ],
        width=160,
        keeper_choice="player",
    )

    payload = _phase2(_pipeline(), result)
    actual = (
        payload["verdict"],
        payload["last_defender"]["id"],
        payload["offside_player_ids"],
    )
    assert actual == ("offside", 3, [1]), (
        "the keeper's attacking-team cluster changed the defending team and line"
    )


def test_bug06_player_edits_and_analyze_preserve_the_stored_attacking_team():
    result = _analysis(
        [_person(90, 1), _person(40, 2)],
        width=200,
        keeper_choice=None,
    )
    players = [
        {
            "id": 0,
            "bbox": [70.0, 300.0, 90.0, 400.0],
            "team": 2,
            "role": "player",
            "confidence": 0.9,
        },
        {
            "id": 1,
            "bbox": [40.0, 300.0, 60.0, 400.0],
            "team": 1,
            "role": "goalkeeper",
            "confidence": 0.9,
        },
    ]

    phases.apply_player_edits(_pipeline(), result, players)
    assert (result.attacking_team, result.defending_team) == (1, 2), (
        "editing teams and a keeper overwrote the user's attacking-team choice"
    )

    result.keeper_choice = "player"
    phases.run_phase2(_pipeline(), result, "right")
    assert result.offside_result is not None
    assert result.offside_result.attacking_team == 1
    assert result.offside_result.defending_team == 2


def test_bug07_unknown_team_context_produces_no_line_and_no_evaluated_players():
    people = [_person(1000, None), _person(700, None), _person(650, None)]
    line = OffsideLineCalculator().calculate_offside_line(
        people,
        attacking_side="right",
        exclude_goalkeeper=True,
        defending_team=None,
    )
    assert line is None, "an attacker with unknown team context set the offside line"

    fallback_line = _manual_line(people[1], 1)
    outcome = OffsideDecision().evaluate(
        people,
        fallback_line,
        attacking_side="right",
        attacking_team=None,
        defending_team=None,
    )
    assert outcome.player_statuses == [], (
        "players were evaluated despite there being no team context"
    )


# ---------------------------------------------------------------------------
# Calibration geometry
# ---------------------------------------------------------------------------


def test_bug08_mirrored_calibration_preserves_line_and_metric_margin():
    for name, mirrored in (("normal", False), ("mirrored", True)):
        calibration = _field_calibration(mirrored)
        people = [_person(800, 1), _person(700, 2), _person(650, 2)]
        line = OffsideLineCalculator().calculate_offside_line(
            people,
            attacking_side="right",
            calibration=calibration,
            exclude_goalkeeper=True,
            defending_team=2,
        )
        assert line is not None
        outcome = OffsideDecision().evaluate(
            people,
            line,
            attacking_side="right",
            calibration=calibration,
            exclude_goalkeeper=True,
            attacking_team=1,
            defending_team=2,
        )
        status = next(
            (item for item in outcome.player_statuses if item.player_idx == 0), None
        )
        line_player_x = people[line.defining_player_idx].bbox[2]
        metric_margin = None if status is None else status.margin_px
        actual = (
            line_player_x,
            outcome.any_offside,
            metric_margin is not None and metric_margin > 0,
        )
        assert actual == (700.0, True, True), (
            f"{name} calibration returned line/offside/positive-margin={actual}"
        )


def test_bug09_rejected_automatic_camera_is_not_effective_for_analyze():
    rejected = _rejected_camera()
    state = CalibrationState()
    state.finish(rejected)
    assert auto_fit(rejected).label == "NOT BELIEVABLE"

    effective = state.effective()
    assert effective.method == "auto_failed", (
        "CalibrationState.effective() returned the rejected full-3D camera"
    )
    assert effective.debug_info.get("rejected")
    assert state.auto is rejected
    assert state.report()["automatic"]["full_3d"] is True


def test_bug09_phase2_rejected_camera_becomes_a_drawn_line_estimate():
    result = _analysis(
        [_person(800, 1), _person(700, 2), _person(650, 2)],
        calibration=_rejected_camera(),
    )

    payload = _phase2(_pipeline(), result)
    assert payload.get("estimate") is True, (
        "a rejected camera still produced a trusted full-3D verdict"
    )
    assert payload["verdict"] == "offside"
    assert payload["margin_m"] is None
    assert payload["line_field_x"] is None
    assert payload["view3d_available"] is False
    message = (payload.get("message") or "").lower()
    assert "estimate" in message
    assert "drawn line" in message
    assert "margin" in message
    assert "rejected" in message


def test_bug10_vanishing_point_comparison_uses_the_slanted_drawn_line():
    calibration = CalibrationResult(
        homography=np.eye(3),
        vanishing_point=(200.0, -300.0),
        method="vanishing_point",
        quality_score=0.5,
    )
    people = [_person(600, 1, y=300.0), _person(700, 2, y=600.0)]
    line = OffsideLineCalculator().calculate_offside_line(
        people,
        attacking_side="right",
        calibration=calibration,
        exclude_goalkeeper=True,
        defending_team=2,
    )
    assert line is not None
    outcome = OffsideDecision().evaluate(
        people,
        line,
        attacking_side="right",
        calibration=calibration,
        exclude_goalkeeper=True,
        attacking_team=1,
        defending_team=2,
    )
    result = _analysis(people, width=1000, height=700, calibration=calibration)
    result.offside_line = line
    result.offside_result = outcome

    payload = phases.serialize_verdict(result)
    actual = (
        payload["verdict"],
        payload.get("estimate"),
        payload["margin_m"],
        payload["line_field_x"],
    )
    assert actual == ("offside", True, None, None), (
        "the attacker was compared with a pixel column instead of the slanted line"
    )
    message = (payload.get("message") or "").lower()
    assert "estimate" in message
    assert "drawn line" in message
    assert "margin" in message


@pytest.mark.parametrize(
    ("mirrored", "defender_x"),
    [(False, 400.0), (True, 500.0)],
)
def test_bug11_own_half_is_never_offside_and_has_no_positive_margin(
    mirrored, defender_x
):
    calibration = _field_calibration(mirrored, full_3d=True)
    attacker = _person(450, 1)
    defender = _person(defender_x, 2)
    people = [attacker, defender]
    defender_field = calibration.project_to_field((defender_x, 400.0))
    assert defender_field is not None
    line = _manual_line(defender, 1, field_x=defender_field[0])

    outcome = OffsideDecision().evaluate(
        people,
        line,
        attacking_side="right",
        calibration=calibration,
        exclude_goalkeeper=True,
        attacking_team=1,
        defending_team=2,
    )
    status = next(
        (item for item in outcome.player_statuses if item.player_idx == 0), None
    )
    result = _analysis(people, calibration=calibration)
    result.offside_line = line
    result.offside_result = outcome
    payload = phases.serialize_verdict(result)

    actual = (
        outcome.any_offside,
        bool(status is not None and getattr(status, "in_own_half", False)),
        payload["margin_m"],
    )
    assert actual == (False, True, None), (
        "an own-half attacker produced an offside call or a positive margin"
    )


# ---------------------------------------------------------------------------
# Body points and detector classes
# ---------------------------------------------------------------------------


def test_bug12_pose_body_point_excludes_arms_and_box_fallback_is_offside():
    attacker = _person(710, 1, keypoints=_body_keypoints(), width=60.0)
    defender = _person(700, 2)
    people = [attacker, defender]
    line = OffsideLineCalculator().calculate_offside_line(
        people,
        attacking_side="right",
        exclude_goalkeeper=True,
        defending_team=2,
    )
    assert line is not None

    with_pose = OffsideDecision().evaluate(
        people,
        line,
        attacking_side="right",
        exclude_goalkeeper=True,
        attacking_team=1,
        defending_team=2,
    )
    assert not with_pose.any_offside, (
        "the attacker's outstretched wrist/box corner counted as an offside body point"
    )

    attacker.keypoints = None
    box_fallback = OffsideDecision().evaluate(
        people,
        line,
        attacking_side="right",
        exclude_goalkeeper=True,
        attacking_team=1,
        defending_team=2,
    )
    assert box_fallback.any_offside


def test_bug12_player_edits_drop_pose_when_the_bbox_moves():
    keypoints = _body_keypoints()
    result = _analysis([_person(710, 1, keypoints=keypoints, width=60.0)], width=800)
    row = {
        "id": 0,
        "bbox": [650.0, 300.0, 710.0, 400.0],
        "team": 1,
        "role": "player",
        "confidence": 0.9,
    }

    phases.apply_player_edits(_pipeline(), result, [row])
    np.testing.assert_array_equal(result.detections[0].keypoints, keypoints)

    moved_row = dict(row, bbox=[650.0, 300.0, 711.0, 400.0])
    phases.apply_player_edits(_pipeline(), result, [moved_row])
    assert result.detections[0].keypoints is None, (
        "a moved box retained pose keypoints matched to the old box"
    )


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Boxes:
    def __init__(self, values):
        self.data = _Array(values)

    def __len__(self):
        return len(self.data.values)


class _DetectionResult:
    def __init__(self, rows):
        self.names = {0: "BALL", 1: "goalkeeper", 2: "player", 3: "referee"}
        self.boxes = _Boxes(rows)


class _DetectionModel:
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, image, **kwargs):
        return [_DetectionResult(self.rows)]


def test_bug13_ball_boxes_are_dropped_before_height_and_player_limits():
    detector = PlayerDetector()
    detector.max_players = 3
    detector._detection_model = _DetectionModel(
        [
            [10, 20, 110, 200, 0.99, 0],  # Tall BALL must not consume a slot.
            [200, 20, 260, 200, 0.90, 1],
            [300, 20, 360, 200, 0.80, 2],
            [400, 20, 460, 200, 0.70, 3],
            [500, 20, 540, 200, 0.60, 2],
        ]
    )

    detections = detector.detect(
        np.zeros((500, 600, 3), dtype=np.uint8), use_pose=False
    )
    actual_boxes = {tuple(det.bbox) for det in detections}
    expected_boxes = {
        (200.0, 20.0, 260.0, 200.0),
        (300.0, 20.0, 360.0, 200.0),
        (400.0, 20.0, 460.0, 200.0),
    }
    assert actual_boxes == expected_boxes, (
        "a ball was retained as a person and consumed the max-players limit"
    )
    assert all(not det.is_goalkeeper for det in detections)


# ---------------------------------------------------------------------------
# Additional boundary and integration coverage
# ---------------------------------------------------------------------------


def test_bug04_only_marked_keeper_visible_cannot_define_a_line():
    people = [_person(800, 1), _person(1000, 2, keeper=True)]
    line = OffsideLineCalculator().calculate_offside_line(
        people,
        attacking_side="right",
        exclude_goalkeeper=True,
        defending_team=2,
    )
    assert line is None

    payload = _phase2(_pipeline(), _analysis(people, keeper_choice="player"))
    assert payload["verdict"] == "unknown"
    assert payload.get("message")


def test_bug08_full_3d_json_margin_has_the_same_positive_sign_when_mirrored():
    payloads = {}
    for name, mirrored in (("normal", False), ("mirrored", True)):
        result = _analysis(
            [_person(800, 1), _person(700, 2), _person(650, 2)],
            calibration=_believable_camera(reverse_angle=mirrored),
            width=1280,
            height=720,
        )
        payloads[name] = _phase2(_pipeline(), result)

    for name, payload in payloads.items():
        assert payload["verdict"] == "offside", name
        assert payload["margin_m"] is not None and payload["margin_m"] > 0, name
        assert payload["line_field_x"] is not None, name
    assert payloads["mirrored"]["margin_m"] == pytest.approx(
        payloads["normal"]["margin_m"]
    )


def test_bug09_api_rejected_camera_estimates_but_screen_keeps_automatic_result(
    client, monkeypatch
):
    pipeline = _pipeline()

    def fake_phase1(_pipeline, image):
        result = _analysis(
            [_person(800, 1), _person(700, 2), _person(650, 2)],
            width=image.shape[1],
            height=image.shape[0],
            keeper_choice=None,
        )
        result.image = image
        result.team_result = TeamAssignmentResult(
            detections=result.detections,
            embeddings=np.zeros((3, 8), dtype=np.float32),
            clustering=None,
            team_colors={1: (255, 0, 0), 2: (0, 0, 255)},
        )
        frame = image.copy()
        return result, {"original": frame, "detection": frame, "teams": frame}

    monkeypatch.setattr(phases, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(phases, "run_phase1", fake_phase1)
    monkeypatch.setattr(
        phases,
        "run_calibration",
        lambda _pipeline, _image: _rejected_camera(),
    )
    uploaded = client.post(
        "/api/sessions", files={"file": ("frame.png", _png_bytes(), "image/png")}
    )
    assert uploaded.status_code == 200, uploaded.text
    session_id = uploaded.json()["id"]
    assert client.post(f"/api/sessions/{session_id}/detect").status_code == 200

    response = client.post(
        f"/api/sessions/{session_id}/analyze",
        json={
            "attacking_side": "right",
            "attacking_team": 1,
            "keeper": "not_in_frame",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["estimate"] is True
    assert payload["margin_m"] is None
    assert payload["line_field_x"] is None
    assert payload["view3d_available"] is False
    assert "rejected" in (payload.get("message") or "").lower()

    calibration = client.get(f"/api/sessions/{session_id}/calibration")
    assert calibration.status_code == 200
    assert calibration.json()["fit"]["label"] == "NOT BELIEVABLE"


def test_bug10_vanishing_point_orders_defenders_along_slanted_lines():
    calibration = CalibrationResult(
        homography=np.eye(3),
        vanishing_point=(200.0, -300.0),
        method="vanishing_point",
        quality_score=0.5,
    )
    people = [
        _person(900, 1, y=600.0),
        _person(700, 2, y=600.0),
        _person(650, 2, y=300.0),
    ]
    line = OffsideLineCalculator().calculate_offside_line(
        people,
        attacking_side="right",
        calibration=calibration,
        exclude_goalkeeper=True,
        defending_team=2,
    )
    assert line is not None
    # At the common row 600, player 2 projects to x=875 and player 1 to
    # x=700, reversing their raw x order. Player 2 is therefore deepest.
    assert line.defining_player_idx == 2
    outcome = OffsideDecision().evaluate(
        people,
        line,
        attacking_side="right",
        calibration=calibration,
        exclude_goalkeeper=True,
        attacking_team=1,
        defending_team=2,
    )
    assert outcome.any_offside


@pytest.mark.parametrize("mirrored", [False, True])
def test_bug11_attacker_in_opponent_half_is_offside(mirrored):
    result = _analysis(
        [_person(600, 1), _person(550, 2), _person(500, 2)],
        calibration=_field_calibration(mirrored),
    )
    phases.run_phase2(_pipeline(), result, "right")
    assert result.offside_result is not None
    status = next(
        item for item in result.offside_result.player_statuses if item.player_idx == 0
    )
    assert result.offside_result.any_offside
    assert getattr(status, "in_own_half", False) is False


@pytest.mark.parametrize("mirrored", [False, True])
def test_bug11_attacker_exactly_on_halfway_is_not_offside(mirrored):
    result = _analysis(
        [_person(525, 1), _person(500, 2), _person(450, 2)],
        calibration=_field_calibration(mirrored),
    )
    phases.run_phase2(_pipeline(), result, "right")
    assert result.offside_result is not None
    status = next(
        item for item in result.offside_result.player_statuses if item.player_idx == 0
    )
    assert not result.offside_result.any_offside
    assert getattr(status, "in_own_half", False) is True


def _pose_with_body_positions(xs: Sequence[float], wrist: Tuple[float, float]) -> np.ndarray:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    torso_and_legs = (5, 6, 11, 12, 13, 14, 15, 16)
    for index, x in zip(torso_and_legs, xs):
        keypoints[index] = (x, 350.0, 0.95)
    keypoints[9] = (*wrist, 0.99)
    return keypoints


@pytest.mark.parametrize(
    ("side", "attacker", "defender", "body_x"),
    [
        (
            "right",
            _person(700, 1),
            PlayerDetection(
                bbox=(660.0, 300.0, 720.0, 400.0),
                confidence=0.9,
                team_id=2,
                keypoints=_pose_with_body_positions(
                    (680, 690, 682, 688, 684, 686, 680, 690), (720, 340)
                ),
            ),
            690.0,
        ),
        (
            "left",
            _person(300, 1, side="left"),
            PlayerDetection(
                bbox=(280.0, 300.0, 340.0, 400.0),
                confidence=0.9,
                team_id=2,
                keypoints=_pose_with_body_positions(
                    (320, 310, 318, 312, 316, 314, 320, 310), (280, 340)
                ),
            ),
            310.0,
        ),
    ],
)
def test_bug12_defender_arm_does_not_set_the_line(side, attacker, defender, body_x):
    people = [attacker, defender]
    line = OffsideLineCalculator().calculate_offside_line(
        people,
        attacking_side=side,
        exclude_goalkeeper=True,
        defending_team=2,
    )
    assert line is not None
    assert line.image_x == pytest.approx(body_x)
    outcome = OffsideDecision().evaluate(
        people,
        line,
        attacking_side=side,
        exclude_goalkeeper=True,
        attacking_team=1,
        defending_team=2,
    )
    assert outcome.any_offside


def test_bug12_sparse_or_outside_pose_falls_back_to_the_box_corner():
    sparse = _pose_with_body_positions((680, 685, 690), (710, 340))
    sparse_detection = _person(710, 1, keypoints=sparse)
    assert get_directional_ground_point(sparse_detection, "right") == (710.0, 400.0)

    outside = _pose_with_body_positions(
        (900, 910, 905, 915, 900, 910, 905, 915), (920, 340)
    )
    outside_detection = _person(710, 1, keypoints=outside)
    assert get_directional_ground_point(outside_detection, "right") == (710.0, 400.0)


class _PoseResult:
    def __init__(self, boxes, keypoints):
        self.boxes = _Boxes(boxes)
        self.keypoints = types.SimpleNamespace(data=_Array(keypoints))


def test_bug12_pose_runs_per_box_and_maps_keypoints_back_to_the_frame():
    """Broadcast players are too small for the full image: pose reads a crop of each."""
    detector = PlayerDetector()
    seen_shapes = []

    def pose_model(crops, **kwargs):
        assert isinstance(crops, list), "pose must run on one crop per box"
        seen_shapes.extend(crop.shape[:2] for crop in crops)
        results = []
        for crop in crops:
            height, width = crop.shape[:2]
            neighbour = np.zeros((17, 3))
            neighbour[:, :2] = (1.0, 1.0)
            target = np.zeros((17, 3))
            target[:, :2] = (width / 2.0, height / 2.0)
            target[:, 2] = 0.9
            results.append(
                _PoseResult(
                    [
                        [0, 0, width * 0.2, height * 0.2, 0.9, 0],
                        [width * 0.1, height * 0.05, width * 0.9, height * 0.95, 0.9, 0],
                    ],
                    np.stack([neighbour, target]),
                )
            )
        return results

    detector._pose_model = pose_model
    detection = PlayerDetection(bbox=(100.0, 200.0, 140.0, 300.0), confidence=0.9)
    image = np.zeros((500, 600, 3), dtype=np.uint8)

    detector._add_pose_keypoints(image, [detection])

    assert seen_shapes == [(120, 60)]
    assert detection.keypoints is not None
    # The crop starts at (90, 190), so its centre (30, 60) is (120, 250) in the frame.
    np.testing.assert_allclose(detection.keypoints[0], (120.0, 250.0, 0.9))
