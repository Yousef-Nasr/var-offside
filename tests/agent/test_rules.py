"""Rule checks, what-if and policy, using a real solver camera and the real line code.

Positions match the agent plan's examples: metres along the pitch, attack to the right.
No model or network is used.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from var_offside.agent import findings as f
from var_offside.agent.checks import run_checks
from var_offside.agent.checks import camera as camera_check
from var_offside.agent.checks import teams as team_check
from var_offside.agent.findings import Evidence, Finding
from var_offside.agent.policy import (
    CLEARED,
    NEEDS_ANSWER,
    NOT_USABLE,
    TOO_CLOSE,
    decide,
    independent_kinds,
    is_too_close,
    review,
    stability_test,
)
from var_offside.agent.scene import Change, Scene
from var_offside.agent.whatif import Call, run_call, what_if
from var_offside.calibration import pitch_solver as ps
from var_offside.calibration.base import CalibrationResult
from var_offside.detection import PlayerDetection
from var_offside.pipeline import AnalysisResult
from var_offside.teams import TeamAssignmentResult

W, H = 1920, 1080
RED, BLUE, GREEN, BLACK = 0, 1, 2, 3


def _camera(cx=8.0, cy=75.0, height=22.0, target=(32.0, 2.0), focal=1900.0) -> ps.Camera:
    position = np.array([cx, cy, -height], dtype=np.float64)
    target_w = np.array([target[0], target[1], 0.0])
    return ps.Camera(float(focal), ps._look_at(position, target_w), position, W / 2.0, H / 2.0)


CAMERA = _camera()


def _pixel(camera: ps.Camera, x: float, y: float, up: float = 0.0) -> np.ndarray:
    world = ps._field_to_world(np.array([[x, y]], dtype=np.float64))[0]
    world[2] = -up
    projection = camera.K @ camera.R @ np.column_stack((np.eye(3), -camera.position))
    h = projection @ np.append(world, 1.0)
    return h[:2] / h[2]


def _calibration(camera: ps.Camera = CAMERA, quality: float = 0.95) -> CalibrationResult:
    result = ps.to_calibration_result(camera, [], W, H, "hand")
    result.method = "pnl"
    result.quality_score = quality
    result.debug_info = {}
    return result


def _vector(kind: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    vector = np.zeros(8)
    vector[kind] = 1.0
    if noise:
        vector += np.random.default_rng(seed).normal(0.0, noise, 8)
    return vector


class Build:
    """Builds a scene by standing people on the pitch."""

    def __init__(self, camera: ps.Camera = CAMERA) -> None:
        self.camera = camera
        self.people: List[PlayerDetection] = []
        self.rows: List[np.ndarray] = []

    def add(
        self,
        x: float,
        team: Optional[int],
        kit: int,
        y: float = 34.0,
        cls: str = "player",
        keeper: bool = False,
        tall: float = 1.8,
    ) -> int:
        """Stand someone at (x, y). The box's front foot corner is exactly that point."""
        foot = _pixel(self.camera, x, y)
        head = _pixel(self.camera, x, y, tall)
        box_height = foot[1] - head[1]
        player_id = len(self.people)
        self.people.append(
            PlayerDetection(
                bbox=(foot[0] - box_height * 0.35, head[1], foot[0], foot[1]),
                confidence=0.9,
                player_id=player_id,
                team_id=team,
                is_goalkeeper=keeper,
                detector_class=cls,
                keypoints=_pose(foot[0], foot[1], head[1], box_height),
            )
        )
        self.rows.append(_vector(kit, 0.01, player_id))
        return player_id

    def scene(
        self,
        side: Optional[str] = "right",
        attacking: Optional[int] = 1,
        keeper: Optional[str] = "player",
        calibration: Optional[CalibrationResult] = None,
        image: Optional[np.ndarray] = None,
    ) -> Scene:
        result = AnalysisResult(
            image=_noise() if image is None else image,
            detections=self.people,
            attacking_team=attacking,
            keeper_choice=keeper,
            team_result=TeamAssignmentResult(
                detections=self.people,
                embeddings=np.array(self.rows),
                clustering=None,
                team_colors={},
            ),
        )
        return Scene.from_result(result, side=side, calibration=calibration or _calibration())


def _pose(x: float, foot_y: float, head_y: float, height: float) -> np.ndarray:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    for index, row in ((5, 0.2), (6, 0.2), (11, 0.5), (12, 0.5), (13, 0.75), (14, 0.75), (15, 0.98), (16, 0.98)):
        keypoints[index] = (x, head_y + height * row, 0.95)
    return keypoints


def _noise() -> np.ndarray:
    return np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)


def _crowd(build: Build, deep: float = 70.0) -> None:
    """Midfield players who take no part in the call, so teams and the frame look normal."""
    for offset, y in ((0.0, 20.0), (4.0, 45.0), (8.0, 28.0)):
        build.add(deep + offset, 1, RED, y=y)
        build.add(deep + offset + 1.0, 2, BLUE, y=y + 6.0)


def _standard(extra=None) -> Tuple[Build, Dict[str, int]]:
    """Attacker #9 at 88 m, defender #4 at 87 m, keeper #14 at 100 m."""
    build = Build()
    _crowd(build)
    ids = {
        "attacker": build.add(88.0, 1, RED, y=40.0),
        "defender": build.add(87.0, 2, BLUE, y=28.0),
        "keeper": build.add(100.0, 2, GREEN, y=33.0, cls="goalkeeper", keeper=True),
    }
    return build, ids


def _codes(found: Sequence[Finding]) -> List[str]:
    return [finding.code for finding in found]


def _vision(detail: str = "Seen in the frame") -> Evidence:
    return Evidence(f.VISION, detail)


# ---------------------------------------------------------------------------
# What-if
# ---------------------------------------------------------------------------


def test_call_is_measured_in_metres_from_the_second_last_defender():
    build, ids = _standard()
    call = run_call(build.scene())
    assert call.verdict == "offside"
    assert call.margin_m == pytest.approx(1.0, abs=0.02)
    assert call.line_player_id == ids["defender"]
    assert call.foremost_attacker_id == ids["attacker"]


def test_what_if_leaves_the_scene_unchanged():
    build, ids = _standard()
    scene = build.scene()
    before = scene.player(ids["defender"]).team_id
    changed = what_if(scene, Change("set_team", player_id=ids["defender"], team=1))
    assert changed.line_player_id != ids["defender"]
    assert scene.player(ids["defender"]).team_id == before
    assert run_call(scene).verdict == "offside"


def test_scene_is_a_copy_of_the_session_result():
    build, ids = _standard()
    scene = build.scene()
    build.people[ids["attacker"]].team_id = 2
    assert scene.player(ids["attacker"]).team_id == 1


@pytest.mark.parametrize(
    ("side", "team", "keeper", "message"),
    [
        (None, 1, "player", "attack direction"),
        ("right", None, "player", "attacking team"),
        ("right", 1, None, "keeper"),
    ],
)
def test_missing_choice_gives_no_call(side, team, keeper, message):
    build, _ids = _standard()
    scene = build.scene(side=side, attacking=team, keeper=keeper)
    if keeper is None:
        scene = scene.apply(Change("set_role", player_id=8, role="player"))
    call = run_call(scene)
    assert call.verdict == "unknown"
    assert message in call.message


def test_what_if_agrees_with_the_app_on_the_same_scene():
    from unittest.mock import MagicMock

    from var_offside.offside.decision import OffsideDecision
    from var_offside.offside.line_calculator import OffsideLineCalculator
    from var_offside.offside.side_detector import SideDetector
    from var_offside.web import phases

    build, _ids = _standard()
    scene = build.scene()
    pipeline = MagicMock()
    pipeline.line_calculator = OffsideLineCalculator()
    pipeline.offside_decision = OffsideDecision()
    pipeline.side_detector = SideDetector()
    pipeline._draw_offside_line_debug.side_effect = lambda image, *args, **kwargs: image
    pipeline.field_overlay.draw_calibration_result.side_effect = lambda image, *args: image
    pipeline.field_overlay.draw_projected_field.side_effect = lambda image, *args: image
    pipeline.visualizer.create_clean_result_view.side_effect = lambda image, *args: image
    result = AnalysisResult(
        image=scene.image,
        detections=list(scene.detections),
        calibration=scene.calibration,
        attacking_team=1,
        keeper_choice="player",
    )
    phases.run_phase2(pipeline, result, "right")
    app = phases.serialize_verdict(result)
    call = run_call(scene)
    assert call.verdict == app["verdict"]
    assert call.margin_m == app["margin_m"]
    assert call.line_player_id == app["last_defender"]["id"]


# ---------------------------------------------------------------------------
# Evidence rules
# ---------------------------------------------------------------------------


def test_same_source_twice_is_one_kind_of_evidence():
    twice = [_vision("first"), _vision("second")]
    assert independent_kinds(twice) == {f.VISION}


def test_flipping_change_needs_vision_or_two_kinds_of_rule():
    build, ids = _standard()
    scene = build.scene()
    change = Change("set_team", player_id=ids["attacker"], team=2)
    assert decide(scene, change, [_vision()]).apply
    assert not decide(scene, change, [Evidence(f.EMBEDDING, "kit")]).apply
    assert decide(scene, change, [Evidence(f.EMBEDDING, "kit"), Evidence(f.GEOMETRY, "place")]).apply


def test_harmless_change_needs_one_clear_answer():
    build, _ids = _standard()
    scene = build.scene()
    change = Change("set_team", player_id=0, team=2)
    decision = decide(scene, change, [Evidence(f.EMBEDDING, "kit")])
    assert decision.apply and not decision.flips_call


def test_no_evidence_changes_nothing():
    build, _ids = _standard()
    assert not decide(build.scene(), Change("set_team", player_id=0, team=2), []).apply


@pytest.mark.parametrize(
    "change",
    [Change("set_side", side="left"), Change("set_attacking_team", team=2)],
)
def test_side_and_team_never_change_without_vision(change):
    build, _ids = _standard()
    scene = build.scene()
    rules_only = [Evidence(f.GEOMETRY, "shape"), Evidence(f.CALIBRATION, "goal"), Evidence(f.BALL, "ball")]
    assert not decide(scene, change, rules_only).apply
    assert decide(scene, change, rules_only + [_vision()]).apply


def test_conflicting_evidence_is_never_applied():
    build, _ids = _standard()
    change = Change("set_side", side="right")
    both = [_vision(), Evidence(f.CALIBRATION, "goal")]
    assert not decide(build.scene(side=None), change, both, conflict=True).apply


def test_your_answer_always_applies():
    build, ids = _standard()
    change = Change("set_team", player_id=ids["attacker"], team=2)
    assert decide(build.scene(), change, [Evidence(f.USER, "You chose it")]).apply


def test_keeper_flip_needs_vision_even_with_two_rule_kinds():
    build, ids = _standard()
    scene = build.scene(keeper="not_in_frame")
    scene = scene.apply(Change("set_keeper_not_in_frame"))
    change = Change("set_role", player_id=ids["keeper"], role="goalkeeper")
    rules = [Evidence(f.DETECTOR_CLASS, "class"), Evidence(f.EMBEDDING, "kit")]
    assert not decide(scene, change, rules).apply
    assert decide(scene, change, rules + [_vision()]).apply


# ---------------------------------------------------------------------------
# Examples from the plan: mistakes that rules can catch
# ---------------------------------------------------------------------------


def test_clean_frame_clears_when_vision_has_looked():
    build, _ids = _standard()
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == CLEARED, outcome.questions
    assert outcome.call.verdict == "offside"
    assert outcome.stability.tested > 0
    assert not outcome.stability.changed


def test_clean_frame_does_not_clear_without_vision():
    build, _ids = _standard()
    outcome = review(build.scene())
    assert outcome.ending == NEEDS_ANSWER
    assert any(question.topic == "vision" for question in outcome.questions)


def test_ball_boy_off_the_pitch_is_asked_about_then_removed_with_a_second_source():
    build = Build()
    _crowd(build)
    attacker = build.add(88.0, 1, RED, y=40.0)
    defender = build.add(85.0, 2, BLUE, y=28.0)
    build.add(100.0, 2, GREEN, y=33.0, cls="goalkeeper", keeper=True)
    boy = build.add(92.0, 2, BLUE, y=72.5)
    scene = build.scene()
    assert run_call(scene).verdict == "onside"
    assert run_call(scene).margin_m == pytest.approx(-4.0, abs=0.05)

    rules_only = review(scene, vision_checked=True)
    assert rules_only.ending == NEEDS_ANSWER
    assert any(question.player_ids == (boy,) for question in rules_only.questions)
    assert rules_only.scene.player(boy) is not None

    remove = Change("remove_player", player_id=boy)
    settled = review(scene, {remove: [_vision("A ball boy")]}, vision_checked=True)
    assert settled.ending == CLEARED, settled.questions
    assert settled.scene.player(boy) is None
    assert settled.call.verdict == "offside"
    assert settled.call.margin_m == pytest.approx(3.0, abs=0.05)
    assert settled.call.line_player_id == defender
    assert [item.flips_call for item in settled.applied] == [True]
    assert attacker in settled.call.offside_ids


def test_doubled_box_is_removed_because_it_cannot_change_the_call():
    build, _ids = _standard()
    first = build.people[0]
    build.people.append(
        PlayerDetection(
            bbox=tuple(value + 1.0 for value in first.bbox),
            confidence=0.5,
            player_id=len(build.people),
            team_id=first.team_id,
        )
    )
    build.rows.append(build.rows[0].copy())
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == CLEARED, outcome.questions
    assert outcome.scene.player(len(build.people) - 1) is None


def test_defender_on_the_wrong_team_flips_the_call_so_it_is_asked_about():
    build = Build()
    _crowd(build)
    attacker = build.add(88.0, 1, RED, y=40.0)
    build.add(86.0, 2, BLUE, y=28.0)
    build.add(100.0, 2, GREEN, y=33.0, cls="goalkeeper", keeper=True)
    wrong = build.add(90.0, 1, BLUE, y=35.0)
    scene = build.scene()
    assert run_call(scene).verdict == "offside"

    outcome = review(scene, vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert any(question.player_ids == (wrong,) for question in outcome.questions)

    switch = Change("set_team", player_id=wrong, team=2)
    settled = review(scene, {switch: [_vision("Blue kit")]}, vision_checked=True)
    assert settled.ending == CLEARED, settled.questions
    assert settled.call.verdict == "onside"
    assert attacker not in settled.call.offside_ids


def test_wrong_team_far_from_the_line_is_fixed_without_asking():
    build, _ids = _standard()
    far = build.add(62.0, 1, BLUE, y=30.0)
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == CLEARED, outcome.questions
    assert outcome.scene.player(far).team_id == 2
    assert outcome.call.verdict == "offside"


def test_wrong_side_is_never_cleared():
    build, _ids = _standard()
    outcome = review(build.scene(side="left"), vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert "goal_in_view_contradicted" in _codes(outcome.findings)


def test_right_side_gets_no_contradiction():
    build, _ids = _standard()
    assert not any("contradicted" in code for code in _codes(run_checks(build.scene())))


def test_side_is_read_correctly_on_a_reverse_angle_camera():
    reverse = _camera(cx=8.0, cy=-75.0, target=(32.0, -2.0))
    build = Build(reverse)
    for offset, y in ((0.0, 20.0), (4.0, 45.0), (8.0, 28.0)):
        build.add(70.0 + offset, 1, RED, y=y)
        build.add(71.0 + offset, 2, BLUE, y=y + 6.0)
    scene = build.scene(side=None, calibration=_calibration(reverse))
    proposals = {item.proposal for item in run_checks(scene) if item.code == "goal_in_view"}
    assert proposals == {Change("set_side", side="left")}


def test_wrong_attacking_team_is_never_cleared():
    build, _ids = _standard()
    # Team 2 forms the back line, so calling it the attacker contradicts the shape.
    for x, y in ((90.0, 22.0), (91.0, 48.0), (92.0, 36.0)):
        build.add(x, 2, BLUE, y=y)
    scene = build.scene(attacking=2)
    outcome = review(scene, vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert "team_shape_contradicted" in _codes(outcome.findings)


def test_side_and_team_are_asked_for_by_rules_alone_even_when_they_agree():
    build, _ids = _standard()
    outcome = review(build.scene(side=None, attacking=None), vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert outcome.scene.side is None
    assert any(question.topic == "side" for question in outcome.questions)


def test_side_applies_when_vision_agrees_with_the_camera():
    build, _ids = _standard()
    scene = build.scene(side=None)
    change = Change("set_side", side="right")
    outcome = review(scene, {change: [_vision("Run to the right")]}, vision_checked=True)
    assert outcome.scene.side == "right"
    assert outcome.ending == CLEARED, outcome.questions


def test_keeper_treated_as_not_in_frame_is_asked_about():
    build = Build()
    _crowd(build)
    build.add(88.0, 1, RED, y=40.0)
    build.add(85.0, 2, BLUE, y=28.0)
    keeper = build.add(100.0, 2, GREEN, y=33.0, cls="goalkeeper")
    scene = build.scene(keeper="not_in_frame")
    assert run_call(scene).margin_m == pytest.approx(-12.0, abs=0.05)

    outcome = review(scene, vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert any(question.player_ids == (keeper,) for question in outcome.questions)

    mark = Change("set_role", player_id=keeper, role="goalkeeper")
    settled = review(scene, {mark: [_vision("Gloves")]}, vision_checked=True)
    assert settled.ending == CLEARED, settled.questions
    assert settled.call.verdict == "offside"
    assert settled.call.margin_m == pytest.approx(3.0, abs=0.05)


def test_referee_setting_the_line_is_corrected_from_class_and_kit():
    build = Build()
    _crowd(build)
    build.add(88.0, 1, RED, y=40.0)
    build.add(85.0, 2, BLUE, y=28.0)
    build.add(100.0, 2, GREEN, y=33.0, cls="goalkeeper", keeper=True)
    referee = build.add(93.0, 2, BLACK, y=45.0, cls="referee")
    scene = build.scene()
    assert run_call(scene).margin_m == pytest.approx(-5.0, abs=0.05)

    outcome = review(scene, vision_checked=True)
    assert outcome.ending == CLEARED, outcome.questions
    assert outcome.call.verdict == "offside"
    assert outcome.call.margin_m == pytest.approx(3.0, abs=0.05)
    assert outcome.scene.player(referee).team_id is None
    assert outcome.applied[0].flips_call


def test_line_setter_cut_off_by_the_frame_is_not_cleared():
    build, ids = _standard()
    det = build.people[ids["defender"]]
    det.bbox = (det.bbox[0], det.bbox[1], float(W), det.bbox[3])
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert "line_setter_cut_off" in _codes(outcome.findings)


def test_arm_that_could_decide_a_tight_call_is_too_close_and_not_a_question():
    build = Build()
    _crowd(build)
    attacker = build.add(87.1, 1, RED, y=40.0)
    build.add(87.0, 2, BLUE, y=28.0)
    build.add(100.0, 2, GREEN, y=33.0, cls="goalkeeper", keeper=True)
    build.people[attacker].keypoints = None
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == TOO_CLOSE
    assert not outcome.questions
    assert any(item.code == "no_body_points" for item in outcome.stability.changed)


def test_far_box_with_no_pose_needs_no_question():
    build, ids = _standard()
    build.people[ids["attacker"]].keypoints = None
    build.people[ids["attacker"]].bbox = _moved(build.people[ids["attacker"]].bbox, 60.0)
    outcome = review(build.scene(), vision_checked=True)
    assert not outcome.stability.changed


def _moved(bbox, dx):
    return (bbox[0] + dx, bbox[1], bbox[2] + dx, bbox[3])


# ---------------------------------------------------------------------------
# Too close, unusable frames and failures
# ---------------------------------------------------------------------------


def test_too_close_compares_the_margin_with_the_uncertainty():
    assert is_too_close(Call("offside", margin_m=0.12), 0.30)
    assert is_too_close(Call("onside", margin_m=-0.12), 0.30)
    assert not is_too_close(Call("onside", margin_m=-2.0), 0.30)
    assert not is_too_close(Call("onside", margin_m=None), 0.30)
    assert not is_too_close(Call("onside", margin_m=-0.1), None)


def test_margin_under_the_uncertainty_ends_too_close(monkeypatch):
    build, _ids = _standard()
    monkeypatch.setattr("var_offside.agent.policy.line_uncertainty_m", lambda *args: 5.0)
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == TOO_CLOSE, outcome.questions
    assert "smaller than" in outcome.reason


def test_two_different_signs_make_a_frame_not_usable():
    build = Build()
    for x in (80.0, 84.0, 88.0):
        build.add(x, 1, RED)
    scene = build.scene(calibration=CalibrationResult(method="auto_failed"))
    outcome = review(scene)
    assert outcome.ending == NOT_USABLE
    assert "pitch lines" in outcome.reason


def test_one_sign_alone_is_a_question_not_a_rejection():
    build = Build()
    for x in (80.0, 84.0, 88.0):
        build.add(x, 1, RED)
    outcome = review(build.scene())
    assert outcome.ending == NEEDS_ANSWER


@pytest.mark.parametrize("method", ["missing", "estimate"])
def test_no_measured_margin_never_clears(method):
    build, _ids = _standard()
    calibration = (
        CalibrationResult(method="auto_failed")
        if method == "missing"
        else CalibrationResult(
            homography=np.linalg.inv(CAMERA.field_to_image()),
            homography_inv=CAMERA.field_to_image(),
            method="pnl",
            quality_score=0.9,
        )
    )
    outcome = review(build.scene(calibration=calibration), vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER


def test_rejected_camera_never_clears():
    build, _ids = _standard()
    k = np.array([[20.0, 0.0, 640.0], [0.0, 20.0, 360.0], [0.0, 0.0, 1.0]])
    position = np.array([0.0, -50.0, -5.0])
    rejected = CalibrationResult(
        homography=np.eye(3),
        homography_inv=np.eye(3),
        camera_matrix=k,
        rotation_matrix=np.eye(3),
        position=position,
        projection_matrix=k @ np.column_stack((np.eye(3), -position)),
        image_size=(H, W),
        method="pnl",
        quality_score=0.95,
    )
    outcome = review(build.scene(calibration=rejected), vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert "no_camera" in _codes(outcome.findings)


def test_a_check_that_fails_stops_the_frame_clearing(monkeypatch):
    from var_offside.agent import checks

    def boom(_scene):
        raise RuntimeError("broken")

    build, _ids = _standard()
    monkeypatch.setattr(checks, "CHECKS", (("teams", boom),) + tuple(checks.CHECKS[1:]))
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert "check_failed" in _codes(outcome.findings)


def test_any_failure_in_the_review_goes_to_a_person(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("broken")

    build, _ids = _standard()
    monkeypatch.setattr("var_offside.agent.policy.run_checks", boom)
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert outcome.questions


def test_untestable_doubt_counts_as_changing_the_call():
    build, _ids = _standard()
    gone = Finding(
        "teams", "x", "x", f.DOUBT, alternatives=(Change("set_team", player_id=999, team=2),)
    )
    assert stability_test(build.scene(), [gone]).changed == (gone,)


def test_missing_embeddings_stop_the_frame_clearing():
    build, _ids = _standard()
    build.rows = [np.zeros(8) for _row in build.rows]
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == NEEDS_ANSWER
    assert "no_embeddings" in _codes(outcome.findings)


# ---------------------------------------------------------------------------
# Camera checks
# ---------------------------------------------------------------------------


def test_uncertainty_for_an_automatic_fit_is_small_for_a_good_fit():
    value = camera_check.line_uncertainty_m(_calibration(), W, H)
    assert value is not None and 0.0 < value < 2.0


def test_uncertainty_grows_with_a_weaker_automatic_fit():
    good = camera_check.line_uncertainty_m(_calibration(quality=0.95), W, H)
    weak = camera_check.line_uncertainty_m(_calibration(quality=0.4), W, H)
    assert weak > good


def test_uncertainty_is_unknown_without_a_camera():
    assert camera_check.line_uncertainty_m(CalibrationResult(method="auto_failed"), W, H) is None


def test_uncertainty_uses_the_pins_of_a_hand_fit():
    pins = []
    for lid in ("r_box_front_far", "r_box_front_near", "r_six_front_far", "r_six_front_near", "r_spot"):
        mark = ps._LANDMARK_BY_ID[lid]
        u, v = _pixel(CAMERA, mark.x, mark.y)
        pins.append(ps.Pin("point", lid, float(u), float(v)))
    calibration = ps.to_calibration_result(CAMERA, pins, W, H, "hand")
    value = camera_check.line_uncertainty_m(calibration, W, H)
    assert value == pytest.approx(ps.estimate_line_error(CAMERA, pins, W, H))


def test_normal_heights_pass_and_a_wrong_lens_is_caught():
    build, _ids = _standard()
    assert "odd_player_heights" not in _codes(run_checks(build.scene()))
    assert "height_drift" not in _codes(run_checks(build.scene()))

    wrong = _camera(focal=1100.0)
    outcome = run_checks(build.scene(calibration=_calibration(wrong)))
    assert {"odd_player_heights", "height_drift"} & set(_codes(outcome))


def test_heights_are_measured_correctly():
    build, _ids = _standard()
    samples = camera_check.player_heights_m(build.scene(), CAMERA)
    assert [height for _y, height in samples] == pytest.approx([1.8] * len(samples), abs=0.02)


# ---------------------------------------------------------------------------
# Team doubt
# ---------------------------------------------------------------------------


def test_team_doubt_uses_embeddings_and_leaves_clean_teams_alone():
    build, _ids = _standard()
    assert not [item for item in run_checks(build.scene()) if item.check == "teams"]


def test_kit_between_both_teams_is_a_doubt_without_a_proposal():
    build, _ids = _standard()
    between = build.add(62.0, 1, RED, y=30.0)
    build.rows[between] = _vector(RED) + _vector(BLUE)
    found = [item for item in run_checks(build.scene()) if item.check == "teams"]
    assert _codes(found) == ["between_teams"]
    assert found[0].proposal is None
    assert found[0].alternatives == (Change("set_team", player_id=between, team=2),)


def test_distance_leaves_the_player_out_of_their_own_team():
    build, _ids = _standard()
    teams = team_check.team_vectors(build.scene())
    own, other = team_check.distance_ratio(0, 1, teams)
    assert own < 0.1 < other


# ---------------------------------------------------------------------------
# What rules can't catch
# ---------------------------------------------------------------------------


def test_rules_cannot_see_a_missed_player_so_vision_is_still_required():
    """Defender #21 at 90 m has no box: the call is wrong and no rule can tell."""
    build, _ids = _standard()
    scene = build.scene()
    assert run_call(scene).verdict == "offside"
    assert not any("miss" in code for code in _codes(run_checks(scene)))

    outcome = review(scene)
    assert outcome.ending == NEEDS_ANSWER
    assert any(question.topic == "vision" for question in outcome.questions)


def test_vision_missing_a_player_too_would_clear_a_wrong_call():
    """The plan's stated limit: the stability test only tests doubts someone raised."""
    build, _ids = _standard()
    outcome = review(build.scene(), vision_checked=True)
    assert outcome.ending == CLEARED
    assert outcome.call.verdict == "offside"  # With the unseen defender it would be onside.


def test_found_missed_player_with_a_detector_box_flips_the_call():
    build, _ids = _standard()
    scene = build.scene()
    foot = _pixel(CAMERA, 90.0, 35.0)
    head = _pixel(CAMERA, 90.0, 35.0, 1.8)
    box = (float(foot[0] - 30.0), float(head[1]), float(foot[0]), float(foot[1]))
    add = Change("add_player", team=2, bbox=box)

    both = [_vision("Person in F3"), Evidence(f.DETECTOR_RERUN, "Box found on the re-run")]
    assert decide(scene, add, both).apply
    assert what_if(scene, add).verdict == "onside"
    assert what_if(scene, add).margin_m == pytest.approx(-2.0, abs=0.05)


def test_uneven_teams_raise_a_note_but_prove_nothing():
    build, _ids = _standard()
    for x, y in ((72.0, 15.0), (74.0, 50.0), (76.0, 25.0), (78.0, 55.0), (79.0, 10.0)):
        build.add(x, 1, RED, y=y)
    assert "uneven_teams" in _codes(run_checks(build.scene()))
