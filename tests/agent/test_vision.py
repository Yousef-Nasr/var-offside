"""Vision checks with a fake session: no key, no network, and no model."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest

from var_offside.agent import findings as f
from var_offside.agent import policy, vision, vision_checks
from var_offside.agent.config import AgentConfig
from var_offside.agent.llm import OpenRouterClient, encode_image
from var_offside.agent.scene import Change, Scene
from tests.agent.test_rules import Build, _calibration
from var_offside.detection import PlayerDetection

ALL_ON = {key: True for key in vision_checks.VISION_AUTO_APPLY}


class Reply:
    status_code = 200

    def __init__(self, content: Any) -> None:
        self.content = content

    def json(self) -> Dict[str, Any]:
        text = self.content if isinstance(self.content, str) else json.dumps(self.content)
        return {"choices": [{"message": {"content": text}}], "usage": {"cost": 0.001}}


class Session:
    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.bodies: List[Dict[str, Any]] = []

    def post(self, _url: str, **kwargs: Any) -> Reply:
        self.bodies.append(kwargs["json"])
        return Reply(self.answers.pop(0))


def answered(task: str, answer: Dict[str, Any]) -> Dict[str, Any]:
    return {"task": task, "status": "answered", "certainty": "clear", "answer": answer, "evidence": ["seen"]}


def cannot_tell(task: str) -> Dict[str, Any]:
    return {"task": task, "status": "cannot_tell", "certainty": "unclear", "answer": None}


def client_for(*answers: Any) -> OpenRouterClient:
    session = Session(*answers)
    client = OpenRouterClient(AgentConfig(api_key="k", model="test/vision"), session=session, sleep=lambda _: None)
    client.test_session = session  # type: ignore[attr-defined]
    return client


def people_scene(**kwargs: Any) -> Scene:
    detections = []
    for pid, (x, team) in enumerate([(100, 1), (300, 1), (500, 2), (700, 2), (900, 1), (1100, 2)]):
        detections.append(
            PlayerDetection(bbox=(x, 300, x + 40, 420), confidence=0.9, player_id=pid, team_id=team)
        )
    image = np.full((720, 1280, 3), 60, dtype=np.uint8)
    defaults = dict(detections=tuple(detections), image=image, width=1280, height=720, next_player_id=6)
    return Scene(**{**defaults, **kwargs})


class FakeDetector:
    """Stands in for the real detector, and records the crop it was given."""

    def __init__(self, boxes: List[tuple]) -> None:
        self.boxes = boxes
        self.last_ball = "the frame's ball"
        self.shapes: List[tuple] = []

    def detect(self, crop: np.ndarray, **_kwargs: Any) -> List[PlayerDetection]:
        self.shapes.append(crop.shape)
        self.last_ball = "the crop's ball"
        return [PlayerDetection(bbox=box, confidence=0.5) for box in self.boxes]


@pytest.fixture
def trust_all(monkeypatch):
    monkeypatch.setattr(vision_checks, "VISION_AUTO_APPLY", ALL_ON)


def codes(found) -> List[str]:
    return [item.code for item in found]


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def test_drawing_never_alters_the_frame():
    scene = people_scene()
    before = scene.image.copy()
    vision.number_frame(scene.image, scene.detections, grid=True)
    vision.crop_player(scene.image, scene.detections[0])
    vision.team_contact_sheet(scene)
    vision.crop_sheet(scene.image, scene.detections)
    np.testing.assert_array_equal(scene.image, before)


def test_number_frame_marks_the_image():
    scene = people_scene()
    assert np.any(vision.number_frame(scene.image, scene.detections) != scene.image)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [("A1", (0.0, 0.0, 160.0, 120.0)), ("h6", (1120.0, 600.0, 1280.0, 720.0)), ("F3", (800.0, 240.0, 960.0, 360.0))],
)
def test_cell_box(cell, expected):
    assert vision.cell_box(cell, 1280, 720) == pytest.approx(expected)


@pytest.mark.parametrize("cell", ["Z1", "A9", "A0", "", "pitch", "A"])
def test_bad_cell_is_rejected(cell):
    assert vision.cell_box(cell, 1280, 720) is None


def test_cell_of_inverts_cell_box():
    assert vision.cell_of(880, 300, 1280, 720) == "F3"
    assert vision.cell_of(5000, 5000, 1280, 720) == "H6"


def test_contact_sheet_has_only_the_two_teams():
    scene = people_scene()
    official = scene.detections[0]
    official.is_referee = True
    assert vision.team_contact_sheet(scene) is not None


# ---------------------------------------------------------------------------
# Detector re-run
# ---------------------------------------------------------------------------


def test_rerun_maps_the_box_back_to_the_frame_and_keeps_the_ball():
    scene = people_scene()
    # Cell F3 is (800, 240)-(960, 360). With half a cell of reach, the crop starts at (720, 180)
    # and is enlarged 2x, so a box at crop (240, 180)-(320, 400) is at frame (840, 270)-(880, 380).
    detector = FakeDetector([(240, 180, 320, 400)])
    box = vision.rerun_detector(detector, scene.image, scene.detections, "F3")
    assert box == pytest.approx((840, 270, 880, 380))
    assert detector.shapes[0][:2] == (480, 640)
    assert detector.last_ball == "the frame's ball"


def test_rerun_ignores_someone_who_already_has_a_box():
    scene = people_scene()
    # Frame (900, 300)-(940, 420) is player #4's box.
    detector = FakeDetector([(360, 240, 440, 480)])
    assert vision.rerun_detector(detector, scene.image, scene.detections, "F3") is None


def test_rerun_ignores_something_too_small_to_be_a_person():
    scene = people_scene()
    detector = FakeDetector([(240, 180, 250, 200)])
    assert vision.rerun_detector(detector, scene.image, scene.detections, "F3") is None


def test_rerun_finds_nothing():
    scene = people_scene()
    assert vision.rerun_detector(FakeDetector([]), scene.image, scene.detections, "F3") is None


# ---------------------------------------------------------------------------
# Unusable answers never clear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("check", "task"),
    [
        (vision_checks.check_false_boxes, "false_boxes"),
        (vision_checks.check_missed_players, "players"),
    ],
)
def test_cannot_tell_is_unchecked(check, task):
    found = check(client_for(cannot_tell(task)), people_scene())
    assert [item.severity for item in found] == [f.UNCHECKED]


@pytest.mark.parametrize(
    ("check", "task", "topics"),
    [
        (vision_checks.check_play, "play", ["side", "attacking_team"]),
        (vision_checks.check_keeper, "keeper", ["keeper"]),
    ],
)
def test_cannot_tell_on_a_choice_is_for_a_person_and_not_a_failure(check, task, topics):
    found = check(client_for(cannot_tell(task)), people_scene())
    assert [(item.check, item.code) for item in found] == [(topic, vision_checks.CANT_TELL) for topic in topics]
    assert not any(item.severity == f.UNCHECKED for item in found)


@pytest.mark.parametrize("bad", ["not json", {"task": "play", "status": "answered", "certainty": "clear", "answer": {"attacking_team": "A", "attacked_goal_side": "right", "verdict": "offside"}}])
def test_invalid_reply_is_unchecked(bad):
    found = vision_checks.check_play(client_for(bad, bad), people_scene())
    assert [item.severity for item in found] == [f.UNCHECKED]


def test_service_failure_is_unchecked():
    class Down:
        status_code = 401

    class DownSession:
        def post(self, _url, **_kwargs):
            return Down()

    client = OpenRouterClient(AgentConfig(api_key="k", model="m"), session=DownSession(), sleep=lambda _: None)
    assert [item.severity for item in vision_checks.check_play(client, people_scene())] == [f.UNCHECKED]


def test_no_key_is_unchecked_and_report_is_incomplete():
    client = OpenRouterClient(AgentConfig(), session=Session(), sleep=lambda _: None)
    report = vision_checks.run_vision(client, people_scene())
    assert not report.complete
    assert any(item.severity == f.UNCHECKED for item in report.findings)


@pytest.mark.parametrize(
    ("check", "task", "answer"),
    [
        (vision_checks.check_false_boxes, "false_boxes", {"not_players": [{"id": 40, "what": "board"}]}),
        (vision_checks.check_false_boxes, "false_boxes", {"duplicates": [[0, 40]]}),
        (vision_checks.check_keeper, "keeper", {"state": "in_frame", "keeper_id": 40}),
        (vision_checks.check_play, "play", {"attacking_team": "A", "attacked_goal_side": "right", "ball_holder_id": 40}),
    ],
)
def test_answer_naming_an_unknown_number_is_discarded(check, task, answer):
    found = check(client_for(answered(task, answer)), people_scene())
    assert [item.severity for item in found] == [f.UNCHECKED]
    assert "#40" in found[0].message


def test_kit_answer_naming_an_unknown_number_is_discarded():
    answer = {"kits": {"group_1": "red", "group_2": "blue"}, "mismatches": [{"id": 40, "matches": "group_2"}]}
    found, _kits = vision_checks.check_kits(client_for(answered("teams", answer)), people_scene())
    assert [item.severity for item in found] == [f.UNCHECKED]


def test_grid_cell_off_the_grid_is_discarded():
    answer = {"unnumbered_people": [{"grid_cell": "Z9"}]}
    found = vision_checks.check_missed_players(client_for(answered("players", answer)), people_scene())
    assert [item.severity for item in found] == [f.UNCHECKED]


# ---------------------------------------------------------------------------
# Answers become findings
# ---------------------------------------------------------------------------


def test_missed_player_with_a_box_has_two_kinds_of_evidence(trust_all):
    answer = {"unnumbered_people": [{"grid_cell": "F3", "same_kit_as": 3}]}
    detector = FakeDetector([(240, 180, 320, 400)])
    found = vision_checks.check_missed_players(client_for(answered("players", answer)), people_scene(), detector)
    assert codes(found) == ["missed_player"]
    change = found[0].proposal
    assert (change.kind, change.team) == ("add_player", 2)
    assert change.bbox == pytest.approx((840, 270, 880, 380))
    assert {item.kind for item in found[0].evidence} == {f.VISION, f.DETECTOR_RERUN}


def test_model_never_supplies_a_box(trust_all):
    answer = {"unnumbered_people": [{"grid_cell": "F3", "same_kit_as": 3}]}
    found = vision_checks.check_missed_players(client_for(answered("players", answer)), people_scene(), FakeDetector([]))
    assert codes(found) == ["possible_missed_player"]
    assert found[0].proposal is None
    assert {item.kind for item in found[0].evidence} == {f.VISION}


def test_missed_player_of_unknown_team_can_only_be_tested(trust_all):
    answer = {"unnumbered_people": [{"grid_cell": "F3"}]}
    detector = FakeDetector([(240, 180, 320, 400)])
    found = vision_checks.check_missed_players(client_for(answered("players", answer)), people_scene(), detector)
    assert found[0].proposal is None
    assert {change.team for change in found[0].alternatives} == {1, 2}


def test_only_a_missed_player_with_no_detector_box_is_never_proposed():
    settings = vision_checks.VISION_AUTO_APPLY
    assert settings["players"] is False and settings["players_with_rerun"] is True
    assert all(settings[key] for key in ("false_boxes", "teams", "side", "attacking_team", "keeper", "officials"))


def test_untrusted_check_only_raises_a_doubt(monkeypatch):
    monkeypatch.setitem(vision_checks.VISION_AUTO_APPLY, "officials", False)
    scene = people_scene()
    scene.detections[0].detector_class = "referee"
    answer = {"people": [{"id": 0, "role": "referee"}]}
    found = vision_checks.check_officials(client_for(answered("officials", answer)), scene)
    assert found[0].proposal is None
    assert found[0].alternatives == (Change("set_role", player_id=0, role="referee"),)


def test_kit_mismatch_proposes_the_switch(trust_all):
    answer = {"kits": {"group_1": "red", "group_2": "blue"}, "mismatches": [{"id": 0, "matches": "group_2"}, {"id": 1, "matches": "neither"}]}
    found, kits = vision_checks.check_kits(client_for(answered("teams", answer)), people_scene())
    assert found[0].proposal == Change("set_team", player_id=0, team=2)
    assert found[1].proposal is None
    assert kits == {1: "red", 2: "blue"}


def test_keeper_answer_outside_the_candidates_is_discarded():
    scene = people_scene(side="right")
    found = vision_checks.check_keeper(client_for(answered("keeper", {"state": "in_frame", "keeper_id": 0})), scene)
    assert [item.severity for item in found] == [f.UNCHECKED]


def test_keeper_can_be_not_in_frame(trust_all):
    found = vision_checks.check_keeper(client_for(answered("keeper", {"state": "not_in_frame"})), people_scene(side="right"))
    assert found[0].proposal == Change("set_keeper_not_in_frame")


def test_a_wrong_side_or_team_already_set_gets_corrected(trust_all, monkeypatch):
    monkeypatch.setattr(vision, "box_letters", lambda _scene: {1: "A", 2: "B"})
    answer = {"attacking_team": "B", "attacked_goal_side": "left"}
    found = vision_checks.check_play(client_for(answered("play", answer)), people_scene(side="right", attacking_team=1))
    assert [item.proposal for item in found] == [Change("set_side", side="left"), Change("set_attacking_team", team=2)]


def test_a_right_side_and_team_already_set_change_nothing(monkeypatch):
    monkeypatch.setattr(vision, "box_letters", lambda _scene: {1: "A", 2: "B"})
    answer = {"attacking_team": "A", "attacked_goal_side": "right"}
    assert vision_checks.check_play(client_for(answered("play", answer)), people_scene(side="right", attacking_team=1)) == []


@pytest.mark.parametrize(("letters", "team"), [({1: "A", 2: "B"}, 1), ({1: "B", 2: "A"}, 2)])
def test_the_chosen_letter_is_read_back_as_the_right_team(monkeypatch, letters, team):
    monkeypatch.setattr(vision, "box_letters", lambda _scene: letters)
    answer = {"attacking_team": "A", "attacked_goal_side": "right"}
    found = vision_checks.check_play(client_for(answered("play", answer)), people_scene())
    assert found[1].proposal == Change("set_attacking_team", team=team)


def test_one_part_it_cant_tell_doesnt_lose_the_other(monkeypatch):
    monkeypatch.setattr(vision, "box_letters", lambda _scene: {1: "A", 2: "B"})
    answer = {"attacking_team": "cannot_tell", "attacked_goal_side": "left"}
    found = vision_checks.check_play(client_for(answered("play", answer)), people_scene())
    assert found[0].proposal == Change("set_side", side="left")
    assert (found[1].check, found[1].code) == ("attacking_team", vision_checks.CANT_TELL)


def test_an_unclear_answer_counts_as_not_knowing():
    reply = answered("play", {"attacking_team": "A", "attacked_goal_side": "left"})
    reply["certainty"] = "unclear"
    found = vision_checks.check_play(client_for(reply), people_scene())
    assert {item.code for item in found} == {vision_checks.CANT_TELL}


def test_play_request_never_asks_for_a_verdict_and_describes_each_letters_kit(monkeypatch):
    monkeypatch.setattr(vision, "box_letters", lambda _scene: {1: "B", 2: "A"})
    client = client_for(answered("play", {"attacking_team": "A", "attacked_goal_side": "right"}))
    vision_checks.check_play(client, people_scene(), {1: "red shirts", 2: "blue shirts"})
    messages = client.test_session.bodies[0]["messages"]
    assert "Never give a verdict" in messages[0]["content"]
    text = json.dumps(messages[1]["content"])
    assert "labelled A are one team. They wear: blue shirts" in text
    assert "labelled B are the other team. They wear: red shirts" in text


def test_letters_stay_fixed_for_a_frame_and_vary_between_frames():
    assert vision.box_letters(people_scene()) == vision.box_letters(people_scene())
    seen = set()
    for shift in range(20):
        boxes = tuple(
            PlayerDetection(bbox=(x + shift, 300, x + shift + 40, 420), confidence=0.9, player_id=pid, team_id=1 + pid % 2)
            for pid, x in enumerate(range(100, 700, 100))
        )
        seen.add(vision.box_letters(people_scene(detections=boxes))[1])
    assert seen == {"A", "B"}


def test_play_frame_colours_each_team_and_leaves_the_frame_alone(monkeypatch):
    monkeypatch.setattr(vision, "box_letters", lambda _scene: {1: "A", 2: "B"})
    scene = people_scene()
    before = scene.image.copy()
    drawn = vision.play_frame(scene)
    np.testing.assert_array_equal(scene.image, before)
    team_1 = scene.detections[0]
    team_2 = scene.detections[2]
    key_height = drawn.shape[0] - scene.image.shape[0]
    assert key_height > 0
    middle = 360 + key_height
    assert tuple(drawn[middle, int(team_1.bbox[0])]) == vision.ORANGE
    assert tuple(drawn[middle, int(team_2.bbox[0])]) == vision.CYAN


def test_play_frame_draws_a_likely_keeper_unsorted(monkeypatch):
    monkeypatch.setattr(vision, "box_letters", lambda _scene: {1: "A", 2: "B"})
    scene = people_scene()
    keeper = scene.detections[5]
    drawn = vision.play_frame(scene, unsorted={keeper.player_id})
    middle = 360 + drawn.shape[0] - scene.image.shape[0]
    assert tuple(drawn[middle, int(keeper.bbox[0])]) == vision.WHITE


# ---------------------------------------------------------------------------
# Blind calibration: tested on the actual request body
# ---------------------------------------------------------------------------


def _decode(url: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)


def test_camera_request_holds_no_teams_line_or_verdict():
    image = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    calibration = _calibration()
    client = client_for(answered("camera", {"lines": [], "overall": "good"}))
    assert vision_checks.check_camera(client, image, calibration) == []

    body = client.test_session.bodies[0]
    system, user = body["messages"]
    text = (system["content"] + user["content"][0]["text"]).lower()
    for banned in ("offside", "verdict", "margin", "team", "attacker", "defender", "keeper", "player", "group_"):
        assert banned not in text, banned

    # The picture is exactly the frame plus pitch lines: no boxes, numbers or line.
    overlay, names = vision.pitch_overlay(image, calibration)
    assert names
    assert user["content"][1]["image_url"]["url"] == encode_image(overlay)


def test_camera_check_cannot_be_given_the_scene():
    import inspect

    assert list(inspect.signature(vision_checks.check_camera).parameters) == ["client", "image", "calibration"]
    assert list(inspect.signature(vision.pitch_overlay).parameters) == ["image", "calibration"]


def test_camera_vision_alone_never_changes_the_camera():
    image = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    calibration = _calibration()
    _overlay, names = vision.pitch_overlay(image, calibration)
    answer = {"lines": [{"name": names[0], "fit": "off", "move": "left"}], "overall": "some_off"}
    found = vision_checks.check_camera(client_for(answered("camera", answer)), image, calibration)
    assert [item.severity for item in found] == [f.DOUBT]
    assert found[0].proposal is None


# ---------------------------------------------------------------------------
# With the policy
# ---------------------------------------------------------------------------


def test_a_mistake_vision_finds_alone_is_applied_even_if_it_flips_the_call():
    """Vision says a player is on the wrong team. A person can still undo it."""
    build = Build()
    build.add(88, 1, 0)
    wrong = build.add(90, 1, 1)
    build.add(86, 2, 1)
    build.add(100, 2, 2, keeper=True)
    scene = build.scene()
    switch = Change("set_team", player_id=wrong, team=2)
    decision = policy.decide(scene, switch, (f.Evidence(f.VISION, "Kit matches Team 2."),))
    assert decision.flips_call and decision.apply


def test_unchecked_vision_stops_a_frame_clearing():
    scene = people_scene(side="right", attacking_team=1, keeper_choice="not_in_frame", calibration=_calibration())
    unchecked = f.Finding("keeper", "vision_unchecked", "No usable answer.", f.UNCHECKED)
    review = policy.review(scene, vision_checked=True, reported=[unchecked])
    assert review.ending == policy.NEEDS_ANSWER
    assert any("No usable answer" in question.text for question in review.questions)


def test_reported_proposal_with_two_kinds_is_applied():
    scene = people_scene(side="right", attacking_team=1, keeper_choice="not_in_frame")
    add = Change("add_player", team=2, bbox=(1200.0, 300.0, 1240.0, 420.0))
    finding = f.Finding(
        "players",
        "missed_player",
        "Vision and the detector agree.",
        f.DOUBT,
        evidence=(f.Evidence(f.VISION, "seen"), f.Evidence(f.DETECTOR_RERUN, "found")),
        proposal=add,
    )
    review = policy.review(scene, vision_checked=True, reported=[finding])
    assert [item.change for item in review.applied] == [add]
    assert len(review.scene.detections) == 7


def test_reported_finding_about_a_removed_player_is_dropped():
    scene = people_scene()
    stale = f.Finding("teams", "x", "gone", f.DOUBT, (99,), alternatives=(Change("set_team", player_id=99, team=1),))
    assert policy._still_apply(scene, [stale]) == []


def test_run_vision_makes_one_call_per_check_in_a_fixed_order():
    scene = people_scene()
    answers = [
        answered("teams", {"kits": {"group_1": "red", "group_2": "blue"}}),
        answered("play", {"attacking_team": "A", "attacked_goal_side": "right"}),
        answered("players", {"unnumbered_people": []}),
        answered("false_boxes", {}),
        answered("keeper", {"state": "not_in_frame"}),
    ]
    client = client_for(*answers)
    report = vision_checks.run_vision(client, scene)
    assert client.usage.calls == len(answers)
    assert report.complete


def test_no_image_is_unchecked():
    report = vision_checks.run_vision(client_for(), people_scene(image=None))
    assert not report.complete


# ---------------------------------------------------------------------------
# Gemini as a people detector: boxes on a 0 to 1000 scale (measured, not used by a run)
# ---------------------------------------------------------------------------


class _BoxSession:
    def __init__(self, content: Any) -> None:
        self.content = content
        self.calls = 0

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls += 1
        return Reply(self.content)


def _people_client(reply_people: Any) -> OpenRouterClient:
    content = {"task": "detect", "status": "answered", "certainty": "clear", "answer": {"people": reply_people}, "evidence": []}
    return OpenRouterClient(AgentConfig(api_key="k", model="m"), session=_BoxSession(content), sleep=lambda _: None)


def _ask_people(client: OpenRouterClient) -> Any:
    from var_offside.agent import prompts, schemas

    return client.ask("detect", prompts.SYSTEM_PROMPT, prompts.DETECT_PEOPLE_PROMPT, schemas.PeopleBoxesCheck)


def test_people_boxes_convert_from_the_0_to_1000_scale_to_pixels():
    answer = _ask_people(_people_client([{"box_2d": [100, 250, 400, 500], "kind": "player"}]))
    box = vision.box_2d_to_pixels(answer.answer.people[0].box_2d, 1920, 1080)
    assert box == pytest.approx((480.0, 108.0, 960.0, 432.0))


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([100, 250, 400, 1001], id="above-1000"),
        pytest.param([-1, 250, 400, 500], id="negative"),
        pytest.param([400, 250, 100, 500], id="y-reversed"),
        pytest.param([100, 500, 400, 250], id="x-reversed"),
        pytest.param([100, 250, 400], id="three-values"),
    ],
)
def test_bad_people_boxes_are_rejected_after_one_retry(bad):
    from var_offside.agent.llm import InvalidAnswer

    client = _people_client([{"box_2d": bad, "kind": "player"}])
    with pytest.raises(InvalidAnswer):
        _ask_people(client)
    assert client._session.calls == 2
