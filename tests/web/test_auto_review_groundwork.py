"""App groundwork for Auto-review: detector class, ball, stable ids, revision, tasks."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from var_offside.calibration import pitch_solver as ps
from var_offside.calibration.base import CalibrationResult
from var_offside.detection import PlayerDetector
from var_offside.detection.player_detector import BallDetection
from var_offside.pipeline import AnalysisResult
from var_offside.teams import TeamAssignmentResult
from var_offside.web import jobs, phases, sessions
from var_offside.web.server import app
from tests.web.test_offside_rules import (
    _DetectionModel,
    _person,
    _phase2,
    _pipeline,
    _png_bytes,
)

W, H = 1920, 1080


# ---------------------------------------------------------------------------
# Detector class and ball
# ---------------------------------------------------------------------------


def _detector(rows, limit=30) -> PlayerDetector:
    detector = PlayerDetector()
    detector.max_players = limit
    detector._detection_model = _DetectionModel(rows)
    return detector


ROWS = [
    [10, 20, 110, 200, 0.99, 0],  # Tall ball
    [200, 20, 260, 200, 0.90, 1],  # Goalkeeper
    [300, 20, 360, 200, 0.80, 2],  # Player
    [400, 20, 460, 200, 0.70, 3],  # Referee
]


def _detect(rows, limit=30):
    detector = _detector(rows, limit)
    people = detector.detect(np.zeros((500, 600, 3), dtype=np.uint8), use_pose=False)
    return detector, people


def test_detector_class_is_kept_as_evidence_and_sets_no_role():
    _detector_used, people = _detect(ROWS)
    assert [det.detector_class for det in people] == ["goalkeeper", "player", "referee"]
    assert not any(det.is_goalkeeper for det in people)
    assert not any(getattr(det, "is_referee", False) for det in people)


def test_ball_position_is_kept_and_the_ball_is_not_a_player():
    detector, people = _detect(ROWS)
    assert len(people) == 3
    assert isinstance(detector.last_ball, BallDetection)
    assert detector.last_ball.center == pytest.approx((60.0, 110.0))
    assert detector.last_ball.confidence == pytest.approx(0.99, abs=1e-5)


def test_short_ball_is_still_kept_though_people_that_short_are_filtered():
    detector, _people = _detect([[10, 20, 14, 24, 0.9, 0], *ROWS[1:]])
    assert detector.last_ball is not None


def test_most_confident_ball_wins_and_a_new_detect_forgets_the_old_one():
    detector, _people = _detect([[10, 20, 20, 30, 0.5, 0], [50, 20, 60, 30, 0.8, 0], *ROWS[1:]])
    assert detector.last_ball.center == pytest.approx((55.0, 25.0))

    detector._detection_model = _DetectionModel(ROWS[1:])
    detector.detect(np.zeros((500, 600, 3), dtype=np.uint8), use_pose=False)
    assert detector.last_ball is None


def test_ball_after_the_player_limit_is_still_seen_and_the_limit_still_holds():
    rows = [*ROWS[1:], [10, 20, 20, 30, 0.2, 0]]
    detector, people = _detect(rows, limit=2)
    assert len(people) == 2
    assert detector.last_ball is not None


def test_phase1_keeps_the_ball_on_the_result():
    pipeline = _pipeline()
    pipeline.player_detector.detect.return_value = [_person(100, None), _person(300, None)]
    pipeline.player_detector.last_ball = BallDetection((1.0, 2.0, 3.0, 4.0), 0.9)
    pipeline.team_assigner.assign_teams.return_value = None
    result, _stages = phases.run_phase1(pipeline, np.zeros((500, 1000, 3), dtype=np.uint8))
    assert result.ball.center == (2.0, 3.0)


# ---------------------------------------------------------------------------
# Stable ids
# ---------------------------------------------------------------------------


def _result(count: int = 4) -> AnalysisResult:
    people = [_person(100 + 100 * index, 1 + index % 2) for index in range(count)]
    for index, detection in enumerate(people):
        detection.player_id = index
    embeddings = np.array([[float(index + 1), 0.0] for index in range(count)])
    return AnalysisResult(
        image=np.zeros((500, 1000, 3), dtype=np.uint8),
        detections=people,
        next_player_id=count,
        team_result=TeamAssignmentResult(
            detections=people, embeddings=embeddings, clustering=None, team_colors={}
        ),
    )


def _rows(result: AnalysisResult):
    return phases.serialize_players(result)


def test_phase1_numbers_players_by_list_position():
    pipeline = _pipeline()
    people = [_person(100, None), _person(300, None)]
    people[0].player_id, people[1].player_id = 7, 3
    pipeline.player_detector.detect.return_value = people
    pipeline.team_assigner.assign_teams.return_value = None
    result, _stages = phases.run_phase1(pipeline, np.zeros((500, 1000, 3), dtype=np.uint8))
    assert [row["id"] for row in _rows(result)] == [0, 1]
    assert result.next_player_id == 2


def test_removing_a_player_keeps_the_other_ids():
    result = _result()
    rows = [row for row in _rows(result) if row["id"] != 1]
    phases.apply_player_edits(_pipeline(), result, rows)
    assert [row["id"] for row in _rows(result)] == [0, 2, 3]


def test_edit_finds_the_right_player_by_id_after_a_removal():
    result = _result()
    result.detections[3].detector_class = "referee"
    keypoints = np.ones((17, 3))
    result.detections[3].keypoints = keypoints
    phases.apply_player_edits(_pipeline(), result, [r for r in _rows(result) if r["id"] != 1])
    # #3 now sits at list position 2. Its own pose and class must still be found.
    phases.apply_player_edits(_pipeline(), result, _rows(result))
    moved = next(det for det in result.detections if det.player_id == 3)
    assert moved.detector_class == "referee"
    assert moved.keypoints is keypoints


def test_added_players_get_new_ids_that_are_never_reused():
    result = _result()
    rows = _rows(result) + [{"id": None, "bbox": [10, 10, 40, 90], "team": 1, "role": "player"}]
    phases.apply_player_edits(_pipeline(), result, rows)
    assert _rows(result)[-1]["id"] == 4

    phases.apply_player_edits(_pipeline(), result, [r for r in _rows(result) if r["id"] != 4])
    rows = _rows(result) + [{"id": None, "bbox": [10, 10, 40, 90], "team": 1, "role": "player"}]
    phases.apply_player_edits(_pipeline(), result, rows)
    assert _rows(result)[-1]["id"] == 5


@pytest.mark.parametrize("bad_id", [99, True, "1"])
def test_unknown_id_is_treated_as_a_new_player(bad_id):
    result = _result()
    rows = _rows(result)[:2] + [{"id": bad_id, "bbox": [10, 10, 40, 90], "team": 1, "role": "player"}]
    phases.apply_player_edits(_pipeline(), result, rows)
    assert [row["id"] for row in _rows(result)] == [0, 1, 4]


def test_a_repeated_id_does_not_give_two_players_the_same_id():
    result = _result()
    first = _rows(result)[0]
    phases.apply_player_edits(_pipeline(), result, [first, dict(first)])
    assert [row["id"] for row in _rows(result)] == [0, 4]


def test_verdict_reports_stable_ids_after_a_removal():
    people = [_person(800, 1), _person(50, 2), _person(700, 2), _person(650, 2)]
    for index, detection in enumerate(people):
        detection.player_id = index
    people.pop(1)
    result = AnalysisResult(
        image=np.zeros((500, 2000, 3), dtype=np.uint8),
        detections=people,
        attacking_team=1,
        keeper_choice="not_in_frame",
        calibration=CalibrationResult(method="auto_failed"),
    )
    payload = _phase2(_pipeline(), result)
    assert payload["verdict"] == "offside"
    assert payload["last_defender"]["id"] == 2
    assert payload["offside_player_ids"] == [0]


def test_embeddings_follow_their_players_through_edits():
    result = _result()
    rows = [row for row in _rows(result) if row["id"] != 1]
    rows.append({"id": None, "bbox": [10, 10, 40, 90], "team": 1, "role": "player"})
    phases.apply_player_edits(_pipeline(), result, rows)
    np.testing.assert_array_equal(
        result.team_result.embeddings[:, 0], [1.0, 3.0, 4.0, 0.0]
    )


def test_players_without_ids_still_use_their_list_position():
    people = [_person(100, 1), _person(200, 2)]
    result = AnalysisResult(image=np.zeros((500, 1000, 3), dtype=np.uint8), detections=people)
    assert [row["id"] for row in _rows(result)] == [0, 1]


# ---------------------------------------------------------------------------
# Scene revision
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    store = sessions.SessionStore()
    monkeypatch.setattr(sessions, "store", store)
    monkeypatch.setattr(phases, "get_pipeline", lambda: _pipeline())

    def fake_phase1(_pipeline_used, image):
        result = AnalysisResult(
            image=image,
            detections=[_person(800, 1), _person(700, 2), _person(650, 2)],
        )
        result.next_player_id = 3
        return result, {"original": image, "detection": image, "teams": image}

    monkeypatch.setattr(phases, "run_phase1", fake_phase1)
    monkeypatch.setattr(
        phases, "run_calibration", lambda _pipeline_used, _image: CalibrationResult(method="auto_failed")
    )
    with TestClient(app) as test_client:
        yield test_client
    store.clear()


def _session(client) -> str:
    response = client.post(
        "/api/sessions", files={"file": ("frame.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_revision_goes_up_when_the_scene_changes_and_only_then(client):
    session_id = _session(client)
    session = sessions.store.get(session_id)
    assert session.revision == 0

    players = client.post(f"/api/sessions/{session_id}/detect").json()["players"]
    assert session.revision == 1

    assert client.put(f"/api/sessions/{session_id}/players", json={"players": players}).status_code == 200
    assert session.revision == 2

    choices = {"attacking_side": "right", "attacking_team": 1, "keeper": "not_in_frame"}
    assert client.post(f"/api/sessions/{session_id}/analyze", json=choices).status_code == 200
    assert session.revision == 3

    # Analyzing the same scene again changes nothing.
    assert client.post(f"/api/sessions/{session_id}/analyze", json=choices).status_code == 200
    assert session.revision == 3

    changed = {**choices, "attacking_team": 2}
    assert client.post(f"/api/sessions/{session_id}/analyze", json=changed).status_code == 200
    assert session.revision == 4

    assert client.delete(f"/api/sessions/{session_id}/calibration").status_code == 200
    assert session.revision == 5


def test_revision_is_not_added_to_existing_response_shapes(client):
    session_id = _session(client)
    payload = client.post(f"/api/sessions/{session_id}/detect").json()
    assert "revision" not in payload


def test_selecting_a_new_frame_bumps_the_revision():
    session = sessions.Session(id="x", filename="a.mp4", kind="video", path="a.mp4", frame_count=5)
    session.select_frame(0)
    assert session.revision == 0
    session.select_frame(2)
    assert session.revision == 1


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def _wait(job, status):
    for _attempt in range(200):
        if job.status == status:
            return
        threading.Event().wait(0.01)
    raise AssertionError(f"job stayed {job.status}")


def test_task_runs_off_the_render_thread_and_reports_progress():
    manager = jobs.JobManager()
    seen = {}

    def work(job):
        seen["thread"] = threading.current_thread().name
        manager.report(job, 40, "Checking")
        seen["progress"] = job.progress

    job = manager.start_task("auto_review", work)
    _wait(job, "done")
    assert seen["progress"] == 40
    assert "var-task-auto_review" in seen["thread"]
    assert job.to_dict() == {"status": "done", "progress": 100, "message": "Done", "file_url": None}


def test_task_failure_is_reported():
    manager = jobs.JobManager()

    def work(_job):
        raise RuntimeError("broken")

    job = manager.start_task("auto_review", work)
    _wait(job, "error")
    assert job.message == "broken"


def test_task_can_be_cancelled():
    manager = jobs.JobManager()
    started = threading.Event()

    def work(job):
        started.set()
        job.cancel_event.wait(5)

    job = manager.start_task("auto_review", work)
    assert started.wait(5)
    manager.cancel(job.id)
    _wait(job, "cancelled")
    assert job.status == "cancelled"


def test_unknown_task_kind_is_refused():
    with pytest.raises(ValueError):
        jobs.JobManager().start_task("image", lambda job: None)


def test_export_route_still_refuses_a_run_kind():
    with pytest.raises(ValueError):
        jobs.JobManager().start("auto_review", AnalysisResult())


# ---------------------------------------------------------------------------
# Attack direction warning
# ---------------------------------------------------------------------------


def _camera(cy: float, target_y: float) -> ps.Camera:
    position = np.array([8.0, cy, -22.0], dtype=np.float64)
    target = np.array([32.0, target_y, 0.0])
    return ps.Camera(1900.0, ps._look_at(position, target), position, W / 2.0, H / 2.0)


def test_end_warning_on_a_normal_camera():
    camera = _camera(75.0, 2.0)
    assert ps._side_warning(camera, W, H, "right") is None
    assert "far end" in ps._side_warning(camera, W, H, "left")


def test_reverse_angle_camera_is_read_the_other_way_round():
    camera = _camera(-75.0, -2.0)
    # This camera shows the same end, but pitch metres run leftward on screen.
    assert ps._side_warning(camera, W, H, "left") is None
    assert "far end" in ps._side_warning(camera, W, H, "right")


def test_no_side_gives_no_warning():
    assert ps._side_warning(_camera(75.0, 2.0), W, H, None) is None
