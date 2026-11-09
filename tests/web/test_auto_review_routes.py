"""The Auto-review routes on the real session and Analyze code. No key, no network, no models."""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pytest

from var_offside.agent import config as agent_config
from var_offside.agent.config import AgentConfig
from var_offside.calibration.base import CalibrationResult
from var_offside.teams import TeamAssignmentResult
from var_offside.web import jobs, phases, sessions
from tests.web.test_offside_rules import (  # noqa: F401
    _analysis,
    _person,
    _pipeline,
    _png_bytes,
    client,
    isolated_state,
)


def _people(width: int = 2000):
    return [
        _person(800, 1),
        _person(700, 2),
        _person(650, 2),
        _person(1000, 2),
        _person(400, 1),
        _person(300, 1),
        _person(200, 2),
    ]


@pytest.fixture
def app_session(client, monkeypatch):  # noqa: F811
    pipeline = _pipeline()

    def fake_phase1(_pipeline, image):
        # A sharp frame: a blank one with no camera has two signs of being unusable.
        image = np.random.default_rng(1).integers(0, 255, image.shape, dtype=np.uint8)
        result = _analysis(_people(), width=image.shape[1], height=image.shape[0], attacking_team=None, keeper_choice=None)
        result.image = image
        result.team_result = TeamAssignmentResult(
            detections=result.detections,
            embeddings=np.zeros((len(result.detections), 8), dtype=np.float32),
            clustering=None,
            team_colors={1: (255, 0, 0), 2: (0, 0, 255)},
        )
        frame = image.copy()
        return result, {"original": frame, "detection": frame, "teams": frame}

    monkeypatch.setattr(phases, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(phases, "run_phase1", fake_phase1)
    monkeypatch.setattr(phases, "run_calibration", lambda _p, _i: CalibrationResult(method="auto_failed"))
    # No key: these tests run with vision off, whatever is on this machine.
    monkeypatch.setattr(agent_config, "load_config", lambda *a, **k: AgentConfig())

    uploaded = client.post("/api/sessions", files={"file": ("frame.png", _png_bytes(), "image/png")})
    assert uploaded.status_code == 200, uploaded.text
    session_id = uploaded.json()["id"]
    assert client.post(f"/api/sessions/{session_id}/detect").status_code == 200
    return session_id


def _follow(client, session_id: str, wanted=("needs_answer", "done", "cancelled", "error"), since: int = 0) -> Dict[str, Any]:  # noqa: F811
    end = time.time() + 10
    while time.time() < end:
        body = client.get(f"/api/sessions/{session_id}/auto-review?since={since}").json()
        if body["status"] in wanted:
            return body
        time.sleep(0.02)
    raise AssertionError(f"Run stayed {body['status']}: {body['events'][-1]}")


def _question(client, session_id: str) -> Dict[str, Any]:  # noqa: F811
    events = _follow(client, session_id, ("needs_answer",))["events"]
    return [event for event in events if event["type"] == "question_asked"][-1]


def _answer(client, session_id: str, choice: str):  # noqa: F811
    question = _question(client, session_id)
    return client.post(f"/api/sessions/{session_id}/auto-review/answers/{question['question_id']}", json={"choice": choice})


def test_a_run_needs_detect_first(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(agent_config, "load_config", lambda *a, **k: AgentConfig())
    uploaded = client.post("/api/sessions", files={"file": ("frame.png", _png_bytes(), "image/png")}).json()
    response = client.post(f"/api/sessions/{uploaded['id']}/auto-review")
    assert response.status_code == 400
    assert "detection" in response.json()["detail"].lower()


def test_status_before_any_run_is_404(client):  # noqa: F811
    uploaded = client.post("/api/sessions", files={"file": ("frame.png", _png_bytes(), "image/png")}).json()
    assert client.get(f"/api/sessions/{uploaded['id']}/auto-review").status_code == 404


def test_unknown_session_is_404(client):  # noqa: F811
    assert client.post("/api/sessions/nope/auto-review").status_code == 404


def test_start_returns_a_job_and_the_detected_players(client, app_session):  # noqa: F811
    response = client.post(f"/api/sessions/{app_session}/auto-review")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"]
    assert len(body["phase1"]["players"]) == 7
    assert body["vision"] is False
    _follow(client, app_session)


def test_a_blank_frame_with_no_camera_is_not_usable(client, app_session, monkeypatch):  # noqa: F811
    session = sessions.store.get_optional(app_session)
    session.phase1_result.image = np.zeros_like(session.phase1_result.image)
    client.post(f"/api/sessions/{app_session}/auto-review")
    result = _follow(client, app_session, ("done",))["result"]
    assert result["ending"] == "not_usable"
    assert result["verdict"] is None


def test_vision_off_never_clears_and_reaches_review(client, app_session):  # noqa: F811
    client.post(f"/api/sessions/{app_session}/auto-review")
    assert _answer(client, app_session, "Right").status_code == 200
    assert _question(client, app_session)["text"] == "Which team is attacking?"
    assert _answer(client, app_session, "Team 1").status_code == 200
    question = _question(client, app_session)
    assert question["choices"][-1] == "Not in frame"
    assert _answer(client, app_session, "Not in frame").status_code == 200

    body = _follow(client, app_session, ("done",))
    result = body["result"]
    assert result["ending"] == "needs_answer"
    assert result["verdict"] is None
    assert (result["attacking_side"], result["attacking_team"], result["keeper"]) == ("right", 1, "not_in_frame")
    assert body["events"][-1]["type"] == "run_ended"


def test_events_can_be_read_from_a_position(client, app_session):  # noqa: F811
    client.post(f"/api/sessions/{app_session}/auto-review")
    everything = _follow(client, app_session)["events"]
    later = client.get(f"/api/sessions/{app_session}/auto-review?since=1").json()["events"]
    assert later[0]["seq"] == 2
    assert len(later) == len(everything) - 1


def test_a_stale_answer_is_409(client, app_session):  # noqa: F811
    client.post(f"/api/sessions/{app_session}/auto-review")
    question = _question(client, app_session)
    session = sessions.store.get_optional(app_session)
    session.bump_revision()
    response = client.post(
        f"/api/sessions/{app_session}/auto-review/answers/{question['question_id']}", json={"choice": "Right"}
    )
    assert response.status_code == 409
    assert "changed" in response.json()["detail"]


def test_a_wrong_answer_is_409(client, app_session):  # noqa: F811
    client.post(f"/api/sessions/{app_session}/auto-review")
    question = _question(client, app_session)
    response = client.post(
        f"/api/sessions/{app_session}/auto-review/answers/{question['question_id']}", json={"choice": "Sideways"}
    )
    assert response.status_code == 409


def _finished_with_choices(client, session_id: str) -> Dict[str, Any]:  # noqa: F811
    client.post(f"/api/sessions/{session_id}/auto-review")
    _answer(client, session_id, "Right")
    _answer(client, session_id, "Team 1")
    _answer(client, session_id, "Not in frame")
    return _follow(client, session_id, ("done",))


def test_undo_of_an_answer_reverts_it(client, app_session):  # noqa: F811
    body = _finished_with_choices(client, app_session)
    side = next(event for event in body["events"] if event["type"] == "side_set")
    response = client.post(f"/api/sessions/{app_session}/auto-review/decisions/{side['undo_id']}/undo")
    assert response.status_code == 200, response.text
    assert response.json()["result"]["attacking_side"] is None


def test_undo_after_a_player_edit_is_409(client, app_session):  # noqa: F811
    body = _finished_with_choices(client, app_session)
    team = next(event for event in body["events"] if event["type"] == "attacking_team_set")
    players = body["result"]["players"]
    assert client.put(f"/api/sessions/{app_session}/players", json={"players": players}).status_code == 200
    response = client.post(f"/api/sessions/{app_session}/auto-review/decisions/{team['undo_id']}/undo")
    assert response.status_code == 409
    assert "changed" in response.json()["detail"]


def test_undo_while_running_is_409(client, app_session):  # noqa: F811
    client.post(f"/api/sessions/{app_session}/auto-review")
    _question(client, app_session)
    assert client.post(f"/api/sessions/{app_session}/auto-review/decisions/d1/undo").status_code == 409


def test_cancelling_the_job_stops_the_run(client, app_session):  # noqa: F811
    job_id = client.post(f"/api/sessions/{app_session}/auto-review").json()["job_id"]
    _question(client, app_session)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    body = _follow(client, app_session, ("cancelled",))
    assert body["result"] is None
    assert jobs.manager.get(job_id).status == "cancelled"


def test_a_new_run_replaces_the_old_one(client, app_session):  # noqa: F811
    client.post(f"/api/sessions/{app_session}/auto-review")
    session = sessions.store.get_optional(app_session)
    _question(client, app_session)
    first = session.auto_review
    client.post(f"/api/sessions/{app_session}/auto-review")
    assert session.auto_review is not first
    time.sleep(0.5)
    assert first.status == "cancelled"
    session.auto_review.stop()


def test_the_report_record_is_kept_for_export(client, app_session):  # noqa: F811
    _finished_with_choices(client, app_session)
    session = sessions.store.get_optional(app_session)
    record = session.auto_review.record
    assert record["review_status"] == "needs_answer"
    assert [item["decided_by"] for item in record["decisions"]] == ["you", "you", "you"]


def test_with_the_switch_off_analyze_is_untouched(client, app_session):  # noqa: F811
    response = client.post(
        f"/api/sessions/{app_session}/analyze",
        json={"attacking_side": "right", "attacking_team": 1, "keeper": "not_in_frame"},
    )
    assert response.status_code == 200
    assert "ending" not in response.json()
    assert sessions.store.get_optional(app_session).auto_review is None
