"""Tests for the Calibrate step's routes: everything heavy is faked."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from var_offside.calibration import pitch_solver as ps
from var_offside.calibration.base import CalibrationResult
from var_offside.config import get_settings
from var_offside.web import jobs, phases, sessions
from var_offside.web.server import app
from tests.web.test_server import _fake_apply_edits, _fake_phase1

# A realistic size: a 2 px placement error means little here.
WIDTH, HEIGHT = 1280, 720


def _true_camera() -> ps.Camera:
    position = np.array([8.0, 75.0, -22.0])
    return ps.Camera(
        1266.0,
        ps._look_at(position, np.array([32.0, 2.0, 0.0])),
        position,
        WIDTH / 2.0,
        HEIGHT / 2.0,
    )


def _auto_result() -> CalibrationResult:
    """A slightly wrong full-3D automatic result, shaped like PnLCalib's."""
    truth = _true_camera()
    values = truth.params().copy()
    values[0] += 0.02
    values[4:7] += np.array([0.5, -0.4, 0.3])
    camera = truth.with_params(values)
    result = ps.to_calibration_result(camera, [], WIDTH, HEIGHT, "auto")
    result.method = "pnl"
    result.quality_score = 0.9
    result.debug_info = {}
    return result


def _pins(ids: List[str]) -> List[Dict[str, Any]]:
    truth = _true_camera()
    rows = []
    for lid in ids:
        mark = ps._LANDMARK_BY_ID[lid]
        h = ps._project_h(
            truth.params(), truth, ps._field_to_world(np.array([[mark.x, mark.y]]))
        )
        rows.append(
            {"kind": "point", "ref": lid, "x": float(h[0, 0] / h[0, 2]), "y": float(h[0, 1] / h[0, 2])}
        )
    return rows


ANALYZE = {"attacking_side": "right", "attacking_team": 1, "keeper": "player"}
FIVE = ["r_box_front_far", "r_box_front_near", "r_six_front_far", "r_six_front_near", "r_spot"]


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    store = sessions.SessionStore()
    monkeypatch.setattr(sessions, "store", store)
    monkeypatch.setattr(jobs, "manager", jobs.JobManager())
    monkeypatch.setattr(get_settings(), "output_dir", tmp_path, raising=False)
    monkeypatch.setattr(phases, "get_pipeline", lambda: object())
    monkeypatch.setattr(phases, "run_phase1", _fake_phase1)
    monkeypatch.setattr(phases, "apply_player_edits", _fake_apply_edits)
    yield
    store.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def seen(monkeypatch) -> Dict[str, Any]:
    """Record the calibration Analyze hands to phase 2."""
    record: Dict[str, Any] = {"calls": 0, "calibration": None}

    def fake_phase2(pipeline, result, side, exclude_goalkeeper=True):
        record["calls"] += 1
        record["calibration"] = result.calibration
        return result, {}

    monkeypatch.setattr(phases, "run_phase2", fake_phase2)
    return record


def _session(client) -> str:
    ok, buffer = cv2.imencode(".png", np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    assert ok
    response = client.post(
        "/api/sessions", files={"file": ("frame.png", buffer.tobytes(), "image/png")}
    )
    assert response.status_code == 200
    return response.json()["id"]


def _ready(client, session_id: str) -> Dict[str, Any]:
    deadline = time.time() + 5
    while time.time() < deadline:
        payload = client.get(f"/api/sessions/{session_id}/calibration").json()
        if payload["status"] == "ready":
            return payload
        time.sleep(0.01)
    raise AssertionError("calibration never became ready")


def test_detect_starts_calibration_once(client, monkeypatch):
    calls = []

    def fake(pipeline, image):
        calls.append(1)
        return _auto_result()

    monkeypatch.setattr(phases, "run_calibration", fake)
    sid = _session(client)
    assert client.post(f"/api/sessions/{sid}/detect").status_code == 200
    payload = _ready(client, sid)
    assert len(calls) == 1
    assert payload["mode"] == "edit" and payload["source"] == "auto"
    assert payload["method"] == "pnl" and payload["full_3d"] is True
    assert payload["fit"]["label"] == "AUTO FIT" and payload["fit"]["can_continue"]
    assert len(payload["field_to_image"]) == 3
    assert {row["id"] for row in payload["pitch"]["landmarks"]} >= {"r_spot", "c_spot"}


def test_screen_shows_running_until_the_result_arrives(client, monkeypatch):
    import threading

    release = threading.Event()

    def slow(pipeline, image):
        release.wait(5)
        return _auto_result()

    monkeypatch.setattr(phases, "run_calibration", slow)
    sid = _session(client)
    first = client.get(f"/api/sessions/{sid}/calibration").json()
    assert first["status"] == "running"
    assert first["fit"]["label"] == "FINDING THE PITCH"
    assert first["fit"]["can_continue"] is False
    release.set()
    assert _ready(client, sid)["fit"]["label"] == "AUTO FIT"


def test_failed_automatic_result_opens_hand_placement(client, monkeypatch):
    def boom(pipeline, image):
        raise RuntimeError("model missing")

    monkeypatch.setattr(phases, "run_calibration", boom)
    sid = _session(client)
    payload = _ready(client, sid)
    assert payload["mode"] == "hand"
    assert payload["method"] == "failed"
    assert payload["field_to_image"] is None
    assert payload["fit"]["label"] == "NOT FOUND"


def test_solve_changes_nothing(client, monkeypatch, seen):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: _auto_result())
    sid = _session(client)
    before = _ready(client, sid)
    trial = client.post(
        f"/api/sessions/{sid}/calibration/solve",
        json={"mode": "edit", "pins": _pins(FIVE)},
    )
    assert trial.status_code == 200
    body = trial.json()
    assert body["fit"]["label"] == "ADJUSTED"
    assert body["field_to_image"] != before["field_to_image"]
    assert _ready(client, sid)["field_to_image"] == before["field_to_image"]


def test_solve_rejects_unknown_markings(client, monkeypatch):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: _auto_result())
    sid = _session(client)
    _ready(client, sid)
    bad = client.post(
        f"/api/sessions/{sid}/calibration/solve",
        json={"mode": "edit", "pins": [{"kind": "point", "ref": "nope", "x": 1, "y": 1}]},
    )
    assert bad.status_code == 400
    assert "nope" in bad.json()["detail"]


def test_hand_mode_counts_what_is_still_needed(client, monkeypatch):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: _auto_result())
    sid = _session(client)
    _ready(client, sid)
    body = client.post(
        f"/api/sessions/{sid}/calibration/solve",
        json={"mode": "hand", "pins": _pins(FIVE[:1])},
    ).json()
    assert body["mode"] == "hand"
    assert body["field_to_image"] is None
    assert body["fit"]["needed"] == 3


def test_apply_reaches_analyze_and_survives_player_edits(client, monkeypatch, seen):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: _auto_result())
    sid = _session(client)
    phase1 = client.post(f"/api/sessions/{sid}/detect").json()
    _ready(client, sid)

    applied = client.put(
        f"/api/sessions/{sid}/calibration", json={"mode": "edit", "pins": _pins(FIVE)}
    )
    assert applied.status_code == 200
    assert applied.json()["source"] == "edited"
    assert applied.json()["method"] == "manual"

    # A player edit must not lose the calibration.
    assert client.put(
        f"/api/sessions/{sid}/players", json={"players": phase1["players"]}
    ).status_code == 200
    payload = _ready(client, sid)
    assert payload["source"] == "edited" and len(payload["pins"]) == 5

    client.post(f"/api/sessions/{sid}/analyze", json=ANALYZE)
    calibration = seen["calibration"]
    assert calibration.method == "manual"
    assert calibration.debug_info["source"] == "edited"
    assert calibration.has_3d_calibration


def test_hand_result_is_labelled_hand(client, monkeypatch, seen):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: CalibrationResult(method="auto_failed"))
    sid = _session(client)
    client.post(f"/api/sessions/{sid}/detect")
    _ready(client, sid)
    applied = client.put(
        f"/api/sessions/{sid}/calibration", json={"mode": "hand", "pins": _pins(FIVE)}
    ).json()
    assert applied["source"] == "hand" and applied["full_3d"] is True
    assert applied["mode"] == "hand"
    client.post(f"/api/sessions/{sid}/analyze", json=ANALYZE)
    assert seen["calibration"].debug_info["source"] == "hand"


def test_refused_fit_is_not_applied(client, monkeypatch):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: CalibrationResult(method="auto_failed"))
    sid = _session(client)
    _ready(client, sid)
    too_few = client.put(
        f"/api/sessions/{sid}/calibration", json={"mode": "hand", "pins": _pins(FIVE[:2])}
    )
    assert too_few.status_code == 400
    assert _ready(client, sid)["source"] == "auto"


def test_delete_goes_back_to_auto(client, monkeypatch, seen):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: _auto_result())
    sid = _session(client)
    client.post(f"/api/sessions/{sid}/detect")
    _ready(client, sid)
    client.put(f"/api/sessions/{sid}/calibration", json={"mode": "edit", "pins": _pins(FIVE)})
    reset = client.delete(f"/api/sessions/{sid}/calibration")
    assert reset.status_code == 200
    assert reset.json()["source"] == "auto" and reset.json()["pins"] == []
    client.post(f"/api/sessions/{sid}/analyze", json=ANALYZE)
    assert seen["calibration"].method == "pnl"


def test_apply_clears_the_old_verdict(client, monkeypatch, seen):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: _auto_result())
    sid = _session(client)
    client.post(f"/api/sessions/{sid}/detect")
    _ready(client, sid)
    client.post(f"/api/sessions/{sid}/analyze", json=ANALYZE)
    session = sessions.store.get(sid)
    assert session.phase2_result is not None
    client.put(f"/api/sessions/{sid}/calibration", json={"mode": "edit", "pins": _pins(FIVE)})
    assert session.phase2_result is None


def test_a_new_frame_starts_from_auto(client, monkeypatch):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: _auto_result())
    sid = _session(client)
    session = sessions.store.get(sid)
    session.kind = "video"
    session.frame_count = 5
    session.reader = type("R", (), {"read": lambda s, i: session.image, "release": lambda s: None})()
    _ready(client, sid)
    client.put(f"/api/sessions/{sid}/calibration", json={"mode": "edit", "pins": _pins(FIVE)})
    assert _ready(client, sid)["source"] == "edited"
    # The same frame keeps the edit, another frame doesn't.
    assert client.post(f"/api/sessions/{sid}/select-frame", json={"index": 0}).status_code == 200
    assert _ready(client, sid)["source"] == "edited"
    assert client.post(f"/api/sessions/{sid}/select-frame", json={"index": 3}).status_code == 200
    assert _ready(client, sid)["source"] == "auto"


def test_verdict_reports_a_hand_calibration():
    result = phases.AnalysisResult()
    result.image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    camera = _true_camera()
    pins = ps.parse_pins(_pins(FIVE))
    result.calibration = ps.to_calibration_result(camera, pins, WIDTH, HEIGHT, "edited")
    verdict = phases.serialize_verdict(result)
    assert verdict["calibration"] == {"method": "manual", "full_3d": True, "edited": True}
    row = next(check for check in verdict["checks"] if check["key"] == "calibration")
    assert row["detail"] == "edited by hand, full 3D"
    assert verdict["line_field_x"] is None


def test_report_records_auto_and_final(client, monkeypatch, seen):
    monkeypatch.setattr(phases, "run_calibration", lambda p, i: _auto_result())
    sid = _session(client)
    client.post(f"/api/sessions/{sid}/detect")
    _ready(client, sid)
    client.put(f"/api/sessions/{sid}/calibration", json={"mode": "edit", "pins": _pins(FIVE)})
    report = sessions.store.get(sid).calibration.report()
    json.dumps(report)  # must be plain JSON
    assert report["source"] == "edited"
    assert report["automatic"]["method"] == "pnl"
    assert report["final"]["method"] == "manual"
    assert len(report["pins"]) == 5


def test_an_impossible_automatic_camera_opens_hand_placement(client, monkeypatch):
    """PnLCalib can score an impossible camera highly: seen on a real frame."""

    def impossible(pipeline, image):
        position = np.array([-63.0, -21.0, -3.0])
        camera = ps.Camera(98.0, ps._look_at(position, np.array([0.0, 0.0, 0.0])), position, WIDTH / 2.0, HEIGHT / 2.0)
        result = ps.to_calibration_result(camera, [], WIDTH, HEIGHT, "auto")
        result.method = "pnl"
        result.quality_score = 0.8
        return result

    monkeypatch.setattr(phases, "run_calibration", impossible)
    sid = _session(client)
    payload = _ready(client, sid)
    assert payload["fit"]["label"] == "NOT BELIEVABLE"
    assert payload["fit"]["can_continue"] is False
    assert payload["mode"] == "hand"
    assert payload["auto_3d"] is False
    assert payload["field_to_image"] is None


def test_team_colours_are_fixed_red_and_blue():
    result = phases.AnalysisResult()
    result.image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    assert phases._team_colors(result) == {"1": "#ff3347", "2": "#2e7dff"}


def test_player_markers_draw_rings_not_boxes():
    from var_offside.detection import PlayerDetection
    from var_offside.web.overlays import TEAM_BGR, draw_player_markers

    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    players = [
        PlayerDetection(bbox=(100.0, 200.0, 160.0, 380.0), confidence=1.0, team_id=1),
        PlayerDetection(bbox=(400.0, 200.0, 460.0, 380.0), confidence=1.0, team_id=2, is_goalkeeper=True),
    ]
    out = draw_player_markers(image, players)
    assert image.sum() == 0, "the input image must not be changed"

    def has_colour(region, bgr):
        return bool(np.any(np.all(np.abs(region.astype(int) - np.array(bgr)) < 12, axis=-1)))

    # A ring at the feet, in the team colour.
    assert has_colour(out[370:395, 80:180], TEAM_BGR[1])
    assert has_colour(out[370:395, 380:480], TEAM_BGR[2])
    # No box edge up the side of the player.
    assert out[250:340, 100:102].sum() == 0
    # A label above the head.
    assert has_colour(out[170:200, 100:160], TEAM_BGR[1])
