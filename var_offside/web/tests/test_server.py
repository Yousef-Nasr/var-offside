"""
Tests for the browser app's FastAPI server.

Everything heavy is faked: the pipeline getter is replaced and the phase
functions are monkeypatched, so these tests run without models, a GPU or network
access.  The live 3D ``LiveScene`` is injected through ``sys.modules``.
"""

from __future__ import annotations

import json
import sys
import time
import types
import zipfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from var_offside.calibration.base import CalibrationResult
from var_offside.config import get_settings
from var_offside.detection import PlayerDetection
from var_offside.offside.decision import OffsideResult, PlayerOffsideStatus
from var_offside.offside.line_calculator import OffsideLine
from var_offside.pipeline import AnalysisResult
from var_offside.teams import TeamAssignmentResult
from var_offside.web import jobs, phases, sessions
from var_offside.web.server import app

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
CAMERA = {
    "target": [0.0, 0.0, 0.0],
    "yaw": 35.0,
    "pitch": 22.0,
    "distance": 18.0,
    "fov": 40.0,
}
OPTIONS = {"line_plane": True, "photo": False, "labels": True, "pulse": True}


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """Give every test a fresh session store, job manager and output dir."""
    fresh_store = sessions.SessionStore()
    monkeypatch.setattr(sessions, "store", fresh_store)
    monkeypatch.setattr(jobs, "manager", jobs.JobManager())
    monkeypatch.setattr(get_settings(), "output_dir", tmp_path, raising=False)
    monkeypatch.setattr(phases, "get_pipeline", lambda: object())
    yield
    fresh_store.clear()


@pytest.fixture
def client():
    """Yield a TestClient for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def _png_bytes() -> bytes:
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[:, :] = (40, 120, 200)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def _fake_phase1(pipeline, image) -> Tuple[AnalysisResult, Dict[str, np.ndarray]]:
    """Minimal phase-1 result with two teams and one goalkeeper."""
    frame = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    result = AnalysisResult()
    result.image = frame
    result.detections = [
        PlayerDetection(bbox=(10.0, 20.0, 50.0, 100.0), confidence=0.91, team_id=1),
        PlayerDetection(bbox=(60.0, 20.0, 100.0, 100.0), confidence=0.85, team_id=2),
        PlayerDetection(
            bbox=(110.0, 20.0, 150.0, 100.0),
            confidence=0.88,
            team_id=2,
            is_goalkeeper=True,
        ),
    ]
    result.attacking_team = 1
    result.defending_team = 2
    result.team_result = TeamAssignmentResult(
        detections=result.detections,
        embeddings=np.zeros((3, 8), dtype=np.float32),
        clustering=None,
        team_colors={1: (255, 0, 0), 2: (0, 0, 255)},
    )
    result.processing_time_ms = 12.0
    stages = {
        "original": frame.copy(),
        "detection": frame.copy(),
        "teams": frame.copy(),
    }
    return result, stages


def _fake_apply_edits(pipeline, result, players) -> Dict[str, np.ndarray]:
    """Rebuild detections from the contract Player list."""
    detections = []
    for player in players:
        detections.append(
            PlayerDetection(
                bbox=tuple(float(value) for value in player["bbox"]),
                confidence=float(player.get("confidence") or 1.0),
                team_id=player.get("team"),
                is_goalkeeper=(player.get("role") == "goalkeeper"),
            )
        )
    result.detections = detections
    if result.team_result is not None:
        result.team_result.detections = detections
    return {"teams": result.image.copy()}


def _fake_phase2(pipeline, result, attacking_side, exclude_goalkeeper=True):
    """Minimal phase-2 result with a full-3D calibration and an offside verdict."""
    calibration = CalibrationResult(
        homography=np.eye(3),
        homography_inv=np.eye(3),
        projection_matrix=np.eye(4),
        quality_score=0.9,
        method="pnl",
    )
    result.calibration = calibration

    line = OffsideLine(
        field_x=100.0,
        image_x=98.0,
        reference_point=(98.0, 100.0),
        defining_player_idx=1,
        defining_body_part=None,
        ignored_player_idx=2,
        confidence=0.85,
    )
    result.offside_line = line

    attacker = result.detections[0]
    status = PlayerOffsideStatus(
        player_idx=0,
        player_detection=attacker,
        is_offside=True,
        most_advanced_part=None,
        advanced_x=100.84,
        margin_px=0.84,
        margin_m=0.84,
        confidence=0.91,
    )
    result.offside_result = OffsideResult(
        offside_line=line,
        player_statuses=[status],
        any_offside=True,
        offside_player_indices=[0],
        attacking_side=attacking_side,
        attacking_team=result.attacking_team,
        defending_team=result.defending_team,
    )
    result.visualization = result.image.copy()
    result.processing_time_ms = 99.0
    frame = result.image
    stages = {
        "calibration": frame.copy(),
        "offside": frame.copy(),
        "birdseye": frame.copy(),
        "result": result.visualization.copy(),
    }
    return result, stages


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Install fake phase implementations."""
    monkeypatch.setattr(phases, "run_phase1", _fake_phase1)
    monkeypatch.setattr(phases, "apply_player_edits", _fake_apply_edits)
    monkeypatch.setattr(phases, "run_phase2", _fake_phase2)


def _upload_image(client: TestClient) -> Dict[str, Any]:
    response = client.post(
        "/api/sessions",
        files={"file": ("frame.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _detect(client: TestClient, session_id: str) -> Dict[str, Any]:
    response = client.post(f"/api/sessions/{session_id}/detect")
    assert response.status_code == 200, response.text
    return response.json()


def _analyze(
    client: TestClient, session_id: str, side: str = "right"
) -> Dict[str, Any]:
    response = client.post(
        f"/api/sessions/{session_id}/analyze",
        json={"attacking_side": side, "attacking_team": 1, "keeper": "player"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in {"done", "error", "cancelled"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Job {job_id} did not finish in time")


# ---------------------------------------------------------------------------
# HTTP tests
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["device"] in {"cuda", "cpu"}


def test_upload_image_and_frame(client):
    session = _upload_image(client)
    assert session["kind"] == "image"
    assert session["width"] == IMAGE_WIDTH
    assert session["height"] == IMAGE_HEIGHT
    assert session["filename"] == "frame.png"
    assert "id" in session

    frame = client.get(f"/api/sessions/{session['id']}/frame.jpg")
    assert frame.status_code == 200
    assert frame.headers["content-type"].startswith("image/jpeg")
    assert frame.content.startswith(b"\xff\xd8")


def test_select_frame_on_image(client):
    session = _upload_image(client)
    response = client.post(
        f"/api/sessions/{session['id']}/select-frame", json={"index": 0}
    )
    assert response.status_code == 200
    assert response.json() == {"index": 0}

    invalid = client.post(
        f"/api/sessions/{session['id']}/select-frame", json={"index": 5}
    )
    assert invalid.status_code == 400


def test_detect_returns_phase1_shape(client, fake_pipeline):
    session = _upload_image(client)
    payload = _detect(client, session["id"])

    assert set(payload) == {
        "players",
        "team_colors",
        "checks",
        "stages",
        "time_ms",
    }
    assert len(payload["players"]) == 3
    first = payload["players"][0]
    assert set(first) == {"id", "bbox", "team", "role", "confidence"}
    assert payload["players"][2]["role"] == "goalkeeper"
    assert payload["team_colors"] == {"1": "#ff3347", "2": "#2e7dff"}
    assert payload["stages"] == ["original", "detection", "teams"]
    by_key = {check["key"]: check for check in payload["checks"]}
    assert by_key["players"]["detail"] == "3"
    assert by_key["teams"]["detail"] == "1 / 2 + GK"
    assert by_key["side"]["status"] == "check"
    assert by_key["side"]["detail"] == "choose left or right"


def test_stage_png_404_then_200(client, fake_pipeline):
    session = _upload_image(client)
    missing = client.get(f"/api/sessions/{session['id']}/stages/teams.png")
    assert missing.status_code == 404

    _detect(client, session["id"])
    stage = client.get(f"/api/sessions/{session['id']}/stages/teams.png")
    assert stage.status_code == 200
    assert stage.headers["content-type"] == "image/png"
    assert stage.content.startswith(b"\x89PNG")


def test_update_players(client, fake_pipeline):
    session = _upload_image(client)
    phase1 = _detect(client, session["id"])

    players = phase1["players"]
    players[0]["team"] = 2
    players[1]["role"] = "goalkeeper"
    players[2]["role"] = "player"
    players.append(
        {
            "id": None,
            "bbox": [5.0, 5.0, 25.0, 45.0],
            "team": 1,
            "role": "player",
            "confidence": 0.5,
        }
    )

    response = client.put(
        f"/api/sessions/{session['id']}/players", json={"players": players}
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    assert len(updated["players"]) == 4
    assert updated["players"][0]["team"] == 2
    assert updated["players"][1]["role"] == "goalkeeper"
    assert updated["players"][3]["id"] == 3


def test_analyze_requires_valid_side(client, fake_pipeline):
    session = _upload_image(client)
    _detect(client, session["id"])

    missing = client.post(f"/api/sessions/{session['id']}/analyze", json={})
    assert missing.status_code == 400
    assert "attacking_side" in missing.json()["detail"]

    invalid = client.post(
        f"/api/sessions/{session['id']}/analyze",
        json={"attacking_side": "up"},
    )
    assert invalid.status_code == 400

    ok = _analyze(client, session["id"], "right")
    assert set(ok) == {
        "verdict",
        "estimate",
        "message",
        "margin_m",
        "attacking_side",
        "last_defender",
        "offside_player_ids",
        "line_image_points",
        "line_field_x",
        "calibration",
        "checks",
        "stages",
        "time_ms",
        "view3d_available",
    }
    assert ok["verdict"] == "offside"
    assert ok["margin_m"] == 0.84
    assert ok["attacking_side"] == "right"
    assert ok["last_defender"] == {"id": 1, "part": None}
    assert ok["offside_player_ids"] == [0]
    assert ok["calibration"] == {"method": "pnl", "full_3d": True, "edited": False}
    assert len(ok["line_image_points"]) == 2
    assert ok["view3d_available"] is True


def test_analyze_without_full_3d_returns_null_margin(client, fake_pipeline, monkeypatch):
    def _no_3d_phase2(pipeline, result, attacking_side, exclude_goalkeeper=True):
        result, stages = _fake_phase2(
            pipeline, result, attacking_side, exclude_goalkeeper
        )
        result.calibration = CalibrationResult(
            homography=np.eye(3),
            homography_inv=np.eye(3),
            quality_score=0.4,
            method="pnl",
        )
        return result, stages

    monkeypatch.setattr(phases, "run_phase2", _no_3d_phase2)

    session = _upload_image(client)
    _detect(client, session["id"])
    verdict = _analyze(client, session["id"])

    assert verdict["margin_m"] is None
    assert verdict["calibration"]["full_3d"] is False


def test_export_image_job_lifecycle(client, fake_pipeline):
    session = _upload_image(client)
    _detect(client, session["id"])
    _analyze(client, session["id"])

    response = client.post(
        f"/api/sessions/{session['id']}/export", json={"kind": "image"}
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    status = _wait_for_job(client, job_id)
    assert status["status"] == "done", status
    assert status["progress"] == 100
    assert status["file_url"].startswith("/api/files/")

    download = client.get(status["file_url"])
    assert download.status_code == 200
    assert download.content.startswith(b"\x89PNG")


def test_export_replay_uses_renderer(client, fake_pipeline, monkeypatch):
    calls: List[str] = []

    def fake_render(
        image_path, analysis_result, output_path, device, should_stop
    ) -> Optional[str]:
        calls.append(image_path)
        with open(output_path, "wb") as handle:
            handle.write(b"fake-mp4")
        return output_path

    monkeypatch.setattr(jobs, "render_replay", fake_render)

    session = _upload_image(client)
    _detect(client, session["id"])
    _analyze(client, session["id"])

    response = client.post(
        f"/api/sessions/{session['id']}/export", json={"kind": "replay"}
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    status = _wait_for_job(client, job_id)
    assert status["status"] == "done", status
    assert calls, "the fake renderer was not called"

    download = client.get(status["file_url"])
    assert download.status_code == 200
    assert download.content == b"fake-mp4"


def test_export_report_zip(client, fake_pipeline):
    session = _upload_image(client)
    _detect(client, session["id"])
    _analyze(client, session["id"])

    response = client.post(
        f"/api/sessions/{session['id']}/export", json={"kind": "report"}
    )
    assert response.status_code == 200, response.text
    status = _wait_for_job(client, response.json()["job_id"])
    assert status["status"] == "done", status

    download = client.get(status["file_url"])
    assert download.status_code == 200
    with zipfile.ZipFile(BytesIO(download.content)) as archive:
        names = set(archive.namelist())
        assert names == {"result.png", "report.json"}
        report = json.loads(archive.read("report.json"))
    assert report["verdict"]["verdict"] == "offside"
    assert report["phase1"]["players"]


def test_export_rejects_unknown_kind_and_missing_analysis(client, fake_pipeline):
    session = _upload_image(client)
    missing = client.post(
        f"/api/sessions/{session['id']}/export", json={"kind": "image"}
    )
    assert missing.status_code == 400

    unknown = client.post(
        f"/api/sessions/{session['id']}/export", json={"kind": "nope"}
    )
    assert unknown.status_code == 400


def test_job_cancel(client, fake_pipeline, monkeypatch):
    started = time.time()

    def slow_render(image_path, analysis_result, output_path, device, should_stop):
        while time.time() - started < 5.0:
            if should_stop():
                return None
            time.sleep(0.02)
        return output_path

    monkeypatch.setattr(jobs, "render_replay", slow_render)

    session = _upload_image(client)
    _detect(client, session["id"])
    _analyze(client, session["id"])
    job_id = client.post(
        f"/api/sessions/{session['id']}/export", json={"kind": "replay"}
    ).json()["job_id"]

    response = client.delete(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json() == {"status": "cancelled"}

    status = _wait_for_job(client, job_id)
    assert status["status"] == "cancelled"


def test_files_route_rejects_traversal(client, fake_pipeline):
    session = _upload_image(client)
    _detect(client, session["id"])
    _analyze(client, session["id"])
    job_id = client.post(
        f"/api/sessions/{session['id']}/export", json={"kind": "image"}
    ).json()["job_id"]
    status = _wait_for_job(client, job_id)
    assert status["status"] == "done"

    for name in (
        "..%2F..%2FAPI_CONTRACT.md",
        "..%5C..%5CAPI_CONTRACT.md",
        "..",
        "unknown.png",
    ):
        response = client.get(f"/api/files/{name}")
        assert response.status_code == 404, name
    assert jobs.manager.file_path("../API_CONTRACT.md") is None


def test_static_root_serves_something(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_unknown_session_returns_404(client):
    response = client.get("/api/sessions/does-not-exist/frame.jpg")
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


# ---------------------------------------------------------------------------
# WebSocket tests
# ---------------------------------------------------------------------------


class FakeLiveScene:
    """Stand-in for ``var_offside.web.live3d.LiveScene``."""

    instances: List["FakeLiveScene"] = []

    def __init__(self, result, image_bgr, device="cuda", on_status=None):
        self.result = result
        self.image_bgr = image_bgr
        self.device = device
        self.on_status = on_status
        self.built = False
        self.upgraded = False
        self.closed = False
        FakeLiveScene.instances.append(self)

    def build_proxy(self) -> None:
        self.built = True

    def upgrade_meshes(self, should_stop=None) -> None:
        self.upgraded = True

    def recover_bodies(self, should_stop=None):
        return ["body"]

    def plant_bodies(self, bodies, should_stop=None) -> None:
        self.upgraded = bodies is not None

    def replay_steps(self, fps=24, kind="replay"):
        return [{}, {}]

    def render_step(self, step, width, height, options=None):
        return np.full((height, width, 3), 60, dtype=np.uint8)

    def preset(self, name: str) -> Dict[str, Any]:
        return dict(CAMERA)

    def render(self, camera, width, height, options, t=0.0):
        return np.full((height, width, 3), 40, dtype=np.uint8)

    def replay(self, width, height, options, fps=24):
        for _ in range(2):
            yield np.full((height, width, 3), 60, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _inject_fake_live_scene(monkeypatch) -> None:
    module = types.ModuleType("var_offside.web.live3d")
    module.LiveScene = FakeLiveScene
    monkeypatch.setitem(sys.modules, "var_offside.web.live3d", module)
    FakeLiveScene.instances = []


def _receive(ws) -> Tuple[str, Any]:
    message = ws.receive()
    assert message.get("type") == "websocket.send", message
    text = message.get("text")
    if text is not None:
        return "text", json.loads(text)
    return "bytes", message.get("bytes")


def test_websocket_live_view(client, fake_pipeline, monkeypatch):
    _inject_fake_live_scene(monkeypatch)

    session = _upload_image(client)
    _detect(client, session["id"])
    _analyze(client, session["id"])

    with client.websocket_connect(f"/api/sessions/{session['id']}/view") as ws:
        statuses: List[str] = []
        camera_messages: List[Dict[str, Any]] = []
        frames: List[bytes] = []

        for _ in range(30):
            kind, payload = _receive(ws)
            if kind == "text":
                if payload["type"] == "status":
                    statuses.append(payload["stage"])
                elif payload["type"] == "camera":
                    camera_messages.append(payload)
            else:
                frames.append(payload)
            if "ready" in statuses and len(frames) >= 2:
                break

        assert "loading" in statuses
        assert "proxy" in statuses
        assert "meshes" in statuses
        assert "ready" in statuses
        assert camera_messages, "no camera message was sent"
        assert set(camera_messages[0]["camera"]) == set(CAMERA)
        assert len(frames) >= 2
        assert frames[0].startswith(b"\xff\xd8")

        # Camera request -> one JPEG frame.
        ws.send_json(
            {
                "type": "camera",
                "camera": CAMERA,
                "width": 320,
                "height": 180,
                "quality": "draft",
                "options": OPTIONS,
            }
        )
        kind, payload = _receive(ws)
        assert kind == "bytes"
        assert payload.startswith(b"\xff\xd8")

        # Preset request -> camera JSON followed by a frame.
        ws.send_json(
            {"type": "preset", "name": "birdseye", "width": 320, "height": 180}
        )
        preset_camera = None
        preset_frame = None
        for _ in range(4):
            kind, payload = _receive(ws)
            if kind == "text" and payload["type"] == "camera":
                preset_camera = payload
            elif kind == "bytes":
                preset_frame = payload
            if preset_camera is not None and preset_frame is not None:
                break
        assert preset_camera is not None
        assert set(preset_camera["camera"]) == set(CAMERA)
        assert preset_frame is not None and preset_frame.startswith(b"\xff\xd8")

        # Replay streams frames and ends with replay_end + camera.
        ws.send_json({"type": "replay", "width": 160, "height": 90})
        replay_frames = 0
        replay_ended = False
        replay_camera = None
        for _ in range(12):
            kind, payload = _receive(ws)
            if kind == "bytes":
                replay_frames += 1
            elif payload["type"] == "replay_end":
                replay_ended = True
            elif payload["type"] == "camera":
                replay_camera = payload
            if replay_ended and replay_camera is not None:
                break
        assert replay_frames == 2
        assert replay_ended
        assert replay_camera is not None

    assert FakeLiveScene.instances
    assert FakeLiveScene.instances[0].closed is True


def test_websocket_requires_analysis(client, monkeypatch):
    _inject_fake_live_scene(monkeypatch)
    session = _upload_image(client)

    with client.websocket_connect(f"/api/sessions/{session['id']}/view") as ws:
        kind, payload = _receive(ws)
        assert kind == "text"
        assert payload["type"] == "status"
        assert payload["stage"] == "error"
