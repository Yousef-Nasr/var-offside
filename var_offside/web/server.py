"""
FastAPI server for the browser-based VAR offside app.

All HTTP routes live under ``/api`` exactly as documented in
``docs/reference/web-api.md``.  The static frontend is mounted at ``/``
when ``var_offside/web/static/`` exists.  The live 3D WebSocket is optional: the
``LiveScene`` class is imported lazily inside the handler so this module and its
tests keep working even when ``live3d.py`` is absent or broken.

Heavy pipeline work runs in a thread pool so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from var_offside.calibration.base import CalibrationResult
from var_offside.config import get_settings
from var_offside.web import jobs, phases, sessions
from var_offside.web.auto_review_routes import router as auto_review_router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
DEFAULT_OPTIONS: Dict[str, bool] = {
    "line_plane": True,
    "photo": False,
    "labels": True,
    "pulse": True,
    "key_only": False,
}
PRESETS = frozenset({"broadcast", "birdseye", "line", "free"})
QUALITIES = frozenset({"draft", "full"})
JPEG_QUALITY_DRAFT = 70
FLIGHT_SECONDS = 0.75
REPLAY_FPS = 24
REPLAY_MAX_WIDTH = 1280
PULSE_INTERVAL_SECONDS = 0.125
JPEG_QUALITY_FULL = 90

# One pipeline instance is shared by the whole server; serialise heavy calls so
# concurrent detect/analyze requests cannot interleave model state.
_PIPELINE_EXEC_LOCK = threading.Lock()

app = FastAPI(
    title="VAR Offside Web",
    version="1.0.0",
    description="Browser front end for the VAR offside pipeline.",
)
app.include_router(auto_review_router)


# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})
_UPLOAD_CHUNK = 1 << 20
# Room for the multipart headers around the file itself.
_MULTIPART_SLACK = 64 * 1024


def _max_upload_mb() -> float:
    server = getattr(get_settings(), "server", None)
    return float(getattr(server, "max_upload_mb", 500))


def _too_large_detail() -> str:
    return (
        f"The file is larger than the {_max_upload_mb():g} MB upload limit. "
        "Use a shorter clip or a single frame, or raise server.max_upload_mb in config.yaml."
    )


def _unsupported_detail(suffix: str) -> str:
    shown = f"'{suffix}' files" if suffix else "Files without an extension"
    return (
        f"{shown} can't be opened. Upload an image (PNG, JPG, BMP, WebP, TIFF) "
        "or a video (MP4, MOV, M4V, AVI, MKV, WebM)."
    )


def _is_supported_upload(filename: str, content_type: Optional[str]) -> bool:
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix in IMAGE_SUFFIXES or suffix in VIDEO_SUFFIXES
    return (content_type or "").split("/")[0] in ("image", "video")


@app.middleware("http")
async def _reject_oversized_uploads(request: Request, call_next):
    """Refuse an upload by its Content-Length before the body is read to disk."""
    if request.method == "POST" and request.url.path == "/api/sessions":
        length = request.headers.get("content-length")
        limit = _max_upload_mb() * 1024 * 1024 + _MULTIPART_SLACK
        if length is not None and length.isdigit() and int(length) > limit:
            return JSONResponse(status_code=413, content={"detail": _too_large_detail()})
    return await call_next(request)


async def _read_upload(file: UploadFile) -> bytes:
    """Read the upload in chunks, stopping as soon as it passes the size limit."""
    limit = _max_upload_mb() * 1024 * 1024
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=_too_large_detail())
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class SelectFrameBody(BaseModel):
    """Body for ``POST /api/sessions/{id}/select-frame``."""

    index: int


class PlayersBody(BaseModel):
    """Body for ``PUT /api/sessions/{id}/players``."""

    players: List[Dict[str, Any]]


class AnalyzeBody(BaseModel):
    """Body for ``POST /api/sessions/{id}/analyze``."""

    attacking_side: Optional[str] = None
    # Loosely typed so a bad value gets the verdict's message, not a 422.
    attacking_team: Any = None
    keeper: Any = None


class CalibrationBody(BaseModel):
    """Body for the trial solve and for applying a calibration."""

    pins: List[Dict[str, Any]] = []
    mode: Optional[str] = None  # "edit" | "hand"
    start: Optional[Dict[str, Any]] = None  # last camera, keeps a drag quick
    attacking_side: Optional[str] = None


class ExportBody(BaseModel):
    """Body for ``POST /api/sessions/{id}/export``."""

    kind: Optional[str] = None
    options: Optional[Dict[str, bool]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_session(session_id: str) -> sessions.Session:
    """Return a session or raise the contract's 404 error."""
    session = sessions.store.get_optional(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _session_payload(session: sessions.Session) -> Dict[str, Any]:
    """Serialise a session to the contract's upload response."""
    payload: Dict[str, Any] = {
        "id": session.id,
        "kind": session.kind,
        "width": int(session.width),
        "height": int(session.height),
        "filename": session.filename,
    }
    if session.kind == "video":
        payload["frame_count"] = int(session.frame_count)
        payload["fps"] = float(session.fps)
    return payload


def _available_stages(session: sessions.Session) -> List[str]:
    """Return the stage keys currently available, in canonical order."""
    with session.lock:
        return [key for key in phases.STAGE_KEYS if key in session.stages]


def _encode_jpeg(frame: np.ndarray, quality: int) -> Optional[bytes]:
    """Encode a BGR frame as JPEG bytes."""
    if frame is None:
        return None
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        return None
    return buffer.tobytes()


def _start_calibration(session: sessions.Session) -> None:
    """Calibrate the selected frame in the background, once per frame."""
    with session.lock:
        state = session.calibration
        if not state.begin():
            return
    frame = session.current_frame()
    # Resolved here, not in the thread, so the thread never outlives its setup.
    pipeline = phases.get_pipeline()
    calibrate = phases.run_calibration

    def work() -> None:
        try:
            if frame is None:
                raise ValueError("Could not read the selected frame")
            with _PIPELINE_EXEC_LOCK:
                result = calibrate(pipeline, frame)
        except Exception as exc:  # noqa: BLE001 - the screen offers hand placement
            logger.warning("Automatic calibration failed: %s", exc)
            result = CalibrationResult(method="auto_failed")
            result.debug_info["error"] = str(exc)
        state.finish(result)

    threading.Thread(target=work, name="var-calibrate", daemon=True).start()


def _run_detect(session: sessions.Session) -> Dict[str, Any]:
    """Thread-pool worker: run phase 1 and store its results."""
    frame = session.current_frame()
    if frame is None:
        raise ValueError("Could not read the selected frame")

    with _PIPELINE_EXEC_LOCK:
        result, stages = phases.run_phase1(phases.get_pipeline(), frame)

    with session.lock:
        session.phase1_result = result
        session.phase2_result = None
        session.stages = dict(stages)
        session.analysis_choices = None
        session.bump_revision()

    # Ready, ideally, by the time the review is done.
    _start_calibration(session)
    return phases.serialize_phase1(result, _available_stages(session))


def _run_player_edits(session: sessions.Session, players: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Thread-pool worker: apply player edits and redraw the review stage."""
    with session.lock:
        result = session.phase1_result
    if result is None:
        raise ValueError("Run detection before editing players")

    with _PIPELINE_EXEC_LOCK:
        stages = phases.apply_player_edits(phases.get_pipeline(), result, players)

    with session.lock:
        session.set_stages(stages)
        session.clear_analysis()
        session.bump_revision()

    return phases.serialize_phase1(result, _available_stages(session))


def _run_analyze(
    session: sessions.Session,
    attacking_side: str,
    attacking_team: Any,
    keeper: Any,
) -> Dict[str, Any]:
    """Thread-pool worker: run phase 2 and store the verdict."""
    with session.lock:
        result = session.phase1_result
    if result is None:
        raise ValueError("Run detection before analysis")

    # ``True == 1`` in Python, but a boolean isn't a team.
    known_team = not isinstance(attacking_team, bool) and attacking_team in (1, 2)
    result.attacking_team = attacking_team if known_team else None
    result.defending_team = (3 - attacking_team) if known_team else None
    result.keeper_choice = keeper if keeper in ("player", "not_in_frame") else None
    choices = (attacking_side, result.attacking_team, result.keeper_choice)
    with session.lock:
        session.clear_analysis()
        if choices != session.analysis_choices:
            session.analysis_choices = choices
            session.bump_revision()

    choice_error = phases.analysis_choice_error(result)
    if choice_error is not None:
        result.offside_line = None
        result.offside_result = None
        result.visualization = None
        result.verdict_message = choice_error
        result.side_result = None
        with session.lock:
            session.phase2_result = result
        verdict = phases.serialize_verdict(result, _available_stages(session))
        verdict["attacking_side"] = attacking_side
        return verdict

    _start_calibration(session)
    with session.lock:
        calibration_state = session.calibration
    calibration_state.wait()
    result.calibration = calibration_state.effective()

    with _PIPELINE_EXEC_LOCK:
        result, stages = phases.run_phase2(
            phases.get_pipeline(), result, attacking_side
        )

    unknown = result.offside_line is None or result.offside_result is None
    with session.lock:
        if unknown:
            session.clear_analysis()
        else:
            session.set_stages(stages)
        session.phase2_result = result

    return phases.serialize_verdict(result, _available_stages(session))


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> Dict[str, str]:
    """Return service status and the configured compute device."""
    return {"status": "ok", "device": get_settings().device}


@app.post("/api/sessions")
async def create_session(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload an image or video and create an in-memory session.

    413 when the file passes ``server.max_upload_mb``; 415 when it isn't an
    image or video, by extension or by content.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded")
    if not _is_supported_upload(file.filename, file.content_type):
        raise HTTPException(
            status_code=415, detail=_unsupported_detail(Path(file.filename).suffix.lower())
        )
    data = await _read_upload(file)
    try:
        session = sessions.store.create(file.filename, data)
    except sessions.UnsupportedMedia as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - unexpected upload failure
        logger.exception("Creating session failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    return _session_payload(session)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a session and everything made from it, including its exports."""
    _get_session(session_id)
    jobs.manager.delete_for_session(session_id)
    sessions.store.close(session_id)
    return {"status": "deleted"}


@app.get("/api/sessions/{session_id}/frame.jpg")
def session_frame(session_id: str, index: Optional[int] = None) -> Response:
    """Return the selected frame (or the requested video frame) as JPEG."""
    session = _get_session(session_id)
    try:
        frame = session.frame_at(index)
    except Exception as exc:  # noqa: BLE001 - invalid index or broken video
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame could not be read")
    data = _encode_jpeg(frame, JPEG_QUALITY_FULL)
    if data is None:
        raise HTTPException(status_code=500, detail="Could not encode the frame")
    return Response(content=data, media_type="image/jpeg")


@app.post("/api/sessions/{session_id}/select-frame")
def select_frame(session_id: str, payload: SelectFrameBody) -> Dict[str, int]:
    """Choose the frame later phases analyse."""
    session = _get_session(session_id)
    try:
        index = session.select_frame(payload.index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"index": index}


@app.post("/api/sessions/{session_id}/detect")
async def detect(session_id: str) -> Dict[str, Any]:
    """Run phase 1 (detection, teams, goalkeepers) for the selected frame."""
    session = _get_session(session_id)
    try:
        return await run_in_threadpool(_run_detect, session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface pipeline failures
        logger.exception("Detection failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/sessions/{session_id}/players")
async def update_players(
    session_id: str, payload: Optional[PlayersBody] = None
) -> Dict[str, Any]:
    """Rebuild detections from the reviewed player list."""
    session = _get_session(session_id)
    if payload is None:
        raise HTTPException(status_code=400, detail="A players list is required")
    try:
        return await run_in_threadpool(_run_player_edits, session, payload.players)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Player edit failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/calibration")
def get_calibration(session_id: str) -> Dict[str, Any]:
    """Return the calibration for the selected frame, starting it if needed."""
    session = _get_session(session_id)
    _start_calibration(session)
    return session.calibration.payload(int(session.width), int(session.height))


@app.post("/api/sessions/{session_id}/calibration/solve")
def solve_calibration(
    session_id: str, payload: Optional[CalibrationBody] = None
) -> Dict[str, Any]:
    """Solve a camera from pins without changing anything."""
    session = _get_session(session_id)
    body = payload or CalibrationBody()
    try:
        return session.calibration.trial(
            body.pins,
            body.mode,
            int(session.width),
            int(session.height),
            start=body.start,
            attacking_side=body.attacking_side,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/sessions/{session_id}/calibration")
def apply_calibration(
    session_id: str, payload: Optional[CalibrationBody] = None
) -> Dict[str, Any]:
    """Apply pins as the frame's calibration.  Run Analyze afterwards."""
    session = _get_session(session_id)
    body = payload or CalibrationBody()
    try:
        session.calibration.apply(
            body.pins,
            body.mode,
            int(session.width),
            int(session.height),
            attacking_side=body.attacking_side,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with session.lock:
        session.clear_analysis()
        session.bump_revision()
    return session.calibration.payload(int(session.width), int(session.height))


@app.delete("/api/sessions/{session_id}/calibration")
def reset_calibration(session_id: str) -> Dict[str, Any]:
    """Drop any hand calibration and go back to the automatic result."""
    session = _get_session(session_id)
    session.calibration.revert()
    with session.lock:
        session.clear_analysis()
        session.bump_revision()
    return session.calibration.payload(int(session.width), int(session.height))


@app.post("/api/sessions/{session_id}/analyze")
async def analyze(
    session_id: str, payload: Optional[AnalyzeBody] = None
) -> Dict[str, Any]:
    """Run phase 2 and return the verdict."""
    session = _get_session(session_id)
    side = payload.attacking_side if payload is not None else None
    if side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="attacking_side must be 'left' or 'right'",
        )
    attacking_team = payload.attacking_team if payload is not None else None
    keeper = payload.keeper if payload is not None else None
    try:
        return await run_in_threadpool(
            _run_analyze, session, side, attacking_team, keeper
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/stages/{key}.png")
def stage_image(session_id: str, key: str) -> Response:
    """Return one debug stage as PNG, or 404 when it has not been produced."""
    session = _get_session(session_id)
    if key not in phases.STAGE_KEYS:
        raise HTTPException(status_code=404, detail="Unknown stage")
    with session.lock:
        image = session.stages.get(key)
    if image is None:
        raise HTTPException(status_code=404, detail="Stage not produced yet")
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the stage")
    return Response(content=buffer.tobytes(), media_type="image/png")


@app.post("/api/sessions/{session_id}/export")
def export(session_id: str, payload: Optional[ExportBody] = None) -> Dict[str, str]:
    """Start an export job for this session."""
    session = _get_session(session_id)
    kind = payload.kind if payload is not None else None
    if kind not in jobs.JOB_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"kind must be one of {', '.join(jobs.JOB_KINDS)}",
        )

    with session.lock:
        phase1 = session.phase1_result
        phase2 = session.phase2_result
        result_stage = session.stages.get("result")
        teams_stage = session.stages.get("teams")
    result = phase2 or phase1
    if result is None:
        raise HTTPException(status_code=400, detail="Run detection before exporting")
    if kind != "image" and phase2 is None:
        raise HTTPException(status_code=400, detail="Run Analyze before exporting")

    overlay = getattr(result, "visualization", None)
    if overlay is None:
        overlay = result_stage if result_stage is not None else teams_stage
    if overlay is None:
        overlay = getattr(result, "image", None)

    report: Optional[Dict[str, Any]] = None
    if kind == "report":
        report = {
            "filename": session.filename,
            "session_id": session.id,
            "phase1": phases.serialize_phase1(phase1) if phase1 is not None else None,
            "verdict": phases.serialize_verdict(phase2) if phase2 is not None else None,
            "calibration": session.calibration.report(),
        }
        if session.auto_review is not None and session.auto_review.record is not None:
            report["auto_review"] = session.auto_review.record

    job = jobs.manager.start(
        kind=kind,
        result=result,
        image=overlay,
        report=report,
        session_id=session.id,
        options=payload.options,
    )
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    """Return one job's status."""
    job = jobs.manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str) -> Dict[str, str]:
    """Cancel a running job."""
    job = jobs.manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled"}


@app.get("/api/files/{name}")
def download_file(name: str) -> FileResponse:
    """Serve a file produced by an export job (traversal is rejected)."""
    path = jobs.manager.file_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=name)


# ---------------------------------------------------------------------------
# Live 3D WebSocket
# ---------------------------------------------------------------------------


def _message_size(message: Dict[str, Any]) -> Tuple[int, int]:
    """Read and clamp the requested render size from a client message."""
    try:
        width = int(message.get("width") or DEFAULT_WIDTH)
    except (TypeError, ValueError):
        width = DEFAULT_WIDTH
    try:
        height = int(message.get("height") or DEFAULT_HEIGHT)
    except (TypeError, ValueError):
        height = DEFAULT_HEIGHT
    width = max(64, min(width, MAX_WIDTH))
    height = max(64, min(height, MAX_HEIGHT))
    return width, height


def _message_options(message: Dict[str, Any]) -> Dict[str, bool]:
    """Merge client render options over the contract defaults."""
    options = dict(DEFAULT_OPTIONS)
    provided = message.get("options")
    if isinstance(provided, dict):
        for key in DEFAULT_OPTIONS:
            if key in provided:
                options[key] = bool(provided[key])
    return options


def _camera_payload(camera: Any) -> Optional[Dict[str, Any]]:
    """Normalise a camera dict to plain JSON numbers.

    ``LiveScene.preset`` returns the contract's camera dict; converting here
    keeps numpy scalars out of the JSON encoder even if the renderer returns
    them.
    """
    if not isinstance(camera, dict):
        return None
    target = camera.get("target")
    if not isinstance(target, (list, tuple)) or len(target) < 3:
        return None
    try:
        return {
            "target": [float(value) for value in target[:3]],
            "yaw": float(camera.get("yaw", 0.0)),
            "pitch": float(camera.get("pitch", 22.0)),
            "distance": float(camera.get("distance", 18.0)),
            "fov": float(camera.get("fov", 40.0)),
        }
    except (TypeError, ValueError):
        return None


def _same_camera(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True when two cameras are close enough that a flight would be invisible."""
    near = all(abs(float(x) - float(y)) < 0.01 for x, y in zip(a["target"], b["target"]))
    return (
        near
        and abs((float(a["yaw"]) - float(b["yaw"]) + 180.0) % 360.0 - 180.0) < 0.05
        and abs(float(a["pitch"]) - float(b["pitch"])) < 0.05
        and abs(float(a["distance"]) - float(b["distance"])) < 0.01
    )


def _interpolate_camera(
    a: Dict[str, Any], b: Dict[str, Any], progress: float
) -> Dict[str, Any]:
    """Eased camera between two orbit cameras: short way round, even zoom speed."""
    u = 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(progress, 1.0)))
    yaw_a = float(a["yaw"])
    yaw_step = (float(b["yaw"]) - yaw_a + 180.0) % 360.0 - 180.0
    dist_a, dist_b = max(float(a["distance"]), 1e-3), max(float(b["distance"]), 1e-3)
    return {
        "target": [
            float(x) + (float(y) - float(x)) * u for x, y in zip(a["target"], b["target"])
        ],
        "yaw": yaw_a + yaw_step * u,
        "pitch": float(a["pitch"]) + (float(b["pitch"]) - float(a["pitch"])) * u,
        "distance": dist_a * (dist_b / dist_a) ** u,
        "fov": float(a["fov"]) + (float(b["fov"]) - float(a["fov"])) * u,
    }


@app.websocket("/api/sessions/{session_id}/view")
async def live_view(websocket: WebSocket, session_id: str) -> None:
    """Stream the live 3D view as JSON control messages plus JPEG frames."""
    await websocket.accept()
    loop = asyncio.get_running_loop()

    async def _error_and_close(message: str) -> None:
        try:
            await websocket.send_json(
                {"type": "status", "stage": "error", "message": message}
            )
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001 - socket may already be gone
            pass

    session = sessions.store.get_optional(session_id)
    if session is None:
        await _error_and_close("Session not found")
        return

    with session.lock:
        result = session.phase2_result
    if result is None:
        await _error_and_close("Run Analyze before opening the 3D view")
        return

    try:
        # Imported lazily so the server and its tests work without live3d.py.
        from var_offside.web.live3d import LiveScene
    except Exception as exc:  # noqa: BLE001 - renderer is a separate deliverable
        logger.warning("LiveScene unavailable: %s", exc)
        await _error_and_close(f"3D renderer unavailable: {exc}")
        return

    send_lock = asyncio.Lock()
    last_status: Dict[str, Optional[str]] = {"stage": None, "message": None}
    ready_sent = {"value": False}

    async def send_json(payload: Dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_bytes(data: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(data)

    async def send_status(stage: str, message: str) -> None:
        last_status["stage"] = stage
        last_status["message"] = message if stage == "proxy" else None
        if stage == "ready":
            ready_sent["value"] = True
        await send_json({"type": "status", "stage": stage, "message": message})

    def on_status(stage: str, message: str) -> None:
        """LiveScene status callback (runs on the 3D worker thread)."""
        if loop.is_closed():
            return

        async def _safe_send() -> None:
            try:
                await send_status(stage, message)
            except Exception:  # noqa: BLE001 - client may have disconnected
                pass

        try:
            asyncio.run_coroutine_threadsafe(_safe_send(), loop)
        except RuntimeError:
            pass

    executor = session.view_executor
    state: Dict[str, Any] = {
        "camera": None,
        "options": dict(DEFAULT_OPTIONS),
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
    }
    connected_at = time.monotonic()
    upgrade_stop = threading.Event()
    replay_state: Dict[str, Any] = {"task": None, "stop": asyncio.Event()}
    scene: Any = None

    def _render(
        camera: Dict[str, Any], width: int, height: int, options: Dict[str, bool]
    ) -> Optional[np.ndarray]:
        elapsed = time.monotonic() - connected_at
        return scene.render(camera, width, height, options, elapsed)

    async def render_and_send(
        message: Dict[str, Any], camera: Optional[Dict[str, Any]] = None
    ) -> None:
        """Render the newest request and send one JPEG frame."""
        width, height = _message_size(message)
        options = _message_options(message)
        quality = message.get("quality", "full")
        if quality not in QUALITIES:
            quality = "full"
        if quality == "draft":
            render_w, render_h = max(width // 2, 32), max(height // 2, 32)
            jpeg_quality = JPEG_QUALITY_DRAFT
        else:
            render_w, render_h = width, height
            jpeg_quality = JPEG_QUALITY_FULL

        active_camera = camera or message.get("camera") or state["camera"]
        if not isinstance(active_camera, dict):
            return

        state.update(
            {
                "camera": active_camera,
                "options": options,
                "width": width,
                "height": height,
            }
        )
        frame = await loop.run_in_executor(
            executor, lambda: _render(active_camera, render_w, render_h, options)
        )
        data = _encode_jpeg(frame, jpeg_quality)
        if data is not None:
            await send_bytes(data)

    # -- Initial load: stand-in scene and first frame -----------------------
    try:
        await send_status("loading", "Preparing 3D view")

        def _create_scene() -> Any:
            instance = LiveScene(
                result=result,
                image_bgr=result.image,
                device=get_settings().device,
                on_status=on_status,
            )
            instance.build_proxy()
            return instance

        scene = await loop.run_in_executor(executor, _create_scene)
        session.view_scene = scene
        await send_status("proxy", "Stand-in players ready")

        raw_camera = await loop.run_in_executor(
            executor, lambda: scene.preset("broadcast")
        )
        camera = _camera_payload(raw_camera) or raw_camera
        state["camera"] = camera
        await send_json({"type": "camera", "camera": camera})
        first = await loop.run_in_executor(
            executor,
            lambda: _render(
                camera, DEFAULT_WIDTH, DEFAULT_HEIGHT, dict(DEFAULT_OPTIONS)
            ),
        )
        first_data = _encode_jpeg(first, JPEG_QUALITY_FULL)
        if first_data is not None:
            await send_bytes(first_data)
    except WebSocketDisconnect:
        await _shutdown_view(scene, executor)
        return
    except Exception as exc:  # noqa: BLE001 - 3D setup may fail
        logger.exception("Live 3D setup failed for session %s", session_id)
        await _error_and_close(str(exc))
        await _shutdown_view(scene, executor)
        return

    # -- Live loop: input is handled from the first frame on ----------------
    pending: Dict[str, Any] = {"message": None}
    pending_event = asyncio.Event()
    meshes_done = asyncio.Event()
    animation: Dict[str, Any] = {"task": None, "stop": None, "fly_back": False}
    activity: Dict[str, Any] = {"last_input": time.monotonic(), "visible": True}

    def animating() -> bool:
        task = animation["task"]
        return task is not None and not task.done()

    async def stop_animation() -> None:
        """Stop the running camera flight or replay and wait for it to end."""
        if not animating():
            return
        animation["stop"].set()
        try:
            await animation["task"]
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the animation reports its own errors
            pass

    async def start_animation(run: Any) -> None:
        await stop_animation()
        stop = asyncio.Event()
        animation["stop"] = stop
        animation["fly_back"] = False
        animation["task"] = asyncio.create_task(run(stop))

    async def fly_to(
        destination: Dict[str, Any],
        width: int,
        height: int,
        options: Dict[str, bool],
        stop: asyncio.Event,
    ) -> None:
        """Glide the camera to ``destination``, then send one full frame."""
        origin = state["camera"]
        state["camera"] = destination
        await send_json({"type": "camera", "camera": destination})

        if isinstance(origin, dict) and not _same_camera(origin, destination):
            started = time.monotonic()
            while not stop.is_set():
                progress = (time.monotonic() - started) / FLIGHT_SECONDS
                if progress >= 1.0:
                    break
                view = _interpolate_camera(origin, destination, progress)
                frame = await loop.run_in_executor(
                    executor,
                    lambda: _render(
                        view, max(width // 2, 32), max(height // 2, 32), options
                    ),
                )
                data = _encode_jpeg(frame, JPEG_QUALITY_DRAFT)
                if data is not None:
                    await send_bytes(data)

        if stop.is_set():
            return
        state.update({"options": options, "width": width, "height": height})
        final = await loop.run_in_executor(
            executor, lambda: _render(destination, width, height, options)
        )
        data = _encode_jpeg(final, JPEG_QUALITY_FULL)
        if data is not None:
            await send_bytes(data)

    async def handle_preset(message: Dict[str, Any]) -> None:
        name = message.get("name")
        if name not in PRESETS:
            name = "broadcast"
        width, height = _message_size(message)
        options = _message_options(message)
        raw = await loop.run_in_executor(executor, lambda: scene.preset(name))
        destination = _camera_payload(raw) or raw
        await start_animation(
            lambda stop: fly_to(destination, width, height, options, stop)
        )

    async def run_replay(message: Dict[str, Any], stop: asyncio.Event) -> None:
        """Stream the cinematic at real speed, skipping frames if rendering lags."""
        width, height = _message_size(message)
        if width > REPLAY_MAX_WIDTH:
            height = max(int(height * REPLAY_MAX_WIDTH / width), 64)
            width = REPLAY_MAX_WIDTH
        options = _message_options(message)
        return_to = state["camera"]

        try:
            # The story needs the real bodies, so wait for them (Stop still works).
            if not meshes_done.is_set():
                waiters = {
                    asyncio.create_task(meshes_done.wait()),
                    asyncio.create_task(stop.wait()),
                }
                _, unfinished = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )
                for waiter in unfinished:
                    waiter.cancel()

            if not stop.is_set():
                steps = await loop.run_in_executor(
                    executor, lambda: scene.replay_steps(REPLAY_FPS)
                )
                started = time.monotonic()
                shown = -1
                while not stop.is_set():
                    index = int((time.monotonic() - started) * REPLAY_FPS)
                    if index >= len(steps):
                        break
                    if index == shown:
                        await asyncio.sleep(0.004)
                        continue
                    shown = index
                    frame = await loop.run_in_executor(
                        executor,
                        lambda: scene.render_step(steps[index], width, height, options),
                    )
                    data = _encode_jpeg(frame, JPEG_QUALITY_FULL)
                    if data is not None:
                        await send_bytes(data)
        except WebSocketDisconnect:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the view alive
            logger.exception("Replay failed for session %s", session_id)
            await send_status("error", f"Replay failed: {exc}")

        await send_json({"type": "replay_end"})
        # New input (a drag or a preset) takes over; otherwise glide back.
        if stop.is_set() and not animation["fly_back"]:
            return
        if not isinstance(return_to, dict):
            return
        raw = await loop.run_in_executor(executor, lambda: scene.preset("line"))
        state["camera"] = _camera_payload(raw) or raw
        glide = asyncio.Event()
        animation["stop"] = glide
        animation["fly_back"] = False
        await fly_to(return_to, state["width"], state["height"], state["options"], glide)

    async def reader() -> None:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                continue
            kind = message.get("type")
            activity["last_input"] = time.monotonic()
            if kind == "camera":
                await stop_animation()
                # Keep only the newest pending request.
                pending["message"] = message
                pending_event.set()
            elif kind == "preset":
                await handle_preset(message)
            elif kind == "replay":
                await start_animation(
                    lambda stop, msg=message: run_replay(msg, stop)
                )
            elif kind == "stop":
                if animating():
                    animation["fly_back"] = True
                    animation["stop"].set()
            elif kind == "visible":
                activity["visible"] = bool(message.get("visible", True))

    async def renderer() -> None:
        while True:
            await pending_event.wait()
            pending_event.clear()
            message = pending["message"]
            pending["message"] = None
            if message is None:
                continue
            try:
                await render_and_send(message)
            except WebSocketDisconnect:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the view alive
                logger.warning("Camera render failed: %s", exc)
                await send_status("error", str(exc))

    async def pulser() -> None:
        """Keep the slow pulse moving while the view is otherwise idle."""
        while True:
            await asyncio.sleep(PULSE_INTERVAL_SECONDS)
            options = state["options"]
            if (
                not activity["visible"]
                or animating()
                or pending["message"] is not None
                or not options.get("pulse", True)
                or not getattr(scene, "has_pulse", False)
                or time.monotonic() - activity["last_input"] < 0.4
            ):
                continue
            await render_and_send(
                {
                    "width": state["width"],
                    "height": state["height"],
                    "options": options,
                    "quality": "full",
                }
            )

    async def mesher() -> None:
        """Recover bodies off the render thread, then swap them in on it."""
        try:
            await send_status("meshes", "Building body meshes...")
            bodies = await loop.run_in_executor(
                None, lambda: scene.recover_bodies(should_stop=upgrade_stop.is_set)
            )
            await loop.run_in_executor(
                executor,
                lambda: scene.plant_bodies(bodies, should_stop=upgrade_stop.is_set),
            )
            if upgrade_stop.is_set():
                return
            if not ready_sent["value"]:
                await send_status("ready", last_status.get("message") or "3D view ready")
            part_name = getattr(scene, "part_name", None)
            if part_name:
                await send_json({"type": "defender_part", "name": part_name})
            if not animating():
                pending["message"] = {
                    "width": state["width"],
                    "height": state["height"],
                    "options": state["options"],
                    "quality": "full",
                }
                pending_event.set()
        except asyncio.CancelledError:
            raise
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - stand-ins stay usable
            logger.exception("Body meshes failed for session %s", session_id)
            await send_status("ready", f"Body meshes unavailable: {exc}")
        finally:
            meshes_done.set()

    mesh_task = asyncio.create_task(mesher())
    tasks = {
        asyncio.create_task(reader()),
        asyncio.create_task(renderer()),
        asyncio.create_task(pulser()),
    }
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exception = None if task.cancelled() else task.exception()
            if exception is not None:
                raise exception
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        # The server is shutting this socket down; only the cleanup below is left.
        logger.debug("Live 3D view cancelled for session %s", session_id)
    except RuntimeError as exc:
        # A send racing the client's disconnect is a normal way to end.
        logger.debug("Live 3D socket closed: %s", exc)
    except Exception as exc:  # noqa: BLE001 - report unexpected view failures
        logger.exception("Live 3D view failed for session %s", session_id)
        await _error_and_close(str(exc))
    finally:
        upgrade_stop.set()
        cancels = [mesh_task, *tasks]
        if animation["task"] is not None:
            cancels.append(animation["task"])
        for task in cancels:
            if not task.done():
                task.cancel()
        close = executor.submit(scene.close)
        try:
            await asyncio.gather(*cancels, return_exceptions=True)
            await asyncio.wrap_future(close)
        except asyncio.CancelledError:
            close.result(timeout=5)
        except Exception as exc:  # noqa: BLE001 - shutdown is best effort
            logger.warning("LiveScene close failed: %s", exc)


async def _shutdown_view(scene: Any, executor: Any) -> None:
    """Close the LiveScene on its own worker thread."""
    if scene is None:
        return
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(executor, scene.close)
    except Exception as exc:  # noqa: BLE001 - shutdown is best effort
        logger.warning("LiveScene close failed: %s", exc)


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api keeps priority)
# ---------------------------------------------------------------------------

if (STATIC_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")
else:  # pragma: no cover - exercised only while the frontend is not built yet

    @app.get("/", response_class=HTMLResponse)
    def frontend_missing() -> str:
        """Small placeholder when the frontend has not been built yet."""
        return (
            "<!doctype html><html><head><title>VAR Offside</title></head>"
            "<body style='font-family:sans-serif;max-width:40rem;margin:4rem auto'>"
            "<h1>VAR Offside API is running</h1>"
            "<p>The web frontend is not installed yet. The API lives under "
            "<code>/api</code>; try <a href='/api/health'>/api/health</a>.</p>"
            "</body></html>"
        )
