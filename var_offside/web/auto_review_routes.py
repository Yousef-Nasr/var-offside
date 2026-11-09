"""HTTP routes for Auto-review. The agent is imported inside them, so the app starts with no key."""

from __future__ import annotations

import copy
import logging
import threading
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from var_offside.web import jobs, phases

logger = logging.getLogger(__name__)

router = APIRouter()


class AnswerBody(BaseModel):
    """Body for ``POST /api/sessions/{id}/auto-review/answers/{qid}``."""

    choice: str


class _LockedDetector:
    """The shared detector, used one caller at a time. The lock covers only the detector."""

    def __init__(self, detector: Any, lock: threading.Lock) -> None:
        self._detector = detector
        self._lock = lock

    @property
    def last_ball(self) -> Any:
        return getattr(self._detector, "last_ball", None)

    @last_ball.setter
    def last_ball(self, value: Any) -> None:
        self._detector.last_ball = value

    def detect(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._detector.detect(*args, **kwargs)


class SessionHost:
    """Lets a run read and change a web session through the app's own functions."""

    def __init__(self, session: Any, server: Any) -> None:
        self.session = session
        self.server = server

    def revision(self) -> int:
        return self.session.revision

    def wait_calibration(self) -> bool:
        """True once the automatic camera fit has finished, even if it found no camera."""
        self.server._start_calibration(self.session)
        return self.session.calibration.wait()

    def snapshot(self) -> Any:
        from var_offside.agent.scene import Scene

        with self.session.lock:
            result = self.session.phase1_result
            calibration = self.session.calibration.effective()
            revision = self.session.revision
        return Scene.from_result(result, calibration=calibration, revision=revision)

    def commit(self, scene: Any) -> None:
        rows = [phases.serialize_player(index, det) for index, det in enumerate(scene.detections)]
        self.server._run_player_edits(self.session, rows)

    def restore(self, scene: Any) -> None:
        """Put a scene back, keeping the pose and embeddings of anyone it brings back."""
        with self.session.lock:
            result = self.session.phase1_result
            detections = copy.deepcopy(list(scene.detections))
            result.detections = detections
            result.next_player_id = max(
                [getattr(result, "next_player_id", 0), *(det.player_id + 1 for det in detections)]
            )
            team_result = result.team_result
            if team_result is not None:
                team_result.detections = detections
                old = getattr(team_result, "embeddings", None)
                width = old.shape[1] if isinstance(old, np.ndarray) and old.ndim == 2 else None
                if width is not None:
                    rows = np.zeros((len(detections), width), dtype=old.dtype)
                    for row, det in enumerate(detections):
                        if det.player_id in scene.embeddings:
                            rows[row] = scene.embeddings[det.player_id]
                    team_result.embeddings = rows
        self.commit(scene)

    def analyze(self, side: Optional[str], team: Optional[int], keeper: Optional[str]) -> Dict[str, Any]:
        return self.server._run_analyze(self.session, side, team, keeper)

    def detector(self) -> Any:
        return _LockedDetector(phases.get_pipeline().player_detector, self.server._PIPELINE_EXEC_LOCK)


def _server() -> Any:
    from var_offside.web import server

    return server


def _run_of(session: Any) -> Any:
    run = session.auto_review
    if run is None:
        raise HTTPException(status_code=404, detail="Auto-review hasn't run on this frame")
    return run


@router.post("/api/sessions/{session_id}/auto-review")
def start_auto_review(session_id: str) -> Dict[str, Any]:
    """Start a run on the detected frame. Follow it with ``GET``."""
    from var_offside.agent.config import load_config
    from var_offside.agent.runner import AutoReviewRun, client_for

    server = _server()
    session = server._get_session(session_id)
    with session.lock:
        result = session.phase1_result
        previous = session.auto_review
    if result is None:
        raise HTTPException(status_code=400, detail="Run detection before Auto-review")
    if previous is not None:
        previous.stop()

    config = load_config()
    holder: Dict[str, Any] = {}

    def cancelled() -> bool:
        job = holder.get("job")
        return job is not None and job.cancel_event.is_set()

    run = AutoReviewRun(SessionHost(session, server), client_for(config), config, cancelled)
    with session.lock:
        session.auto_review = run
    holder["job"] = jobs.manager.start_task("auto_review", lambda _job: run.execute(), session_id=session.id)
    return {
        "job_id": holder["job"].id,
        "phase1": phases.serialize_phase1(result, server._available_stages(session)),
        "vision": bool(config.vision_available),
    }


@router.get("/api/sessions/{session_id}/auto-review")
def get_auto_review(session_id: str, since: int = 0) -> Dict[str, Any]:
    """New events since ``since``, and the result once the run has ended."""
    session = _server()._get_session(session_id)
    return _run_of(session).snapshot(since)


@router.post("/api/sessions/{session_id}/auto-review/answers/{question_id}")
def answer_auto_review(session_id: str, question_id: str, payload: AnswerBody) -> Dict[str, str]:
    """A person's answer to the question the run is waiting on."""
    from var_offside.agent.runner import NotPending, StaleRun

    session = _server()._get_session(session_id)
    try:
        _run_of(session).answer(question_id, payload.choice)
    except StaleRun as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NotPending as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "running"}


@router.post("/api/sessions/{session_id}/auto-review/decisions/{decision_id}/undo")
def undo_auto_review(session_id: str, decision_id: str) -> Dict[str, Any]:
    """Revert one correction and rerun the call."""
    from var_offside.agent.runner import NotPending, StaleRun

    session = _server()._get_session(session_id)
    try:
        return _run_of(session).undo(decision_id)
    except StaleRun as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NotPending as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
