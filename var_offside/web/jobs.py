"""
Background export jobs with progress and cancellation.

Exports run on plain daemon threads, and their state is exposed over
``/api/jobs/{id}``.  Cancellation reuses
the ``should_stop`` callables accepted by the 3D render entry points.

Exports are written to ``get_settings().output_dir`` and served by
``GET /api/files/{name}``.  Only filenames produced by a job are served, which
also blocks path traversal.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2

from var_offside.config import get_settings
from var_offside.web.sessions import RENDER_THREAD

logger = logging.getLogger(__name__)

EXPORT_FPS = 30
EXPORT_WIDTH = 1280
EXPORT_HEIGHT = 720

JOB_KINDS = ("image", "replay", "birdseye", "debug", "report")
# Background work with no output file, which the export route can't start.
TASK_KINDS = ("auto_review",)
TERMINAL_STATUSES = frozenset({"done", "error", "cancelled"})


def _output_dir() -> Path:
    """Return (and create) the configured export directory."""
    directory = Path(get_settings().output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Renderer entry points (lazy so importing jobs never imports pyrender)
# ---------------------------------------------------------------------------


def _render_live_video(
    kind: str,
    analysis_result: Any,
    output_path: str,
    device: str,
    should_stop: Callable[[], bool],
    progress: Optional[Callable[[float], None]] = None,
    options: Optional[Dict[str, bool]] = None,
) -> Optional[str]:
    """
    Write a cinematic from :class:`LiveScene` frames.

    Each frame is a separate job on the shared render thread, so the live 3D view
    keeps working during an export.  Body recovery reuses what the view recovered.
    """
    from var_offside.web.live3d import LiveScene

    def on_render_thread(call: Callable[[], Any]) -> Any:
        return RENDER_THREAD.submit(call).result()

    def build() -> Any:
        instance = LiveScene(
            result=analysis_result, image_bgr=analysis_result.image, device=device
        )
        instance.build_proxy()
        return instance

    scene = on_render_thread(build)
    writer = None
    try:
        bodies = scene.recover_bodies(should_stop=should_stop)
        on_render_thread(lambda: scene.plant_bodies(bodies, should_stop=should_stop))
        steps = on_render_thread(lambda: scene.replay_steps(EXPORT_FPS, kind))
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            EXPORT_FPS,
            (EXPORT_WIDTH, EXPORT_HEIGHT),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open a video writer for {output_path}")
        for i, step in enumerate(steps):
            if should_stop():
                return None
            frame = on_render_thread(
                lambda s=step: scene.render_step(s, EXPORT_WIDTH, EXPORT_HEIGHT, options)
            )
            writer.write(frame)
            if progress is not None:
                progress((i + 1) / len(steps))
        return output_path
    finally:
        if writer is not None:
            writer.release()
        on_render_thread(scene.close)


def render_replay(
    image_path: str,
    analysis_result: Any,
    output_path: str,
    device: str,
    should_stop: Callable[[], bool],
    progress: Optional[Callable[[float], None]] = None,
    options: Optional[Dict[str, bool]] = None,
) -> Optional[str]:
    """Render the cinematic replay, the same story the 3D view plays."""
    return _render_live_video(
        "replay", analysis_result, output_path, device, should_stop, progress, options
    )


def render_birdseye(
    image_path: str,
    analysis_result: Any,
    output_path: str,
    device: str,
    should_stop: Callable[[], bool],
    progress: Optional[Callable[[float], None]] = None,
    options: Optional[Dict[str, bool]] = None,
) -> Optional[str]:
    """Render the zoom-out to the bird's-eye view, with the verdict."""
    return _render_live_video(
        "birdseye", analysis_result, output_path, device, should_stop, progress, options
    )


def render_debug(
    image_path: str, analysis_result: Any, output_path: str, device: str, should_stop
) -> Optional[str]:
    """Render the 2D-projection debug video.

    It opens its own GL window, so it must run on the render thread, and the live
    view waits until it finishes.
    """
    from var_offside.visualization.var_3d import render_debug_3d_video

    return RENDER_THREAD.submit(
        render_debug_3d_video,
        image_path=image_path,
        analysis_result=analysis_result,
        output_path=output_path,
        device=device,
        should_stop=should_stop,
    ).result()


def _to_h264(path: Path) -> None:
    """Re-encode an mp4 as H.264 so browsers and phones can play it.

    OpenCV's bundled writer only offers mp4v.  Without ffmpeg, or if it fails, the
    original file stays in place.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001 - optional dependency
            logger.warning("ffmpeg not found; keeping the mp4v video: %s", path)
            return
    temp = path.with_name(path.stem + "_h264" + path.suffix)
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=300)
        os.replace(temp, path)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("H.264 conversion failed, keeping the mp4v video: %s", exc)
        if temp.exists():
            temp.unlink()


def _renderer_for(kind: str) -> Optional[Callable[..., Optional[str]]]:
    """Return the render entry point for ``kind`` (late-bound for tests)."""
    return {
        "replay": render_replay,
        "birdseye": render_birdseye,
        "debug": render_debug,
    }.get(kind)

_EXTENSIONS = {
    "image": ".png",
    "replay": ".mp4",
    "birdseye": ".mp4",
    "debug": ".mp4",
    "report": ".zip",
}


@dataclass
class Job:
    """State of one background export."""

    id: str
    kind: str
    status: str = "running"
    progress: int = 0
    message: str = "Queued"
    file_name: Optional[str] = None
    error: Optional[str] = None
    options: Optional[Dict[str, bool]] = None
    session_id: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the contract's ``GET /jobs/{id}`` shape."""
        file_url = f"/api/files/{self.file_name}" if self.file_name else None
        return {
            "status": self.status,
            "progress": int(self.progress),
            "message": self.message,
            "file_url": file_url,
        }


class JobManager:
    """Registry and runner for export jobs."""

    def __init__(self) -> None:
        """Create an empty job registry."""
        self._jobs: Dict[str, Job] = {}
        self._files: set[str] = set()
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(
        self,
        kind: str,
        result: Any,
        image: Any = None,
        report: Optional[Dict[str, Any]] = None,
        session_id: str = "session",
        options: Optional[Dict[str, bool]] = None,
    ) -> Job:
        """Start an export job in a daemon thread.

        Args:
            kind: One of :data:`JOB_KINDS`.
            result: Phase-2 ``AnalysisResult`` (or phase-1 for image exports).
            image: Fallback BGR frame when the result has no visualization.
            report: Extra JSON metadata written into a report zip.
            session_id: Session id used to build a readable output filename.

        Returns:
            The created :class:`Job`.

        Raises:
            ValueError: If ``kind`` is unknown.
        """
        if kind not in JOB_KINDS:
            raise ValueError(f"Unknown export kind: {kind}")

        job = Job(id=uuid.uuid4().hex, kind=kind, options=options, session_id=session_id)
        with self._lock:
            self._jobs[job.id] = job

        thread = threading.Thread(
            target=self._run,
            args=(job, result, image, report, session_id),
            name=f"var-export-{kind}-{job.id[:8]}",
            daemon=True,
        )
        thread.start()
        return job

    def start_task(self, kind: str, work: Callable[[Job], None], session_id: Optional[str] = None) -> Job:
        """Run ``work(job)`` in a daemon thread, outside the render thread.

        ``work`` reports through :meth:`report` and should stop early once
        ``job.cancel_event`` is set.

        Raises:
            ValueError: If ``kind`` isn't one of :data:`TASK_KINDS`.
        """
        if kind not in TASK_KINDS:
            raise ValueError(f"Unknown task kind: {kind}")

        job = Job(id=uuid.uuid4().hex, kind=kind, session_id=session_id)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(
            target=self._run_task,
            args=(job, work),
            name=f"var-task-{kind}-{job.id[:8]}",
            daemon=True,
        ).start()
        return job

    def report(self, job: Job, progress: int, message: str) -> None:
        """Update a running task's progress."""
        self._update(job, progress=progress, message=message)

    def _run_task(self, job: Job, work: Callable[[Job], None]) -> None:
        """Execute one task; never raises out of the thread."""
        try:
            work(job)
            if job.cancel_event.is_set():
                self._mark_cancelled(job, None)
            else:
                self._update(job, status="done", progress=100, message="Done")
        except Exception as exc:  # noqa: BLE001 - report through job status
            logger.exception("Task %s (%s) failed", job.id, job.kind)
            if job.cancel_event.is_set():
                self._mark_cancelled(job, None)
            else:
                self._update(job, status="error", message=str(exc), error=str(exc))

    def get(self, job_id: str) -> Optional[Job]:
        """Return a job by id, or ``None`` when unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[Job]:
        """Request cancellation of a running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status not in TERMINAL_STATUSES:
                job.cancel_event.set()
                job.status = "cancelled"
                job.message = "Cancelled"
            return job

    def delete_for_session(self, session_id: str) -> None:
        """Stop a session's jobs and delete its export files.

        A job still writing can hold its file open. It's cancelled here, and it removes
        the file itself when it stops.
        """
        with self._lock:
            owned = [job for job in self._jobs.values() if job.session_id == session_id]
            for job in owned:
                if job.status not in TERMINAL_STATUSES:
                    job.cancel_event.set()
                    job.status = "cancelled"
                    job.message = "Cancelled"
                del self._jobs[job.id]
                self._files.discard(job.file_name)
        for job in owned:
            if job.kind not in _EXTENSIONS:
                continue
            path = _output_dir() / self._file_name(job, session_id)
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.info("Export %s will be removed when its job stops: %s", path, exc)

    # -- file serving ------------------------------------------------------

    def file_path(self, name: str) -> Optional[Path]:
        """Resolve a produced file name, rejecting traversal and unknown names."""
        if not name or name != Path(name).name:
            return None
        if "/" in name or "\\" in name or "\x00" in name:
            return None
        with self._lock:
            if name not in self._files:
                return None

        base = _output_dir().resolve()
        path = (base / name).resolve()
        if path.parent != base or not path.is_file():
            return None
        return path

    # -- worker ------------------------------------------------------------

    def _run(
        self,
        job: Job,
        result: Any,
        image: Any,
        report: Optional[Dict[str, Any]],
        session_id: str,
    ) -> None:
        """Execute one job; never raises out of the thread."""
        try:
            self._update(job, progress=1, message="Starting export")
            name = self._file_name(job, session_id)
            path = _output_dir() / name

            if _renderer_for(job.kind) is not None:
                self._run_render(job, result, image, path)
            elif job.kind == "image":
                self._run_image(job, result, image, path)
            else:
                self._run_report(job, result, image, report, path)

            if job.cancel_event.is_set():
                self._mark_cancelled(job, path)
                return
            if job.status == "error":
                return

            with self._lock:
                job.file_name = name
                job.status = "done"
                job.progress = 100
                job.message = "Export ready"
                self._files.add(name)
            logger.info("Export %s (%s) ready: %s", job.id, job.kind, path)
        except Exception as exc:  # noqa: BLE001 - report through job status
            logger.exception("Export job %s failed", job.id)
            if job.cancel_event.is_set():
                self._mark_cancelled(job, None)
            else:
                self._update(job, status="error", message=str(exc), error=str(exc))

    def _run_render(self, job: Job, result: Any, image: Any, path: Path) -> None:
        """Run one of the 3D render entry points."""
        if result is None or getattr(result, "offside_result", None) is None:
            self._update(
                job,
                status="error",
                message="Run the full offside analysis before generating 3D",
                error="No offside result",
            )
            return

        frame = getattr(result, "image", None)
        if frame is None:
            frame = image
        if frame is None:
            self._update(
                job,
                status="error",
                message="No analysed frame available for 3D rendering",
                error="No frame",
            )
            return

        self._update(
            job,
            progress=10,
            message="Preparing frame for 3D (first run loads the body model)...",
        )
        fd, temp_path = tempfile.mkstemp(prefix="var_web_3d_", suffix=".png")
        os.close(fd)
        written = cv2.imwrite(temp_path, frame)
        if not written:
            self._update(
                job, status="error", message="Could not write the frame for 3D"
            )
            return
        try:
            renderer = _renderer_for(job.kind)
            if renderer is None:
                self._update(job, status="error", message="Unknown renderer")
                return
            self._update(job, progress=20, message="Rendering 3D video...")
            extra: Dict[str, Any] = {}
            if "progress" in inspect.signature(renderer).parameters:
                extra["progress"] = lambda done: self._update(
                    job, progress=20 + int(done * 70)
                )
            if "options" in inspect.signature(renderer).parameters:
                extra["options"] = job.options
            output = renderer(
                image_path=temp_path,
                analysis_result=result,
                output_path=str(path),
                device=get_settings().device,
                should_stop=job.cancel_event.is_set,
                **extra,
            )
            if job.cancel_event.is_set():
                return
            if not output or not os.path.isfile(output):
                self._update(
                    job,
                    status="error",
                    message=(
                        "3D rendering produced no output. This frame may lack full "
                        "camera calibration (PnL), which the 3D scene needs."
                    ),
                    error="No output",
                )
                return
            if Path(output) != path and os.path.isfile(output):
                try:
                    os.replace(output, path)
                except OSError:
                    pass
            self._update(job, progress=92, message="Finalising video...")
            _to_h264(path)
        finally:
            if os.path.isfile(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _run_image(self, job: Job, result: Any, image: Any, path: Path) -> None:
        """Write the result overlay as a PNG."""
        self._update(job, progress=20, message="Writing overlay image")
        frame = getattr(result, "visualization", None) if result is not None else None
        if frame is None:
            frame = image
        if frame is None and result is not None:
            frame = getattr(result, "image", None)
        if frame is None:
            self._update(job, status="error", message="No analysed frame to export")
            return
        if not cv2.imwrite(str(path), frame):
            self._update(job, status="error", message="Could not write the PNG")
            return
        self._update(job, progress=90, message="Image written")

    def _run_report(
        self,
        job: Job,
        result: Any,
        image: Any,
        report: Optional[Dict[str, Any]],
        path: Path,
    ) -> None:
        """Write a zip containing ``result.png`` and ``report.json``."""
        self._update(job, progress=20, message="Building report")
        frame = getattr(result, "visualization", None) if result is not None else None
        if frame is None:
            frame = image
        if frame is None and result is not None:
            frame = getattr(result, "image", None)
        if frame is None:
            self._update(job, status="error", message="No analysed frame to export")
            return

        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            self._update(job, status="error", message="Could not encode the PNG")
            return

        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(report or {}),
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("result.png", encoded.tobytes())
            archive.writestr(
                "report.json", json.dumps(payload, indent=2, default=str)
            )
        self._update(job, progress=90, message="Report written")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _file_name(job: Job, session_id: str) -> str:
        """Build a unique, path-safe output filename."""
        prefix = "".join(ch for ch in (session_id or "")[:8] if ch.isalnum()) or "session"
        return f"{prefix}_{job.kind}_{job.id[:8]}{_EXTENSIONS[job.kind]}"

    def _update(self, job: Job, **fields: Any) -> None:
        """Update job fields unless the job was cancelled in the meantime."""
        with self._lock:
            if job.status == "cancelled" and fields.get("status") in {
                "done",
                "error",
            }:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def _mark_cancelled(self, job: Job, path: Optional[Path]) -> None:
        """Mark a job cancelled and remove a partial output file."""
        with self._lock:
            job.status = "cancelled"
            job.message = "Cancelled"
            job.progress = min(job.progress, 99)
        if path is not None and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


# Process-wide manager used by the FastAPI handlers.
manager = JobManager()
