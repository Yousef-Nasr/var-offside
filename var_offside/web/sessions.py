"""
In-memory session store for the browser app.

Each uploaded image or video becomes a :class:`Session`: the file is saved in a
temporary directory, the current frame can be decoded on demand and the phase
results plus debug stage images are kept in memory.  Sessions are process-local
and are never persisted; restarting the server drops them.

Video reading uses OpenCV directly (the desktop app's ``VideoHandler`` is a Qt
wrapper around the same API).  A per-session lock serialises frame access and a
single-worker executor is kept for the live 3D scene, whose pyrender/OpenGL
context is thread-bound.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import cv2
import numpy as np

from var_offside.web.calibration_stage import CalibrationState

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
)
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
)


class VideoReader:
    """Minimal OpenCV video reader with random frame access."""

    def __init__(self, path: str):
        """Open ``path`` and read its metadata.

        Args:
            path: Path to a video file.

        Raises:
            ValueError: If OpenCV cannot open the file.
        """
        self.path = path
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise ValueError("Could not open the uploaded video")

        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if self.frame_count <= 0:
            self.frame_count = 1
        if self.fps <= 0.0:
            self.fps = 30.0

    def read(self, index: int) -> Optional[np.ndarray]:
        """Read frame ``index`` (clamped to the available range)."""
        if self._capture is None:
            return None
        index = max(0, min(int(index), self.frame_count - 1))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        """Release the underlying capture."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None


# One render thread for the whole process, not one per session. On Windows,
# pyglet caches its WGL function lookups on first use, and a lookup from a thread
# with no current GL context is cached as missing, which breaks every later scene.
# Everything that opens a GL window, video exports included, must run here.
RENDER_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="var-live3d")


def _log_close_failure(future: Any) -> None:
    """Log a failed LiveScene close; shutdown is best effort."""
    exc = future.exception()
    if exc is not None:
        logger.warning("Closing live scene failed: %s", exc)


@dataclass
class Session:
    """One uploaded media file plus its analysis state."""

    id: str
    filename: str
    kind: str  # "image" | "video"
    path: str
    width: int = 0
    height: int = 0
    image: Optional[np.ndarray] = None
    reader: Optional[VideoReader] = None
    frame_count: int = 1
    fps: float = 0.0
    frame_index: int = 0

    # Analysis state
    phase1_result: Any = None
    phase2_result: Any = None
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    calibration: CalibrationState = field(default_factory=CalibrationState)

    # Threading / live 3D state
    lock: threading.RLock = field(default_factory=threading.RLock)
    view_executor: ThreadPoolExecutor = field(default_factory=lambda: RENDER_THREAD)
    view_scene: Any = None

    @property
    def is_video(self) -> bool:
        """Whether this session holds a video."""
        return self.kind == "video"

    def current_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the selected frame, or ``None`` if it is missing."""
        with self.lock:
            if self.kind == "image":
                return None if self.image is None else self.image.copy()
            if self.reader is None:
                return None
            frame = self.reader.read(self.frame_index)
            return None if frame is None else frame.copy()

    def frame_at(self, index: Optional[int] = None) -> Optional[np.ndarray]:
        """Return a copy of frame ``index`` without changing the selection.

        Images ignore ``index``.  Videos clamp it to the available range.
        """
        with self.lock:
            if self.kind == "image":
                return None if self.image is None else self.image.copy()
            if self.reader is None:
                return None
            target = self.frame_index if index is None else int(index)
            frame = self.reader.read(target)
            return None if frame is None else frame.copy()

    def select_frame(self, index: int) -> int:
        """Select the frame later phases analyse and return its index."""
        with self.lock:
            limit = self.frame_count if self.kind == "video" else 1
            index = int(index)
            if index < 0 or index >= limit:
                raise ValueError(f"index must be between 0 and {max(limit - 1, 0)}")
            if index != self.frame_index:
                # Broadcast cameras move, so each frame starts from the automatic result.
                self.calibration = CalibrationState()
            self.frame_index = index
            return index

    def set_stages(self, stages: Optional[Dict[str, np.ndarray]]) -> None:
        """Store or replace debug stage images."""
        with self.lock:
            if stages:
                self.stages.update(stages)

    def clear_analysis(self) -> None:
        """Drop phase-2 output and the stages produced by it."""
        with self.lock:
            self.phase2_result = None
            for key in ("calibration", "offside", "birdseye", "result"):
                self.stages.pop(key, None)

    def close(self) -> None:
        """Release the video capture and close the live scene."""
        with self.lock:
            if self.reader is not None:
                self.reader.release()
                self.reader = None
            scene = self.view_scene
            self.view_scene = None
        if scene is not None:
            # GL contexts are thread-bound: close on the thread that made it.
            future = self.view_executor.submit(scene.close)
            future.add_done_callback(_log_close_failure)


class SessionStore:
    """Thread-safe, in-memory ``id -> Session`` registry."""

    def __init__(self) -> None:
        """Create the store and its private temporary directory."""
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self._temp_dir = Path(tempfile.mkdtemp(prefix="var_offside_web_"))
        logger.info("Session store using %s", self._temp_dir)

    @property
    def temp_dir(self) -> Path:
        """Directory holding uploaded files."""
        return self._temp_dir

    def create(self, filename: str, data: bytes) -> Session:
        """Create a session from uploaded bytes.

        Args:
            filename: Original client filename (used only for display/suffix).
            data: Raw uploaded bytes.

        Returns:
            The created :class:`Session`.

        Raises:
            ValueError: If the bytes are neither a decodable image nor a video.
        """
        if not data:
            raise ValueError("The uploaded file is empty")

        session_id = uuid.uuid4().hex
        safe_name = Path(filename or "upload").name
        suffix = Path(safe_name).suffix.lower()
        if not suffix or len(suffix) > 10:
            suffix = ".bin"
        target = self._temp_dir / f"{session_id}{suffix}"
        target.write_bytes(data)

        image = None
        try:
            buffer = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001 - not an image, try video
            logger.debug("Image decode failed for %s: %s", safe_name, exc)

        if image is not None:
            session = Session(
                id=session_id,
                filename=safe_name,
                kind="image",
                path=str(target),
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                image=image,
                frame_count=1,
                fps=0.0,
            )
        else:
            try:
                reader = VideoReader(str(target))
            except ValueError:
                try:
                    target.unlink()
                except OSError:
                    pass
                raise ValueError(
                    "Uploaded file is not a supported image or video"
                ) from None
            session = Session(
                id=session_id,
                filename=safe_name,
                kind="video",
                path=str(target),
                width=reader.width,
                height=reader.height,
                reader=reader,
                frame_count=reader.frame_count,
                fps=reader.fps,
            )

        with self._lock:
            self._sessions[session_id] = session
        logger.info(
            "Created %s session %s (%s, %dx%d)",
            session.kind,
            session_id,
            safe_name,
            session.width,
            session.height,
        )
        return session

    def get(self, session_id: str) -> Session:
        """Return a session by id.

        Raises:
            KeyError: If the id is unknown.
        """
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def get_optional(self, session_id: str) -> Optional[Session]:
        """Return a session by id, or ``None`` when unknown."""
        with self._lock:
            return self._sessions.get(session_id)

    def close(self, session_id: str) -> bool:
        """Close and remove one session."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True

    def clear(self) -> None:
        """Close every session (used by tests and shutdown)."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def __iter__(self) -> Iterator[Session]:
        """Iterate over a snapshot of the live sessions."""
        with self._lock:
            return iter(list(self._sessions.values()))


# Process-wide store used by the FastAPI handlers.
store = SessionStore()
