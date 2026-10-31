"""
Video Handler - Video loading and frame extraction.
"""

import cv2
import numpy as np
from typing import Optional


class VideoHandler:
    """Wrapper around cv2.VideoCapture for on-demand frame seeking."""

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._path: Optional[str] = None
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0

    @property
    def is_loaded(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def path(self) -> Optional[str]:
        return self._path

    def load(self, path: str) -> bool:
        """
        Load a video file.

        Returns:
            True if loaded successfully.
        """
        self.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False

        self._cap = cap
        self._path = path
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Seek to and read a specific frame.

        Args:
            frame_idx: 0-based frame index.

        Returns:
            BGR numpy array, or None on failure.
        """
        if not self.is_loaded:
            return None

        frame_idx = max(0, min(frame_idx, self._frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._path = None
            self._frame_count = 0
