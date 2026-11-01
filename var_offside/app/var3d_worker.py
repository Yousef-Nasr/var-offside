"""
Background worker that renders a 3D VAR replay video without blocking the GUI.

The offside pipeline gates its 3D step on ``result.image_path`` being a real
file (SAM 3D Body loads the frame from disk).  GUI runs operate on in-memory
frames, so here we write the analysed frame to a temporary PNG, point the
result at it, and call the standalone render entry points.
"""

from __future__ import annotations

import os
import tempfile

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from var_offside.pipeline import AnalysisResult
from var_offside.visualization.var_3d import RenderCancelled


# Human-readable render modes -> entry function names
RENDER_MODES = {
    "cinematic": "Cinematic replay (broadcast → 3D sweep)",
    "zoomout": "Zoom-out to 3D bird's-eye",
    "debug": "Debug overlay (2D projection)",
}


class Var3DWorker(QThread):
    """Render a 3D VAR replay in a worker thread."""

    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str)  # output video path
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: AnalysisResult = None
        self._output_path: str = ""
        self._mode: str = "cinematic"
        self._device: str = "cuda"
        self._temp_image: str = None
        self._orig_image_path: str = None
        self._stop_requested = False

    def request_stop(self):
        """Ask a running render to stop before the next frame."""
        self._stop_requested = True

    def setup(
        self,
        result: AnalysisResult,
        output_path: str,
        mode: str = "cinematic",
        device: str = "cuda",
    ):
        self._result = result
        self._output_path = output_path
        self._mode = mode if mode in RENDER_MODES else "cinematic"
        self._device = device
        self._stop_requested = False

    def run(self):
        try:
            if self._result is None or self._result.offside_result is None:
                self.error.emit("Run the full offside analysis before generating 3D.")
                return

            frame = self._result.image
            if frame is None:
                self.error.emit("No analysed frame available for 3D rendering.")
                return

            # Persist the analysed frame so SAM 3D Body can load it.
            self.progress.emit(5, "Preparing frame for 3D...")
            fd, temp_path = tempfile.mkstemp(prefix="var3d_frame_", suffix=".png")
            os.close(fd)
            cv2.imwrite(temp_path, frame)
            self._temp_image = temp_path
            self._orig_image_path = self._result.image_path
            self._result.image_path = temp_path

            self.progress.emit(
                15, "Loading 3D model and reconstructing players (first run is slow)..."
            )

            from var_offside.visualization.var_3d import (
                render_var_3d_video,
                render_zoomout_3d_video,
                render_debug_3d_video,
            )

            should_stop = lambda: self._stop_requested

            if self._mode == "zoomout":
                out = render_zoomout_3d_video(
                    image_path=temp_path,
                    analysis_result=self._result,
                    output_path=self._output_path,
                    device=self._device,
                    should_stop=should_stop,
                )
            elif self._mode == "debug":
                out = render_debug_3d_video(
                    image_path=temp_path,
                    analysis_result=self._result,
                    output_path=self._output_path,
                    device=self._device,
                    should_stop=should_stop,
                )
            else:
                out = render_var_3d_video(
                    image_path=temp_path,
                    analysis_result=self._result,
                    output_path=self._output_path,
                    device=self._device,
                    should_stop=should_stop,
                )

            if not out or not os.path.isfile(out):
                self.error.emit(
                    "3D rendering produced no output. This frame may lack full "
                    "camera calibration (PnL), which the 3D scene needs. Try a "
                    "clearer frame with visible pitch lines."
                )
                return

            self.progress.emit(100, "3D replay ready")
            self.finished_ok.emit(out)
        except RenderCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            import traceback

            traceback.print_exc()
            self.error.emit(f"3D rendering failed: {exc}")
        finally:
            # The temp frame is deleted below; don't leave the shared result
            # pointing at a file that no longer exists.
            if self._temp_image and self._result is not None:
                if self._result.image_path == self._temp_image:
                    self._result.image_path = self._orig_image_path
            if self._temp_image and os.path.isfile(self._temp_image):
                try:
                    os.remove(self._temp_image)
                except OSError:
                    pass
                self._temp_image = None
