"""
Lightweight MP4 player dialog built on OpenCV + QTimer.

Used to preview rendered 3D VAR replays inside the app.  We decode with
OpenCV rather than QtMultimedia so playback does not depend on system codecs
being registered with Qt.
"""

from __future__ import annotations

import os

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QFileDialog,
    QSizePolicy,
)


class VideoPlayerDialog(QDialog):
    """Modal-ish dialog that loops an MP4 clip with basic transport controls."""

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self._path = video_path
        self.setWindowTitle(f"3D Replay — {os.path.basename(video_path)}")
        self.resize(900, 620)

        self._cap = cv2.VideoCapture(video_path)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 1 else 30.0
        self._playing = True
        self._loop = True

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._timer.start(int(1000 / self._fps))

    def _build_ui(self):
        root = QVBoxLayout(self)

        self.video_label = QLabel("Loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #111;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(640, 360)
        root.addWidget(self.video_label, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self._frame_count - 1))
        self.slider.sliderPressed.connect(lambda: self._set_playing(False))
        self.slider.sliderMoved.connect(self._seek)
        root.addWidget(self.slider)

        controls = QHBoxLayout()
        self.btn_play = QPushButton("Pause")
        self.btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self.btn_play)

        self.btn_loop = QPushButton("Loop: On")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(True)
        self.btn_loop.clicked.connect(self._toggle_loop)
        controls.addWidget(self.btn_loop)

        self.lbl_time = QLabel("0 / 0")
        controls.addWidget(self.lbl_time)

        controls.addStretch(1)

        self.btn_save = QPushButton("Save Video As...")
        self.btn_save.clicked.connect(self._save_as)
        controls.addWidget(self.btn_save)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        controls.addWidget(self.btn_close)

        root.addLayout(controls)

    # ----- playback -----------------------------------------------------

    def _next_frame(self):
        if not self._playing or self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    return
            else:
                self._set_playing(False)
                return
        self._show_frame(frame)
        pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.blockSignals(True)
        self.slider.setValue(max(0, pos - 1))
        self.slider.blockSignals(False)
        self.lbl_time.setText(f"{pos} / {self._frame_count}")

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def _seek(self, value: int):
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            ok, frame = self._cap.read()
            if ok:
                self._show_frame(frame)
            self.lbl_time.setText(f"{value} / {self._frame_count}")

    def _toggle_play(self):
        self._set_playing(not self._playing)

    def _set_playing(self, playing: bool):
        self._playing = playing
        self.btn_play.setText("Pause" if playing else "Play")

    def _toggle_loop(self):
        self._loop = self.btn_loop.isChecked()
        self.btn_loop.setText("Loop: On" if self._loop else "Loop: Off")

    def _save_as(self):
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Video As", os.path.basename(self._path), "MP4 (*.mp4)"
        )
        if not dest:
            return
        try:
            import shutil

            shutil.copyfile(self._path, dest)
        except OSError:
            pass

    def closeEvent(self, event):
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().closeEvent(event)
