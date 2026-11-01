"""
Main Window - Layout, toolbar, menus, and wiring for the VAR Offside app.

Two-phase pipeline flow:
  Phase 1 (Run):      detection + teams   ->  pause for user review
  Phase 2 (Continue): calibration + offside + bird's eye + result
"""

import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QToolBar,
    QAction,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QSplitter,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

from var_offside.app.stage_viewer import StageViewer
from var_offside.app.controls_panel import ControlsPanel, IMAGE_FILTER, VIDEO_FILTER
from var_offside.app.pipeline_worker import PipelineWorker
from var_offside.app.var3d_worker import Var3DWorker
from var_offside.app.video_player import VideoPlayerDialog
from var_offside.app.video_handler import VideoHandler
from var_offside.app.player_editor import PlayerEditorDialog
from var_offside.pipeline import AnalysisResult


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VAR Offside Analyzer")
        self.resize(1280, 800)

        # State
        self._current_image: np.ndarray = None
        self._source_path: str = None  # path of the loaded image / video
        self._phase1_result: AnalysisResult = (
            None  # after detection + optional grouping
        )
        self._last_result: AnalysisResult = None  # after full pipeline
        self._video = VideoHandler()

        # Worker
        self._worker = PipelineWorker()
        self._worker.stage_image_ready.connect(self._on_stage_image)
        self._worker.progress.connect(self._on_progress)
        self._worker.phase1_complete.connect(self._on_phase1_complete)
        self._worker.pipeline_complete.connect(self._on_complete)
        self._worker.offside_data_ready.connect(self._on_offside_data)
        self._worker.error.connect(self._on_error)

        # 3D replay worker
        self._var3d_worker = Var3DWorker()
        self._var3d_worker.progress.connect(self._on_progress)
        self._var3d_worker.finished_ok.connect(self._on_3d_ready)
        self._var3d_worker.error.connect(self._on_error)
        self._var3d_worker.cancelled.connect(self._on_3d_cancelled)

        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        self.stage_viewer = StageViewer()
        splitter.addWidget(self.stage_viewer)

        self.controls = ControlsPanel()
        splitter.addWidget(self.controls)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout.addWidget(splitter)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_open_image = QAction("Open Image", self)
        self.act_open_image.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open_image.triggered.connect(self._action_open_image)
        tb.addAction(self.act_open_image)

        self.act_open_video = QAction("Open Video", self)
        self.act_open_video.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_open_video.triggered.connect(self._action_open_video)
        tb.addAction(self.act_open_video)

        tb.addSeparator()

        self.act_run = QAction("Run", self)
        self.act_run.setShortcut(QKeySequence("F5"))
        self.act_run.triggered.connect(self._action_run)
        tb.addAction(self.act_run)

        self.act_continue = QAction("Continue", self)
        self.act_continue.setShortcut(QKeySequence("F6"))
        self.act_continue.setEnabled(False)
        self.act_continue.triggered.connect(self._action_continue)
        tb.addAction(self.act_continue)

        self.act_stop = QAction("Stop", self)
        self.act_stop.setShortcut(QKeySequence("Escape"))
        self.act_stop.setEnabled(False)
        self.act_stop.triggered.connect(self._action_stop)
        tb.addAction(self.act_stop)

        tb.addSeparator()

        self.act_save = QAction("Save", self)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save.triggered.connect(self._action_save_current)
        tb.addAction(self.act_save)

    def _build_statusbar(self):
        self.statusBar().showMessage("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _connect_signals(self):
        c = self.controls
        c.image_selected.connect(self._load_image)
        c.video_selected.connect(self._load_video)
        c.frame_changed.connect(self._seek_frame)
        c.run_requested.connect(self._action_run)
        c.continue_requested.connect(self._action_continue)
        c.stop_requested.connect(self._action_stop)
        c.edit_players_requested.connect(self._action_edit_players)
        c.save_current_requested.connect(self._action_save_current)
        c.save_all_requested.connect(self._action_save_all)
        c.generate_3d_requested.connect(self._action_generate_3d)

    # ------------------------------------------------------------------
    # Image / video loading
    # ------------------------------------------------------------------

    def _load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", f"Could not load image:\n{path}")
            return
        self._video.release()
        self._source_path = path
        self.controls.hide_video_controls()
        self._set_current_image(img)
        self.controls.set_filename(path)
        self.controls.reset_for_new_source()
        self._phase1_result = None
        self._last_result = None
        self.statusBar().showMessage(f"Loaded: {os.path.basename(path)}")

    def _load_video(self, path: str):
        if not self._video.load(path):
            QMessageBox.warning(self, "Error", f"Could not open video:\n{path}")
            return
        self._source_path = path
        self.controls.set_video_range(self._video.frame_count)
        self.controls.set_filename(path)
        self.controls.reset_for_new_source()
        self._phase1_result = None
        self._last_result = None
        frame = self._video.get_frame(0)
        if frame is not None:
            self._set_current_image(frame)
        self.statusBar().showMessage(
            f"Video: {os.path.basename(path)} ({self._video.frame_count} frames, "
            f"{self._video.fps:.1f} fps)"
        )

    def _seek_frame(self, idx: int):
        if not self._video.is_loaded:
            return
        frame = self._video.get_frame(idx)
        if frame is not None:
            self._set_current_image(frame)

    def _set_current_image(self, image: np.ndarray):
        self._current_image = image
        self.stage_viewer.set_stage_image("original", image)
        self.stage_viewer.switch_to_stage("original")

    # ------------------------------------------------------------------
    # Toolbar / control actions
    # ------------------------------------------------------------------

    def _action_open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", IMAGE_FILTER)
        if path:
            self._load_image(path)

    def _action_open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", VIDEO_FILTER)
        if path:
            self._load_video(path)

    def _action_run(self):
        """Phase 1: detection + optional player grouping."""
        if self._current_image is None:
            QMessageBox.information(self, "No Image", "Load an image or video first.")
            return
        if self._worker.isRunning():
            return

        self._set_running(True)
        self._worker.setup_phase1(
            image=self._current_image,
            enable_team_classification=self.controls.get_enable_team_classification(),
            use_automation=self.controls.get_use_automation(),
        )
        self._worker.start()

    def _action_continue(self):
        """Phase 2: calibration + offside analysis."""
        if self._phase1_result is None:
            QMessageBox.information(self, "No Data", "Run detection first (Phase 1).")
            return
        attacking_side = self.controls.get_attacking_side()
        if (
            attacking_side is None
            and self.controls.get_use_automation()
            and self._phase1_result.automation_result is not None
        ):
            attacking_side = self._phase1_result.automation_result.goal_side
            self.controls.set_attacking_side(attacking_side)
        if attacking_side is None:
            QMessageBox.information(
                self,
                "Attacking Side Required",
                "Choose Left or Right before continuing.",
            )
            return
        if self._worker.isRunning():
            return

        self._set_running(True)
        self._worker.setup_phase2(
            phase1_result=self._phase1_result,
            attacking_side=attacking_side,
            exclude_goalkeeper=self.controls.get_exclude_goalkeeper(),
        )
        self._worker.start()

    def _action_stop(self):
        if self._worker.isRunning():
            self._worker.request_stop()
            self.statusBar().showMessage("Stopping...")
        if self._var3d_worker.isRunning():
            self._var3d_worker.request_stop()
            self.statusBar().showMessage("Stopping 3D replay...")

    def _action_save_current(self):
        img = self.stage_viewer.get_current_stage_image()
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, img)
            self.statusBar().showMessage(f"Saved: {path}")

    def _action_save_all(self):
        if self._last_result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Save All Stages To")
        if not folder:
            return
        from var_offside.app.stage_viewer import STAGES

        for key, _name in STAGES:
            viewer = self.stage_viewer.get_viewer(key)
            img = viewer.get_current_image()
            if img is not None:
                out = os.path.join(folder, f"{key}.png")
                cv2.imwrite(out, img)
        self.statusBar().showMessage(f"All stages saved to {folder}")

    def _action_generate_3d(self, mode: str):
        """Render a 3D VAR replay from the completed analysis, in a worker."""
        if self._last_result is None or self._last_result.offside_result is None:
            QMessageBox.information(
                self, "No Result", "Run the full offside analysis (Continue) first."
            )
            return
        if self._worker.isRunning() or self._var3d_worker.isRunning():
            return

        # Output path next to the source (or the configured output dir).
        from var_offside.config import get_settings

        settings = get_settings()
        if self._source_path:
            stem = os.path.splitext(os.path.basename(self._source_path))[0]
        else:
            stem = "frame"
        out_dir = str(settings.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        suffix = {"cinematic": "3d", "zoomout": "3d_zoomout", "debug": "3d_debug"}.get(
            mode, "3d"
        )
        output_path = os.path.join(out_dir, f"{stem}_{suffix}.mp4")

        self._set_running(True)
        self.statusBar().showMessage("Generating 3D replay...")
        self._var3d_worker.setup(
            result=self._last_result,
            output_path=output_path,
            mode=mode,
            device=settings.device,
        )
        self._var3d_worker.start()

    def _on_3d_ready(self, video_path: str):
        """3D render finished — play it in a dialog."""
        self._set_running(False)
        self.controls.update_results(self._last_result)
        self.statusBar().showMessage(f"3D replay saved: {video_path}")
        try:
            dlg = VideoPlayerDialog(video_path, parent=self)
            dlg.exec_()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Playback",
                f"Rendered to:\n{video_path}\n\nCould not open the player: {exc}",
            )

    def _on_3d_cancelled(self):
        """3D render was stopped by the user."""
        self._set_running(False)
        self.statusBar().showMessage("3D replay cancelled")

    def _action_edit_players(self):
        """Open the player editor dialog (available after phase 1)."""
        result = self._phase1_result or self._last_result
        if result is None or not result.detections:
            QMessageBox.information(self, "No Data", "Run detection first.")
            return

        team_colors = {1: (255, 60, 60), 2: (60, 60, 255)}
        if result.team_result and result.team_result.team_colors:
            team_colors.update(result.team_result.team_colors)

        dlg = PlayerEditorDialog(
            image=result.image,
            detections=result.detections,
            team_colors=team_colors,
            parent=self,
        )
        if dlg.exec_() == PlayerEditorDialog.Accepted:
            edited = dlg.get_edited_detections()
            # If we haven't run phase 2 yet, just update phase1_result
            if self._last_result is None:
                self._phase1_result.detections = edited
                # Re-render teams tab with updated detections
                self._refresh_teams_tab(edited)
                if self.controls.get_use_automation():
                    self.statusBar().showMessage(
                        "Players updated. Automation side is kept; click Continue."
                    )
                else:
                    self.statusBar().showMessage(
                        "Players updated. Choose attack side and click Continue."
                    )
            else:
                # Full result exists — rerun from offside stage
                self._rerun_with_edits(edited)

    def _refresh_teams_tab(self, detections):
        """Re-draw the player visualisation after player edits (no model re-run)."""
        result = self._phase1_result
        if result is None:
            return
        pipeline = self._worker.get_pipeline()
        if result.team_result is not None:
            team_img = pipeline.team_assigner.visualize_teams(
                result.image, result.team_result
            )
        else:
            team_img = pipeline._draw_detections_debug(
                result.image, detections, "Players"
            )
        self.stage_viewer.set_stage_image("teams", team_img)
        self.stage_viewer.switch_to_stage("teams")

    def _rerun_with_edits(self, edited_detections):
        if self._worker.isRunning():
            return
        attacking_side = self.controls.get_attacking_side()
        if (
            attacking_side is None
            and self.controls.get_use_automation()
            and self._last_result is not None
            and self._last_result.automation_result is not None
        ):
            attacking_side = self._last_result.automation_result.goal_side
            self.controls.set_attacking_side(attacking_side)
        if attacking_side is None:
            QMessageBox.information(
                self,
                "Attacking Side Required",
                "Choose Left or Right before rerunning the line analysis.",
            )
            return
        self._set_running(True)
        self._worker.setup_rerun(
            cached_result=self._last_result,
            edited_detections=edited_detections,
            attacking_side=attacking_side,
            exclude_goalkeeper=self.controls.get_exclude_goalkeeper(),
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    def _on_stage_image(self, stage_key: str, image: np.ndarray):
        self.stage_viewer.set_stage_image(stage_key, image)

    def _on_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.statusBar().showMessage(message)

    def _on_phase1_complete(self, result: AnalysisResult):
        """Phase 1 done — detections finished, pause for review."""
        self._phase1_result = result
        self._last_result = None  # full result not ready yet
        self._set_running(False)

        self.controls.enter_phase1_review(result)
        if result.automation_result is not None:
            self.controls.set_attacking_side(result.automation_result.goal_side)
        self.act_continue.setEnabled(True)

        # Switch to the most relevant review tab
        if result.automation_result is not None:
            self.stage_viewer.switch_to_stage("automation")
        else:
            self.stage_viewer.switch_to_stage("teams")

        if result.errors:
            self.statusBar().showMessage(
                f"Phase 1 done with errors: {'; '.join(result.errors)}"
            )
        else:
            if result.automation_result is not None:
                self.statusBar().showMessage(
                    "Phase 1 complete - review automation results, edit if needed, then Continue"
                )
            else:
                self.statusBar().showMessage(
                    "Phase 1 complete - review players, edit if needed, set side, then Continue"
                )

    def _on_complete(self, result: AnalysisResult):
        """Phase 2 / full pipeline complete."""
        self._last_result = result
        self._set_running(False)
        self.controls.update_results(result)
        self.act_continue.setEnabled(False)

        if result.errors:
            self.statusBar().showMessage(
                f"Done with errors: {'; '.join(result.errors)}"
            )
        else:
            self.statusBar().showMessage(
                f"Pipeline complete ({result.processing_time_ms:.0f} ms)"
            )

    def _on_offside_data(self, offside_result, calibration):
        """Feed structured offside data to the overlay viewer."""
        self.stage_viewer.offside_viewer.set_offside_data(offside_result, calibration)

    def _on_error(self, msg: str):
        self._set_running(False)
        QMessageBox.critical(self, "Pipeline Error", msg)
        self.statusBar().showMessage(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_running(self, running: bool):
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setValue(0)
        self.act_run.setEnabled(not running)
        self.act_continue.setEnabled(not running and self.act_continue.isEnabled())
        self.act_stop.setEnabled(running)
        self.act_open_image.setEnabled(not running)
        self.act_open_video.setEnabled(not running)
        self.controls.set_running(running)

    def closeEvent(self, event):
        if self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)
        if self._var3d_worker.isRunning():
            self._var3d_worker.request_stop()
            self._var3d_worker.wait(3000)
        self._video.release()
        super().closeEvent(event)
