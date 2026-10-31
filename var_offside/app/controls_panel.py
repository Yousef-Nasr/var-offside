"""
Controls Panel - Right-side panel with source, video, offside options, and results.

Supports the two-phase pipeline flow:
  Phase 1 (Run)      -> detection + optional grouping -> pause for review
  Phase 2 (Continue) -> calibration + line analysis + result
"""

import os
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QSlider,
    QButtonGroup,
    QRadioButton,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QSpacerItem,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal


# Supported file filters
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mkv *.mov)"


class ControlsPanel(QWidget):
    """Right-hand control panel (fixed 260 px wide)."""

    # Signals
    image_selected = pyqtSignal(str)
    video_selected = pyqtSignal(str)
    frame_changed = pyqtSignal(int)
    run_requested = pyqtSignal()  # phase 1
    continue_requested = pyqtSignal()  # phase 2
    stop_requested = pyqtSignal()
    edit_players_requested = pyqtSignal()
    save_current_requested = pyqtSignal()
    save_all_requested = pyqtSignal()
    generate_3d_requested = pyqtSignal(str)  # emits the selected render mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self._manual_team_classification = False
        self._manual_exclude_goalkeeper = True
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # -- Source --
        src_group = QGroupBox("Source")
        src_lay = QVBoxLayout(src_group)

        self.btn_browse_image = QPushButton("Browse Image...")
        self.btn_browse_image.clicked.connect(self._browse_image)
        src_lay.addWidget(self.btn_browse_image)

        self.btn_browse_video = QPushButton("Browse Video...")
        self.btn_browse_video.clicked.connect(self._browse_video)
        src_lay.addWidget(self.btn_browse_video)

        self.lbl_filename = QLabel("No file loaded")
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setStyleSheet("color: #aaa; font-size: 11px;")
        src_lay.addWidget(self.lbl_filename)

        root.addWidget(src_group)

        # -- Video controls (hidden by default) --
        self.video_group = QGroupBox("Video")
        vid_lay = QVBoxLayout(self.video_group)

        nav_row = QHBoxLayout()
        self.btn_prev_frame = QPushButton("<<")
        self.btn_prev_frame.setFixedWidth(40)
        self.btn_prev_frame.clicked.connect(self._prev_frame)
        nav_row.addWidget(self.btn_prev_frame)

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.slider_frame.valueChanged.connect(self._on_slider)
        nav_row.addWidget(self.slider_frame)

        self.btn_next_frame = QPushButton(">>")
        self.btn_next_frame.setFixedWidth(40)
        self.btn_next_frame.clicked.connect(self._next_frame)
        nav_row.addWidget(self.btn_next_frame)

        vid_lay.addLayout(nav_row)

        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_frame_info.setAlignment(Qt.AlignCenter)
        vid_lay.addWidget(self.lbl_frame_info)

        self.video_group.setVisible(False)
        root.addWidget(self.video_group)

        # ============================================================
        # Phase 1: Run  (detection + optional grouping)
        # ============================================================
        self.btn_run = QPushButton(">>> Run Detection")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #2a7d2a; color: white; padding: 10px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3a9d3a; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.btn_run.clicked.connect(self.run_requested.emit)
        root.addWidget(self.btn_run)

        # -- Phase-1 review hint (hidden until phase 1 done) --
        self.lbl_review_hint = QLabel(
            "Review detections below.\n"
            "Edit players, choose the attacking side, then click Continue."
        )
        self.lbl_review_hint.setWordWrap(True)
        self.lbl_review_hint.setStyleSheet(
            "color: #ffcc00; font-size: 11px; padding: 4px;"
            "border: 1px solid #665500; background: #332a00;"
        )
        self.lbl_review_hint.setVisible(False)
        root.addWidget(self.lbl_review_hint)

        # -- Edit Players (enabled after phase 1) --
        self.btn_edit_players = QPushButton("Edit Players...")
        self.btn_edit_players.setEnabled(False)
        self.btn_edit_players.clicked.connect(self.edit_players_requested.emit)
        root.addWidget(self.btn_edit_players)

        # -- Offside Options --
        self.opt_group = QGroupBox("Offside Options")
        opt_lay = QVBoxLayout(self.opt_group)

        # Attacking side
        self.lbl_attacking_side = QLabel("Attacking Side:")
        opt_lay.addWidget(self.lbl_attacking_side)
        atk_side_row = QHBoxLayout()
        self.bg_atk_side = QButtonGroup(self)
        for label, val in [("Left", "left"), ("Right", "right")]:
            rb = QRadioButton(label)
            rb.setProperty("side_val", val)
            self.bg_atk_side.addButton(rb)
            atk_side_row.addWidget(rb)
        opt_lay.addLayout(atk_side_row)

        # Exclude GK
        self.chk_exclude_gk = QCheckBox("Exclude Goalkeeper")
        self.chk_exclude_gk.setChecked(True)
        opt_lay.addWidget(self.chk_exclude_gk)

        self.chk_team_classification = QCheckBox("Enable Team Classification")
        self.chk_team_classification.setChecked(False)
        self.chk_team_classification.toggled.connect(
            self._on_team_classification_toggled
        )
        opt_lay.addWidget(self.chk_team_classification)

        self.chk_use_automation = QCheckBox("Use Automation")
        self.chk_use_automation.setChecked(False)
        self.chk_use_automation.toggled.connect(self._on_use_automation_toggled)
        opt_lay.addWidget(self.chk_use_automation)

        root.addWidget(self.opt_group)

        # ============================================================
        # Phase 2: Continue  (calibration + offside)
        # ============================================================
        self.btn_continue = QPushButton(">>> Continue Analysis")
        self.btn_continue.setStyleSheet(
            "QPushButton { background-color: #1a5da8; color: white; padding: 10px; font-weight: bold; }"
            "QPushButton:hover { background-color: #2a7dc8; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self.continue_requested.emit)
        root.addWidget(self.btn_continue)

        # -- Results --
        self.results_group = QGroupBox("Results")
        res_lay = QVBoxLayout(self.results_group)

        self.lbl_players = QLabel("Players: --")
        res_lay.addWidget(self.lbl_players)
        self.lbl_teams = QLabel("Mode: --")
        res_lay.addWidget(self.lbl_teams)
        self.lbl_side = QLabel("Side: --")
        res_lay.addWidget(self.lbl_side)

        self.lbl_verdict = QLabel("")
        self.lbl_verdict.setAlignment(Qt.AlignCenter)
        self.lbl_verdict.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 8px;"
        )
        res_lay.addWidget(self.lbl_verdict)

        self.lbl_time = QLabel("Time: --")
        res_lay.addWidget(self.lbl_time)

        self.results_group.setVisible(False)
        root.addWidget(self.results_group)

        # ============================================================
        # 3D Replay  (available after full analysis)
        # ============================================================
        self.threed_group = QGroupBox("3D VAR Replay")
        td_lay = QVBoxLayout(self.threed_group)

        self.combo_3d_mode = QComboBox()
        self.combo_3d_mode.addItem("Cinematic replay", "cinematic")
        self.combo_3d_mode.addItem("Zoom-out bird's-eye", "zoomout")
        self.combo_3d_mode.addItem("Debug overlay (2D)", "debug")
        td_lay.addWidget(self.combo_3d_mode)

        self.btn_generate_3d = QPushButton("Generate 3D Replay")
        self.btn_generate_3d.setStyleSheet(
            "QPushButton { background-color: #7a3fa0; color: white; padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #9a5fc0; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.btn_generate_3d.setEnabled(False)
        self.btn_generate_3d.setToolTip(
            "Run the full offside analysis first. Needs full camera "
            "calibration (PnL) for a real 3D pitch."
        )
        self.btn_generate_3d.clicked.connect(self._emit_generate_3d)
        td_lay.addWidget(self.btn_generate_3d)

        self.threed_group.setVisible(False)
        root.addWidget(self.threed_group)

        self._on_use_automation_toggled(self.chk_use_automation.isChecked())

        # -- Save --
        save_row = QHBoxLayout()
        self.btn_save_current = QPushButton("Save Current")
        self.btn_save_current.clicked.connect(self.save_current_requested.emit)
        self.btn_save_current.setEnabled(False)
        save_row.addWidget(self.btn_save_current)

        self.btn_save_all = QPushButton("Save All")
        self.btn_save_all.clicked.connect(self.save_all_requested.emit)
        self.btn_save_all.setEnabled(False)
        save_row.addWidget(self.btn_save_all)
        root.addLayout(save_row)

        root.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_attacking_side(self):
        """Return 'left', 'right', or None when not chosen yet."""
        btn = self.bg_atk_side.checkedButton()
        return btn.property("side_val") if btn else None

    def get_exclude_goalkeeper(self) -> bool:
        return self.chk_exclude_gk.isChecked()

    def get_enable_team_classification(self) -> bool:
        return self.chk_team_classification.isChecked()

    def get_use_automation(self) -> bool:
        return self.chk_use_automation.isChecked()

    def set_attacking_side(self, side: str):
        """Select the attacking-side radio button programmatically."""
        for button in self.bg_atk_side.buttons():
            if button.property("side_val") == side:
                button.setChecked(True)
                return

    def clear_attacking_side(self):
        """Clear the attacking-side selection."""
        for button in self.bg_atk_side.buttons():
            button.setAutoExclusive(False)
            button.setChecked(False)
            button.setAutoExclusive(True)

    def set_attacking_side_enabled(self, enabled: bool):
        """Enable or disable manual attacking-side selection."""
        for button in self.bg_atk_side.buttons():
            button.setEnabled(enabled)

    def set_filename(self, path: str):
        self.lbl_filename.setText(os.path.basename(path))

    def set_video_range(self, frame_count: int):
        self.video_group.setVisible(True)
        self.slider_frame.setMaximum(max(0, frame_count - 1))
        self.slider_frame.setValue(0)
        self.lbl_frame_info.setText(f"Frame: 0 / {frame_count}")
        # Scrub to the kick moment, then analyse that single frame.
        self.btn_run.setText(">>> Analyze This Frame")

    def hide_video_controls(self):
        self.video_group.setVisible(False)
        self.btn_run.setText(">>> Run Detection")

    # -- Phase-state transitions ------------------------------------------

    def enter_phase1_review(self, result):
        """
        Called when phase 1 (detection + teams) completes.
        Show detection counts, enable Edit Players and Continue, show hint.
        """
        self.lbl_review_hint.setVisible(True)
        self.btn_edit_players.setEnabled(True)
        self.btn_continue.setEnabled(True)
        self.btn_run.setEnabled(True)

        # Show partial results
        self.results_group.setVisible(True)
        self.lbl_players.setText(f"Players: {result.num_players_detected}")
        classification = "on" if result.team_result is not None else "off"
        automation = "on" if result.automation_result is not None else "off"
        self.lbl_teams.setText(
            "Geometry mode: bbox bottom line | "
            f"team classification {classification} | automation {automation}"
        )
        if result.automation_result is not None:
            self.set_attacking_side_enabled(False)
            self.lbl_side.setText(
                f"Side: suggested {result.automation_result.goal_side} from automation"
            )
            self.lbl_verdict.setText(
                "Automation filled the side and goalkeeper filtering"
            )
        else:
            self.set_attacking_side_enabled(True)
            self.lbl_side.setText("Side: choose left or right below")
            self.lbl_verdict.setText("Choose attacking side to continue")
        self.lbl_verdict.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 8px;"
            "background-color: #333; color: #ffcc00;"
        )
        self.lbl_time.setText(f"Phase 1: {result.processing_time_ms:.0f} ms")

    def update_results(self, result):
        """Populate full results after phase 2 / full pipeline completes."""
        self.lbl_review_hint.setVisible(False)
        self.results_group.setVisible(True)
        self.lbl_players.setText(f"Players: {result.num_players_detected}")

        classification = "on" if result.team_result is not None else "off"
        automation = "on" if result.automation_result is not None else "off"
        self.lbl_teams.setText(
            "Geometry mode: bbox bottom line | "
            f"team classification {classification} | automation {automation}"
        )

        if result.offside_result is not None:
            self.lbl_side.setText(
                f"Attack side: {result.offside_result.attacking_side}"
            )
        elif result.side_result and result.attacking_team in (1, 2):
            self.lbl_side.setText(
                f"Attack side: {result.side_result.get_attacking_side(result.attacking_team)}"
            )
        elif result.side_result:
            self.lbl_side.setText(f"Attack side: {result.side_result.team1_attacking}")

        if result.offside_result:
            if result.offside_result.any_offside:
                n = result.offside_result.num_ahead
                self.lbl_verdict.setText(f"AHEAD OF LINE ({n})")
                self.lbl_verdict.setStyleSheet(
                    "font-size: 16px; font-weight: bold; padding: 8px;"
                    "background-color: #8b0000; color: white;"
                )
            else:
                self.lbl_verdict.setText("NO PLAYER AHEAD")
                self.lbl_verdict.setStyleSheet(
                    "font-size: 16px; font-weight: bold; padding: 8px;"
                    "background-color: #006400; color: white;"
                )

        self.lbl_time.setText(f"Time: {result.processing_time_ms:.0f} ms")
        self.btn_edit_players.setEnabled(True)
        self.btn_save_current.setEnabled(True)
        self.btn_save_all.setEnabled(True)
        self.btn_continue.setEnabled(False)

        # 3D replay becomes available once we have an offside result.
        has_offside = result.offside_result is not None
        self.threed_group.setVisible(has_offside)
        self.btn_generate_3d.setEnabled(has_offside)

    def set_running(self, running: bool):
        """Toggle UI state while any pipeline phase is running."""
        self.btn_run.setEnabled(not running)
        self.btn_continue.setEnabled(not running and self.btn_continue.isEnabled())
        self.btn_browse_image.setEnabled(not running)
        self.btn_browse_video.setEnabled(not running)
        self.chk_use_automation.setEnabled(not running)
        self.chk_team_classification.setEnabled(not running)
        self.btn_edit_players.setEnabled(
            not running and self.btn_edit_players.isEnabled()
        )
        # Derive from visibility (set once an offside result exists) rather
        # than the button's own state, so a failed 3D render can be retried.
        self.btn_generate_3d.setEnabled(
            not running and not self.threed_group.isHidden()
        )

    def reset_for_new_source(self):
        """Reset controls when a new image/video is loaded."""
        self.lbl_review_hint.setVisible(False)
        self.btn_edit_players.setEnabled(False)
        self.btn_continue.setEnabled(False)
        self.btn_save_current.setEnabled(False)
        self.btn_save_all.setEnabled(False)
        self.results_group.setVisible(False)
        self.threed_group.setVisible(False)
        self.btn_generate_3d.setEnabled(False)
        self.lbl_verdict.setText("")
        self.clear_attacking_side()
        self.set_attacking_side_enabled(not self.chk_use_automation.isChecked())

    def _on_team_classification_toggled(self, enabled: bool):
        if self.chk_use_automation.isChecked():
            self.lbl_teams.setText("Mode: automation + required team classification")
            return
        if enabled:
            self.lbl_teams.setText("Mode: geometry + team classification")
        else:
            self.lbl_teams.setText("Mode: geometry only")

    def _on_use_automation_toggled(self, enabled: bool):
        if enabled:
            self._manual_team_classification = self.chk_team_classification.isChecked()
            self._manual_exclude_goalkeeper = self.chk_exclude_gk.isChecked()
            self.chk_team_classification.setChecked(True)
            self.chk_exclude_gk.setChecked(True)
            self.chk_team_classification.setEnabled(False)
            self.chk_exclude_gk.setEnabled(False)
            self.clear_attacking_side()
            self.set_attacking_side_enabled(False)
            self.lbl_attacking_side.setText("Attacking Side (automatic):")
        else:
            self.chk_team_classification.setEnabled(True)
            self.chk_exclude_gk.setEnabled(True)
            self.chk_team_classification.setChecked(self._manual_team_classification)
            self.chk_exclude_gk.setChecked(self._manual_exclude_goalkeeper)
            self.set_attacking_side_enabled(True)
            self.lbl_attacking_side.setText("Attacking Side:")
        self._on_team_classification_toggled(self.chk_team_classification.isChecked())

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _emit_generate_3d(self):
        mode = self.combo_3d_mode.currentData() or "cinematic"
        self.generate_3d_requested.emit(mode)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", IMAGE_FILTER)
        if path:
            self.hide_video_controls()
            self.set_filename(path)
            self.image_selected.emit(path)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", VIDEO_FILTER)
        if path:
            self.set_filename(path)
            self.video_selected.emit(path)

    def _on_slider(self, value: int):
        total = self.slider_frame.maximum() + 1
        self.lbl_frame_info.setText(f"Frame: {value} / {total}")
        self.frame_changed.emit(value)

    def _prev_frame(self):
        v = self.slider_frame.value()
        if v > 0:
            self.slider_frame.setValue(v - 1)

    def _next_frame(self):
        v = self.slider_frame.value()
        if v < self.slider_frame.maximum():
            self.slider_frame.setValue(v + 1)
