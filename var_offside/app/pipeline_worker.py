"""
Pipeline Worker - QThread worker with **two-phase** execution.

Phase 1 (Run):  Detection  ->  optional grouping   ->  PAUSE
    User reviews detections, edits players, selects attacking side.
Phase 2 (Continue):  Calibration  ->  GK  ->  Side  ->  Line  ->  Bird's Eye  ->  Result

Partial rerun (after player edits) skips detection/teams and reruns from
the offside calculation with the edited detections.
"""

import time
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from typing import Optional, List

from var_offside.pipeline import OffsidePipeline, AnalysisResult
from var_offside.detection import PlayerDetection


class PipelineWorker(QThread):
    """Two-phase pipeline worker."""

    # Signals
    stage_image_ready = pyqtSignal(str, object)  # (stage_key, np.ndarray)
    progress = pyqtSignal(int, str)  # (percent, message)
    phase1_complete = pyqtSignal(object)  # AnalysisResult (partial)
    pipeline_complete = pyqtSignal(object)  # AnalysisResult (full)
    offside_data_ready = pyqtSignal(
        object, object
    )  # (OffsideResult, CalibrationResult)
    error = pyqtSignal(str)  # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pipeline: Optional[OffsidePipeline] = None

        # Configuration (set before start)
        self._image: Optional[np.ndarray] = None
        self._attacking_team: Optional[int] = None
        self._attacking_side: Optional[str] = None
        self._exclude_goalkeeper: bool = True
        self._enable_team_classification: bool = False
        self._stop_requested = False

        # Run mode
        self._mode: str = "phase1"  # "phase1" | "phase2" | "rerun"

        # Cached intermediate result from phase 1
        self._phase1_result: Optional[AnalysisResult] = None

        # For partial reruns with edited detections
        self._edited_detections: Optional[List[PlayerDetection]] = None

    # ------------------------------------------------------------------
    # Public API (called from main thread before .start())
    # ------------------------------------------------------------------

    def setup_phase1(
        self,
        image: np.ndarray,
        enable_team_classification: bool = False,
    ):
        """Configure phase 1: detection + optional grouping."""
        self._image = image
        self._enable_team_classification = enable_team_classification
        self._mode = "phase1"
        self._phase1_result = None
        self._edited_detections = None
        self._stop_requested = False

    def setup_phase2(
        self,
        phase1_result: AnalysisResult,
        attacking_team: Optional[int] = None,
        attacking_side: Optional[str] = None,
        exclude_goalkeeper: bool = True,
    ):
        """Configure phase 2: calibration + offside (after user review)."""
        self._phase1_result = phase1_result
        self._image = phase1_result.image
        self._attacking_team = attacking_team
        self._attacking_side = attacking_side
        self._exclude_goalkeeper = exclude_goalkeeper
        self._enable_team_classification = phase1_result.team_result is not None
        self._mode = "phase2"
        self._edited_detections = None
        self._stop_requested = False

    def setup_rerun(
        self,
        cached_result: AnalysisResult,
        edited_detections: List[PlayerDetection],
        attacking_team: Optional[int] = None,
        attacking_side: Optional[str] = None,
        exclude_goalkeeper: bool = True,
    ):
        """Configure a partial rerun from the offside stage using edited detections."""
        self._image = cached_result.image
        self._phase1_result = cached_result
        self._edited_detections = edited_detections
        self._attacking_team = attacking_team
        self._attacking_side = attacking_side
        self._exclude_goalkeeper = exclude_goalkeeper
        self._enable_team_classification = cached_result.team_result is not None
        self._mode = "rerun"
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def get_pipeline(self) -> OffsidePipeline:
        """Lazy-init and return the pipeline (heavy model loading)."""
        if self._pipeline is None:
            self._pipeline = OffsidePipeline()
        return self._pipeline

    # ------------------------------------------------------------------
    # Thread entry
    # ------------------------------------------------------------------

    def run(self):
        try:
            if self._mode == "phase1":
                self._run_phase1()
            elif self._mode == "phase2":
                self._run_phase2()
            elif self._mode == "rerun":
                self._run_rerun()
        except Exception as exc:
            traceback.print_exc()
            self.error.emit(str(exc))

    # ------------------------------------------------------------------
    # Phase 1:  Original -> Detection -> Teams  (then pause)
    # ------------------------------------------------------------------

    def _run_phase1(self):
        pipeline = self.get_pipeline()
        image = self._image
        if image is None:
            self.error.emit("No image loaded")
            return

        result = AnalysisResult()
        result.image = image
        start = time.time()

        # -- Original (0%) --
        self.progress.emit(0, "Original image")
        self.stage_image_ready.emit("original", image)
        if self._stop_requested:
            return

        # -- Detection (15%) --
        self.progress.emit(10, "Detecting players...")
        try:
            result.detections = pipeline.player_detector.detect(image)
        except Exception as exc:
            self.error.emit(f"Detection failed: {exc}")
            return
        if self._stop_requested:
            return

        det_img = pipeline._draw_detections_debug(
            image, result.detections, "Detections"
        )
        self.stage_image_ready.emit("detection", det_img)
        self.progress.emit(15, f"Detected {len(result.detections)} players")

        if not result.detections:
            result.errors.append("No players detected")
            result.processing_time_ms = (time.time() - start) * 1000
            self.phase1_complete.emit(result)
            return

        self.progress.emit(20, "Preparing player groups...")

        if self._enable_team_classification:
            try:
                result.team_result = pipeline.team_assigner.assign_teams(
                    image, result.detections
                )
            except Exception as exc:
                result.team_result = None
                result.errors.append(f"Team assignment skipped: {exc}")
        else:
            result.team_result = None

        if self._stop_requested:
            return

        if result.team_result is not None:
            team_img = pipeline.team_assigner.visualize_teams(image, result.team_result)
        else:
            team_img = pipeline._draw_detections_debug(
                image, result.detections, "Players"
            )
        self.stage_image_ready.emit("teams", team_img)

        result.processing_time_ms = (time.time() - start) * 1000
        self.progress.emit(30, "Phase 1 complete - review players and set side")
        self.phase1_complete.emit(result)

    # ------------------------------------------------------------------
    # Phase 2:  Calibration -> GK -> Side -> Offside -> Bird's Eye -> Result
    # ------------------------------------------------------------------

    def _run_phase2(self):
        pipeline = self.get_pipeline()
        result = self._phase1_result
        if result is None:
            self.error.emit("No phase-1 result available")
            return
        image = result.image
        start = time.time()

        # -- Calibration (50%) --
        self.progress.emit(35, "Calibrating field...")
        try:
            result.calibration = pipeline.calibrator.calibrate(image)
        except Exception as exc:
            self.error.emit(f"Calibration failed: {exc}")
            return
        if self._stop_requested:
            return

        calib_img = pipeline.field_overlay.draw_calibration_result(
            image, result.calibration
        )
        if result.calibration.has_3d_calibration:
            calib_img = pipeline.field_overlay.draw_projected_field(
                calib_img, result.calibration
            )
        self.stage_image_ready.emit("calibration", calib_img)
        self.progress.emit(
            50, f"Calibration done (quality={result.calibration.quality_score:.2f})"
        )

        # -- GK identification --
        gk_indices = (None, None)
        if self._exclude_goalkeeper:
            self.progress.emit(55, "Identifying goalkeepers...")
            h, w = image.shape[:2]
            team_ids = None
            if result.team_result is not None:
                team_ids = [d.team_id for d in result.detections]
            gk_indices = pipeline.goalkeeper_filter.identify_goalkeepers(
                image,
                result.detections,
                team_ids,
                w,
                result.team_result.embeddings if result.team_result else None,
            )
            if gk_indices[0] is not None:
                result.detections[gk_indices[0]].is_goalkeeper = True
            if gk_indices[1] is not None:
                result.detections[gk_indices[1]].is_goalkeeper = True
        else:
            self.progress.emit(55, "Skipping goalkeepers...")
        if self._stop_requested:
            return

        # -- Offside stages (shared) --
        self._run_offside_stages(pipeline, result, image, gk_indices, start)

    # ------------------------------------------------------------------
    # Partial rerun (after player editing, skips detection/teams/calib)
    # ------------------------------------------------------------------

    def _run_rerun(self):
        pipeline = self.get_pipeline()
        cached = self._phase1_result
        if cached is None:
            self.error.emit("No cached result for rerun")
            return
        image = cached.image
        start = time.time()

        result = AnalysisResult()
        result.image = image
        result.detections = list(self._edited_detections)
        result.team_result = cached.team_result
        result.calibration = cached.calibration
        result.attacking_team = cached.attacking_team
        result.defending_team = cached.defending_team

        self.progress.emit(50, "Rerunning offside with edited players...")

        # Rebuild GK indices from is_goalkeeper flags
        gk1, gk2 = None, None
        for i, d in enumerate(result.detections):
            if d.is_goalkeeper:
                if d.team_id == 1 and gk1 is None:
                    gk1 = i
                elif d.team_id == 2 and gk2 is None:
                    gk2 = i
        gk_indices = (gk1, gk2)

        self._run_offside_stages(pipeline, result, image, gk_indices, start)

    # ------------------------------------------------------------------
    # Shared offside stages (used by phase 2 and rerun)
    # ------------------------------------------------------------------

    def _run_offside_stages(self, pipeline, result, image, gk_indices, start):
        h, w = image.shape[:2]

        # -- Side selection --
        self.progress.emit(60, "Using selected attacking side...")
        if self._attacking_side not in ("left", "right"):
            self.error.emit("Choose attacking side (left or right) before continuing")
            return

        result.side_result = pipeline.side_detector.detect_with_manual_override(
            result.attacking_team if result.attacking_team in (1, 2) else 1,
            self._attacking_side,
        )
        if self._stop_requested:
            return

        atk_side = self._attacking_side

        # -- Line + decision --
        self.progress.emit(65, "Calculating line player...")
        result.offside_line = pipeline.line_calculator.calculate_offside_line(
            result.detections,
            attacking_side=atk_side,
            calibration=result.calibration,
            exclude_goalkeeper=self._exclude_goalkeeper,
            defending_team=result.defending_team,
        )

        if result.offside_line is None:
            result.errors.append("Could not calculate line player")
            result.processing_time_ms = (time.time() - start) * 1000
            self.pipeline_complete.emit(result)
            return
        if self._stop_requested:
            return

        result.offside_result = pipeline.offside_decision.evaluate(
            result.detections,
            result.offside_line,
            attacking_side=atk_side,
            calibration=result.calibration,
            exclude_goalkeeper=self._exclude_goalkeeper,
            attacking_team=result.attacking_team,
            defending_team=result.defending_team,
        )

        offside_img = image.copy()
        self.stage_image_ready.emit("offside", offside_img)
        # Emit structured data so the overlay viewer can add interactive items
        self.offside_data_ready.emit(result.offside_result, result.calibration)
        self.progress.emit(70, "Line analysis complete")

        # -- Bird's Eye (85%) --
        self.progress.emit(80, "Generating bird's eye view...")
        birdseye_img = self._draw_birdseye(result)
        if birdseye_img is not None:
            self.stage_image_ready.emit("birdseye", birdseye_img)
            self.progress.emit(85, "Bird's eye ready")
        else:
            self.stage_image_ready.emit("birdseye", None)
            self.progress.emit(85, "Bird's eye skipped")
        if self._stop_requested:
            return

        # -- Clean result frame (100%) --
        self.progress.emit(90, "Creating result frame...")
        result.visualization = pipeline.visualizer.create_clean_result_view(
            image,
            result.offside_result,
            result.calibration,
        )
        self.stage_image_ready.emit("result", result.visualization)

        result.processing_time_ms = (time.time() - start) * 1000
        self.progress.emit(100, "Complete")
        self.pipeline_complete.emit(result)

    # ------------------------------------------------------------------
    # Bird's eye helper
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_birdseye(result: AnalysisResult):
        """Render a bird's eye view of the pitch with players."""
        try:
            from birdseye_var.pitch import PitchDrawer
        except ImportError:
            return None
        from var_offside.offside.line_calculator import project_detection_to_field

        calibration = result.calibration
        if calibration is None or not calibration.is_valid:
            return None
        if calibration.method == "vanishing_point":
            return None

        drawer = PitchDrawer(pixels_per_meter=10.0)
        pitch = drawer.draw()

        team_colors = {1: (255, 60, 60), 2: (60, 60, 255)}
        if result.team_result and result.team_result.team_colors:
            team_colors.update(result.team_result.team_colors)

        attacking_side = (
            result.offside_result.attacking_side
            if result.offside_result is not None
            else None
        )

        for idx, det in enumerate(result.detections):
            field_pt = None
            if attacking_side is not None:
                field_pt = project_detection_to_field(det, attacking_side, calibration)
            if field_pt is None:
                bx, by = det.bottom_center
                field_pt = calibration.project_to_field((bx, by))
            if field_pt is None:
                continue
            fx, fy = field_pt
            if fx < -5 or fx > 110 or fy < -5 or fy > 73:
                continue
            color = team_colors.get(det.team_id, (128, 128, 128))
            label = "GK" if det.is_goalkeeper else None
            if result.offside_line is not None:
                if idx == result.offside_line.ignored_player_idx:
                    label = "IGN"
                elif idx == result.offside_line.defining_player_idx:
                    label = "LINE"
            drawer.draw_player(pitch, fx, fy, color=color, label=label)

        if result.offside_line and result.offside_line.field_x is not None:
            is_offside = (
                result.offside_result.any_offside if result.offside_result else False
            )
            line_color = (0, 0, 255) if is_offside else (0, 200, 0)
            drawer.draw_offside_line(
                pitch, result.offside_line.field_x, color=line_color, label="Offside"
            )

        return pitch
