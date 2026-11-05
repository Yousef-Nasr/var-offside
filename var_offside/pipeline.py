"""
Main offside detection pipeline.
Orchestrates all components for end-to-end offside analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union

from var_offside.config import get_settings
from var_offside.detection import PlayerDetector, PlayerDetection, GoalkeeperFilter
from var_offside.teams import TeamAssigner, TeamAssignmentResult
from var_offside.calibration import get_calibrator, CalibrationResult
from var_offside.offside import (
    SideDetector,
    AttackingSideResult,
    OffsideLineCalculator,
    OffsideLine,
    OffsideDecision,
    OffsideResult,
    MeshOffsideCalculator,
)
from var_offside.visualization import OffsideVisualizer, PlayerVisualizer, FieldOverlay
from var_offside.visualization.line_projection import project_line


@dataclass
class AnalysisResult:
    """Complete analysis result."""

    # Input information
    image_path: Optional[str] = None
    image: Optional[np.ndarray] = None

    # Detection results
    detections: List[PlayerDetection] = field(default_factory=list)

    # Team assignment
    team_result: Optional[TeamAssignmentResult] = None

    # Calibration
    calibration: Optional[CalibrationResult] = None

    # Side detection
    side_result: Optional[AttackingSideResult] = None

    # Offside analysis
    offside_line: Optional[OffsideLine] = None
    offside_result: Optional[OffsideResult] = None
    attacking_team: Optional[int] = None
    defending_team: Optional[int] = None
    keeper_choice: Optional[str] = None
    verdict_message: Optional[str] = None

    # Visualization
    visualization: Optional[np.ndarray] = None

    # 3D VAR video path (set when generate_3d_video=True)
    var_3d_video_path: Optional[str] = None

    # Debug 3D video path (set when generate_debug_3d_video=True)
    debug_3d_video_path: Optional[str] = None

    # Debug 3D image path (zoomed-out still with full pitch projection)
    debug_3d_image_path: Optional[str] = None

    # Zoom-out VAR video path (set when generate_zoomout_video=True)
    zoomout_video_path: Optional[str] = None

    # Processing metadata
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def is_offside(self) -> bool:
        """Whether any player is ahead of the detected line."""
        if self.offside_result:
            return self.offside_result.any_offside
        return False

    @property
    def num_players_detected(self) -> int:
        """Number of players detected."""
        return len(self.detections)


class OffsidePipeline:
    """
    Main pipeline for offside detection.

    Pipeline stages:
    1. Load image
    2. Calibrate field (get homography)
    3. Detect players (YOLO Pose)
    4. Assign teams (ResNet embeddings + K-Means)
    5. Identify goalkeepers
    6. Detect attacking side
    7. Calculate line player
    8. Evaluate players ahead of the line
    9. Generate visualization
    """

    def __init__(self, calibration_method: str = "auto", device: Optional[str] = None):
        """
        Initialize the offside pipeline.

        Args:
            calibration_method: Calibration method ('pnl', 'vanishing_point', 'manual', 'auto')
            device: Compute device ('cuda' or 'cpu')
        """
        self.settings = get_settings()
        if device:
            self.settings.device = device

        # Initialize components
        self.player_detector = PlayerDetector()
        self.team_assigner = TeamAssigner()
        self.goalkeeper_filter = GoalkeeperFilter()
        self.calibrator = get_calibrator(calibration_method)
        self.side_detector = SideDetector()
        self.line_calculator = OffsideLineCalculator()
        self.offside_decision = OffsideDecision()
        # 3D mesh-based offside (HF `mirrash7/VAR` method), used as a fallback.
        self.mesh_offside_calculator = MeshOffsideCalculator(
            device=self.settings.device
        )

        # Visualization
        self.visualizer = OffsideVisualizer()
        self.player_viz = PlayerVisualizer()
        self.field_overlay = FieldOverlay()

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        attacking_team: Optional[int] = None,
        attacking_side: Optional[str] = None,
        enable_team_classification: bool = False,
        exclude_goalkeeper: bool = True,
        offside_method: str = "geometry",
        generate_visualization: bool = True,
        generate_3d_video: bool = False,
        generate_debug_3d_video: bool = False,
        generate_zoomout_video: bool = False,
        save_result: bool = False,
        output_path: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> AnalysisResult:
        """
        Run complete offside analysis on an image.

        Args:
            image: Image path or numpy array
            attacking_team: Team id for the attackers when known (1 or 2)
            attacking_side: Manual attacking direction ('left' or 'right')
            enable_team_classification: Whether to run team clustering
            exclude_goalkeeper: Whether to exclude detected goalkeepers
            offside_method: Offside computation method:
                'geometry' (default) uses the bbox-corner + homography line;
                'mesh_3d' uses the 3D body-mesh method (HF mirrash7/VAR),
                measuring the furthest-forward body point per player;
                'auto' runs geometry and falls back to mesh_3d when the
                geometry line has no reliable field projection.
            generate_visualization: Whether to generate visualization
            generate_3d_video: Whether to generate a 3D VAR-style video
            generate_debug_3d_video: Whether to generate a debug 3D video (static broadcast camera, no orbit)
            generate_zoomout_video: Whether to generate a zoom-out VAR video (pitch overlay + zoom to full pitch)
            save_result: Whether to save output images
            output_path: Custom output path
            verbose: Print debug information
            debug: Save debug images for each stage

        Returns:
            AnalysisResult with all analysis data
        """
        import time

        start_time = time.time()

        result = AnalysisResult()

        # Setup debug output directory
        debug_dir = None
        if debug:
            debug_dir = self.settings.output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Debug] Saving stage images to: {debug_dir}")

        # 1. Load image
        if isinstance(image, (str, Path)):
            result.image_path = str(image)
            result.image = cv2.imread(str(image))
            if result.image is None:
                result.errors.append(f"Could not load image: {image}")
                return result
        else:
            result.image = image

        image_np = result.image
        h, w = image_np.shape[:2]
        effective_team_classification = (
            enable_team_classification or attacking_team in (1, 2)
        )

        # Save original image for debug
        if debug:
            cv2.imwrite(str(debug_dir / "1_original.jpg"), image_np)
            print(f"[Debug] Saved: 1_original.jpg")

        try:
            # 2. Calibrate field
            result.calibration = self.calibrator.calibrate(image_np)

            if debug:
                calib_img = self.field_overlay.draw_calibration_result(
                    image_np, result.calibration
                )
                # Also draw projected field lines if 3D calibration available
                if result.calibration.has_3d_calibration:
                    calib_img = self.field_overlay.draw_projected_field(
                        calib_img, result.calibration
                    )
                cv2.imwrite(str(debug_dir / "2_calibration.jpg"), calib_img)
                print(
                    f"[Debug] Saved: 2_calibration.jpg (quality={result.calibration.quality_score:.2f})"
                )

            # 3. Detect players
            if verbose:
                print("[Pipeline] Running player detection...")
            result.detections = self.player_detector.detect(image_np, verbose=verbose)

            if debug:
                det_img = self._draw_detections_debug(
                    image_np, result.detections, "Raw Detections"
                )
                cv2.imwrite(str(debug_dir / "3_detections.jpg"), det_img)
                print(
                    f"[Debug] Saved: 3_detections.jpg ({len(result.detections)} players)"
                )

            if not result.detections:
                result.errors.append("No players detected")
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result

            # 4. Assign teams (optional; geometry mode does not depend on it)
            if effective_team_classification:
                try:
                    result.team_result = self.team_assigner.assign_teams(
                        image_np, result.detections
                    )
                except Exception as exc:
                    result.team_result = None
                    result.errors.append(f"Team assignment skipped: {exc}")
            else:
                result.team_result = None

            if debug and result.team_result is not None:
                team_img = self.team_assigner.visualize_teams(
                    image_np, result.team_result
                )
                cv2.imwrite(str(debug_dir / "4_teams.jpg"), team_img)
                print(f"[Debug] Saved: 4_teams.jpg")

            # 5. Identify goalkeepers
            gk_indices = (None, None)
            if exclude_goalkeeper:
                team_ids = None
                if result.team_result is not None:
                    team_ids = [d.team_id for d in result.detections]
                gk_indices = self.goalkeeper_filter.identify_goalkeepers(
                    image_np,
                    result.detections,
                    team_ids,
                    w,
                    result.team_result.embeddings if result.team_result else None,
                )

                if gk_indices[0] is not None:
                    result.detections[gk_indices[0]].is_goalkeeper = True
                if gk_indices[1] is not None:
                    result.detections[gk_indices[1]].is_goalkeeper = True

            if attacking_team in (1, 2):
                result.attacking_team = attacking_team
                result.defending_team = 1 if attacking_team == 2 else 2

            if debug:
                gk_img = self._draw_goalkeepers_debug(
                    image_np, result.detections, gk_indices
                )
                cv2.imwrite(str(debug_dir / "5_goalkeepers.jpg"), gk_img)
                print(f"[Debug] Saved: 5_goalkeepers.jpg")

            if attacking_side not in ("left", "right"):
                result.errors.append(
                    "attacking_side must be 'left' or 'right'"
                )
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result

            if result.attacking_team in (1, 2):
                result.side_result = self.side_detector.detect_with_manual_override(
                    result.attacking_team,
                    attacking_side,
                )
            else:
                result.side_result = AttackingSideResult(
                    team1_attacking=attacking_side,
                    team2_attacking="right" if attacking_side == "left" else "left",
                    confidence=1.0,
                    method="manual_geometry",
                )
            atk_side = attacking_side

            if debug:
                side_img = self._draw_side_debug(
                    image_np, result.side_result, result.detections
                )
                cv2.imwrite(str(debug_dir / "6_attacking_side.jpg"), side_img)
                print(f"[Debug] Saved: 6_attacking_side.jpg (attack side={atk_side})")

            # 7. Calculate offside line
            result.offside_line = self.line_calculator.calculate_offside_line(
                result.detections,
                attacking_side=atk_side,
                calibration=result.calibration,
                exclude_goalkeeper=exclude_goalkeeper,
                defending_team=result.defending_team,
            )

            if result.offside_line is None:
                result.errors.append("Could not calculate offside line")
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result

            if debug:
                line_img = self._draw_offside_line_debug(
                    image_np,
                    result.offside_line,
                    result.detections,
                    result.calibration,
                )
                cv2.imwrite(str(debug_dir / "7_offside_line.jpg"), line_img)
                print(
                    f"[Debug] Saved: 7_offside_line.jpg (x={result.offside_line.image_x:.0f})"
                )

            # 8. Evaluate offside
            result.offside_result = self.offside_decision.evaluate(
                result.detections,
                result.offside_line,
                attacking_side=atk_side,
                calibration=result.calibration,
                exclude_goalkeeper=exclude_goalkeeper,
                attacking_team=result.attacking_team,
                defending_team=result.defending_team,
            )

            # 8b. Optional 3D mesh-based offside (HF mirrash7/VAR method).
            #     'mesh_3d' forces it; 'auto' only falls back when the geometry
            #     line lacks a reliable field projection.
            geometry_line_unreliable = result.offside_line.field_x is None
            want_mesh = offside_method == "mesh_3d" or (
                offside_method == "auto" and geometry_line_unreliable
            )
            if want_mesh:
                try:
                    mesh_line, mesh_result = self.mesh_offside_calculator.compute(
                        image_np,
                        result.detections,
                        attacking_side=atk_side,
                        calibration=result.calibration,
                        exclude_goalkeeper=exclude_goalkeeper,
                        attacking_team=result.attacking_team,
                        defending_team=result.defending_team,
                        line_defender_idx=result.offside_line.defining_player_idx,
                    )
                    if mesh_line is not None and mesh_result is not None:
                        result.offside_line = mesh_line
                        result.offside_result = mesh_result
                        if verbose:
                            print("[Pipeline] Used 3D mesh-based offside method")
                    else:
                        result.errors.append(
                            "3D mesh offside unavailable; kept geometry result"
                        )
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"3D mesh offside failed: {exc}")

            # 9. Generate visualization
            if generate_visualization:
                result.visualization = self.visualizer.create_clean_result_view(
                    image_np, result.offside_result, result.calibration
                )

                if debug:
                    cv2.imwrite(
                        str(debug_dir / "8_final_result.jpg"), result.visualization
                    )
                    print(f"[Debug] Saved: 8_final_result.jpg")

                # Save if requested
                if save_result:
                    self._save_result(result, output_path)

            # 10. Generate 3D VAR video
            if generate_3d_video and result.image_path:
                from var_offside.visualization.var_3d import render_var_3d_video

                video_out = output_path or str(
                    self.settings.output_dir
                    / f"{Path(result.image_path).stem}_offside_3d.mp4"
                )
                if not video_out.endswith(".mp4"):
                    video_out = str(Path(video_out).with_suffix(".mp4"))

                result.var_3d_video_path = render_var_3d_video(
                    image_path=result.image_path,
                    analysis_result=result,
                    output_path=video_out,
                    device=self.settings.device,
                )

            # 11. Generate debug 3D video
            if generate_debug_3d_video and result.image_path:
                from var_offside.visualization.var_3d import render_debug_3d_video

                debug_video_out = output_path or str(
                    self.settings.output_dir
                    / f"{Path(result.image_path).stem}_debug_3d.mp4"
                )
                if not debug_video_out.endswith(".mp4"):
                    debug_video_out = str(Path(debug_video_out).with_suffix(".mp4"))

                result.debug_3d_video_path = render_debug_3d_video(
                    image_path=result.image_path,
                    analysis_result=result,
                    output_path=debug_video_out,
                    device=self.settings.device,
                )

            # 12. Generate zoom-out VAR video
            if generate_zoomout_video and result.image_path:
                from var_offside.visualization.var_3d import render_zoomout_3d_video

                zoomout_out = output_path or str(
                    self.settings.output_dir
                    / f"{Path(result.image_path).stem}_zoomout.mp4"
                )
                if not zoomout_out.endswith(".mp4"):
                    zoomout_out = str(Path(zoomout_out).with_suffix(".mp4"))

                result.zoomout_video_path = render_zoomout_3d_video(
                    image_path=result.image_path,
                    analysis_result=result,
                    output_path=zoomout_out,
                    device=self.settings.device,
                )

        except Exception as e:
            result.errors.append(f"Pipeline error: {str(e)}")
            import traceback

            traceback.print_exc()

        result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    def _draw_detections_debug(
        self, image: np.ndarray, detections: List[PlayerDetection], title: str = ""
    ) -> np.ndarray:
        """Draw detection boxes for debug visualization."""
        output = image.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            color = (0, 255, 0)  # Green
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                f"#{i} ({det.confidence:.2f})",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Title
        cv2.putText(
            output,
            f"{title} ({len(detections)} players)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return output

    def _draw_goalkeepers_debug(
        self, image: np.ndarray, detections: List[PlayerDetection], gk_indices: tuple
    ) -> np.ndarray:
        """Draw players with goalkeepers highlighted."""
        output = image.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(c) for c in det.bbox]

            if det.is_goalkeeper:
                color = (0, 255, 255)  # Yellow for GK
                label = f"GK Team {det.team_id}"
                thickness = 3
            else:
                color = (255, 0, 0) if det.team_id == 1 else (0, 0, 255)
                label = f"T{det.team_id}"
                thickness = 2

            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        cv2.putText(
            output,
            "Goalkeepers Identified",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return output

    def _draw_side_debug(
        self, image: np.ndarray, side_result, detections: List[PlayerDetection]
    ) -> np.ndarray:
        """Draw attacking side visualization."""
        output = image.copy()
        h, w = image.shape[:2]

        # Draw center line
        cv2.line(output, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

        # Draw arrows for attacking direction
        arrow_y = 50
        if side_result.team1_attacking == "left":
            cv2.arrowedLine(
                output,
                (w // 2 + 100, arrow_y),
                (100, arrow_y),
                (255, 0, 0),
                3,
                tipLength=0.3,
            )
            cv2.putText(
                output,
                "Team 1",
                (w // 2 + 110, arrow_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
        else:
            cv2.arrowedLine(
                output,
                (w // 2 - 100, arrow_y),
                (w - 100, arrow_y),
                (255, 0, 0),
                3,
                tipLength=0.3,
            )
            cv2.putText(
                output,
                "Team 1",
                (w // 2 - 180, arrow_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        arrow_y = 90
        if side_result.team2_attacking == "left":
            cv2.arrowedLine(
                output,
                (w // 2 + 100, arrow_y),
                (100, arrow_y),
                (0, 0, 255),
                3,
                tipLength=0.3,
            )
            cv2.putText(
                output,
                "Team 2",
                (w // 2 + 110, arrow_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        else:
            cv2.arrowedLine(
                output,
                (w // 2 - 100, arrow_y),
                (w - 100, arrow_y),
                (0, 0, 255),
                3,
                tipLength=0.3,
            )
            cv2.putText(
                output,
                "Team 2",
                (w // 2 - 180, arrow_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Draw players
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            color = (255, 0, 0) if det.team_id == 1 else (0, 0, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        return output

    def _draw_offside_line_debug(
        self,
        image: np.ndarray,
        offside_line,
        detections: List[PlayerDetection],
        calibration: Optional[CalibrationResult] = None,
    ) -> np.ndarray:
        """Draw offside line on image."""
        output = image.copy()
        h, w = image.shape[:2]

        # Draw offside line
        points = project_line(
            offside_line.field_x,
            offside_line.image_x,
            offside_line.reference_point[1],
            h,
            w,
            calibration,
        )
        for i in range(len(points) - 1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
            cv2.line(output, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # Draw players
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            if det.is_goalkeeper:
                color = (0, 255, 255)
            else:
                color = (255, 0, 0) if det.team_id == 1 else (0, 0, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Mark defining player
        if offside_line.defining_player_idx < len(detections):
            def_det = detections[offside_line.defining_player_idx]
            cx, cy = [int(v) for v in offside_line.reference_point]
            cv2.circle(output, (cx, cy), 15, (0, 255, 0), 3)

        cv2.putText(
            output,
            f"Line Player (x={int(offside_line.image_x)})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        return output

    def analyze_with_manual_calibration(
        self, image: Union[str, Path, np.ndarray], **kwargs
    ) -> AnalysisResult:
        """
        Run analysis with interactive manual calibration.

        Args:
            image: Input image
            **kwargs: Additional arguments for analyze()

        Returns:
            AnalysisResult
        """
        from .calibration import ManualCalibration

        # Load image
        if isinstance(image, (str, Path)):
            image_np = cv2.imread(str(image))
        else:
            image_np = image

        # Run manual calibration
        manual_calib = ManualCalibration()
        calib_result = manual_calib.calibrate(image_np)

        # Store original calibrator
        original_calibrator = self.calibrator

        # Create a dummy calibrator that returns the manual result
        class PrecomputedCalibrator:
            def __init__(self, result):
                self._result = result

            def calibrate(self, img):
                return self._result

            def get_name(self):
                return "manual"

        self.calibrator = PrecomputedCalibrator(calib_result)

        # Run analysis
        result = self.analyze(image, **kwargs)

        # Restore original calibrator
        self.calibrator = original_calibrator

        return result

    def _save_result(self, result: AnalysisResult, output_path: Optional[str] = None):
        """Save analysis result to disk."""
        if output_path is None:
            output_dir = self.settings.output_dir

            if result.image_path:
                base_name = Path(result.image_path).stem
            else:
                import datetime

                base_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            output_path = output_dir / f"{base_name}_offside.jpg"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if result.visualization is not None:
            cv2.imwrite(str(output_path), result.visualization)
            print(f"Saved visualization to: {output_path}")

    def get_summary(self, result: AnalysisResult) -> str:
        """Get human-readable summary of analysis."""
        lines = []
        lines.append("=" * 50)
        lines.append("VAR OFFSIDE ANALYSIS RESULT")
        lines.append("=" * 50)

        if result.image_path:
            lines.append(f"Image: {result.image_path}")

        lines.append(f"Players detected: {result.num_players_detected}")

        if result.team_result:
            team1_count = sum(1 for d in result.detections if d.team_id == 1)
            team2_count = sum(1 for d in result.detections if d.team_id == 2)
            lines.append(
                f"Team 1: {team1_count} players | Team 2: {team2_count} players"
            )

        if result.calibration:
            lines.append(
                f"Calibration: {result.calibration.method} (quality: {result.calibration.quality_score:.2f})"
            )

        if result.side_result:
            if result.offside_result is not None:
                lines.append(f"Attack side: {result.offside_result.attacking_side}")
            elif result.attacking_team in (1, 2):
                lines.append(
                    "Attack side: "
                    f"{result.side_result.get_attacking_side(result.attacking_team)}"
                )
            else:
                lines.append(f"Attack side: {result.side_result.team1_attacking}")

        if result.attacking_team in (1, 2):
            lines.append(
                f"Attacking team: {result.attacking_team} | Defending team: {result.defending_team}"
            )

        if result.offside_result:
            lines.append("")
            if result.offside_result.any_offside:
                lines.append("Players ahead of the line detected")
                lines.append(
                    f"   {result.offside_result.num_ahead} player(s) ahead of the line"
                )
            else:
                lines.append("No player is ahead of the line")
                lines.append("   All evaluated players are behind the line")

        if result.errors:
            lines.append("")
            lines.append("Errors:")
            for error in result.errors:
                lines.append(f"  - {error}")

        lines.append("")
        lines.append(f"Processing time: {result.processing_time_ms:.1f}ms")
        lines.append("=" * 50)

        return "\n".join(lines)
