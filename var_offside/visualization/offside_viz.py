"""
VAR-style offside visualization.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from var_offside.offside.line_calculator import (
    get_directional_ground_point,
)
from var_offside.offside.decision import OffsideResult
from var_offside.calibration.base import CalibrationResult
from var_offside.config import get_settings
from var_offside.visualization.player_viz import PlayerVisualizer
from var_offside.visualization.line_projection import project_line


class OffsideVisualizer:
    """
    Create VAR-style offside visualizations.

    Features:
    - Offside line drawing
    - Player body part markers
    - Offside/onside indicators
    - Split-view comparison
    """

    def __init__(self):
        """Initialize offside visualizer."""
        settings = get_settings()
        self.viz_config = settings.visualization

        self.offside_color = self.viz_config.offside_line_color  # Red
        self.onside_color = self.viz_config.onside_line_color  # Green
        self.clean_line_color = (255, 255, 0)  # UEFA-style cyan in BGR
        self.clean_player_color = (255, 220, 80)
        self.line_thickness = 3

        self.player_viz = PlayerVisualizer()

    def visualize_offside_result(
        self,
        image: np.ndarray,
        result: OffsideResult,
        calibration: Optional[CalibrationResult] = None,
        show_all_players: bool = True,
        show_measurements: bool = True,
    ) -> np.ndarray:
        """
        Create complete offside visualization.

        Args:
            image: Input image
            result: Offside analysis result
            calibration: Optional calibration for perspective line
            show_all_players: Show all players or just relevant ones
            show_measurements: Show distance measurements

        Returns:
            Annotated visualization image
        """
        output = image.copy()
        h, w = image.shape[:2]

        # Draw line
        output = self._draw_offside_line(output, result, h, calibration)

        # Draw players with offside status
        output = self._draw_players_with_status(output, result, show_measurements)

        # Add header with result
        output = self._draw_header(output, result)

        return output

    def _draw_offside_line(
        self,
        image: np.ndarray,
        result: OffsideResult,
        image_height: int,
        calibration: Optional[CalibrationResult] = None,
        show_pattern: bool = True,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw the offside line projected onto the pitch perspective."""
        output = image.copy()
        line = result.offside_line
        h, w = image.shape[:2]

        color = color or (
            self.offside_color if result.any_offside else self.onside_color
        )

        image_y = line.reference_point[1]

        pts = self._compute_offside_line_points(
            line.field_x,
            line.image_x,
            image_y,
            h,
            w,
            calibration,
        )

        # Draw the polyline
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                pt1 = (int(pts[i][0]), int(pts[i][1]))
                pt2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                cv2.line(output, pt1, pt2, color, self.line_thickness, cv2.LINE_AA)

        # Draw small markers along the polyline for a dashed/patterned effect
        if show_pattern and len(pts) >= 2:
            output = self._add_line_pattern_along(output, pts, color)

        return output

    def _compute_offside_line_points(
        self,
        field_x: float,
        image_x: float,
        image_y: Optional[float],
        image_h: int,
        image_w: int,
        calibration: Optional[CalibrationResult],
    ) -> List[Tuple[float, float]]:
        """
        Compute perspective-correct image points for the offside line.

        Priority:
          1. 3D projection matrix  (PnLCalib)
          2. Vanishing point       (converge toward VP)
          3. Homography            (field -> image)
          4. Vertical fallback     (straight line at image_x)

        VP is checked *before* the homography because the VP calibration
        also stores a rough homography that produces nearly-vertical lines.
        Using the VP directly gives a perspective-correct result.
        """
        return project_line(field_x, image_x, image_y, image_h, image_w, calibration)

    @staticmethod
    def _add_line_pattern_along(
        image: np.ndarray,
        pts: List[Tuple[float, float]],
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Draw small markers spaced evenly along the polyline."""
        output = image.copy()

        # Walk along the polyline at regular pixel intervals
        spacing = 40
        accum = 0.0
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if seg_len < 1e-6:
                continue
            dx, dy = (x2 - x1) / seg_len, (y2 - y1) / seg_len
            while accum < seg_len:
                mx = int(x1 + dx * accum)
                my = int(y1 + dy * accum)
                cv2.circle(output, (mx, my), 5, color, -1)
                accum += spacing
            accum -= seg_len  # carry remainder into next segment

        return output

    def _draw_players_with_status(
        self, image: np.ndarray, result: OffsideResult, show_measurements: bool
    ) -> np.ndarray:
        """Draw attacking players with their offside status."""
        output = image.copy()

        for status in result.player_statuses:
            det = status.player_detection

            # Color based on offside status
            if status.is_offside:
                color = self.offside_color
                label = "AHEAD"
            else:
                color = self.onside_color
                label = "BEHIND"

            # Draw bounding box
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

            # Draw most advanced body part
            px, py = [
                int(v) for v in get_directional_ground_point(det, result.attacking_side)
            ]

            # Draw marker
            cv2.circle(output, (px, py), 10, color, -1)
            cv2.circle(output, (px, py), 12, (255, 255, 255), 2)

            # Draw horizontal measurement to the anchor x for readability.
            line_x = int(result.offside_line.image_x)
            cv2.line(output, (px, py), (line_x, py), color, 2)

            # Draw label
            label_y = y1 - 15 if y1 > 40 else y2 + 25

            text = label
            if show_measurements:
                text += f" ({abs(status.margin_px):.0f}px)"

            cv2.putText(
                output, text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        return output

    def _draw_header(self, image: np.ndarray, result: OffsideResult) -> np.ndarray:
        """Draw result header on image."""
        output = image.copy()
        h, w = image.shape[:2]

        # Create header bar
        header_height = 50
        header = np.zeros((header_height, w, 3), dtype=np.uint8)

        if result.any_offside:
            header[:] = (0, 0, 150)  # Dark red
            text = f"AHEAD OF LINE - {result.num_ahead} player(s)"
        else:
            header[:] = (0, 100, 0)  # Dark green
            text = "NO PLAYER AHEAD OF LINE"

        cv2.putText(
            header, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        # Team info
        cv2.putText(
            header,
            f"Attack side: {result.attacking_side}",
            (w - 280, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Combine header with image
        output = np.vstack([header, image])

        return output

    def create_split_view(
        self, original: np.ndarray, annotated: np.ndarray
    ) -> np.ndarray:
        """
        Create side-by-side comparison view.

        Args:
            original: Original image
            annotated: Annotated image with offside visualization

        Returns:
            Combined split view
        """
        # Ensure same height
        h1, w1 = original.shape[:2]
        h2, w2 = annotated.shape[:2]

        if h1 != h2:
            # Resize to match heights
            scale = h1 / h2
            annotated = cv2.resize(annotated, (int(w2 * scale), h1))

        # Add labels
        cv2.putText(
            original,
            "ORIGINAL",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated,
            "VAR ANALYSIS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Combine horizontally
        divider = np.ones((h1, 3, 3), dtype=np.uint8) * 255
        combined = np.hstack([original, divider, annotated])

        return combined

    def create_detailed_view(
        self,
        image: np.ndarray,
        result: OffsideResult,
        calibration: Optional[CalibrationResult] = None,
    ) -> np.ndarray:
        """
        Create detailed analysis view with multiple panels.

        Args:
            image: Input image
            result: Offside result
            calibration: Calibration result

        Returns:
            Multi-panel visualization
        """
        return self.create_clean_result_view(image, result, calibration)

    def create_clean_result_view(
        self,
        image: np.ndarray,
        result: OffsideResult,
        calibration: Optional[CalibrationResult] = None,
    ) -> np.ndarray:
        """Create a clean broadcast-style frame with only the line and line player."""
        output = image.copy()
        h, w = output.shape[:2]

        output = self._draw_offside_line(
            output,
            result,
            h,
            calibration,
            show_pattern=False,
            color=self.clean_line_color,
        )
        output = self._draw_last_player_highlight(output, result)
        output = self._draw_clean_legend(output)
        output = self._draw_clean_header(output, result)
        return output

    def _draw_last_player_highlight(
        self,
        image: np.ndarray,
        result: OffsideResult,
    ) -> np.ndarray:
        """Highlight only the line-defining player with a clean spotlight treatment."""
        return self._draw_line_player_spotlight(image, result)

    def _draw_line_player_spotlight(
        self,
        image: np.ndarray,
        result: OffsideResult,
    ) -> np.ndarray:
        output = image.copy()
        line = result.offside_line
        px, py = [int(v) for v in line.reference_point]
        color = self.clean_player_color
        marker_center = (px, py)

        overlay = output.copy()
        cv2.circle(
            overlay,
            marker_center,
            10,
            color,
            -1,
            cv2.LINE_AA,
        )
        output = cv2.addWeighted(overlay, 0.10, output, 0.90, 0)

        cv2.circle(
            output,
            marker_center,
            8,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            output,
            marker_center,
            2,
            color,
            -1,
            cv2.LINE_AA,
        )
        return output

    def _draw_clean_legend(self, image: np.ndarray) -> np.ndarray:
        """Draw a small legend for the clean result frame."""
        output = image.copy()
        h, w = output.shape[:2]
        legend_w = 210
        legend_h = 58
        x = 16
        y = max(16, h - legend_h - 16)

        cv2.rectangle(output, (x, y), (x + legend_w, y + legend_h), (18, 18, 18), -1)
        cv2.rectangle(
            output,
            (x, y),
            (x + legend_w, y + legend_h),
            (80, 80, 80),
            1,
        )

        line_y = y + 20
        cv2.line(
            output,
            (x + 12, line_y),
            (x + 42, line_y),
            self.clean_line_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            "Offside line",
            (x + 52, line_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        marker_y = y + 42
        cv2.circle(
            output,
            (x + 27, marker_y),
            6,
            self.clean_player_color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            output,
            (x + 27, marker_y),
            2,
            self.clean_player_color,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            "Last player",
            (x + 52, marker_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        return output

    def _draw_clean_header(
        self, image: np.ndarray, result: OffsideResult
    ) -> np.ndarray:
        """Draw a compact clean header for the final result frame."""
        output = image.copy()
        h, w = output.shape[:2]
        header_h = 58
        header = np.zeros((header_h, w, 3), dtype=np.uint8)
        header[:] = (16, 16, 16)

        accent = self.clean_line_color
        cv2.rectangle(header, (0, 0), (10, header_h), accent, -1)

        title = "LINE ANALYSIS"
        subtitle = f"Attack side: {result.attacking_side}"
        cv2.putText(
            header,
            title,
            (22, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            subtitle,
            (22, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )

        line_text = f"Line X: {result.offside_line.image_x:.0f}px"
        if result.offside_line.field_x is not None:
            line_text += f" | Field X: {result.offside_line.field_x:.2f}m"
        text_size = cv2.getTextSize(line_text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)[0]
        cv2.putText(
            header,
            line_text,
            (max(22, w - text_size[0] - 18), 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        return np.vstack([header, output])

    def _create_info_panel(self, result: OffsideResult, width: int) -> np.ndarray:
        """Create info panel with detailed results."""
        panel_height = 100
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray

        y = 25

        # Title
        cv2.putText(
            panel,
            "LINE ANALYSIS",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        y += 30

        # Player statuses
        x = 20
        for status in result.player_statuses:
            color = self.offside_color if status.is_offside else self.onside_color
            tag = "ahead" if status.is_offside else "behind"
            text = f"P{status.player_idx}: {abs(status.margin_px):.0f}px {tag}"

            cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            x += 200
            if x > width - 200:
                x = 20
                y += 25

        return panel
