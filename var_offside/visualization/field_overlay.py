"""
Field line overlay visualization.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from var_offside.calibration.base import CalibrationResult
from var_offside.config import get_settings


class FieldOverlay:
    """
    Overlay field lines and calibration results on images.
    """
    
    def __init__(self):
        """Initialize field overlay."""
        settings = get_settings()
        self.line_color = (0, 255, 0)  # Green
        self.line_thickness = 2
    
    def draw_field_lines(
        self,
        image: np.ndarray,
        lines: List[Tuple[float, float, float, float]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detected field lines on image.
        
        Args:
            image: Input image
            lines: List of (x1, y1, x2, y2) lines
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with lines drawn
        """
        output = image.copy()
        
        for line in lines:
            x1, y1, x2, y2 = [int(c) for c in line]
            cv2.line(output, (x1, y1), (x2, y2), color, thickness)
        
        return output
    
    def draw_vanishing_point(
        self,
        image: np.ndarray,
        vanishing_point: Tuple[float, float],
        lines: Optional[List[Tuple[float, float, float, float]]] = None,
        color: Tuple[int, int, int] = (0, 0, 255)
    ) -> np.ndarray:
        """
        Draw vanishing point and lines converging to it.
        
        Args:
            image: Input image
            vanishing_point: (x, y) vanishing point
            lines: Optional lines to draw
            color: Point color (BGR)
            
        Returns:
            Annotated image
        """
        output = image.copy()
        h, w = image.shape[:2]
        
        vx, vy = int(vanishing_point[0]), int(vanishing_point[1])
        
        # Draw vanishing point if visible
        if 0 <= vx < w and 0 <= vy < h:
            cv2.circle(output, (vx, vy), 10, color, -1)
            cv2.circle(output, (vx, vy), 15, color, 2)
        
        # Draw lines to vanishing point
        if lines:
            for line in lines:
                x1, y1, x2, y2 = [int(c) for c in line]
                # Extend line toward vanishing point
                cv2.line(output, (x1, y1), (vx, vy), (255, 255, 0), 1)
        
        # Label
        cv2.putText(
            output, f"VP: ({vx}, {vy})",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )
        
        return output
    
    def draw_calibration_result(
        self,
        image: np.ndarray,
        calibration: CalibrationResult
    ) -> np.ndarray:
        """
        Draw complete calibration result on image.
        
        Args:
            image: Input image
            calibration: Calibration result
            
        Returns:
            Annotated image
        """
        output = image.copy()
        
        # Draw detected field lines
        if calibration.field_lines:
            output = self.draw_field_lines(output, calibration.field_lines)
        
        # Draw vanishing point
        if calibration.vanishing_point:
            output = self.draw_vanishing_point(
                output, 
                calibration.vanishing_point,
                calibration.field_lines
            )
        
        # Draw quality score
        quality_color = (
            (0, 255, 0) if calibration.quality_score > 0.7
            else (0, 255, 255) if calibration.quality_score > 0.4
            else (0, 0, 255)
        )
        
        cv2.putText(
            output,
            f"Calibration Quality: {calibration.quality_score:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, quality_color, 2
        )
        
        cv2.putText(
            output,
            f"Method: {calibration.method}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )
        
        return output
    
    def draw_grid_overlay(
        self,
        image: np.ndarray,
        calibration: CalibrationResult,
        grid_step: float = 10.0,
        field_length: float = 105.0,
        field_width: float = 68.0
    ) -> np.ndarray:
        """
        Draw a projected field grid overlay.
        
        Args:
            image: Input image
            calibration: Calibration result with homography
            grid_step: Grid spacing in meters
            field_length: Field length in meters
            field_width: Field width in meters
            
        Returns:
            Image with grid overlay
        """
        if not calibration.is_valid:
            return image
        
        output = image.copy()
        color = (0, 200, 200)  # Yellow-ish
        
        # Draw vertical lines (parallel to goal line)
        for x in np.arange(0, field_length + grid_step, grid_step):
            points = []
            for y in np.arange(0, field_width + 1, 5):
                img_pt = calibration.project_to_image((x, y))
                if img_pt:
                    points.append(img_pt)
            
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                cv2.line(output, pt1, pt2, color, 1)
        
        # Draw horizontal lines (parallel to sideline)
        for y in np.arange(0, field_width + grid_step, grid_step):
            points = []
            for x in np.arange(0, field_length + 1, 5):
                img_pt = calibration.project_to_image((x, y))
                if img_pt:
                    points.append(img_pt)
            
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                cv2.line(output, pt1, pt2, color, 1)
        
        return output
    
    def draw_projected_field(
        self,
        image: np.ndarray,
        calibration: CalibrationResult,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw projected field lines using 3D projection matrix from PnLCalib.
        
        Args:
            image: Input image
            calibration: Calibration result with projection matrix
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with projected field overlay
        """
        output = image.copy()
        
        # Check if we have a projection matrix
        if not calibration.has_3d_calibration:
            # Fall back to drawing detected lines
            if calibration.field_lines:
                return self.draw_field_lines(output, calibration.field_lines, color, thickness)
            return output
        
        P = calibration.projection_matrix
        
        # Standard field line coordinates (in meters, centered)
        # Using PnLCalib's coordinate system (origin at field center)
        lines_coords = [
            # Penalty areas
            [[0., 54.16, 0.], [16.5, 54.16, 0.]],
            [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
            [[16.5, 13.84, 0.], [0., 13.84, 0.]],
            [[88.5, 54.16, 0.], [105., 54.16, 0.]],
            [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
            [[88.5, 13.84, 0.], [105., 13.84, 0.]],
            # Goals
            [[0., 37.66, -2.44], [0., 30.34, -2.44]],
            [[0., 37.66, 0.], [0., 37.66, -2.44]],
            [[0., 30.34, 0.], [0., 30.34, -2.44]],
            [[105., 37.66, -2.44], [105., 30.34, -2.44]],
            [[105., 30.34, 0.], [105., 30.34, -2.44]],
            [[105., 37.66, 0.], [105., 37.66, -2.44]],
            # Field boundaries
            [[52.5, 0., 0.], [52.5, 68, 0.]],  # Center line
            [[0., 68., 0.], [105., 68., 0.]],  # Top sideline
            [[0., 0., 0.], [0., 68., 0.]],     # Left goal line
            [[105., 0., 0.], [105., 68., 0.]], # Right goal line
            [[0., 0., 0.], [105., 0., 0.]],    # Bottom sideline
            # Goal areas
            [[0., 43.16, 0.], [5.5, 43.16, 0.]],
            [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
            [[5.5, 24.84, 0.], [0., 24.84, 0.]],
            [[99.5, 43.16, 0.], [105., 43.16, 0.]],
            [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
            [[99.5, 24.84, 0.], [105., 24.84, 0.]],
        ]
        
        h, w = image.shape[:2]
        
        for line in lines_coords:
            w1, w2 = line[0], line[1]
            
            # Adjust to centered coordinates (PnLCalib uses center at 52.5, 34)
            p1 = np.array([w1[0] - 52.5, w1[1] - 34, w1[2], 1.0])
            p2 = np.array([w2[0] - 52.5, w2[1] - 34, w2[2], 1.0])
            
            i1 = P @ p1
            i2 = P @ p2
            
            # Skip if behind camera
            if i1[2] <= 0 or i2[2] <= 0:
                continue
            
            i1 /= i1[2]
            i2 /= i2[2]
            
            # Clip to image bounds (with margin)
            margin = max(w, h)
            if (abs(i1[0]) > margin * 2 or abs(i1[1]) > margin * 2 or
                abs(i2[0]) > margin * 2 or abs(i2[1]) > margin * 2):
                continue
            
            pt1 = (int(i1[0]), int(i1[1]))
            pt2 = (int(i2[0]), int(i2[1]))
            
            cv2.line(output, pt1, pt2, color, thickness)
        
        # Draw center circle
        radius = 9.15
        center = np.array([0, 0, 0, 1.0])  # Center of field
        pts = []
        for ang in np.linspace(0, 360, 72):
            ang_rad = np.deg2rad(ang)
            pt = np.array([radius * np.sin(ang_rad), radius * np.cos(ang_rad), 0, 1.0])
            proj = P @ pt
            if proj[2] > 0:
                proj /= proj[2]
                pts.append([int(proj[0]), int(proj[1])])
        
        if len(pts) > 2:
            pts_array = np.array(pts, np.int32)
            cv2.polylines(output, [pts_array], True, color, thickness)
        
        return output

