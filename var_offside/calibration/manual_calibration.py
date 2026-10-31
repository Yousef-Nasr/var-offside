"""
Manual calibration via point correspondences.
User draws lines/points to establish field-image mapping.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Callable

from var_offside.calibration.base import BaseCalibration, CalibrationResult


# FIFA standard field dimensions (meters)
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

# Standard field keypoints (x, y in meters)
FIELD_KEYPOINTS = {
    "left_goal_top": (0, FIELD_WIDTH / 2 - 7.32 / 2),
    "left_goal_bottom": (0, FIELD_WIDTH / 2 + 7.32 / 2),
    "left_penalty_top": (16.5, FIELD_WIDTH / 2 - 20.15),
    "left_penalty_bottom": (16.5, FIELD_WIDTH / 2 + 20.15),
    "left_penalty_spot": (11, FIELD_WIDTH / 2),
    "center_top": (FIELD_LENGTH / 2, 0),
    "center_bottom": (FIELD_LENGTH / 2, FIELD_WIDTH),
    "center_spot": (FIELD_LENGTH / 2, FIELD_WIDTH / 2),
    "right_penalty_top": (FIELD_LENGTH - 16.5, FIELD_WIDTH / 2 - 20.15),
    "right_penalty_bottom": (FIELD_LENGTH - 16.5, FIELD_WIDTH / 2 + 20.15),
    "right_penalty_spot": (FIELD_LENGTH - 11, FIELD_WIDTH / 2),
    "right_goal_top": (FIELD_LENGTH, FIELD_WIDTH / 2 - 7.32 / 2),
    "right_goal_bottom": (FIELD_LENGTH, FIELD_WIDTH / 2 + 7.32 / 2),
}


class ManualCalibration(BaseCalibration):
    """
    Manual calibration using user-provided point correspondences.
    
    The user clicks on known field landmarks in the image, and
    the system computes the homography from these correspondences.
    """
    
    def __init__(self):
        """Initialize manual calibration."""
        self.image_points: List[Tuple[float, float]] = []
        self.field_points: List[Tuple[float, float]] = []
        self.current_keypoint_idx = 0
        self.keypoint_names = list(FIELD_KEYPOINTS.keys())
        self._window_name = "Manual Calibration"
        self._callback: Optional[Callable] = None
    
    def get_name(self) -> str:
        return "manual"
    
    def calibrate(self, image: np.ndarray) -> CalibrationResult:
        """
        Perform manual calibration.
        
        Opens an interactive window for the user to click on field landmarks.
        
        Args:
            image: BGR input image
            
        Returns:
            CalibrationResult with homography
        """
        result = CalibrationResult(method=self.get_name())
        
        # Reset points
        self.image_points = []
        self.field_points = []
        self.current_keypoint_idx = 0
        
        # Run interactive calibration
        self._run_interactive(image)
        
        # Need at least 4 points for homography
        if len(self.image_points) < 4:
            result.debug_info["error"] = f"Need at least 4 points, got {len(self.image_points)}"
            return result
        
        # Compute homography
        src = np.array(self.image_points, dtype=np.float32)
        dst = np.array(self.field_points, dtype=np.float32)
        
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        
        if H is not None:
            result.homography = H
            try:
                result.homography_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                pass
            
            # Calculate quality based on reprojection error
            result.quality_score = self._calculate_reprojection_quality(
                src, dst, H
            )
            
            # Store keypoints mapping
            result.keypoints = {
                name: (img_pt, field_pt)
                for name, img_pt, field_pt in zip(
                    self.keypoint_names[:len(self.image_points)],
                    self.image_points,
                    self.field_points
                )
            }
        
        return result
    
    def calibrate_with_points(
        self,
        image_points: List[Tuple[float, float]],
        field_points: List[Tuple[float, float]]
    ) -> CalibrationResult:
        """
        Calibrate using pre-defined point correspondences.
        
        Args:
            image_points: List of (x, y) points in image coordinates
            field_points: List of (x, y) points in field coordinates
            
        Returns:
            CalibrationResult
        """
        result = CalibrationResult(method=self.get_name())
        
        if len(image_points) < 4 or len(field_points) < 4:
            result.debug_info["error"] = "Need at least 4 point correspondences"
            return result
        
        src = np.array(image_points, dtype=np.float32)
        dst = np.array(field_points, dtype=np.float32)
        
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        
        if H is not None:
            result.homography = H
            try:
                result.homography_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                pass
            
            result.quality_score = self._calculate_reprojection_quality(src, dst, H)
        
        return result
    
    def _run_interactive(self, image: np.ndarray):
        """
        Run interactive calibration window.
        
        Args:
            image: Input image
        """
        self._display_image = image.copy()
        
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)
        
        print("\n=== Manual Calibration ===")
        print("Click on visible field landmarks in order.")
        print("Press 'n' to skip a landmark.")
        print("Press 'd' to finish.")
        print("Press 'u' to undo last point.")
        print("Press 'q' to cancel.\n")
        
        while True:
            # Draw current state
            display = self._draw_state(image)
            cv2.imshow(self._window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Cancel
                self.image_points = []
                self.field_points = []
                break
            elif key == ord('d'):
                # Done
                break
            elif key == ord('n'):
                # Skip current landmark
                self.current_keypoint_idx += 1
                if self.current_keypoint_idx >= len(self.keypoint_names):
                    break
            elif key == ord('u'):
                # Undo
                if self.image_points:
                    self.image_points.pop()
                    self.field_points.pop()
                    self.current_keypoint_idx = max(0, self.current_keypoint_idx - 1)
        
        cv2.destroyWindow(self._window_name)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_keypoint_idx < len(self.keypoint_names):
                # Add point
                name = self.keypoint_names[self.current_keypoint_idx]
                field_point = FIELD_KEYPOINTS[name]
                
                self.image_points.append((float(x), float(y)))
                self.field_points.append(field_point)
                
                print(f"Added: {name} at image ({x}, {y}) -> field {field_point}")
                
                self.current_keypoint_idx += 1
    
    def _draw_state(self, image: np.ndarray) -> np.ndarray:
        """Draw current calibration state."""
        display = image.copy()
        
        # Draw existing points
        for i, (img_pt, field_pt) in enumerate(zip(self.image_points, self.field_points)):
            x, y = int(img_pt[0]), int(img_pt[1])
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(
                display,
                f"{i+1}",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2
            )
        
        # Show current landmark to click
        if self.current_keypoint_idx < len(self.keypoint_names):
            name = self.keypoint_names[self.current_keypoint_idx]
            cv2.putText(
                display,
                f"Click on: {name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2
            )
        else:
            cv2.putText(
                display,
                "Press 'd' to finish",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )
        
        # Show point count
        cv2.putText(
            display,
            f"Points: {len(self.image_points)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )
        
        return display
    
    def _calculate_reprojection_quality(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        H: np.ndarray
    ) -> float:
        """
        Calculate quality based on reprojection error.
        
        Args:
            src: Source (image) points
            dst: Destination (field) points
            H: Homography matrix
            
        Returns:
            Quality score (0-1)
        """
        # Project source points
        src_h = np.hstack([src, np.ones((len(src), 1))])
        projected = (H @ src_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        
        # Calculate mean reprojection error
        errors = np.linalg.norm(projected - dst, axis=1)
        mean_error = errors.mean()
        
        # Convert to quality (lower error = higher quality)
        # Assume 1 meter error = 0.5 quality
        quality = np.exp(-mean_error / 2.0)
        
        return min(quality, 1.0)
    
    def create_field_template(
        self,
        width: int = 1050,
        height: int = 680
    ) -> np.ndarray:
        """
        Create a top-down field template for reference.
        
        Args:
            width: Template width in pixels
            height: Template height in pixels
            
        Returns:
            Field template image
        """
        # Scale factors
        sx = width / FIELD_LENGTH
        sy = height / FIELD_WIDTH
        
        template = np.zeros((height, width, 3), dtype=np.uint8)
        template[:] = (0, 100, 0)  # Dark green
        
        # Draw field lines
        color = (255, 255, 255)
        
        # Outer boundary
        cv2.rectangle(template, (0, 0), (width-1, height-1), color, 2)
        
        # Center line
        cx = width // 2
        cv2.line(template, (cx, 0), (cx, height), color, 2)
        
        # Center circle
        cv2.circle(template, (cx, height // 2), int(9.15 * sx), color, 2)
        
        # Penalty areas
        pa_width = int(16.5 * sx)
        pa_height = int(40.3 * sy)
        pa_y = (height - pa_height) // 2
        
        # Left penalty area
        cv2.rectangle(template, (0, pa_y), (pa_width, pa_y + pa_height), color, 2)
        
        # Right penalty area
        cv2.rectangle(template, (width - pa_width, pa_y), (width, pa_y + pa_height), color, 2)
        
        # Draw keypoint markers
        for name, (fx, fy) in FIELD_KEYPOINTS.items():
            px = int(fx * sx)
            py = int(fy * sy)
            cv2.circle(template, (px, py), 5, (0, 0, 255), -1)
        
        return template
