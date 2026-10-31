"""
Abstract base class for field calibration methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np


@dataclass
class CalibrationResult:
    """Result of field calibration."""

    # Homography matrix (image -> field coordinates)
    homography: Optional[np.ndarray] = None

    # Inverse homography (field -> image coordinates)
    homography_inv: Optional[np.ndarray] = None

    # Detected field lines in image coordinates
    field_lines: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # Vanishing point (if computed)
    vanishing_point: Optional[Tuple[float, float]] = None

    # Detected keypoints (image coords -> field coords mapping)
    keypoints: dict = field(default_factory=dict)

    # Raw keypoints in image/model coordinates for debug overlays
    keypoints_raw: dict = field(default_factory=dict)

    # Image/model sizes used by the calibrator for debug reprojection
    image_size: Optional[Tuple[int, int]] = None  # (height, width)
    model_size: Optional[Tuple[int, int]] = None  # (height, width)

    # Calibration quality score (0-1)
    quality_score: float = 0.0

    # Method used for calibration
    method: str = "unknown"

    # Debug information
    debug_info: dict = field(default_factory=dict)

    # 3D camera calibration parameters (from PnLCalib)
    camera_matrix: Optional[np.ndarray] = None  # K matrix (intrinsics)
    projection_matrix: Optional[np.ndarray] = None  # P = K @ [R | t]
    rotation_matrix: Optional[np.ndarray] = None  # R matrix
    position: Optional[np.ndarray] = None  # Camera position in world coords

    @property
    def is_valid(self) -> bool:
        """Check if calibration is valid."""
        return self.homography is not None or self.projection_matrix is not None

    @property
    def has_3d_calibration(self) -> bool:
        """Check if 3D camera calibration is available."""
        return self.projection_matrix is not None

    def project_to_field(
        self, point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Project image point to field coordinates.

        Args:
            point: (x, y) in image coordinates

        Returns:
            (x, y) in field coordinates (meters), or None
        """
        if (
            self.camera_matrix is not None
            and self.rotation_matrix is not None
            and self.position is not None
        ):
            field_point = self._project_to_field_3d(point)
            if field_point is not None:
                return field_point

        if self.homography is None:
            return None

        pt = np.array([point[0], point[1], 1.0])
        transformed = self.homography @ pt

        if abs(transformed[2]) < 1e-10:
            return None

        return (transformed[0] / transformed[2], transformed[1] / transformed[2])

    def _project_to_field_3d(
        self,
        point: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """Back-project an image point to the ground plane using 3D camera pose."""
        try:
            k_inv = np.linalg.inv(self.camera_matrix)
        except np.linalg.LinAlgError:
            return None

        pixel_h = np.array([point[0], point[1], 1.0], dtype=np.float64)
        ray_camera = k_inv @ pixel_h

        # x_cam = R @ (X_world - C), so world ray direction is R^T @ ray_camera.
        ray_world = self.rotation_matrix.T @ ray_camera
        camera_center = np.array(self.position, dtype=np.float64).reshape(3)

        if abs(ray_world[2]) < 1e-10:
            return None

        scale = -camera_center[2] / ray_world[2]
        if scale <= 0:
            return None

        world_point = camera_center + scale * ray_world

        # PnLCalib uses a field-centered world frame.
        return (
            float(world_point[0] + 52.5),
            float(world_point[1] + 34.0),
        )

    def project_to_image(
        self, field_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Project field point to image coordinates.

        Args:
            field_point: (x, y) in field coordinates (meters)

        Returns:
            (x, y) in image coordinates, or None
        """
        if self.homography_inv is None:
            return None

        pt = np.array([field_point[0], field_point[1], 1.0])
        transformed = self.homography_inv @ pt

        if abs(transformed[2]) < 1e-10:
            return None

        return (transformed[0] / transformed[2], transformed[1] / transformed[2])


class BaseCalibration(ABC):
    """
    Abstract base class for field calibration methods.

    All calibration methods should inherit from this class and
    implement the calibrate() method.
    """

    @abstractmethod
    def calibrate(self, image: np.ndarray) -> CalibrationResult:
        """
        Perform field calibration on an image.

        Args:
            image: BGR input image

        Returns:
            CalibrationResult with homography and metadata
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this calibration method."""
        pass

    def visualize(self, image: np.ndarray, result: CalibrationResult) -> np.ndarray:
        """
        Visualize calibration results on image.

        Args:
            image: Input image
            result: Calibration result

        Returns:
            Annotated image
        """
        import cv2

        output = image.copy()

        # Draw detected field lines
        for line in result.field_lines:
            x1, y1, x2, y2 = [int(c) for c in line]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw vanishing point
        if result.vanishing_point is not None:
            vx, vy = [int(c) for c in result.vanishing_point]
            # Only draw if within reasonable bounds
            h, w = image.shape[:2]
            if -w < vx < 2 * w and -h < vy < 2 * h:
                cv2.circle(output, (vx, vy), 10, (0, 0, 255), -1)
                cv2.putText(
                    output,
                    "VP",
                    (vx + 15, vy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Draw quality score
        cv2.putText(
            output,
            f"Quality: {result.quality_score:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        return output
