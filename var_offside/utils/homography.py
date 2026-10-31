"""
Homography transformation utilities.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def compute_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> Optional[np.ndarray]:
    """
    Compute homography matrix from point correspondences.
    
    Args:
        src_points: Source points (N, 2) in image coordinates
        dst_points: Destination points (N, 2) in field coordinates
        
    Returns:
        3x3 homography matrix or None if failed
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        return None
    
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    return H


def apply_homography(
    H: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """
    Apply homography transformation to points.
    
    Args:
        H: 3x3 homography matrix
        points: Points to transform (N, 2)
        
    Returns:
        Transformed points (N, 2)
    """
    points = np.array(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, 2)
    
    # Add homogeneous coordinate
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Apply transformation
    transformed = (H @ points_h.T).T
    
    # Convert back to Cartesian
    transformed = transformed[:, :2] / transformed[:, 2:3]
    
    return transformed


def project_point(
    H: np.ndarray,
    point: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Project a single point using homography.
    
    Args:
        H: Homography matrix
        point: (x, y) point to project
        
    Returns:
        Projected (x, y) point
    """
    result = apply_homography(H, np.array([point]))
    return (result[0, 0], result[0, 1])


def project_line(
    H: np.ndarray,
    field_x: float,
    image_height: int,
    num_points: int = 100
) -> List[Tuple[float, float]]:
    """
    Project a vertical line from field coordinates to image.
    Used for drawing offside lines.
    
    Args:
        H: Homography matrix (field to image)
        field_x: X coordinate on field (in meters)
        image_height: Height of image
        num_points: Number of points to sample
        
    Returns:
        List of (x, y) points in image coordinates
    """
    # Sample points along the vertical line on field
    field_y = np.linspace(0, 68, num_points)  # FIFA field width
    field_points = np.column_stack([np.full_like(field_y, field_x), field_y])
    
    # Project to image
    image_points = apply_homography(H, field_points)
    
    return [(p[0], p[1]) for p in image_points]


def get_inverse_homography(H: np.ndarray) -> Optional[np.ndarray]:
    """
    Get inverse of homography matrix.
    
    Args:
        H: 3x3 homography matrix
        
    Returns:
        Inverse homography or None if singular
    """
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None


def warp_image(
    image: np.ndarray,
    H: np.ndarray,
    output_size: Tuple[int, int]
) -> np.ndarray:
    """
    Warp image using homography.
    
    Args:
        image: Input image
        H: Homography matrix
        output_size: (width, height) of output
        
    Returns:
        Warped image
    """
    return cv2.warpPerspective(image, H, output_size)


def draw_field_grid(
    image: np.ndarray,
    H_inv: np.ndarray,
    field_length: float = 105.0,
    field_width: float = 68.0,
    grid_step: float = 10.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1
) -> np.ndarray:
    """
    Draw field coordinate grid on image.
    
    Args:
        image: Input image
        H_inv: Inverse homography (field -> image)
        field_length: Field length in meters
        field_width: Field width in meters
        grid_step: Grid step in meters
        color: Line color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with grid overlay
    """
    output = image.copy()
    
    # Draw vertical lines
    for x in np.arange(0, field_length + grid_step, grid_step):
        points = []
        for y in np.arange(0, field_width + 1, 5):
            pt = project_point(H_inv, (x, y))
            points.append(pt)
        
        for i in range(len(points) - 1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i+1][0]), int(points[i+1][1]))
            cv2.line(output, pt1, pt2, color, thickness)
    
    # Draw horizontal lines
    for y in np.arange(0, field_width + grid_step, grid_step):
        points = []
        for x in np.arange(0, field_length + 1, 5):
            pt = project_point(H_inv, (x, y))
            points.append(pt)
        
        for i in range(len(points) - 1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i+1][0]), int(points[i+1][1]))
            cv2.line(output, pt1, pt2, color, thickness)
    
    return output
