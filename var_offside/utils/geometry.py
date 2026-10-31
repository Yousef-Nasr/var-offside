"""
Geometric utility functions for line and point operations.
"""

import numpy as np
from typing import Tuple, Optional, List


def calculate_line_intersection(
    line1: Tuple[float, float, float, float],
    line2: Tuple[float, float, float, float]
) -> Optional[Tuple[float, float]]:
    """
    Calculate intersection point of two lines.
    
    Args:
        line1: (x1, y1, x2, y2) endpoints of first line
        line2: (x1, y1, x2, y2) endpoints of second line
        
    Returns:
        (x, y) intersection point or None if parallel
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Convert to homogeneous coordinates
    l1 = np.cross([x1, y1, 1], [x2, y2, 1])
    l2 = np.cross([x3, y3, 1], [x4, y4, 1])
    
    # Intersection in homogeneous coordinates
    intersection = np.cross(l1, l2)
    
    # Check for parallel lines
    if abs(intersection[2]) < 1e-10:
        return None
    
    # Convert back to Cartesian
    x = intersection[0] / intersection[2]
    y = intersection[1] / intersection[2]
    
    return (x, y)


def point_to_line_distance(
    point: Tuple[float, float],
    line: Tuple[float, float, float, float]
) -> float:
    """
    Calculate perpendicular distance from point to line.
    
    Args:
        point: (x, y) point coordinates
        line: (x1, y1, x2, y2) line endpoints
        
    Returns:
        Distance from point to line
    """
    px, py = point
    x1, y1, x2, y2 = line
    
    # Line direction vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Length of line segment
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-10:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    
    # Perpendicular distance
    distance = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length
    
    return distance


def get_line_angle(line: Tuple[float, float, float, float]) -> float:
    """
    Get angle of line in degrees (0-180).
    
    Args:
        line: (x1, y1, x2, y2) line endpoints
        
    Returns:
        Angle in degrees
    """
    x1, y1, x2, y2 = line
    angle = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle) % 180
    return angle_deg


def project_point_to_line(
    point: Tuple[float, float],
    line: Tuple[float, float, float, float]
) -> Tuple[float, float]:
    """
    Project a point onto a line.
    
    Args:
        point: (x, y) point to project
        line: (x1, y1, x2, y2) line endpoints
        
    Returns:
        (x, y) projected point on line
    """
    px, py = point
    x1, y1, x2, y2 = line
    
    # Line direction vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        return (x1, y1)
    
    # Parameter t for projection
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    
    # Projected point
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return (proj_x, proj_y)


def extend_line_to_bounds(
    line: Tuple[float, float, float, float],
    width: int,
    height: int
) -> Tuple[float, float, float, float]:
    """
    Extend a line segment to image boundaries.
    
    Args:
        line: (x1, y1, x2, y2) line endpoints
        width: Image width
        height: Image height
        
    Returns:
        Extended line endpoints
    """
    x1, y1, x2, y2 = line
    
    # Line in ax + by + c = 0 form
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    # Find intersections with image boundaries
    points = []
    
    # Top edge (y = 0)
    if abs(b) > 1e-10:
        x = -c / a if abs(a) > 1e-10 else x1
        if 0 <= x <= width:
            points.append((x, 0))
    
    # Bottom edge (y = height)
    if abs(a) > 1e-10:
        x = -(b * height + c) / a
        if 0 <= x <= width:
            points.append((x, height))
    
    # Left edge (x = 0)
    if abs(b) > 1e-10:
        y = -c / b
        if 0 <= y <= height:
            points.append((0, y))
    
    # Right edge (x = width)
    if abs(b) > 1e-10:
        y = -(a * width + c) / b
        if 0 <= y <= height:
            points.append((width, y))
    
    if len(points) >= 2:
        return (points[0][0], points[0][1], points[1][0], points[1][1])
    
    return line


def get_horizontal_position(
    point: Tuple[float, float],
    vanishing_point: Tuple[float, float],
    image_width: int,
    direction: str = "left"
) -> float:
    """
    Get the horizontal field position of a point relative to vanishing point.
    Used for offside line calculation.
    
    Args:
        point: (x, y) point coordinates
        vanishing_point: (x, y) vanishing point
        image_width: Width of image
        direction: "left" or "right" - which direction is attacking
        
    Returns:
        Normalized position (0-1) where 0 is goal line
    """
    px, py = point
    vx, vy = vanishing_point
    
    if direction == "left":
        # Attacking left, so leftmost is 0
        position = px / image_width
    else:
        # Attacking right, so rightmost is 0  
        position = 1.0 - (px / image_width)
    
    return position


def lines_from_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float, float, float]]:
    """
    Create line segments from a list of points.
    
    Args:
        points: List of (x, y) points
        
    Returns:
        List of lines (x1, y1, x2, y2)
    """
    lines = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        lines.append((x1, y1, x2, y2))
    return lines
