"""
Classical computer vision vanishing point calibration.
Fallback when PnLCalib is not available.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from var_offside.calibration.base import BaseCalibration, CalibrationResult
from var_offside.config import get_settings
from var_offside.utils.geometry import calculate_line_intersection, get_line_angle, point_to_line_distance


class VanishingPointCalibration(BaseCalibration):
    """
    Field calibration using classical CV vanishing point detection.
    
    Pipeline:
    1. Segment green field
    2. Detect edges and lines (Hough transform)
    3. Filter lines by angle and merge collinear segments
    4. Estimate vanishing point using RANSAC
    5. Compute approximate homography
    """
    
    def __init__(self):
        """Initialize vanishing point calibration."""
        settings = get_settings()
        self.config = settings.calibration
        
        # HSV range for green field
        self.green_lower = np.array(self.config.green_lower)
        self.green_upper = np.array(self.config.green_upper)
        
        # Hough parameters
        self.hough_threshold = self.config.hough_threshold
        self.min_line_length = self.config.min_line_length
        self.max_line_gap = self.config.max_line_gap
        
        # RANSAC parameters
        self.ransac_iterations = self.config.ransac_iterations
        self.inlier_threshold = self.config.inlier_threshold
    
    def get_name(self) -> str:
        return "vanishing_point"
    
    def calibrate(self, image: np.ndarray) -> CalibrationResult:
        """
        Calibrate using vanishing point detection.
        
        Args:
            image: BGR input image
            
        Returns:
            CalibrationResult
        """
        result = CalibrationResult(method=self.get_name())
        h, w = image.shape[:2]
        
        try:
            # 1. Segment green field
            segmented, mask = self._segment_green(image)
            
            # 2. Detect edges
            edges = self._detect_edges(segmented)
            
            # 3. Detect lines
            lines = self._detect_lines(edges)
            
            if lines is None or len(lines) == 0:
                result.debug_info["error"] = "No lines detected"
                return result
            
            # 4. Filter and merge lines
            filtered_lines = self._filter_lines(lines, image)
            
            if len(filtered_lines) < 2:
                result.debug_info["error"] = "Not enough lines after filtering"
                return result
            
            result.field_lines = filtered_lines
            
            # 5. Estimate vanishing point
            vanishing_point = self._estimate_vanishing_point(filtered_lines, image)
            
            if vanishing_point is not None:
                result.vanishing_point = vanishing_point
                result.quality_score = self._calculate_quality(
                    filtered_lines, vanishing_point
                )
            
            # 6. Estimate homography (simplified approach)
            # For accurate homography, we need more correspondences
            # This provides a basic estimate for visualization
            if result.vanishing_point is not None:
                result.homography = self._estimate_homography(
                    result.vanishing_point, w, h
                )
                if result.homography is not None:
                    try:
                        result.homography_inv = np.linalg.inv(result.homography)
                    except np.linalg.LinAlgError:
                        pass
            
            result.debug_info.update({
                "num_lines_raw": len(lines),
                "num_lines_filtered": len(filtered_lines),
                "segmentation_mask": mask
            })
            
        except Exception as e:
            result.debug_info["error"] = str(e)
        
        return result
    
    def _segment_green(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment green field from image.
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (segmented_image, mask)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for green colors
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask
        segmented = cv2.bitwise_and(image, image, mask=mask)
        
        return segmented, mask
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def _detect_lines(
        self,
        edges: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detect lines using Hough transform.
        
        Args:
            edges: Edge image
            
        Returns:
            Array of lines (N, 1, 4) or None
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        return lines
    
    def _filter_lines(
        self,
        lines: np.ndarray,
        image: np.ndarray
    ) -> List[Tuple[float, float, float, float]]:
        """
        Filter and merge detected lines.
        
        Args:
            lines: Raw detected lines
            image: Original image (for size reference)
            
        Returns:
            List of filtered lines
        """
        h, w = image.shape[:2]
        min_angle = self.config.min_line_angle
        max_angle = self.config.max_line_angle
        
        filtered = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = get_line_angle((x1, y1, x2, y2))
            
            # Filter by angle (keep non-horizontal lines for vanishing point)
            if min_angle <= angle <= max_angle:
                # Calculate length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Filter short lines
                min_length = min(h, w) * 0.1
                if length >= min_length:
                    filtered.append((float(x1), float(y1), float(x2), float(y2)))
        
        # Merge collinear lines
        merged = self._merge_collinear_lines(filtered)
        
        return merged
    
    def _merge_collinear_lines(
        self,
        lines: List[Tuple[float, float, float, float]],
        angle_threshold: float = 5.0,
        distance_threshold: float = 20.0
    ) -> List[Tuple[float, float, float, float]]:
        """
        Merge lines that are roughly collinear.
        
        Args:
            lines: Input lines
            angle_threshold: Max angle difference in degrees
            distance_threshold: Max perpendicular distance
            
        Returns:
            Merged lines
        """
        if len(lines) <= 1:
            return lines
        
        # Group lines by angle
        lines_with_angle = [(line, get_line_angle(line)) for line in lines]
        lines_with_angle.sort(key=lambda x: x[1])
        
        merged = []
        used = [False] * len(lines_with_angle)
        
        for i, (line1, angle1) in enumerate(lines_with_angle):
            if used[i]:
                continue
            
            group = [line1]
            used[i] = True
            
            for j, (line2, angle2) in enumerate(lines_with_angle[i+1:], i+1):
                if used[j]:
                    continue
                
                # Check angle similarity
                if abs(angle1 - angle2) < angle_threshold:
                    # Check distance
                    mid1 = ((line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2)
                    dist = point_to_line_distance(mid1, line2)
                    
                    if dist < distance_threshold:
                        group.append(line2)
                        used[j] = True
            
            # Merge group into single line
            if group:
                all_points = []
                for line in group:
                    all_points.extend([(line[0], line[1]), (line[2], line[3])])
                
                # Find extreme points
                all_points = np.array(all_points)
                
                # Use PCA to find line direction and endpoints
                if len(all_points) >= 2:
                    mean = all_points.mean(axis=0)
                    centered = all_points - mean
                    
                    # SVD for direction
                    _, _, vh = np.linalg.svd(centered)
                    direction = vh[0]
                    
                    # Project points onto direction
                    projections = centered @ direction
                    
                    # Get extreme points
                    min_idx = projections.argmin()
                    max_idx = projections.argmax()
                    
                    merged.append((
                        all_points[min_idx, 0],
                        all_points[min_idx, 1],
                        all_points[max_idx, 0],
                        all_points[max_idx, 1]
                    ))
        
        return merged
    
    def _estimate_vanishing_point(
        self,
        lines: List[Tuple[float, float, float, float]],
        image: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate vanishing point using RANSAC.
        
        Args:
            lines: Detected field lines
            image: Original image
            
        Returns:
            (x, y) vanishing point
        """
        h, w = image.shape[:2]
        
        if len(lines) < 2:
            return None
        
        best_point = None
        best_inliers = 0
        
        for _ in range(self.ransac_iterations):
            # Sample 2 random lines
            idx = np.random.choice(len(lines), size=2, replace=False)
            line1, line2 = lines[idx[0]], lines[idx[1]]
            
            # Find intersection
            intersection = calculate_line_intersection(line1, line2)
            
            if intersection is None:
                continue
            
            x, y = intersection
            
            # Check if point is in reasonable range
            margin = max(w, h) * 2
            if not (-margin < x < w + margin and -margin < y < h + margin):
                continue
            
            # Count inliers
            inliers = 0
            for line in lines:
                dist = point_to_line_distance(intersection, line)
                if dist < self.inlier_threshold:
                    inliers += 1
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_point = intersection
        
        return best_point
    
    def _calculate_quality(
        self,
        lines: List[Tuple[float, float, float, float]],
        vanishing_point: Tuple[float, float]
    ) -> float:
        """
        Calculate calibration quality score.
        
        Args:
            lines: Detected lines
            vanishing_point: Estimated vanishing point
            
        Returns:
            Quality score (0-1)
        """
        if not lines:
            return 0.0
        
        # Count lines passing near vanishing point
        inliers = 0
        for line in lines:
            dist = point_to_line_distance(vanishing_point, line)
            if dist < self.inlier_threshold:
                inliers += 1
        
        # Quality based on inlier ratio
        quality = inliers / len(lines)
        
        return min(quality, 1.0)
    
    def _estimate_homography(
        self,
        vanishing_point: Tuple[float, float],
        width: int,
        height: int
    ) -> Optional[np.ndarray]:
        """
        Estimate a basic homography from vanishing point.
        
        This is a simplified estimation. For accurate results,
        use PnLCalib or manual correspondences.
        
        Args:
            vanishing_point: Detected vanishing point
            width: Image width
            height: Image height
            
        Returns:
            Approximate homography matrix
        """
        # This is a placeholder for more sophisticated homography estimation
        # In practice, you'd need more correspondences or camera parameters
        
        # Create a simple perspective transformation based on vanishing point
        vx, vy = vanishing_point
        
        # Define source points (image corners + vanishing point influence)
        src_points = np.float32([
            [0, height],
            [width, height],
            [width * 0.2, height * 0.4],
            [width * 0.8, height * 0.4]
        ])
        
        # Define destination points (field coordinates, simplified)
        dst_points = np.float32([
            [0, 68],
            [105, 68],
            [0, 0],
            [105, 0]
        ])
        
        try:
            H = cv2.getPerspectiveTransform(src_points, dst_points)
            return H
        except cv2.error:
            return None
