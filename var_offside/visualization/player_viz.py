"""
Player visualization with team colors and keypoints.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from var_offside.detection.player_detector import PlayerDetection
from var_offside.detection.body_parts import BodyPartsExtractor
from var_offside.config import get_settings


class PlayerVisualizer:
    """
    Visualize detected players with team colors and pose keypoints.
    """
    
    def __init__(self):
        """Initialize player visualizer."""
        settings = get_settings()
        self.viz_config = settings.visualization
        
        self.team_colors = {
            1: self.viz_config.team1_color,
            2: self.viz_config.team2_color,
        }
        self.gk_color = self.viz_config.goalkeeper_color
        
        # COCO skeleton connections
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        self.body_parts = BodyPartsExtractor()
    
    def draw_players(
        self,
        image: np.ndarray,
        detections: List[PlayerDetection],
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        show_keypoints: bool = True,
        show_labels: bool = True,
        show_confidence: bool = False
    ) -> np.ndarray:
        """
        Draw all detected players on image.
        
        Args:
            image: Input image
            detections: Player detections
            team_colors: Optional custom team colors
            show_keypoints: Draw pose keypoints
            show_labels: Show team/player labels
            show_confidence: Show detection confidence
            
        Returns:
            Annotated image
        """
        output = image.copy()
        colors = team_colors or self.team_colors
        
        for i, det in enumerate(detections):
            color = self._get_player_color(det, colors)
            
            # Draw bounding box
            output = self._draw_bbox(output, det, color, show_labels, show_confidence)
            
            # Draw keypoints
            if show_keypoints and det.keypoints is not None:
                output = self._draw_keypoints(output, det.keypoints, color)
        
        return output
    
    def _get_player_color(
        self,
        detection: PlayerDetection,
        colors: Dict[int, Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
        """Get color for a player based on team and role."""
        if detection.is_goalkeeper:
            return self.gk_color
        
        team_id = detection.team_id or 0
        return colors.get(team_id, (255, 255, 255))
    
    def _draw_bbox(
        self,
        image: np.ndarray,
        detection: PlayerDetection,
        color: Tuple[int, int, int],
        show_label: bool,
        show_confidence: bool
    ) -> np.ndarray:
        """Draw bounding box for a player."""
        output = image.copy()
        x1, y1, x2, y2 = [int(c) for c in detection.bbox]
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if show_label:
            label_parts = []
            
            if detection.is_goalkeeper:
                label_parts.append("GK")
            
            if detection.team_id:
                label_parts.append(f"T{detection.team_id}")
            
            if detection.player_id is not None:
                label_parts.append(f"#{detection.player_id}")
            
            if show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            
            label = " ".join(label_parts)
            
            # Label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                output,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 5, y1),
                color, -1
            )
            
            # Label text
            cv2.putText(
                output, label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2
            )
        
        return output
    
    def _draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int],
        threshold: float = 0.3
    ) -> np.ndarray:
        """Draw pose keypoints and skeleton."""
        output = image.copy()
        
        # Draw skeleton connections
        for start_idx, end_idx in self.skeleton:
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            
            if start[2] > threshold and end[2] > threshold:
                pt1 = (int(start[0]), int(start[1]))
                pt2 = (int(end[0]), int(end[1]))
                cv2.line(output, pt1, pt2, color, 2)
        
        # Draw keypoint circles
        for i, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if conf > threshold:
                cv2.circle(output, (int(x), int(y)), 4, color, -1)
        
        return output
    
    def draw_player_highlight(
        self,
        image: np.ndarray,
        detection: PlayerDetection,
        color: Tuple[int, int, int] = (0, 255, 255),
        highlight_part: str = None
    ) -> np.ndarray:
        """
        Draw a highlighted player (e.g., for offside visualization).
        
        Args:
            image: Input image
            detection: Player to highlight
            color: Highlight color
            highlight_part: Specific body part to highlight
            
        Returns:
            Annotated image
        """
        output = image.copy()
        
        # Draw thicker bounding box
        x1, y1, x2, y2 = [int(c) for c in detection.bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 4)
        
        # Highlight specific body part if requested
        if highlight_part and detection.keypoints is not None:
            kpts_dict = self.body_parts.get_all_keypoint_positions(detection.keypoints)
            
            if highlight_part in kpts_dict:
                x, y, conf = kpts_dict[highlight_part]
                if conf > 0.3:
                    cv2.circle(output, (int(x), int(y)), 12, color, 3)
                    cv2.circle(output, (int(x), int(y)), 8, color, -1)
        
        return output
