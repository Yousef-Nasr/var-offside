"""
Body parts extraction from pose keypoints.
Determines the most advanced playable body part for offside calculation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum

from var_offside.config import get_settings


class BodyPartType(Enum):
    """Types of playable body parts for offside."""
    HEAD = "head"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


@dataclass
class PlayableBodyPart:
    """A playable body part for offside determination."""
    part_type: BodyPartType
    position: Tuple[float, float]  # (x, y) in image coordinates
    confidence: float
    
    @property
    def x(self) -> float:
        return self.position[0]
    
    @property
    def y(self) -> float:
        return self.position[1]


class BodyPartsExtractor:
    """
    Extract playable body parts from YOLO Pose keypoints.
    
    COCO Keypoint Format (17 keypoints):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder
    7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist
    11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee
    15: left_ankle, 16: right_ankle
    """
    
    # Keypoint indices
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # Mapping from keypoint index to body part type
    KEYPOINT_TO_PART = {
        NOSE: BodyPartType.HEAD,
        LEFT_SHOULDER: BodyPartType.LEFT_SHOULDER,
        RIGHT_SHOULDER: BodyPartType.RIGHT_SHOULDER,
        LEFT_KNEE: BodyPartType.LEFT_KNEE,
        RIGHT_KNEE: BodyPartType.RIGHT_KNEE,
        LEFT_ANKLE: BodyPartType.LEFT_FOOT,
        RIGHT_ANKLE: BodyPartType.RIGHT_FOOT,
    }
    
    def __init__(self, confidence_threshold: Optional[float] = None):
        """
        Initialize body parts extractor.
        
        Args:
            confidence_threshold: Minimum keypoint confidence
        """
        settings = get_settings()
        self.confidence_threshold = (
            confidence_threshold or 
            settings.pose.keypoint_confidence_threshold
        )
    
    def extract_playable_parts(
        self,
        keypoints: np.ndarray
    ) -> List[PlayableBodyPart]:
        """
        Extract all playable body parts from keypoints.
        
        Args:
            keypoints: YOLO Pose keypoints array (17, 3) - x, y, confidence
            
        Returns:
            List of PlayableBodyPart objects with sufficient confidence
        """
        parts = []
        
        for kpt_idx, part_type in self.KEYPOINT_TO_PART.items():
            x, y, conf = keypoints[kpt_idx]
            
            if conf >= self.confidence_threshold:
                part = PlayableBodyPart(
                    part_type=part_type,
                    position=(float(x), float(y)),
                    confidence=float(conf)
                )
                parts.append(part)
        
        return parts
    
    def get_most_advanced_part(
        self,
        keypoints: np.ndarray,
        direction: str = "left"
    ) -> Optional[PlayableBodyPart]:
        """
        Get the most advanced (furthest in attacking direction) playable body part.
        This is the part used for offside determination.
        
        Args:
            keypoints: YOLO Pose keypoints array (17, 3)
            direction: "left" (attacking left) or "right" (attacking right)
            
        Returns:
            The most advanced PlayableBodyPart, or None if no valid parts
        """
        parts = self.extract_playable_parts(keypoints)
        
        if not parts:
            return None
        
        if direction == "left":
            # Attacking left, so leftmost (smallest x) is most advanced
            return min(parts, key=lambda p: p.x)
        else:
            # Attacking right, so rightmost (largest x) is most advanced
            return max(parts, key=lambda p: p.x)
    
    def get_head_position(
        self,
        keypoints: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Get head position from keypoints.
        Uses nose keypoint as proxy for head.
        
        Args:
            keypoints: Keypoints array
            
        Returns:
            (x, y) head position or None
        """
        x, y, conf = keypoints[self.NOSE]
        if conf >= self.confidence_threshold:
            return (float(x), float(y))
        return None
    
    def get_feet_positions(
        self,
        keypoints: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Get both feet positions.
        
        Args:
            keypoints: Keypoints array
            
        Returns:
            List of (x, y) positions for each visible foot
        """
        positions = []
        
        for ankle_idx in [self.LEFT_ANKLE, self.RIGHT_ANKLE]:
            x, y, conf = keypoints[ankle_idx]
            if conf >= self.confidence_threshold:
                positions.append((float(x), float(y)))
        
        return positions
    
    def get_all_keypoint_positions(
        self,
        keypoints: np.ndarray
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Get all keypoint positions as a dictionary.
        
        Args:
            keypoints: Keypoints array (17, 3)
            
        Returns:
            Dictionary mapping keypoint name to (x, y, confidence)
        """
        names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        return {
            name: (float(keypoints[i, 0]), float(keypoints[i, 1]), float(keypoints[i, 2]))
            for i, name in enumerate(names)
        }
