"""
YOLO-based player detection with pose estimation.
Uses two-stage approach: detection first, then pose estimation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path

from ultralytics import YOLO

from var_offside.config import get_settings

# Pose runs on a crop around each box: padding as a share of the box, and input size.
POSE_CROP_PAD_X = 0.25
POSE_CROP_PAD_Y = 0.1
POSE_CROP_SIZE = 256


@dataclass(frozen=True)
class BallDetection:
    """Where the detector saw the ball. The ball is never a player."""
    bbox: Tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        """Bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class PlayerDetection:
    """Container for a single player detection."""
    # Bounding box (x1, y1, x2, y2)
    bbox: Tuple[float, float, float, float]
    
    # Confidence score
    confidence: float
    
    # Keypoints array (17, 3) - x, y, confidence for each keypoint
    keypoints: Optional[np.ndarray] = None
    
    # Player ID (assigned later)
    player_id: Optional[int] = None
    
    # Team ID (assigned later)
    team_id: Optional[int] = None
    
    # Is this player a goalkeeper?
    is_goalkeeper: bool = False
    # The detector's own class name, as evidence only: it never sets a role by itself
    detector_class: Optional[str] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Get bottom center of bounding box (foot position approximation)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)
    
    @property
    def width(self) -> float:
        """Bounding box width."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Bounding box height."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height


class PlayerDetector:
    """
    Two-stage player detector:
    1. Use custom YOLO model to detect players (optimized for football)
    2. Use YOLO Pose to extract keypoints from each detection
    """
    
    def __init__(
        self,
        detection_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None
    ):
        """
        Initialize the player detector.
        
        Args:
            detection_model_path: Path to custom YOLO detection model
            pose_model_path: Path to YOLO Pose model
        """
        settings = get_settings()
        self.detection_model_path = detection_model_path or settings.models.yolo_detection
        self.pose_model_path = pose_model_path or settings.models.yolo_pose
        self.confidence_threshold = settings.detection.confidence_threshold
        self.iou_threshold = settings.detection.iou_threshold
        self.min_height = settings.detection.min_player_height
        self.max_players = settings.detection.max_players
        self.device = settings.device
        
        # Models will be loaded lazily
        self._detection_model: Optional[YOLO] = None
        self._pose_model: Optional[YOLO] = None
    
    @property
    def detection_model(self) -> YOLO:
        """Lazy load the detection YOLO model."""
        if self._detection_model is None:
            self._detection_model = YOLO(self.detection_model_path)
        return self._detection_model
    
    @property
    def pose_model(self) -> YOLO:
        """Lazy load the YOLO Pose model."""
        if self._pose_model is None:
            self._pose_model = YOLO(self.pose_model_path)
        return self._pose_model
    
    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        filter_by_height: bool = True,
        verbose: bool = False,
        use_pose: bool = True
    ) -> List[PlayerDetection]:
        """
        Detect players in an image using two-stage approach.
        
        Args:
            image: BGR input image
            confidence: Override confidence threshold
            filter_by_height: Filter out small detections
            verbose: Print debug information
            use_pose: Whether to run pose estimation on detections
            
        Returns:
            List of PlayerDetection objects
        """
        conf = confidence or self.confidence_threshold
        
        if verbose:
            print(f"[PlayerDetector] Stage 1: Running detection on image {image.shape}")
            print(f"[PlayerDetector] Detection model: {self.detection_model_path}")
            print(f"[PlayerDetector] Confidence threshold: {conf}")
        
        # Stage 1: Detect players with custom detection model
        detection_results = self.detection_model(
            image,
            conf=conf,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        self.last_ball = None
        
        for result in detection_results:
            if verbose:
                print(f"[PlayerDetector] Raw detection boxes: {result.boxes}")
                if result.boxes is not None:
                    print(f"[PlayerDetector] Number of raw detections: {len(result.boxes)}")
                    # Print class names if available
                    if hasattr(result, 'names'):
                        print(f"[PlayerDetector] Class names: {result.names}")
            
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            boxes = result.boxes.data.cpu().numpy()
            
            if verbose:
                print(f"[PlayerDetector] Processing {len(boxes)} boxes")
            
            for i, box in enumerate(boxes):
                # YOLO format: x1, y1, x2, y2, conf, class
                if len(box) >= 6:
                    x1, y1, x2, y2, box_conf, cls = box[:6]
                else:
                    x1, y1, x2, y2 = box[:4]
                    box_conf = box[4] if len(box) > 4 else 1.0
                    cls = 0  # Default to class 0
                
                class_name = result.names.get(int(cls), str(int(cls))) if hasattr(result, 'names') else str(int(cls))
                class_name = str(class_name).strip().lower()
                if class_name == "ball":
                    if self.last_ball is None or box_conf > self.last_ball.confidence:
                        self.last_ball = BallDetection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(box_conf),
                        )
                    continue

                # Keep scanning past the limit, so a later ball box is still seen.
                if len(detections) >= self.max_players:
                    continue

                if verbose:
                    print(f"[PlayerDetector] Box {i}: class={class_name}({int(cls)}), conf={box_conf:.2f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                
                bbox = (float(x1), float(y1), float(x2), float(y2))
                height = y2 - y1
                
                # Filter small detections
                if filter_by_height and height < self.min_height:
                    if verbose:
                        print(f"[PlayerDetector]   -> Skipped (height {height:.0f} < {self.min_height})")
                    continue
                
                detection = PlayerDetection(
                    bbox=bbox,
                    confidence=float(box_conf),
                    keypoints=None,
                    player_id=len(detections),
                    detector_class=class_name,
                )
                
                detections.append(detection)
                
                if verbose:
                    print(f"[PlayerDetector]   -> Added player {len(detections)}")
                
        
        if verbose:
            print(f"[PlayerDetector] Stage 1 complete: {len(detections)} players detected")
        
        # Stage 2: Run pose estimation on each detection
        if use_pose and detections:
            if verbose:
                print(f"[PlayerDetector] Stage 2: Running pose estimation...")
            
            detections = self._add_pose_keypoints(image, detections, verbose)
        
        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        if verbose:
            print(f"[PlayerDetector] Total detections: {len(detections)}")
        
        return detections
    
    def _add_pose_keypoints(
        self,
        image: np.ndarray,
        detections: List[PlayerDetection],
        verbose: bool = False
    ) -> List[PlayerDetection]:
        """
        Add pose keypoints to detections by running pose estimation on each box.

        Broadcast players are too small for the pose model to find on the full
        image, so it runs on a crop around each box instead.

        Args:
            image: Full image
            detections: List of player detections
            verbose: Print debug info

        Returns:
            Updated detections with keypoints
        """
        h, w = image.shape[:2]

        crops = []
        cropped = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            pad_x = POSE_CROP_PAD_X * (x2 - x1)
            pad_y = POSE_CROP_PAD_Y * (y2 - y1)
            left, top = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
            right, bottom = int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
            if right <= left or bottom <= top:
                continue
            crops.append(image[top:bottom, left:right])
            cropped.append((det, left, top))

        if not crops:
            return detections

        pose_results = self.pose_model(
            crops,
            conf=0.25,  # Lower threshold for pose
            imgsz=POSE_CROP_SIZE,
            device=self.device,
            verbose=False
        )

        for pose_result, (det, left, top) in zip(pose_results, cropped):
            if pose_result.keypoints is None or pose_result.boxes is None:
                continue

            pose_boxes = pose_result.boxes.data.cpu().numpy()
            pose_keypoints = pose_result.keypoints.data.cpu().numpy()

            # A crop can hold neighbours too: keep the pose matching this box.
            best_match_idx = None
            best_iou = 0.3  # Minimum IOU threshold for matching
            for j, pose_box in enumerate(pose_boxes):
                iou = self._calculate_iou(
                    det.bbox,
                    (pose_box[0] + left, pose_box[1] + top, pose_box[2] + left, pose_box[3] + top),
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j

            if best_match_idx is not None and best_match_idx < len(pose_keypoints):
                keypoints = pose_keypoints[best_match_idx].copy()
                keypoints[:, 0] += left
                keypoints[:, 1] += top
                det.keypoints = keypoints
                if verbose:
                    print(f"[PlayerDetector] Matched pose to player {det.player_id} (IOU={best_iou:.2f})")

        return detections

    def _calculate_iou(
        self,
        box1: Tuple[float, float, float, float],
        box2: Union[Tuple[float, float, float, float], np.ndarray]
    ) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_from_path(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> List[PlayerDetection]:
        """
        Detect players from an image file.
        
        Args:
            image_path: Path to image
            **kwargs: Additional arguments passed to detect()
            
        Returns:
            List of PlayerDetection objects
        """
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.detect(image, **kwargs)

