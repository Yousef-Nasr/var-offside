"""
Goalkeeper detection and filtering.
Identifies goalkeepers to exclude from offside calculation.
"""

import numpy as np
from typing import List, Optional, Tuple, cast
from dataclasses import dataclass

from var_offside.detection.player_detector import PlayerDetection


@dataclass
class GoalkeeperResult:
    """Result of goalkeeper detection."""

    goalkeeper_idx: Optional[int]  # Index of goalkeeper in detection list
    confidence: float  # Confidence in this being the goalkeeper
    method: str  # Method used for detection


class GoalkeeperFilter:
    """
    Identify and filter out goalkeepers from player detections.

    Methods:
    1. Position-based: Closest to the goal line
    2. Color-based: Different jersey color from teammates
    3. Combined: Use multiple signals
    """

    def __init__(
        self,
        position_weight: float = 0.7,
        color_weight: float = 0.3,
        edge_margin_ratio: float = 0.15,
    ):
        """
        Initialize goalkeeper filter.

        Args:
            position_weight: Weight for position-based detection
            color_weight: Weight for color-based detection
            edge_margin_ratio: Ratio of image width to consider as goal area
        """
        self.position_weight = position_weight
        self.color_weight = color_weight
        self.edge_margin_ratio = edge_margin_ratio

    def identify_goalkeeper_by_position(
        self,
        detections: List[PlayerDetection],
        team_ids: List[int],
        target_team: int,
        image_width: int,
        side: str = "left",
    ) -> Optional[int]:
        """
        Identify goalkeeper by position (closest to goal line).

        Args:
            detections: List of player detections
            team_ids: Team ID for each detection
            target_team: Which team to find goalkeeper for
            image_width: Width of image
            side: Which side the goal is on ("left" or "right")

        Returns:
            Index of detected goalkeeper, or None
        """
        # Get team players
        team_indices = [i for i, tid in enumerate(team_ids) if tid == target_team]

        if not team_indices:
            return None

        # Find player closest to the goal line
        goal_x = 0 if side == "left" else image_width

        closest_idx = None
        min_distance = float("inf")

        for idx in team_indices:
            player = detections[idx]
            center_x = player.center[0]
            distance = abs(center_x - goal_x)

            if distance < min_distance:
                min_distance = distance
                closest_idx = idx

        # Check if the closest player is actually near the goal
        edge_threshold = image_width * self.edge_margin_ratio
        if min_distance > edge_threshold:
            return None

        return closest_idx

    def identify_goalkeeper_without_team(
        self,
        detections: List[PlayerDetection],
        image_width: int,
        side: str = "left",
        exclude_idx: Optional[int] = None,
    ) -> Optional[int]:
        """
        Identify goalkeeper by goal-line proximity without team labels.

        This is the fast-path heuristic used when team classification is disabled.
        """
        if not detections:
            return None

        closest_idx = None
        min_distance = float("inf")
        edge_threshold = image_width * self.edge_margin_ratio

        for idx, player in enumerate(detections):
            if exclude_idx is not None and idx == exclude_idx:
                continue

            x1, _y1, x2, _y2 = player.bbox
            distance = x1 if side == "left" else (image_width - x2)

            if distance < min_distance:
                min_distance = distance
                closest_idx = idx

        if closest_idx is None or min_distance > edge_threshold:
            return None

        return closest_idx

    def identify_goalkeeper_by_color(
        self,
        image: np.ndarray,
        detections: List[PlayerDetection],
        team_ids: List[int],
        target_team: int,
        embeddings: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """
        Identify goalkeeper by color difference from teammates.

        Goalkeeper typically wears a different colored jersey from outfield players.
        Find the player with most different color/embedding from team average.

        Args:
            image: BGR image
            detections: Player detections
            team_ids: Team assignments
            target_team: Target team
            embeddings: Optional pre-computed embeddings

        Returns:
            Index of goalkeeper, or None
        """
        team_indices = [i for i, tid in enumerate(team_ids) if tid == target_team]

        if len(team_indices) < 2:
            return None

        # If embeddings provided, use them
        if embeddings is not None:
            team_embeddings = embeddings[team_indices]

            # Find player most different from team centroid
            centroid = team_embeddings.mean(axis=0)
            distances = np.linalg.norm(team_embeddings - centroid, axis=1)

            # The goalkeeper should be an outlier
            if distances.max() > distances.mean() + distances.std():
                outlier_local_idx = distances.argmax()
                return team_indices[outlier_local_idx]

        # Fall back to color histogram comparison
        import cv2

        histograms = []
        for idx in team_indices:
            bbox = detections[idx].bbox
            x1, y1, x2, y2 = [int(c) for c in bbox]

            # Get upper body (jersey area)
            height = y2 - y1
            jersey_y2 = y1 + int(height * 0.5)
            crop = image[y1:jersey_y2, x1:x2]

            if crop.size == 0:
                histograms.append(None)
                continue

            # Compute color histogram
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)

        # Find most different histogram
        valid_indices = [i for i, h in enumerate(histograms) if h is not None]
        if len(valid_indices) < 2:
            return None

        valid_hists = [histograms[i] for i in valid_indices]

        # Compare each to the average of others
        max_diff = 0
        outlier_local_idx = None

        for i, hist in enumerate(valid_hists):
            others = [h for j, h in enumerate(valid_hists) if j != i]
            avg_other = np.mean(others, axis=0)
            diff = np.linalg.norm(hist - avg_other)

            if diff > max_diff:
                max_diff = diff
                outlier_local_idx = valid_indices[i]

        if outlier_local_idx is not None:
            return team_indices[outlier_local_idx]

        return None

    def identify_goalkeepers(
        self,
        image: np.ndarray,
        detections: List[PlayerDetection],
        team_ids: Optional[List[int]],
        image_width: int,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Identify goalkeepers for both teams.

        Args:
            image: BGR image
            detections: Player detections
            team_ids: Team assignments (1 or 2), if available
            image_width: Image width
            embeddings: Optional player embeddings

        Returns:
            Tuple of (team1_gk_idx, team2_gk_idx)
        """
        valid_team_ids = (
            team_ids is not None
            and len(team_ids) == len(detections)
            and any(tid == 1 for tid in team_ids)
            and any(tid == 2 for tid in team_ids)
        )

        if not valid_team_ids:
            gk_left = self.identify_goalkeeper_without_team(
                detections,
                image_width=image_width,
                side="left",
            )
            gk_right = self.identify_goalkeeper_without_team(
                detections,
                image_width=image_width,
                side="right",
                exclude_idx=gk_left,
            )
            return (gk_left, gk_right)

        team_ids = cast(List[int], team_ids)

        # Team 1 goalkeeper (assume on left side)
        gk1 = self.identify_goalkeeper_by_position(
            detections, team_ids, target_team=1, image_width=image_width, side="left"
        )

        # If position didn't work, try color
        if gk1 is None:
            gk1 = self.identify_goalkeeper_by_color(
                image, detections, team_ids, target_team=1, embeddings=embeddings
            )

        # Team 2 goalkeeper (assume on right side)
        gk2 = self.identify_goalkeeper_by_position(
            detections, team_ids, target_team=2, image_width=image_width, side="right"
        )

        if gk2 is None:
            gk2 = self.identify_goalkeeper_by_color(
                image, detections, team_ids, target_team=2, embeddings=embeddings
            )

        return (gk1, gk2)

    def filter_goalkeepers(
        self, detections: List[PlayerDetection], goalkeeper_indices: List[int]
    ) -> List[PlayerDetection]:
        """
        Mark goalkeepers in detection list.

        Args:
            detections: Player detections
            goalkeeper_indices: Indices of goalkeepers

        Returns:
            Updated detections with is_goalkeeper flag set
        """
        for idx in goalkeeper_indices:
            if idx is not None and 0 <= idx < len(detections):
                detections[idx].is_goalkeeper = True

        return detections

    def get_outfield_players(
        self, detections: List[PlayerDetection]
    ) -> List[PlayerDetection]:
        """
        Get only outfield players (exclude goalkeepers).

        Args:
            detections: All player detections

        Returns:
            List of non-goalkeeper players
        """
        return [d for d in detections if not d.is_goalkeeper]
