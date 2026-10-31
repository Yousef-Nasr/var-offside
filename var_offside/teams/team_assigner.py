"""
Main team assignment logic.
Orchestrates embedding extraction and clustering.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from var_offside.detection.player_detector import PlayerDetection
from var_offside.teams.embeddings import EmbeddingExtractor
from var_offside.teams.clustering import TeamClusterer, ClusteringResult
from var_offside.config import get_settings


@dataclass
class TeamAssignmentResult:
    """Result of team assignment."""
    detections: List[PlayerDetection]  # Updated with team IDs
    embeddings: np.ndarray  # Player embeddings
    clustering: ClusteringResult  # Clustering details
    team_colors: Dict[int, Tuple[int, int, int]]  # Detected team colors


class TeamAssigner:
    """
    Assign players to teams using visual embeddings.
    """
    
    def __init__(
        self,
        n_teams: int = 2,
        device: Optional[str] = None
    ):
        """
        Initialize team assigner.
        
        Args:
            n_teams: Number of teams (usually 2)
            device: Compute device
        """
        settings = get_settings()
        
        self.n_teams = n_teams
        self.embedding_extractor = EmbeddingExtractor(device)
        self.clusterer = TeamClusterer(n_teams)
        
        # Visualization colors
        self.team_colors = {
            1: settings.visualization.team1_color,
            2: settings.visualization.team2_color,
        }
    
    def assign_teams(
        self,
        image: np.ndarray,
        detections: List[PlayerDetection]
    ) -> TeamAssignmentResult:
        """
        Assign team IDs to detected players.
        
        Args:
            image: BGR input image
            detections: List of player detections
            
        Returns:
            TeamAssignmentResult with updated detections
        """
        if len(detections) < self.n_teams:
            # Not enough players
            for i, det in enumerate(detections):
                det.team_id = i + 1
            return TeamAssignmentResult(
                detections=detections,
                embeddings=np.array([]),
                clustering=ClusteringResult(
                    labels=np.array([d.team_id for d in detections]),
                    centroids=np.array([]),
                    confidence=np.ones(len(detections))
                ),
                team_colors=self.team_colors
            )
        
        # Extract embeddings
        bboxes = [d.bbox for d in detections]
        embeddings = self.embedding_extractor.extract_from_detections(image, bboxes)
        
        # Filter out zero embeddings (invalid crops)
        valid_mask = np.any(embeddings != 0, axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < self.n_teams:
            # Not enough valid embeddings
            for i, det in enumerate(detections):
                det.team_id = (i % self.n_teams) + 1
            return TeamAssignmentResult(
                detections=detections,
                embeddings=embeddings,
                clustering=ClusteringResult(
                    labels=np.array([d.team_id for d in detections]),
                    centroids=np.array([]),
                    confidence=np.ones(len(detections))
                ),
                team_colors=self.team_colors
            )
        
        # Cluster valid embeddings
        valid_embeddings = embeddings[valid_indices]
        clustering = self.clusterer.cluster(valid_embeddings)
        
        # Assign team IDs to all detections
        # Invalid ones get assigned to nearest team by position
        team_labels = np.zeros(len(detections), dtype=int)
        team_labels[valid_indices] = clustering.labels
        
        # Assign invalid detections to teams based on proximity
        for i in range(len(detections)):
            if not valid_mask[i]:
                # Find nearest valid detection
                det = detections[i]
                min_dist = float('inf')
                nearest_team = 1
                
                for j in valid_indices:
                    other_det = detections[j]
                    dist = np.sqrt(
                        (det.center[0] - other_det.center[0])**2 +
                        (det.center[1] - other_det.center[1])**2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_team = team_labels[j]
                
                team_labels[i] = nearest_team
        
        # Update detections with team IDs
        for i, det in enumerate(detections):
            det.team_id = int(team_labels[i])
        
        # Detect team colors for visualization
        detected_colors = self._detect_team_colors(image, detections)
        
        return TeamAssignmentResult(
            detections=detections,
            embeddings=embeddings,
            clustering=ClusteringResult(
                labels=team_labels,
                centroids=clustering.centroids,
                confidence=np.ones(len(detections))  # Simplified
            ),
            team_colors=detected_colors
        )
    
    def _detect_team_colors(
        self,
        image: np.ndarray,
        detections: List[PlayerDetection]
    ) -> Dict[int, Tuple[int, int, int]]:
        """
        Detect dominant jersey colors for each team.
        
        Args:
            image: BGR image
            detections: Player detections with team IDs
            
        Returns:
            Dictionary mapping team ID to BGR color
        """
        team_colors = {}
        
        for team_id in [1, 2]:
            team_players = [d for d in detections if d.team_id == team_id]
            
            if not team_players:
                team_colors[team_id] = self.team_colors[team_id]
                continue
            
            # Collect jersey pixels
            all_pixels = []
            
            for player in team_players[:5]:  # Sample up to 5 players
                x1, y1, x2, y2 = [int(c) for c in player.bbox]
                height = y2 - y1
                jersey_y2 = y1 + int(height * 0.5)
                
                crop = image[y1:jersey_y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Get pixels (reshape to Nx3)
                pixels = crop.reshape(-1, 3)
                all_pixels.append(pixels)
            
            if all_pixels:
                all_pixels = np.vstack(all_pixels)
                
                # Find dominant color (simple mean)
                dominant = all_pixels.mean(axis=0).astype(int)
                team_colors[team_id] = tuple(dominant.tolist())
            else:
                team_colors[team_id] = self.team_colors[team_id]
        
        return team_colors
    
    def visualize_teams(
        self,
        image: np.ndarray,
        result: TeamAssignmentResult,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Visualize team assignments on image.
        
        Args:
            image: Input image
            result: Team assignment result
            show_labels: Show team labels
            
        Returns:
            Annotated image
        """
        output = image.copy()
        
        for det in result.detections:
            if det.team_id is None:
                continue
            
            color = result.team_colors.get(det.team_id, (255, 255, 255))
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            if show_labels:
                label = f"Team {det.team_id}"
                if det.is_goalkeeper:
                    label = "GK " + label
                
                cv2.putText(
                    output, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2
                )
        
        return output
