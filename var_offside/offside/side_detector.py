"""
Attacking side detection.
Determines which direction teams are attacking.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from var_offside.detection.player_detector import PlayerDetection


@dataclass
class AttackingSideResult:
    """Result of attacking side detection."""
    # Team 1's attacking direction ("left" or "right")
    team1_attacking: str
    
    # Team 2's attacking direction (opposite of team 1)
    team2_attacking: str
    
    # Confidence in this determination (0-1)
    confidence: float
    
    # Method used for detection
    method: str
    
    # Debug information
    debug_info: dict = None
    
    def __post_init__(self):
        if self.debug_info is None:
            self.debug_info = {}
    
    def get_defending_side(self, team_id: int) -> str:
        """Get the side a team is defending (where their goal is)."""
        if team_id == 1:
            return "right" if self.team1_attacking == "left" else "left"
        else:
            return "right" if self.team2_attacking == "left" else "left"
    
    def get_attacking_side(self, team_id: int) -> str:
        """Get the side a team is attacking towards."""
        return self.team1_attacking if team_id == 1 else self.team2_attacking


class SideDetector:
    """
    Detect which side each team is attacking.
    
    Uses multiple signals:
    1. Goalkeeper positions (GK near their own goal)
    2. Player distribution (attackers tend to be more advanced)
    3. Field orientation from calibration
    """
    
    def __init__(self):
        """Initialize side detector."""
        pass
    
    def detect_side(
        self,
        detections: List[PlayerDetection],
        image_width: int,
        goalkeeper_indices: Tuple[Optional[int], Optional[int]] = (None, None)
    ) -> AttackingSideResult:
        """
        Detect attacking sides for both teams.
        
        Args:
            detections: Player detections with team assignments
            image_width: Width of image
            goalkeeper_indices: (team1_gk_idx, team2_gk_idx)
            
        Returns:
            AttackingSideResult
        """
        # Try goalkeeper-based detection first
        if goalkeeper_indices[0] is not None or goalkeeper_indices[1] is not None:
            result = self._detect_by_goalkeeper(
                detections, image_width, goalkeeper_indices
            )
            if result.confidence > 0.7:
                return result
        
        # Fall back to distribution-based detection
        return self._detect_by_distribution(detections, image_width)
    
    def _detect_by_goalkeeper(
        self,
        detections: List[PlayerDetection],
        image_width: int,
        gk_indices: Tuple[Optional[int], Optional[int]]
    ) -> AttackingSideResult:
        """
        Detect side based on goalkeeper positions.
        
        Goalkeeper should be near their own goal (defending side).
        
        Args:
            detections: Player detections
            image_width: Image width
            gk_indices: Goalkeeper indices for each team
            
        Returns:
            AttackingSideResult
        """
        team1_gk_idx, team2_gk_idx = gk_indices
        center_x = image_width / 2
        
        team1_attacking = "right"  # Default
        team2_attacking = "left"   # Default
        confidence = 0.5
        
        debug = {}
        
        if team1_gk_idx is not None and team1_gk_idx < len(detections):
            gk1 = detections[team1_gk_idx]
            gk1_x = gk1.center[0]
            debug["team1_gk_x"] = gk1_x
            
            if gk1_x < center_x:
                # GK1 is on left, so team 1 defends left, attacks right
                team1_attacking = "right"
                team2_attacking = "left"
                confidence = 0.9
            else:
                # GK1 is on right, so team 1 defends right, attacks left
                team1_attacking = "left"
                team2_attacking = "right"
                confidence = 0.9
        
        if team2_gk_idx is not None and team2_gk_idx < len(detections):
            gk2 = detections[team2_gk_idx]
            gk2_x = gk2.center[0]
            debug["team2_gk_x"] = gk2_x
            
            # Verify consistency with team 1
            if gk2_x < center_x:
                # GK2 on left -> team 2 defends left, attacks right
                if team1_attacking == "left":
                    # Consistent
                    confidence = min(confidence + 0.1, 1.0)
            else:
                # GK2 on right -> team 2 defends right, attacks left
                if team1_attacking == "right":
                    # Consistent
                    confidence = min(confidence + 0.1, 1.0)
        
        return AttackingSideResult(
            team1_attacking=team1_attacking,
            team2_attacking=team2_attacking,
            confidence=confidence,
            method="goalkeeper",
            debug_info=debug
        )
    
    def _detect_by_distribution(
        self,
        detections: List[PlayerDetection],
        image_width: int
    ) -> AttackingSideResult:
        """
        Detect side based on player distribution.
        
        Analyzes the average position of each team to infer attacking direction.
        
        Args:
            detections: Player detections
            image_width: Image width
            
        Returns:
            AttackingSideResult
        """
        center_x = image_width / 2
        
        # Get positions for each team
        team1_positions = []
        team2_positions = []
        
        for det in detections:
            if det.is_goalkeeper:
                continue  # Skip goalkeepers for distribution analysis
            
            if det.team_id == 1:
                team1_positions.append(det.center[0])
            elif det.team_id == 2:
                team2_positions.append(det.center[0])
        
        debug = {
            "team1_positions": team1_positions,
            "team2_positions": team2_positions
        }
        
        if not team1_positions or not team2_positions:
            # Default if we don't have enough data
            return AttackingSideResult(
                team1_attacking="right",
                team2_attacking="left",
                confidence=0.3,
                method="distribution_default",
                debug_info=debug
            )
        
        # Calculate average positions
        team1_avg_x = np.mean(team1_positions)
        team2_avg_x = np.mean(team2_positions)
        
        debug["team1_avg_x"] = team1_avg_x
        debug["team2_avg_x"] = team2_avg_x
        
        # Team with higher average x is attacking right
        if team1_avg_x > team2_avg_x:
            team1_attacking = "right"
            team2_attacking = "left"
        else:
            team1_attacking = "left"
            team2_attacking = "right"
        
        # Confidence based on separation
        separation = abs(team1_avg_x - team2_avg_x) / image_width
        confidence = min(0.5 + separation, 0.8)
        
        debug["separation"] = separation
        
        return AttackingSideResult(
            team1_attacking=team1_attacking,
            team2_attacking=team2_attacking,
            confidence=confidence,
            method="distribution",
            debug_info=debug
        )
    
    def detect_with_manual_override(
        self,
        attacking_team: int,
        attacking_side: str
    ) -> AttackingSideResult:
        """
        Create result with manual specification.
        
        Args:
            attacking_team: Which team (1 or 2) is attacking
            attacking_side: Which side ("left" or "right") they're attacking
            
        Returns:
            AttackingSideResult
        """
        other_side = "left" if attacking_side == "right" else "right"
        
        if attacking_team == 1:
            return AttackingSideResult(
                team1_attacking=attacking_side,
                team2_attacking=other_side,
                confidence=1.0,
                method="manual"
            )
        else:
            return AttackingSideResult(
                team1_attacking=other_side,
                team2_attacking=attacking_side,
                confidence=1.0,
                method="manual"
            )
