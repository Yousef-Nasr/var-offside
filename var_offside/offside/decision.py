"""
Line-based player evaluation.
Determines which players are ahead of the detected line.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from var_offside.calibration.base import CalibrationResult
from var_offside.detection.player_detector import PlayerDetection
from var_offside.detection.body_parts import PlayableBodyPart
from var_offside.offside.line_calculator import (
    OffsideLine,
    get_directional_ground_point,
    project_detection_to_field,
)


@dataclass
class PlayerOffsideStatus:
    """Line-relative status for a single player."""

    player_idx: int
    player_detection: PlayerDetection

    # Is this player ahead of the line?
    is_offside: bool

    # The body part that is most advanced (unused in geometry mode)
    most_advanced_part: Optional[PlayableBodyPart]

    # Position of the evaluated point
    advanced_x: float

    # Distance past the line (positive = ahead of line)
    margin_px: float

    # Confidence in this determination
    confidence: float

    # Distance past the line in metres (set by the 3D mesh-based method, which
    # measures the furthest-forward body point rather than the bbox corner).
    margin_m: Optional[float] = None


@dataclass
class OffsideResult:
    """Complete line analysis result."""

    # The calculated offside line
    offside_line: OffsideLine

    # Status for each evaluated player
    player_statuses: List[PlayerOffsideStatus] = field(default_factory=list)

    # Overall result: is there any player ahead of the line?
    any_offside: bool = False

    # List of player indices ahead of the line
    offside_player_indices: List[int] = field(default_factory=list)

    # Attacking direction
    attacking_side: str = ""

    # Team context when available
    attacking_team: Optional[int] = None
    defending_team: Optional[int] = None

    @property
    def ahead_player_indices(self) -> List[int]:
        """Indices of players ahead of the line."""
        return self.offside_player_indices

    @property
    def num_ahead(self) -> int:
        """Number of players ahead of the line."""
        return len(self.offside_player_indices)

    @property
    def any_ahead(self) -> bool:
        """Whether any player is ahead of the line."""
        return self.any_offside

    @property
    def num_offside(self) -> int:
        """Backward-compatible alias for number of players ahead of line."""
        return len(self.offside_player_indices)

    @property
    def most_offside_player(self) -> Optional[PlayerOffsideStatus]:
        """Get the player who is furthest ahead of the line."""
        offside_players = [
            status for status in self.player_statuses if status.is_offside
        ]
        if not offside_players:
            return None
        return max(offside_players, key=lambda s: s.margin_px)


class OffsideDecision:
    """
    Evaluate players relative to a detected line.
    """

    def __init__(self, margin_tolerance_px: int = 3):
        """
        Initialize line-relative decision maker.

        Args:
            margin_tolerance_px: Tolerance in pixels (for "on the line" cases)
        """
        self.margin_tolerance = margin_tolerance_px

    def evaluate(
        self,
        detections: List[PlayerDetection],
        offside_line: OffsideLine,
        attacking_side: str,
        calibration: Optional[CalibrationResult] = None,
        exclude_goalkeeper: bool = True,
        attacking_team: Optional[int] = None,
        defending_team: Optional[int] = None,
    ) -> OffsideResult:
        """
        Evaluate which detected players are ahead of the line.

        Args:
            detections: All player detections
            offside_line: Calculated offside line
            attacking_side: Direction of attack
            calibration: Optional field calibration for field-space comparison
            exclude_goalkeeper: Whether to skip detections marked as goalkeepers

        Returns:
            OffsideResult with all statuses
        """
        result = OffsideResult(
            offside_line=offside_line,
            attacking_side=attacking_side,
            attacking_team=attacking_team,
            defending_team=defending_team,
        )

        # Evaluate only attacking players when team context is available.
        for i, det in enumerate(detections):
            if i == offside_line.defining_player_idx:
                continue

            if exclude_goalkeeper and det.is_goalkeeper:
                continue

            if not self._should_evaluate_player(
                det,
                attacking_team=attacking_team,
                defending_team=defending_team,
            ):
                continue

            status = self._evaluate_player(
                det,
                i,
                offside_line,
                attacking_side,
                calibration,
            )
            result.player_statuses.append(status)

            if status.is_offside:
                result.offside_player_indices.append(i)

        result.any_offside = len(result.offside_player_indices) > 0

        return result

    @staticmethod
    def _should_evaluate_player(
        detection: PlayerDetection,
        attacking_team: Optional[int],
        defending_team: Optional[int],
    ) -> bool:
        """Return whether this detection should be checked against the line."""
        if attacking_team in (1, 2):
            return detection.team_id == attacking_team

        if defending_team in (1, 2) and detection.team_id in (1, 2):
            return detection.team_id != defending_team

        return True

    def _evaluate_player(
        self,
        detection: PlayerDetection,
        player_idx: int,
        offside_line: OffsideLine,
        attacking_side: str,
        calibration: Optional[CalibrationResult],
    ) -> PlayerOffsideStatus:
        """
        Evaluate a single player's position relative to the line.

        Args:
            detection: Player detection
            player_idx: Index in detections list
            offside_line: Offside line
            attacking_side: Attack direction

        Returns:
            PlayerOffsideStatus
        """
        # Geometry mode compares players using the direction-aligned ground point.
        most_advanced = None
        advanced_x = float(get_directional_ground_point(detection, attacking_side)[0])
        confidence = float(detection.confidence)

        field_point = project_detection_to_field(detection, attacking_side, calibration)
        if field_point is not None and offside_line.field_x is not None:
            advanced_x = float(field_point[0])
            line_x = offside_line.field_x
        else:
            line_x = offside_line.image_x

        if attacking_side == "left":
            # Attacking left, ahead if player x is smaller than line x.
            margin = line_x - advanced_x
        else:
            # Attacking right, ahead if player x is larger than line x.
            margin = advanced_x - line_x

        is_offside = margin > self.margin_tolerance

        return PlayerOffsideStatus(
            player_idx=player_idx,
            player_detection=detection,
            is_offside=is_offside,
            most_advanced_part=most_advanced,
            advanced_x=advanced_x,
            margin_px=margin,
            confidence=confidence,
        )

    def get_summary(self, result: OffsideResult) -> str:
        """
        Get a human-readable summary of the line analysis.

        Args:
            result: Offside result

        Returns:
            Summary string
        """
        lines = []
        lines.append("=== Line Analysis ===")
        lines.append(f"Attacking Direction: {result.attacking_side}")
        lines.append(f"Line X: {result.offside_line.image_x:.1f}px")
        lines.append("")

        if result.any_offside:
            lines.append("Players ahead of the line detected")
            lines.append(f"Count: {result.num_ahead}")

            for status in result.player_statuses:
                if status.is_offside:
                    lines.append(
                        f"  - Player {status.player_idx}: "
                        f"{status.margin_px:.1f}px ahead"
                    )
        else:
            lines.append("No player is ahead of the line")

            for status in result.player_statuses:
                lines.append(
                    f"  - Player {status.player_idx}: "
                    f"{abs(status.margin_px):.1f}px behind"
                )

        return "\n".join(lines)
