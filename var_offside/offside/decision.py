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
    HALFWAY_X_M,
    OffsideLine,
    attack_field_sign,
    get_directional_ground_point,
    project_detection_to_field,
    x_on_ray_at_row,
)

# Pitch points carry float error; on the halfway line counts as the own half.
_ON_HALFWAY_M = 1e-6


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

    # In their own half, so never offside whatever the line says
    in_own_half: bool = False


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

    def __init__(
        self,
        margin_tolerance_px: int = 3,
        margin_tolerance_m: float = 0.0,
    ):
        """
        Initialize line-relative decision maker.

        Args:
            margin_tolerance_px: "On the line" tolerance when players are
                compared in image pixels (no pitch calibration).
            margin_tolerance_m: "On the line" tolerance when players are
                compared in pitch metres (calibrated frames).
        """
        self.margin_tolerance = margin_tolerance_px
        self.margin_tolerance_m = margin_tolerance_m

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

        sign = offside_line.attack_field_sign
        if sign is None:
            sign = attack_field_sign(detections, attacking_side, calibration)

        # Only attacking players are evaluated, and only with team context.
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
                sign,
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

        return False

    def _evaluate_player(
        self,
        detection: PlayerDetection,
        player_idx: int,
        offside_line: OffsideLine,
        attacking_side: str,
        calibration: Optional[CalibrationResult],
        field_sign: Optional[int],
    ) -> PlayerOffsideStatus:
        """
        Evaluate a single player's position relative to the line.

        Args:
            detection: Player detection
            player_idx: Index in detections list
            offside_line: Offside line
            attacking_side: Attack direction
            calibration: Optional field calibration
            field_sign: Direction of the attacked goal along pitch x, if known

        Returns:
            PlayerOffsideStatus
        """
        point = get_directional_ground_point(detection, attacking_side)

        field_point = None
        if field_sign is not None:
            field_point = project_detection_to_field(
                detection, attacking_side, calibration
            )

        in_own_half = bool(
            field_point is not None
            and (field_point[0] - HALFWAY_X_M) * field_sign <= _ON_HALFWAY_M
        )

        if field_point is not None and offside_line.field_x is not None:
            # Both positions are pitch metres, so the tolerance must be too.
            advanced_x = float(field_point[0])
            margin = (advanced_x - offside_line.field_x) * field_sign
            tolerance = self.margin_tolerance_m
        else:
            advanced_x = float(point[0])
            line_x = offside_line.image_x
            vanishing_point = calibration.vanishing_point if calibration else None
            if vanishing_point is not None:
                # The drawn line slants, so read it at this player's row.
                line_x = x_on_ray_at_row(
                    offside_line.reference_point, vanishing_point, point[1]
                )
            margin = line_x - advanced_x if attacking_side == "left" else advanced_x - line_x
            tolerance = self.margin_tolerance

        return PlayerOffsideStatus(
            player_idx=player_idx,
            player_detection=detection,
            is_offside=bool(margin > tolerance) and not in_own_half,
            most_advanced_part=None,
            advanced_x=advanced_x,
            margin_px=margin,
            confidence=float(detection.confidence),
            in_own_half=in_own_half,
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
