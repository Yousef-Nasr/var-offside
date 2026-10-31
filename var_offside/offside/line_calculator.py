"""
Offside line calculation.
Determines the line from the last visible players' bbox ground points.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from var_offside.detection.player_detector import PlayerDetection
from var_offside.detection.body_parts import PlayableBodyPart
from var_offside.calibration.base import CalibrationResult


def get_directional_ground_point(
    detection: PlayerDetection,
    attacking_side: str,
) -> Tuple[float, float]:
    """Return the bbox bottom corner that matches the attack direction."""
    x1, _y1, x2, y2 = detection.bbox
    if attacking_side == "left":
        return (float(x1), float(y2))
    return (float(x2), float(y2))


def project_detection_to_field(
    detection: PlayerDetection,
    attacking_side: str,
    calibration: Optional[CalibrationResult],
) -> Optional[Tuple[float, float]]:
    """Project the direction-aligned ground point to field coordinates."""
    if calibration is None or not calibration.is_valid:
        return None
    if calibration.method == "vanishing_point":
        # VP calibration renders the line from image geometry directly; its
        # placeholder homography is not stable enough for player ordering.
        return None
    return calibration.project_to_field(
        get_directional_ground_point(detection, attacking_side)
    )


def get_directional_ordered_indices(
    detections: List[PlayerDetection],
    attacking_side: str,
    calibration: Optional[CalibrationResult] = None,
) -> List[int]:
    """Return detection indices ordered from closest to the attacked goal."""
    reverse = attacking_side == "right"

    projected = []
    fallback = []
    for idx, detection in enumerate(detections):
        field_point = project_detection_to_field(detection, attacking_side, calibration)
        if field_point is not None:
            projected.append((idx, field_point[0]))
        else:
            fallback.append(
                (idx, get_directional_ground_point(detection, attacking_side)[0])
            )

    if projected:
        projected.sort(key=lambda item: item[1], reverse=reverse)
        ordered_indices = [idx for idx, _value in projected]
        if fallback:
            fallback.sort(key=lambda item: item[1], reverse=reverse)
            ordered_indices.extend(idx for idx, _value in fallback)
        return ordered_indices

    fallback.sort(key=lambda item: item[1], reverse=reverse)
    return [idx for idx, _value in fallback]


def get_team_filtered_indices(
    detections: List[PlayerDetection],
    team_id: Optional[int],
) -> List[int]:
    """Return indices limited to one team when team ids are available."""
    if team_id not in (1, 2):
        return list(range(len(detections)))

    team_indices = [idx for idx, det in enumerate(detections) if det.team_id == team_id]
    return team_indices or list(range(len(detections)))


def get_ignored_goalkeeper_idx(
    detections: List[PlayerDetection],
    ordered_indices: List[int],
    exclude_goalkeeper: bool,
) -> Optional[int]:
    """Return the closest goalkeeper to ignore when configured to do so."""
    if not exclude_goalkeeper:
        return None

    for idx in ordered_indices:
        if detections[idx].is_goalkeeper:
            return idx
    return None


@dataclass
class OffsideLine:
    """Representation of the offside line."""

    # X position in field coordinates (meters from goal line)
    field_x: Optional[float]

    # X position in image coordinates
    image_x: float

    # Reference point used to anchor the projected line in image coordinates
    reference_point: Tuple[float, float]

    # The player who defines this line after applying the exclusion rule
    defining_player_idx: int

    # Which body part defines the line
    defining_body_part: Optional[PlayableBodyPart]

    # Player ignored as the goalkeeper / last edge player
    ignored_player_idx: Optional[int] = None

    # Line endpoints in image coordinates for visualization
    image_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    # Confidence in this line
    confidence: float = 1.0


class OffsideLineCalculator:
    """
    Calculate the offside line position.

    In geometry mode the line is anchored to the directional bbox-bottom point
    of the relevant defending player for the selected attacking direction.

    With team context, the default line is the second-last defender. When
    goalkeeper exclusion is enabled, the closest non-goalkeeper defender is used.
    """

    def __init__(self):
        """Initialize offside line calculator."""
        pass

    def calculate_offside_line(
        self,
        detections: List[PlayerDetection],
        attacking_side: str,
        calibration: Optional[CalibrationResult] = None,
        exclude_goalkeeper: bool = True,
        defending_team: Optional[int] = None,
    ) -> Optional[OffsideLine]:
        """
        Calculate the offside line for a given play.

        Args:
            detections: All player detections
            attacking_side: Direction of attack ("left" or "right")
            calibration: Optional calibration result for field projection
            exclude_goalkeeper: Whether to skip detections marked as goalkeepers
            defending_team: Team id for the side defining the line

        Returns:
            OffsideLine or None if cannot be calculated
        """
        team_indices = get_team_filtered_indices(detections, defending_team)
        team_detections = [detections[idx] for idx in team_indices]
        if not team_detections:
            return None

        local_order = get_directional_ordered_indices(
            team_detections,
            attacking_side,
            calibration=calibration,
        )
        ordered_indices = [team_indices[idx] for idx in local_order]

        ignored_player_idx = get_ignored_goalkeeper_idx(
            detections,
            ordered_indices,
            exclude_goalkeeper,
        )

        if exclude_goalkeeper:
            candidate_indices = [
                idx for idx in ordered_indices if not detections[idx].is_goalkeeper
            ]
        else:
            candidate_indices = ordered_indices

        if not candidate_indices:
            return None

        line_rank = (
            0 if exclude_goalkeeper else (1 if len(candidate_indices) > 1 else 0)
        )
        idx = candidate_indices[line_rank]
        det = detections[idx]
        reference_point = get_directional_ground_point(det, attacking_side)
        image_x = reference_point[0]

        field_point = project_detection_to_field(det, attacking_side, calibration)
        field_x = field_point[0] if field_point is not None else None

        return OffsideLine(
            field_x=field_x,
            image_x=image_x,
            reference_point=reference_point,
            defining_player_idx=idx,
            ignored_player_idx=ignored_player_idx,
            defining_body_part=None,
            confidence=float(det.confidence),
        )

    def get_offside_line_image_points(
        self,
        offside_line: OffsideLine,
        image_height: int,
        calibration: Optional[CalibrationResult] = None,
    ) -> List[Tuple[float, float]]:
        """
        Get points to draw the offside line in the image.

        The line passes through the stored reference point and
        converges toward the vanishing point (if available), giving a
        perspective-correct rendering.

        Args:
            offside_line: Calculated offside line
            image_height: Height of image
            calibration: Calibration result

        Returns:
            List of (x, y) points forming the line
        """
        x = offside_line.image_x

        if calibration and calibration.vanishing_point:
            vx, vy = calibration.vanishing_point

            ref_y = offside_line.reference_point[1]

            dx = float(x) - vx
            dy = ref_y - vy

            if abs(dy) > 1e-6:
                t_top = (0.0 - vy) / dy
                x_top = vx + t_top * dx
                t_bot = (float(image_height) - vy) / dy
                x_bot = vx + t_bot * dx
                return [(float(x_top), 0.0), (float(x_bot), float(image_height))]

        # Vertical fallback
        return [(x, 0), (x, image_height)]
