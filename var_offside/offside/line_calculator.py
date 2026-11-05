"""
Offside line calculation.
Determines the line from the last visible players' body points on the ground.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from var_offside.config import get_settings
from var_offside.detection.player_detector import PlayerDetection
from var_offside.detection.body_parts import PlayableBodyPart
from var_offside.calibration.base import CalibrationResult

HALFWAY_X_M = 52.5

# COCO keypoints that count for offside: head, shoulders, hips, knees, ankles.
# Elbows (7, 8) and wrists (9, 10) are arms, which the law does not count.
_NON_ARM_KEYPOINTS = (0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16)
_TORSO_LEG_KEYPOINTS = frozenset((5, 6, 11, 12, 13, 14, 15, 16))
_MIN_TORSO_LEG_KEYPOINTS = 4
# Pose is matched to boxes by IoU, so a point well outside the box is a neighbour's.
_KEYPOINT_BOX_SLACK = 0.1

_SIGN_SAMPLE_PX = 40.0
_MIN_ROW_DISTANCE_PX = 1.0


def _most_advanced_body_x(
    detection: PlayerDetection, attacking_side: str
) -> Optional[float]:
    """Image x of the most advanced non-arm keypoint, or None when pose is unusable."""
    keypoints = detection.keypoints
    if keypoints is None:
        return None
    keypoints = np.asarray(keypoints)
    if keypoints.shape != (17, 3):
        return None

    threshold = get_settings().pose.keypoint_confidence_threshold
    x1, y1, x2, y2 = detection.bbox
    slack_x = (x2 - x1) * _KEYPOINT_BOX_SLACK
    slack_y = (y2 - y1) * _KEYPOINT_BOX_SLACK

    xs = []
    torso_and_legs = 0
    for index in _NON_ARM_KEYPOINTS:
        x, y, confidence = keypoints[index]
        if confidence < threshold:
            continue
        if not (x1 - slack_x <= x <= x2 + slack_x and y1 - slack_y <= y <= y2 + slack_y):
            continue
        xs.append(float(x))
        if index in _TORSO_LEG_KEYPOINTS:
            torso_and_legs += 1

    if torso_and_legs < _MIN_TORSO_LEG_KEYPOINTS:
        return None
    return min(xs) if attacking_side == "left" else max(xs)


def get_directional_ground_point(
    detection: PlayerDetection,
    attacking_side: str,
) -> Tuple[float, float]:
    """Return the player's most advanced body point, dropped to the feet row.

    Uses pose keypoints without the arms.  Falls back to the bbox bottom corner
    that matches the attack direction when pose is missing or too sparse.
    """
    x1, _y1, x2, y2 = detection.bbox
    body_x = _most_advanced_body_x(detection, attacking_side)
    if body_x is not None:
        return (body_x, float(y2))
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


def attack_field_sign(
    detections: Sequence[PlayerDetection],
    attacking_side: str,
    calibration: Optional[CalibrationResult],
) -> Optional[int]:
    """Which way the attacked goal lies along pitch x, from the calibration.

    A reverse-angle camera or a mirrored fit runs pitch metres the other way on
    screen, so the direction is measured around the players rather than assumed.

    Returns:
        +1 for increasing pitch x, -1 for decreasing, None when it can't be measured.
    """
    if calibration is None or not calibration.is_valid:
        return None
    if calibration.method == "vanishing_point":
        return None

    votes = 0
    for detection in detections:
        x, y = get_directional_ground_point(detection, attacking_side)
        left = calibration.project_to_field((x - _SIGN_SAMPLE_PX, y))
        right = calibration.project_to_field((x + _SIGN_SAMPLE_PX, y))
        if left is None or right is None:
            continue
        change = right[0] - left[0]
        if not math.isfinite(change) or change == 0.0:
            continue
        votes += 1 if change > 0.0 else -1

    if votes == 0:
        return None
    sign = 1 if votes > 0 else -1
    return sign if attacking_side == "right" else -sign


def x_on_ray_at_row(
    point: Tuple[float, float],
    vanishing_point: Tuple[float, float],
    row: float,
) -> float:
    """Slide an image point along its ray through the vanishing point to a row."""
    x, y = point
    vx, vy = vanishing_point
    from_vp = y - vy
    if abs(from_vp) < _MIN_ROW_DISTANCE_PX or from_vp * (row - vy) <= 0.0:
        return float(x)
    return float(vx + (row - vy) * (x - vx) / from_vp)


def _image_ordered_indices(
    detections: Sequence[PlayerDetection],
    indices: Sequence[int],
    attacking_side: str,
    calibration: Optional[CalibrationResult],
) -> List[int]:
    """Order players by image position, along the drawn lines when they slant."""
    points = {
        idx: get_directional_ground_point(detections[idx], attacking_side)
        for idx in indices
    }
    vanishing_point = calibration.vanishing_point if calibration is not None else None
    if vanishing_point is not None and points:
        row = max(point[1] for point in points.values())
        values = {
            idx: x_on_ray_at_row(point, vanishing_point, row)
            for idx, point in points.items()
        }
    else:
        values = {idx: point[0] for idx, point in points.items()}
    return sorted(indices, key=values.get, reverse=attacking_side == "right")


def get_directional_ordered_indices(
    detections: List[PlayerDetection],
    attacking_side: str,
    calibration: Optional[CalibrationResult] = None,
) -> List[int]:
    """Return detection indices ordered from closest to the attacked goal."""
    sign = attack_field_sign(detections, attacking_side, calibration)

    projected = []
    fallback = []
    for idx, detection in enumerate(detections):
        field_point = (
            project_detection_to_field(detection, attacking_side, calibration)
            if sign is not None
            else None
        )
        if field_point is not None:
            projected.append((idx, field_point[0] * sign))
        else:
            fallback.append(idx)

    projected.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _value in projected] + _image_ordered_indices(
        detections, fallback, attacking_side, calibration
    )


def get_team_filtered_indices(
    detections: List[PlayerDetection],
    team_id: Optional[int],
) -> List[int]:
    """Return the indices of one team; empty when the team is unknown or absent."""
    if team_id not in (1, 2):
        return []
    return [idx for idx, det in enumerate(detections) if det.team_id == team_id]


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

    # Direction of the attacked goal in pitch coordinates
    attack_field_sign: Optional[int] = None


class OffsideLineCalculator:
    """
    Calculate the offside line position.

    The line is anchored to the most advanced body point of the relevant
    defending player for the selected attacking direction.

    With a keeper marked, all defenders including the keeper are ranked and the
    second one sets the line.  With no keeper marked, ``exclude_goalkeeper`` means
    the keeper isn't in frame, so the last visible defender sets it; otherwise the
    second visible defender does.  Without a known defending team there is no line.
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
        keeper_idx: Optional[int] = None,
    ) -> Optional[OffsideLine]:
        """
        Calculate the offside line for a given play.

        Args:
            detections: All player detections
            attacking_side: Direction of attack ("left" or "right")
            calibration: Optional calibration result for field projection
            exclude_goalkeeper: With no keeper marked, whether the last visible
                defender sets the line (the keeper isn't in frame)
            defending_team: Team id for the side defining the line
            keeper_idx: Index of the keeper, who counts as a defender whatever
                their team id.  When None, a marked keeper is any goalkeeper in
                the defending team.

        Returns:
            OffsideLine or None if cannot be calculated
        """
        team_indices = get_team_filtered_indices(detections, defending_team)
        if not team_indices:
            return None
        if keeper_idx is not None and not 0 <= keeper_idx < len(detections):
            return None
        if keeper_idx is not None and keeper_idx not in team_indices:
            team_indices.append(keeper_idx)

        team_detections = [detections[idx] for idx in team_indices]
        local_order = get_directional_ordered_indices(
            team_detections,
            attacking_side,
            calibration=calibration,
        )
        ordered_indices = [team_indices[idx] for idx in local_order]

        has_keeper = keeper_idx is not None or any(
            detections[idx].is_goalkeeper for idx in ordered_indices
        )
        if has_keeper and len(ordered_indices) < 2:
            return None

        use_second = has_keeper or (not exclude_goalkeeper and len(ordered_indices) > 1)
        idx = ordered_indices[1 if use_second else 0]
        ignored_player_idx = ordered_indices[0] if use_second else None
        det = detections[idx]
        reference_point = get_directional_ground_point(det, attacking_side)
        image_x = reference_point[0]

        sign = attack_field_sign(team_detections, attacking_side, calibration)
        field_point = (
            project_detection_to_field(det, attacking_side, calibration)
            if sign is not None
            else None
        )
        field_x = field_point[0] if field_point is not None else None

        return OffsideLine(
            field_x=field_x,
            image_x=image_x,
            reference_point=reference_point,
            defining_player_idx=idx,
            ignored_player_idx=ignored_player_idx,
            defining_body_part=None,
            confidence=float(det.confidence),
            attack_field_sign=sign,
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
