"""Are the boxes on real, separate players? Rules can't prove that someone is missing."""

from __future__ import annotations

from statistics import median
from typing import List, Sequence

from var_offside.agent.findings import DOUBT, GEOMETRY, NOTE, Evidence, Finding
from var_offside.agent.scene import Change, Scene

DUPLICATE_IOU = 0.7
OUTSIDE_PITCH_M = 3.0
SMALL_BOX_RATIO = 0.35
LARGE_BOX_RATIO = 2.5
MAX_TEAM_SIZE = 11
UNEVEN_TEAMS = 4
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0.0 else 0.0


def _removal(code: str, message: str, player_id: int, severity: str = DOUBT) -> Finding:
    remove = Change("remove_player", player_id=player_id)
    return Finding(
        "players",
        code,
        message,
        severity,
        (player_id,),
        (Evidence(GEOMETRY, message),),
        proposal=remove,
        alternatives=(remove,),
    )


def _duplicates(scene: Scene) -> List[Finding]:
    findings = []
    people = scene.detections
    for i, first in enumerate(people):
        for second in people[i + 1 :]:
            if first.team_id != second.team_id or _iou(first.bbox, second.bbox) < DUPLICATE_IOU:
                continue
            extra = min((first, second), key=lambda det: det.confidence)
            kept = second if extra is first else first
            findings.append(
                _removal(
                    "duplicate_box",
                    f"#{extra.player_id} and #{kept.player_id} are boxes on the same person.",
                    extra.player_id,
                )
            )
    return findings


def _outside_pitch(scene: Scene) -> List[Finding]:
    calibration = scene.calibration
    if calibration is None or not calibration.is_valid or calibration.method == "vanishing_point":
        return []
    findings = []
    for det in scene.detections:
        point = calibration.project_to_field(det.bottom_center)
        if point is None:
            continue
        x, y = point
        limit = OUTSIDE_PITCH_M
        if -limit <= x <= FIELD_LENGTH + limit and -limit <= y <= FIELD_WIDTH + limit:
            continue
        findings.append(
            _removal(
                "outside_pitch",
                f"#{det.player_id} stands outside the pitch, so may not be in play.",
                det.player_id,
            )
        )
    return findings


def _odd_sizes(scene: Scene) -> List[Finding]:
    heights = [det.bbox[3] - det.bbox[1] for det in scene.detections]
    if len(heights) < 5:
        return []
    typical = median(heights)
    findings = []
    for det, height in zip(scene.detections, heights):
        if SMALL_BOX_RATIO * typical <= height <= LARGE_BOX_RATIO * typical:
            continue
        remove = Change("remove_player", player_id=det.player_id)
        message = f"#{det.player_id}'s box is an unusual size for a person here."
        findings.append(
            Finding(
                "players",
                "odd_box_size",
                message,
                DOUBT,
                (det.player_id,),
                (Evidence(GEOMETRY, message),),
                alternatives=(remove,),
            )
        )
    return findings


def _team_counts(scene: Scene) -> List[Finding]:
    counts = {team: sum(det.team_id == team for det in scene.detections) for team in (1, 2)}
    findings = []
    for team, count in counts.items():
        if count > MAX_TEAM_SIZE:
            findings.append(
                Finding("players", "team_too_big", f"Team {team} has {count} players.", NOTE)
            )
    if abs(counts[1] - counts[2]) >= UNEVEN_TEAMS:
        findings.append(
            Finding(
                "players",
                "uneven_teams",
                f"Teams are uneven ({counts[1]} and {counts[2]}), so someone may be missed.",
                NOTE,
            )
        )
    return findings


def check(scene: Scene) -> List[Finding]:
    """Doubled boxes, boxes off the pitch, odd sizes and unlikely team counts."""
    return _duplicates(scene) + _outside_pitch(scene) + _odd_sizes(scene) + _team_counts(scene)
