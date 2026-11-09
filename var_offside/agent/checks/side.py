"""Which way is the attack, and who is attacking?

A wrong answer here reverses the whole check, so a contradiction blocks the frame
whatever the stability test says.
"""

from __future__ import annotations

from statistics import median
from typing import List, Optional

from var_offside.agent.findings import BALL, BLOCKER, CALIBRATION, GEOMETRY, Evidence, Finding
from var_offside.agent.scene import Change, Scene, role_of
from var_offside.offside.line_calculator import attack_field_sign, get_directional_ordered_indices

GOAL_IN_VIEW_M = 15.0
HALF_LENGTH = 52.5
DEEPEST_PLAYERS = 4
BALL_REACH_BOX_HEIGHTS = 1.5


def _side_from_goal_in_view(scene: Scene) -> Optional[str]:
    """The screen direction of the goal the camera shows, when one is clearly in view."""
    calibration = scene.calibration
    # With attack "right", the sign says which way screen-right runs along pitch x.
    screen_right = attack_field_sign(list(scene.detections), "right", calibration)
    if screen_right is None:
        return None
    xs = []
    for det in scene.detections:
        point = calibration.project_to_field(det.bottom_center)
        if point is not None:
            xs.append(point[0])
    if not xs:
        return None
    offset = median(xs) - HALF_LENGTH
    if abs(offset) < GOAL_IN_VIEW_M:
        return None
    goal_sign = 1 if offset > 0.0 else -1
    return "right" if goal_sign == screen_right else "left"


def _side_from_keeper(scene: Scene) -> Optional[str]:
    """The side a marked or detected keeper stands on, from the middle of the players."""
    keepers = [
        det
        for det in scene.detections
        if det.is_goalkeeper or det.detector_class == "goalkeeper"
    ]
    if len(keepers) != 1 or len(scene.detections) < 3:
        return None
    middle = median(det.center[0] for det in scene.detections)
    return "right" if keepers[0].center[0] > middle else "left"


def _team_from_shape(scene: Scene) -> Optional[int]:
    """The attacking team, read from which team has the most deepest outfield players."""
    outfield = [det for det in scene.detections if role_of(det) == "player" and det.team_id in (1, 2)]
    ordered = get_directional_ordered_indices(outfield, scene.side, scene.calibration)
    deepest = [outfield[index].team_id for index in ordered[:DEEPEST_PLAYERS]]
    if len(deepest) < DEEPEST_PLAYERS:
        return None
    counts = {team: deepest.count(team) for team in (1, 2)}
    if counts[1] == counts[2]:
        return None
    defending = 1 if counts[1] > counts[2] else 2
    return 3 - defending


def _team_from_ball(scene: Scene) -> Optional[int]:
    """The team of the player nearest the ball, when someone is close enough to have it."""
    if scene.ball is None:
        return None
    bx, by = scene.ball.center
    nearest, distance = None, float("inf")
    for det in scene.detections:
        if role_of(det) != "player" or det.team_id not in (1, 2):
            continue
        x, y = det.bottom_center
        gap = ((x - bx) ** 2 + (y - by) ** 2) ** 0.5
        if gap < distance:
            nearest, distance = det, gap
    if nearest is None or distance > BALL_REACH_BOX_HEIGHTS * (nearest.bbox[3] - nearest.bbox[1]):
        return None
    return nearest.team_id


def _opinion(code: str, message: str, kind: str, change: Change) -> Finding:
    return Finding("side", code, message, evidence=(Evidence(kind, message),), proposal=change)


def _contradiction(code: str, message: str, kind: str, change: Change) -> Finding:
    return Finding(
        "side", code, message, BLOCKER, evidence=(Evidence(kind, message),), alternatives=(change,)
    )


def _check_value(scene_value, value, code, message, kind, change) -> List[Finding]:
    """An opinion when nothing is set, or a blocker when it contradicts what's set."""
    if value is None:
        return []
    if scene_value is None:
        return [_opinion(code, message, kind, change)]
    if scene_value != value:
        return [_contradiction(f"{code}_contradicted", f"{message} That contradicts the choice made.", kind, change)]
    return []


def check(scene: Scene) -> List[Finding]:
    """Evidence for the attack direction, and for the attacking team once it's known."""
    findings = []
    goal_side = _side_from_goal_in_view(scene)
    findings += _check_value(
        scene.side,
        goal_side,
        "goal_in_view",
        f"The goal in view is to the {goal_side}.",
        CALIBRATION,
        Change("set_side", side=goal_side),
    )
    keeper_side = _side_from_keeper(scene)
    findings += _check_value(
        scene.side,
        keeper_side,
        "keeper_position",
        f"The keeper stands to the {keeper_side}.",
        GEOMETRY,
        Change("set_side", side=keeper_side),
    )
    if scene.side not in ("left", "right"):
        return findings

    shape_team = _team_from_shape(scene)
    findings += _check_value(
        scene.attacking_team,
        shape_team,
        "team_shape",
        f"The deepest players look like a back line, so Team {shape_team} attacks.",
        GEOMETRY,
        Change("set_attacking_team", team=shape_team),
    )
    ball_team = _team_from_ball(scene)
    findings += _check_value(
        scene.attacking_team,
        ball_team,
        "ball_nearby",
        f"A Team {ball_team} player is nearest the ball.",
        BALL,
        Change("set_attacking_team", team=ball_team),
    )
    return findings
