"""Is someone in the teams really an official? Your rule keeps referees as detected people."""

from __future__ import annotations

from typing import List, Set

import numpy as np

from var_offside.agent.checks.teams import MIN_TEAM_PLAYERS, _unit, team_vectors
from var_offside.agent.findings import DETECTOR_CLASS, DOUBT, EMBEDDING, GEOMETRY, Evidence, Finding
from var_offside.agent.scene import Change, Scene, role_of

# How much further from both teams than a typical player is from their own.
KIT_OUTLIER_FACTOR = 2.0
MIN_OUTLIER_DISTANCE = 0.5
TOUCHLINE_M = 3.0
FIELD_WIDTH = 68.0


def kit_outliers(scene: Scene) -> Set[int]:
    """People whose kit is far from both teams, compared with how tight the teams are."""
    teams = team_vectors(scene)
    if len(teams[1]) < MIN_TEAM_PLAYERS or len(teams[2]) < MIN_TEAM_PLAYERS:
        return set()

    centres = {team: _unit(np.mean(list(teams[team].values()), axis=0)) for team in (1, 2)}
    spread = [
        float(np.linalg.norm(vector - centres[team]))
        for team in (1, 2)
        for vector in teams[team].values()
    ]
    limit = max(KIT_OUTLIER_FACTOR * float(np.median(spread)), MIN_OUTLIER_DISTANCE)

    outliers = set()
    for det in scene.detections:
        vector = scene.embeddings.get(det.player_id)
        if vector is None:
            continue
        unit = _unit(vector)
        if min(float(np.linalg.norm(unit - centre)) for centre in centres.values()) > limit:
            outliers.add(det.player_id)
    return outliers


def _near_touchline(scene: Scene, det) -> bool:
    calibration = scene.calibration
    if calibration is None or not calibration.is_valid or calibration.method == "vanishing_point":
        return False
    point = calibration.project_to_field(det.bottom_center)
    return point is not None and (point[1] < TOUCHLINE_M or point[1] > FIELD_WIDTH - TOUCHLINE_M)


def check(scene: Scene) -> List[Finding]:
    """Propose the referee role from the detector class and an odd kit, never from position alone."""
    outliers = kit_outliers(scene)
    findings = []
    for det in scene.detections:
        pid = det.player_id
        if role_of(det) != "player" or det.detector_class == "goalkeeper":
            continue
        evidence = []
        if det.detector_class == "referee":
            evidence.append(Evidence(DETECTOR_CLASS, f"The detector labels #{pid} a referee."))
        if pid in outliers:
            evidence.append(Evidence(EMBEDDING, f"#{pid}'s kit matches neither team."))
        if not evidence:
            continue
        if _near_touchline(scene, det):
            evidence.append(Evidence(GEOMETRY, f"#{pid} is along the touchline."))
        mark = Change("set_role", player_id=pid, role="referee")
        findings.append(
            Finding(
                "officials",
                "official_candidate",
                f"#{pid} may be an official.",
                DOUBT,
                (pid,),
                tuple(evidence),
                # A kit outlier alone may be the keeper, so it's only a doubt.
                proposal=mark if det.detector_class == "referee" else None,
                alternatives=(mark,),
            )
        )
    return findings
