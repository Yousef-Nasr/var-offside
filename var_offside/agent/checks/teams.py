"""Does each player's kit embedding fit their team? Team clustering itself is untouched."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from var_offside.agent.findings import DOUBT, EMBEDDING, UNCHECKED, Evidence, Finding
from var_offside.agent.scene import Change, Scene, role_of

# A distance ratio to own team / other team. Above 1 the other team is closer.
WRONG_TEAM_RATIO = 1.1
BETWEEN_TEAMS_RATIO = 0.9
MIN_TEAM_PLAYERS = 3


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def team_vectors(scene: Scene) -> Dict[int, Dict[int, np.ndarray]]:
    """Unit embeddings by team and player id, leaving out keepers and officials."""
    teams: Dict[int, Dict[int, np.ndarray]] = {1: {}, 2: {}}
    for det in scene.detections:
        vector = scene.embeddings.get(det.player_id)
        if vector is not None and det.team_id in teams and role_of(det) == "player":
            teams[det.team_id][det.player_id] = _unit(vector)
    return teams


def distance_ratio(
    player_id: int, team: int, teams: Dict[int, Dict[int, np.ndarray]]
) -> Optional[Tuple[float, float]]:
    """Distances to the player's own team and the other team, each without the player."""
    vector = teams[team].get(player_id)
    own = [v for pid, v in teams[team].items() if pid != player_id]
    other = list(teams[3 - team].values())
    if vector is None or len(own) < MIN_TEAM_PLAYERS - 1 or len(other) < MIN_TEAM_PLAYERS - 1:
        return None
    return (
        float(np.linalg.norm(vector - _unit(np.mean(own, axis=0)))),
        float(np.linalg.norm(vector - _unit(np.mean(other, axis=0)))),
    )


def check(scene: Scene) -> List[Finding]:
    """Flag players closer to the other team, or about the same from both."""
    teams = team_vectors(scene)
    if len(teams[1]) < MIN_TEAM_PLAYERS or len(teams[2]) < MIN_TEAM_PLAYERS:
        return [
            Finding("teams", "no_embeddings", "Kit embeddings aren't available to check teams.", UNCHECKED)
        ]

    findings = []
    for team in (1, 2):
        for player_id in teams[team]:
            distances = distance_ratio(player_id, team, teams)
            if distances is None or distances[1] <= 0.0:
                continue
            ratio = distances[0] / distances[1]
            if ratio < BETWEEN_TEAMS_RATIO:
                continue
            switch = Change("set_team", player_id=player_id, team=3 - team)
            wrong = ratio >= WRONG_TEAM_RATIO
            message = (
                f"#{player_id}'s kit is closer to Team {3 - team}."
                if wrong
                else f"#{player_id}'s kit fits both teams about equally."
            )
            findings.append(
                Finding(
                    "teams",
                    "closer_to_other_team" if wrong else "between_teams",
                    message,
                    DOUBT,
                    (player_id,),
                    (Evidence(EMBEDDING, message),),
                    proposal=switch if wrong else None,
                    alternatives=(switch,),
                )
            )
    return findings
