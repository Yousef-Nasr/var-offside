"""Who is the keeper? Each candidate is tested rather than trusted: the old finder was right on 24%."""

from __future__ import annotations

from typing import Dict, List, Tuple

from var_offside.agent.checks.officials import kit_outliers
from var_offside.agent.findings import DETECTOR_CLASS, DOUBT, EMBEDDING, GEOMETRY, Evidence, Finding
from var_offside.agent.scene import Change, Scene, role_of
from var_offside.offside.line_calculator import get_directional_ordered_indices

DEEPEST_CANDIDATES = 3


def _candidates(scene: Scene) -> Dict[int, Tuple[Evidence, ...]]:
    """People with some sign of being the keeper, and why."""
    signs: Dict[int, List[Evidence]] = {}
    outliers = kit_outliers(scene)
    deepest = set()
    if scene.side in ("left", "right"):
        order = get_directional_ordered_indices(list(scene.detections), scene.side, scene.calibration)
        deepest = {scene.detections[index].player_id for index in order[:DEEPEST_CANDIDATES]}

    for det in scene.detections:
        if role_of(det) == "referee":
            continue
        pid = det.player_id
        if det.detector_class == "goalkeeper":
            signs.setdefault(pid, []).append(
                Evidence(DETECTOR_CLASS, f"The detector labels #{pid} a goalkeeper.")
            )
        if pid in outliers and (not deepest or pid in deepest):
            signs.setdefault(pid, []).append(
                Evidence(EMBEDDING, f"#{pid}'s kit differs from both teams.")
            )
    for pid, evidence in signs.items():
        if pid in deepest:
            evidence.append(Evidence(GEOMETRY, f"#{pid} is among the deepest players."))
    return {pid: tuple(evidence) for pid, evidence in signs.items()}


def check(scene: Scene) -> List[Finding]:
    """Propose a keeper when there's none, and doubt a choice that signs disagree with."""
    candidates = _candidates(scene)
    marked = next((det.player_id for det in scene.detections if det.is_goalkeeper), None)
    tests = [Change("set_role", player_id=pid, role="goalkeeper") for pid in candidates]
    if scene.keeper_choice != "not_in_frame":
        tests.append(Change("set_keeper_not_in_frame"))

    findings = []
    for pid, evidence in candidates.items():
        if pid == marked:
            continue
        mark = Change("set_role", player_id=pid, role="goalkeeper")
        findings.append(
            Finding(
                "keeper",
                "keeper_candidate",
                f"#{pid} may be the keeper.",
                DOUBT,
                (pid,),
                evidence,
                proposal=mark if marked is None else None,
                alternatives=tuple(test for test in tests if test != mark) + (mark,),
            )
        )
    return findings
