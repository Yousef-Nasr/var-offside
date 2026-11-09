"""Can this frame be judged at all?"""

from __future__ import annotations

from typing import List

import cv2

from var_offside.agent.findings import CALIBRATION, FRAME, UNUSABLE, Evidence, Finding
from var_offside.agent.scene import Scene

MIN_PEOPLE = 6
MIN_WIDTH_PX = 640
MIN_SHARPNESS = 15.0


def _sign(code: str, message: str, kind: str = FRAME) -> Finding:
    return Finding("frame", code, message, UNUSABLE, evidence=(Evidence(kind, message),))


def check(scene: Scene) -> List[Finding]:
    """Signs that the shot can't support a call. Policy needs two different kinds."""
    findings = []
    people = len(scene.detections)
    if people < MIN_PEOPLE:
        findings.append(_sign("too_few_people", f"Only {people} people were found."))
    if scene.image is not None and scene.image.size:
        if scene.width < MIN_WIDTH_PX:
            findings.append(_sign("too_small", f"The frame is only {scene.width} px wide."))
        gray = cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
            findings.append(_sign("blurred", "The frame is too blurred to place feet."))
    if scene.calibration is None or not scene.calibration.is_valid:
        findings.append(_sign("no_pitch", "No pitch lines were found.", CALIBRATION))
    return findings
