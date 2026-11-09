"""Can the camera be trusted, and how far could it be off at the line?"""

from __future__ import annotations

from statistics import median
from typing import List, Optional, Sequence

import numpy as np

from var_offside.agent.findings import BLOCKER, CALIBRATION, Evidence, Finding
from var_offside.agent.scene import Scene, role_of
from var_offside.calibration import pitch_solver as ps
from var_offside.calibration.base import CalibrationResult

MIN_HEIGHT_SAMPLES = 5
PLAYER_HEIGHT_M = (1.5, 2.1)
HEIGHT_DRIFT_M = 0.35
MIN_PIXELS_FOR_PINS = 4
# PnLCalib's quality is 1 - error / 20 px.
QUALITY_ERROR_PX = 20.0


def _in_view_pins(camera: ps.Camera, width: int, height: int) -> List[ps.Pin]:
    """Landmarks the camera shows, as if each had been placed at its projected pixel."""
    pins = []
    for mark in ps.LANDMARKS:
        world = ps._field_to_world(np.array([[mark.x, mark.y]], dtype=np.float64))  # noqa: SLF001
        h = ps._project_h(camera.params(), camera, world)[0]  # noqa: SLF001
        if h[2] <= 1e-6:
            continue
        u, v = h[0] / h[2], h[1] / h[2]
        if 0.0 <= u < width and 0.0 <= v < height:
            pins.append(ps.Pin("point", mark.id, float(u), float(v)))
    return pins


def line_uncertainty_m(calibration: Optional[CalibrationResult], width: int, height: int) -> Optional[float]:
    """How far the line could be off, in metres, or None when it can't be worked out.

    Hand fits use their own pins. An automatic fit has no pins, so the landmarks it
    shows stand in for them, and the figure grows with the fit's reported error.
    """
    camera = ps.Camera.from_calibration(calibration)
    if camera is None or width <= 0 or height <= 0:
        return None

    placed = (calibration.debug_info or {}).get("pins")
    pins = (
        ps.parse_pins(placed)
        if placed
        else _in_view_pins(camera, width, height)
    )
    if len(pins) < MIN_PIXELS_FOR_PINS:
        return None

    value = ps.estimate_line_error(camera, pins, width, height)
    if not np.isfinite(value):
        return None
    if placed:
        return float(value)
    error_px = (1.0 - float(calibration.quality_score)) * QUALITY_ERROR_PX
    return float(value) * max(1.0, error_px / ps._PLACEMENT_SIGMA_PX)  # noqa: SLF001


def _height_m(camera: ps.Camera, foot_xy: Sequence[float], head_row: float) -> Optional[float]:
    """Height of someone standing at a field point whose head shows at this image row."""
    world = ps._field_to_world(np.array([foot_xy], dtype=np.float64))[0]  # noqa: SLF001
    projection = camera.K @ camera.R @ np.column_stack((np.eye(3), -camera.position))
    foot = projection @ np.append(world, 1.0)
    # World z is negative above the pitch, so height h adds -h to z.
    up_row, up_depth = projection[1, 2], projection[2, 2]
    below = up_row - head_row * up_depth
    if abs(below) < 1e-9:
        return None
    return float((foot[1] - head_row * foot[2]) / below)


def player_heights_m(scene: Scene, camera: ps.Camera) -> List[tuple]:
    """(distance from the camera along the pitch width, height) for each outfield person."""
    samples = []
    for det in scene.detections:
        if role_of(det) == "referee":
            continue
        point = scene.calibration.project_to_field(det.bottom_center)
        if point is None:
            continue
        height = _height_m(camera, point, det.bbox[1])
        if height is not None and np.isfinite(height):
            samples.append((point[1], height))
    return samples


def _height_findings(scene: Scene, camera: ps.Camera) -> List[Finding]:
    """Evidence the fit never saw: wrong people's heights mean a wrong camera."""
    samples = player_heights_m(scene, camera)
    if len(samples) < MIN_HEIGHT_SAMPLES:
        return []
    typical = median(height for _position, height in samples)
    low, high = PLAYER_HEIGHT_M
    if not low <= typical <= high:
        message = f"Players come out {typical:.2f} m tall, so the camera looks wrong."
        return [_blocker("odd_player_heights", message)]

    samples.sort()
    third = max(1, len(samples) // 3)
    near = median(height for _position, height in samples[:third])
    far = median(height for _position, height in samples[-third:])
    if abs(near - far) > HEIGHT_DRIFT_M:
        message = f"Player heights drift across the pitch ({near:.2f} m to {far:.2f} m), so the camera looks wrong."
        return [_blocker("height_drift", message)]
    return []


def _blocker(code: str, message: str) -> Finding:
    return Finding("camera", code, message, BLOCKER, evidence=(Evidence(CALIBRATION, message),))


def check(scene: Scene) -> List[Finding]:
    """A rejected or missing camera, an unknown uncertainty, and player-height checks."""
    calibration = scene.calibration
    if calibration is None or not calibration.is_valid:
        rejected = (getattr(calibration, "debug_info", None) or {}).get("rejected")
        return [_blocker("no_camera", rejected or "There's no calibrated camera, so no metres.")]
    camera = ps.Camera.from_calibration(calibration)
    if camera is None:
        return [_blocker("no_3d_camera", "There's no full 3D camera, so no measured margin.")]

    problem = ps.camera_problem(camera, scene.width or int(2.0 * camera.cx))
    if problem:
        return [_blocker("camera_problem", problem)]

    findings = _height_findings(scene, camera)
    if line_uncertainty_m(calibration, scene.width, scene.height) is None:
        findings.append(_blocker("unknown_uncertainty", "The error at the line can't be worked out."))
    return findings
