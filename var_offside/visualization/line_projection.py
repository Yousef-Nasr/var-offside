"""Shared helpers for projecting the line analysis geometry into image space."""

from typing import List, Optional, Tuple

import numpy as np

from var_offside.calibration.base import CalibrationResult


PITCH_WIDTH = 68.0
LINE_SAMPLES = 30


def project_line(
    field_x: Optional[float],
    image_x: float,
    image_y: Optional[float],
    image_h: int,
    image_w: int,
    calibration: Optional[CalibrationResult],
) -> List[Tuple[float, float]]:
    """Project the analysis line into image space using the best available geometry."""
    if (
        field_x is not None
        and calibration is not None
        and calibration.has_3d_calibration
    ):
        pts = _project_via_3d(field_x, calibration.projection_matrix, image_h, image_w)
        if len(pts) >= 2:
            return pts

    if calibration is not None and calibration.vanishing_point is not None:
        ref_y = image_y if image_y is not None else float(image_h)
        return _project_via_vp(image_x, ref_y, calibration.vanishing_point, image_h)

    if (
        field_x is not None
        and calibration is not None
        and calibration.homography_inv is not None
    ):
        pts = _project_via_homography(field_x, calibration, image_h, image_w)
        if len(pts) >= 2:
            return pts

    return [(float(image_x), 0.0), (float(image_x), float(image_h))]


def _project_via_homography(
    field_x: float,
    calibration: CalibrationResult,
    image_h: int,
    image_w: int,
) -> List[Tuple[float, float]]:
    pts = []
    for y in np.linspace(-5, PITCH_WIDTH + 5, LINE_SAMPLES):
        img_pt = calibration.project_to_image((field_x, y))
        if img_pt is None:
            continue
        ix, iy = img_pt
        margin = max(image_w, image_h)
        if abs(ix) < margin * 2 and abs(iy) < margin * 2:
            pts.append((float(ix), float(iy)))
    return pts


def _project_via_3d(
    field_x: float,
    projection_matrix,
    image_h: int,
    image_w: int,
) -> List[Tuple[float, float]]:
    pts = []
    for y in np.linspace(-5, PITCH_WIDTH + 5, LINE_SAMPLES):
        point = np.array([field_x - 52.5, y - 34.0, 0.0, 1.0])
        proj = projection_matrix @ point
        if proj[2] <= 0:
            continue
        ix = proj[0] / proj[2]
        iy = proj[1] / proj[2]
        margin = max(image_w, image_h)
        if abs(ix) < margin * 2 and abs(iy) < margin * 2:
            pts.append((float(ix), float(iy)))
    return pts


def _project_via_vp(
    image_x: float,
    image_y: float,
    vanishing_point: Tuple[float, float],
    image_h: int,
) -> List[Tuple[float, float]]:
    vx, vy = vanishing_point
    dx = float(image_x) - vx
    dy = float(image_y) - vy

    if abs(dy) < 1e-6:
        return [(float(image_x), 0.0), (float(image_x), float(image_h))]

    t_top = (0.0 - vy) / dy
    x_top = vx + t_top * dx
    t_bot = (float(image_h) - vy) / dy
    x_bot = vx + t_bot * dx
    return [(float(x_top), 0.0), (float(x_bot), float(image_h))]
