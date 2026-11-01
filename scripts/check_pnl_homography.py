"""
Check that PnLCalibration's ground-plane homography matches its camera.

For a grid of field points that project in front of the calibrated camera,
compare the projection through ``projection_matrix`` with the one through
``homography_inv`` (field -> image), then map back with ``homography``
(image -> field) and report the maximum pixel and metre error.

    python scripts/check_pnl_homography.py [image]

Exits with PASS when the maximum pixel error is below 1 px, else FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_image(arg: str) -> Path:
    path = Path(arg)
    if path.is_absolute() or path.is_file():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.is_file() else path


def _project_camera(projection_matrix: np.ndarray, field_x: float, field_y: float):
    """Project an absolute field ground point through P (centred PnL world)."""
    point = np.array([field_x - 52.5, field_y - 34.0, 0.0, 1.0], dtype=np.float64)
    proj = projection_matrix @ point
    if proj[2] <= 1e-10:
        return None
    return np.array([proj[0] / proj[2], proj[1] / proj[2]])


def _project_homography(homography: np.ndarray, field_x: float, field_y: float):
    point = np.array([field_x, field_y, 1.0], dtype=np.float64)
    proj = homography @ point
    if abs(proj[2]) < 1e-10:
        return None
    return np.array([proj[0] / proj[2], proj[1] / proj[2]])


def _backproject(homography: np.ndarray, pixel_x: float, pixel_y: float):
    point = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
    proj = homography @ point
    if abs(proj[2]) < 1e-10:
        return None
    return np.array([proj[0] / proj[2], proj[1] / proj[2]])


def main() -> int:
    from var_offside.calibration.pnl_calibration import PnLCalibration

    image_arg = sys.argv[1] if len(sys.argv) > 1 else "images/test7.png"
    image_path = _resolve_image(image_arg)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"FAIL: could not read image {image_path}")
        return 1

    print(f"Calibrating {image_path} ...")
    result = PnLCalibration().calibrate(image)

    if result.projection_matrix is None:
        print("FAIL: PnLCalibration returned no projection_matrix")
        return 1
    if result.homography is None or result.homography_inv is None:
        print("FAIL: PnLCalibration returned no homography/homography_inv")
        return 1

    max_pixel_error = 0.0
    max_metre_error = 0.0
    compared = 0

    for field_x in np.arange(0.0, 105.0 + 1e-9, 7.5):
        for field_y in np.arange(0.0, 68.0 + 1e-9, 6.8):
            p_cam = _project_camera(result.projection_matrix, field_x, field_y)
            if p_cam is None:
                continue  # behind the camera / horizon
            p_h = _project_homography(result.homography_inv, field_x, field_y)
            if p_h is None:
                continue

            pixel_error = float(np.linalg.norm(p_cam - p_h))
            max_pixel_error = max(max_pixel_error, pixel_error)

            back = _backproject(result.homography, float(p_h[0]), float(p_h[1]))
            if back is None:
                continue
            metre_error = float(
                np.linalg.norm(back - np.array([field_x, field_y], dtype=np.float64))
            )
            max_metre_error = max(max_metre_error, metre_error)
            compared += 1

    print(f"Field points compared: {compared}")
    print(f"Max pixel error: {max_pixel_error:.4f} px")
    print(f"Max metre error: {max_metre_error:.4f} m")

    if compared == 0:
        print("FAIL: no field points projected in front of the camera")
        return 1
    if max_pixel_error < 1.0:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
