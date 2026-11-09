"""
Solve a broadcast camera from pitch markings placed by hand.

Used by the browser app's Calibrate step.  The input is a list of *pins*: a pitch
landmark placed on the photo (a point pin), or a spot the photo shows a named pitch
line passing through (a line pin).  The output is a full camera in the same
coordinates as PnLCalib, so a hand result works everywhere an automatic one does:
the offside line, the 3D view, the replay and the exports.

Conventions match :mod:`var_offside.calibration.pnl_calibration`:

* Field metres: x along the pitch (0-105), y across it (0-68, near touchline 68).
* World metres: field minus the pitch centre, z pointing down, so a camera above the
  pitch has a negative z.
* ``x_cam = R @ (X_world - C)`` and ``P = K @ R @ [I | -C]``.

There are two ways to solve:

* **Editing** an automatic camera: start from it and change it as little as possible
  to meet the pins, by keeping the rest of the pitch softly where it was.  One pin is
  enough.
* **From scratch**: search focal lengths and viewpoints, then refine.  Needs at least
  seven constraints (a point pin gives two, a line pin one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from var_offside.calibration.base import CalibrationResult

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

# Rule distances, in metres.
_BOX_DEPTH = 16.5
_BOX_HALF = 20.16
_SIX_DEPTH = 5.5
_SIX_HALF = 9.16
_SPOT = 11.0
_CIRCLE = 9.15
_GOAL_HALF = 3.66
_ARC_HALF = math.sqrt(_CIRCLE**2 - (_BOX_DEPTH - _SPOT) ** 2)

_MID_Y = FIELD_WIDTH / 2.0

# A camera has seven unknowns: focal length, three for rotation, three for position.
CONSTRAINTS_NEEDED = 7

# Hand placement error assumed when estimating how far the line could be off.
_PLACEMENT_SIGMA_PX = 2.0
# Above these, an estimated line error is flagged as weak or refused.
# In simulation the estimate runs high on minimal placements (it covers the whole
# view, not only the area near the pins).  Good spreads stayed under 0.9 m and bunched
# ones started at 1.5 m, which sets the refusal point.
LINE_ERROR_WEAK_M = 0.5
LINE_ERROR_BAD_M = 1.0
# Average pin miss allowed, as a share of the image width (0.3% of 1920 is about 6 px).
_MISS_TOLERANCE = 0.003
# How strongly an edit holds the rest of the pitch still, relative to a pin.
_ANCHOR_WEIGHT_FEW_PINS = 0.05
_ANCHOR_WEIGHT_MANY_PINS = 0.012


# ---------------------------------------------------------------------------
# Pitch model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Landmark:
    """A pitch marking that can be pointed at: an intersection or a spot."""

    id: str
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class PitchLine:
    """A straight pitch marking, given by its two ends in field metres."""

    id: str
    name: str
    a: Tuple[float, float]
    b: Tuple[float, float]


def _build_landmarks() -> Tuple[Landmark, ...]:
    """Every landmark on a standard pitch, both ends and the centre."""
    points: List[Landmark] = []
    for code, side, goal_x, sign in (("r", "Right", FIELD_LENGTH, -1.0), ("l", "Left", 0.0, 1.0)):
        rows = (
            ("box_front_far", "penalty box, far front corner", _BOX_DEPTH, _MID_Y - _BOX_HALF),
            ("box_front_near", "penalty box, near front corner", _BOX_DEPTH, _MID_Y + _BOX_HALF),
            ("box_goal_far", "penalty box meets goal line, far side", 0.0, _MID_Y - _BOX_HALF),
            ("box_goal_near", "penalty box meets goal line, near side", 0.0, _MID_Y + _BOX_HALF),
            ("six_front_far", "6-yard box, far front corner", _SIX_DEPTH, _MID_Y - _SIX_HALF),
            ("six_front_near", "6-yard box, near front corner", _SIX_DEPTH, _MID_Y + _SIX_HALF),
            ("six_goal_far", "6-yard box meets goal line, far side", 0.0, _MID_Y - _SIX_HALF),
            ("six_goal_near", "6-yard box meets goal line, near side", 0.0, _MID_Y + _SIX_HALF),
            ("post_far", "foot of the far goal post", 0.0, _MID_Y - _GOAL_HALF),
            ("post_near", "foot of the near goal post", 0.0, _MID_Y + _GOAL_HALF),
            ("spot", "penalty spot", _SPOT, _MID_Y),
            ("arc_far", "penalty arc meets box line, far side", _BOX_DEPTH, _MID_Y - _ARC_HALF),
            ("arc_near", "penalty arc meets box line, near side", _BOX_DEPTH, _MID_Y + _ARC_HALF),
            ("corner_far", "far corner of the pitch", 0.0, 0.0),
            ("corner_near", "near corner of the pitch", 0.0, FIELD_WIDTH),
        )
        for suffix, label, depth, y in rows:
            points.append(
                Landmark(f"{code}_{suffix}", f"{side} {label}", goal_x + sign * depth, y)
            )
    half = FIELD_LENGTH / 2.0
    points.extend(
        (
            Landmark("c_spot", "Centre spot", half, _MID_Y),
            Landmark("c_far", "Halfway line meets far touchline", half, 0.0),
            Landmark("c_near", "Halfway line meets near touchline", half, FIELD_WIDTH),
            Landmark("c_circle_far", "Centre circle meets halfway line, far side", half, _MID_Y - _CIRCLE),
            Landmark("c_circle_near", "Centre circle meets halfway line, near side", half, _MID_Y + _CIRCLE),
        )
    )
    return tuple(points)


def _build_lines() -> Tuple[PitchLine, ...]:
    """Every straight marking that can be dragged onto the photo."""
    half = FIELD_LENGTH / 2.0
    lines: List[PitchLine] = [
        PitchLine("far_touchline", "Far touchline", (0.0, 0.0), (FIELD_LENGTH, 0.0)),
        PitchLine("near_touchline", "Near touchline", (0.0, FIELD_WIDTH), (FIELD_LENGTH, FIELD_WIDTH)),
        PitchLine("halfway", "Halfway line", (half, 0.0), (half, FIELD_WIDTH)),
    ]
    for code, side, goal_x, sign in (("r", "Right", FIELD_LENGTH, -1.0), ("l", "Left", 0.0, 1.0)):
        box_x = goal_x + sign * _BOX_DEPTH
        six_x = goal_x + sign * _SIX_DEPTH
        lines.extend(
            (
                PitchLine(f"{code}_goal", f"{side} goal line", (goal_x, 0.0), (goal_x, FIELD_WIDTH)),
                PitchLine(f"{code}_box_front", f"{side} penalty box, front line", (box_x, _MID_Y - _BOX_HALF), (box_x, _MID_Y + _BOX_HALF)),
                PitchLine(f"{code}_box_far", f"{side} penalty box, far side", (goal_x, _MID_Y - _BOX_HALF), (box_x, _MID_Y - _BOX_HALF)),
                PitchLine(f"{code}_box_near", f"{side} penalty box, near side", (goal_x, _MID_Y + _BOX_HALF), (box_x, _MID_Y + _BOX_HALF)),
                PitchLine(f"{code}_six_front", f"{side} 6-yard box, front line", (six_x, _MID_Y - _SIX_HALF), (six_x, _MID_Y + _SIX_HALF)),
                PitchLine(f"{code}_six_far", f"{side} 6-yard box, far side", (goal_x, _MID_Y - _SIX_HALF), (six_x, _MID_Y - _SIX_HALF)),
                PitchLine(f"{code}_six_near", f"{side} 6-yard box, near side", (goal_x, _MID_Y + _SIX_HALF), (six_x, _MID_Y + _SIX_HALF)),
            )
        )
    return tuple(lines)


LANDMARKS: Tuple[Landmark, ...] = _build_landmarks()
LINES: Tuple[PitchLine, ...] = _build_lines()
_LANDMARK_BY_ID = {item.id: item for item in LANDMARKS}
_LINE_BY_ID = {item.id: item for item in LINES}


def _arc(cx: float, cy: float, start: float, end: float, steps: int = 32) -> List[List[float]]:
    """Sample a circular marking as a polyline."""
    angles = np.linspace(start, end, steps + 1)
    return [
        [round(cx + _CIRCLE * math.cos(t), 3), round(cy + _CIRCLE * math.sin(t), 3)]
        for t in angles
    ]


def pitch_model() -> Dict[str, Any]:
    """JSON description of the pitch for the browser to draw and pick from."""
    reach = math.acos((_BOX_DEPTH - _SPOT) / _CIRCLE)
    arcs = [
        _arc(FIELD_LENGTH / 2.0, _MID_Y, 0.0, 2.0 * math.pi, 64),
        _arc(_SPOT, _MID_Y, -reach, reach),
        _arc(FIELD_LENGTH - _SPOT, _MID_Y, math.pi - reach, math.pi + reach),
    ]
    return {
        "length": FIELD_LENGTH,
        "width": FIELD_WIDTH,
        "landmarks": [
            {"id": item.id, "name": item.name, "x": item.x, "y": item.y}
            for item in LANDMARKS
        ],
        "lines": [
            {"id": item.id, "name": item.name, "a": list(item.a), "b": list(item.b)}
            for item in LINES
        ],
        "arcs": arcs,
    }


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pin:
    """One placement on the photo, in image pixels."""

    kind: str  # "point" | "line"
    ref: str  # landmark id or line id
    x: float
    y: float


def parse_pins(rows: Optional[Sequence[Dict[str, Any]]]) -> List[Pin]:
    """Validate client pins.

    Raises:
        ValueError: With a readable message for a malformed or unknown pin.
    """
    pins: List[Pin] = []
    for row in rows or ():
        kind = row.get("kind")
        ref = row.get("ref")
        if kind == "point" and ref not in _LANDMARK_BY_ID:
            raise ValueError(f"Unknown pitch landmark: {ref}")
        if kind == "line" and ref not in _LINE_BY_ID:
            raise ValueError(f"Unknown pitch line: {ref}")
        if kind not in ("point", "line"):
            raise ValueError("A pin's kind must be 'point' or 'line'")
        try:
            x, y = float(row["x"]), float(row["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A pin needs numeric x and y") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("A pin needs finite x and y")
        pins.append(Pin(str(kind), str(ref), x, y))
    return pins


def pin_name(pin: Pin) -> str:
    """Human name of what a pin refers to."""
    source = _LANDMARK_BY_ID if pin.kind == "point" else _LINE_BY_ID
    return source[pin.ref].name


def constraint_count(pins: Sequence[Pin]) -> int:
    """How many camera unknowns the pins can fix."""
    return sum(2 if pin.kind == "point" else 1 for pin in pins)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    """A pinhole camera with fixed principal point and pixel aspect."""

    focal: float
    rvec: np.ndarray
    position: np.ndarray
    cx: float
    cy: float
    aspect: float = 1.0  # fy / fx

    @property
    def K(self) -> np.ndarray:
        """Intrinsic matrix."""
        return np.array(
            [[self.focal, 0.0, self.cx], [0.0, self.focal * self.aspect, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def R(self) -> np.ndarray:
        """World-to-camera rotation."""
        return cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64))[0]

    def params(self) -> np.ndarray:
        """Optimiser vector: log focal, rotation, position."""
        return np.concatenate(([math.log(self.focal)], self.rvec, self.position))

    def with_params(self, values: np.ndarray) -> "Camera":
        """A copy with new optimiser values."""
        return Camera(
            focal=math.exp(float(values[0])),
            rvec=np.array(values[1:4], dtype=np.float64),
            position=np.array(values[4:7], dtype=np.float64),
            cx=self.cx,
            cy=self.cy,
            aspect=self.aspect,
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON form, sent to the browser and back as a warm start."""
        return {
            "focal": float(self.focal),
            "rvec": [float(v) for v in self.rvec],
            "position": [float(v) for v in self.position],
            "cx": float(self.cx),
            "cy": float(self.cy),
            "aspect": float(self.aspect),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["Camera"]:
        """Rebuild a camera; anything malformed returns ``None``."""
        if not isinstance(data, dict):
            return None
        try:
            camera = cls(
                focal=float(data["focal"]),
                rvec=np.array([float(v) for v in data["rvec"]], dtype=np.float64),
                position=np.array([float(v) for v in data["position"]], dtype=np.float64),
                cx=float(data["cx"]),
                cy=float(data["cy"]),
                aspect=float(data.get("aspect", 1.0)),
            )
        except (KeyError, TypeError, ValueError):
            return None
        valid = (
            camera.focal > 0
            and camera.rvec.shape == (3,)
            and camera.position.shape == (3,)
            and np.all(np.isfinite(camera.rvec))
            and np.all(np.isfinite(camera.position))
        )
        return camera if valid else None

    @classmethod
    def from_calibration(cls, calibration: Optional[CalibrationResult]) -> Optional["Camera"]:
        """Read the camera out of a full calibration result."""
        if calibration is None or not calibration.has_3d_calibration:
            return None
        K, R, C = calibration.camera_matrix, calibration.rotation_matrix, calibration.position
        if K is None or R is None or C is None:
            return None
        K = np.asarray(K, dtype=np.float64)
        return cls(
            focal=float(K[0, 0]),
            rvec=cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel(),
            position=np.asarray(C, dtype=np.float64).reshape(3).copy(),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            aspect=float(K[1, 1] / K[0, 0]),
        )

    def field_to_image(self) -> np.ndarray:
        """Homography from field metres to image pixels, normalised."""
        R, K = self.R, self.K
        camera_h = K @ np.column_stack((R[:, 0], R[:, 1], -(R @ self.position)))
        to_world = np.array(
            [[1.0, 0.0, -FIELD_LENGTH / 2.0], [0.0, 1.0, -FIELD_WIDTH / 2.0], [0.0, 0.0, 1.0]]
        )
        H = camera_h @ to_world
        return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def _field_to_world(points: np.ndarray) -> np.ndarray:
    """Field metres (N x 2) to ground-plane world metres (N x 3)."""
    world = np.zeros((len(points), 3), dtype=np.float64)
    world[:, 0] = points[:, 0] - FIELD_LENGTH / 2.0
    world[:, 1] = points[:, 1] - FIELD_WIDTH / 2.0
    return world


def _project_h(values: np.ndarray, template: Camera, world: np.ndarray) -> np.ndarray:
    """Homogeneous image points (N x 3) for optimiser ``values``."""
    focal = math.exp(float(values[0]))
    R = cv2.Rodrigues(np.asarray(values[1:4], dtype=np.float64))[0]
    cam = (world - values[4:7]) @ R.T
    out = np.empty_like(cam)
    out[:, 0] = focal * cam[:, 0] + template.cx * cam[:, 2]
    out[:, 1] = focal * template.aspect * cam[:, 1] + template.cy * cam[:, 2]
    out[:, 2] = cam[:, 2]
    return out


@dataclass
class _Problem:
    """Pin geometry prepared once for fast residual evaluation."""

    point_world: np.ndarray
    point_px: np.ndarray
    line_a: np.ndarray
    line_b: np.ndarray
    line_px: np.ndarray
    anchor_world: np.ndarray
    anchor_px: np.ndarray
    anchor_weight: float = 0.0


def _build_problem(
    pins: Sequence[Pin],
    anchors: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    anchor_weight: float = 0.0,
) -> _Problem:
    """Sort pins into arrays."""
    point_field, point_px, line_a, line_b, line_px = [], [], [], [], []
    for pin in pins:
        if pin.kind == "point":
            mark = _LANDMARK_BY_ID[pin.ref]
            point_field.append((mark.x, mark.y))
            point_px.append((pin.x, pin.y))
        else:
            line = _LINE_BY_ID[pin.ref]
            line_a.append(line.a)
            line_b.append(line.b)
            line_px.append((pin.x, pin.y))
    empty = np.zeros((0, 2))
    anchor_world, anchor_px = (
        (np.zeros((0, 3)), empty) if anchors is None else anchors
    )
    return _Problem(
        point_world=_field_to_world(np.array(point_field).reshape(-1, 2)),
        point_px=np.array(point_px, dtype=np.float64).reshape(-1, 2),
        line_a=_field_to_world(np.array(line_a).reshape(-1, 2)),
        line_b=_field_to_world(np.array(line_b).reshape(-1, 2)),
        line_px=np.array(line_px, dtype=np.float64).reshape(-1, 2),
        anchor_world=anchor_world,
        anchor_px=anchor_px,
        anchor_weight=anchor_weight,
    )


def _pin_residuals(values: np.ndarray, template: Camera, problem: _Problem) -> np.ndarray:
    """Pixel misses of the point pins (x, y pairs) then the line pins."""
    parts: List[np.ndarray] = []
    if len(problem.point_world):
        h = _project_h(values, template, problem.point_world)
        depth = np.where(np.abs(h[:, 2]) < 1e-6, 1e-6, h[:, 2])
        parts.append(((h[:, :2] / depth[:, None]) - problem.point_px).ravel())
    if len(problem.line_a):
        # Homogeneous lines stay valid when one end is behind the camera.
        lines = np.cross(
            _project_h(values, template, problem.line_a),
            _project_h(values, template, problem.line_b),
        )
        norm = np.hypot(lines[:, 0], lines[:, 1])
        norm = np.where(norm < 1e-9, 1e-9, norm)
        parts.append(
            (lines[:, 0] * problem.line_px[:, 0] + lines[:, 1] * problem.line_px[:, 1] + lines[:, 2])
            / norm
        )
    return np.concatenate(parts) if parts else np.zeros(0)


def _all_residuals(values: np.ndarray, template: Camera, problem: _Problem) -> np.ndarray:
    """Pin misses plus the weighted pull toward the anchors."""
    misses = _pin_residuals(values, template, problem)
    if not len(problem.anchor_world):
        return misses
    h = _project_h(values, template, problem.anchor_world)
    depth = np.where(np.abs(h[:, 2]) < 1e-6, 1e-6, h[:, 2])
    anchors = ((h[:, :2] / depth[:, None]) - problem.anchor_px).ravel() * problem.anchor_weight
    return np.concatenate((misses, anchors))


_PARAM_SCALE = np.array([0.05, 0.01, 0.01, 0.01, 2.0, 2.0, 2.0])


def _refine(template: Camera, problem: _Problem, max_nfev: int = 120) -> Optional[Tuple[Camera, float]]:
    """Levenberg-Marquardt from ``template``.  Returns ``(camera, cost)``."""
    start = template.params()
    size = len(_all_residuals(start, template, problem))
    if size < len(start):
        return None
    try:
        solved = least_squares(
            _all_residuals,
            start,
            args=(template, problem),
            method="lm",
            x_scale=_PARAM_SCALE,
            max_nfev=max_nfev,
            xtol=1e-7,
            ftol=1e-7,
        )
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        return None
    if not np.all(np.isfinite(solved.x)) or not -2.0 < solved.x[0] - math.log(template.focal) < 2.0:
        return None
    return template.with_params(solved.x), float(solved.cost)


def camera_problem(camera: Camera, width: int, pins: Sequence[Pin] = ()) -> Optional[str]:
    """Why a camera can't be right, in plain words, or ``None`` when it could be."""
    height_m = -float(camera.position[2])
    if height_m < 2.0:
        return "That would put the camera below the pitch. Check the points."
    if height_m > 150.0:
        return "That would put the camera too high above the pitch. Check the points."
    if not 0.25 * width <= camera.focal <= 15.0 * width:
        return "No real lens fits these points. Check the points."
    R = camera.R
    # Image "down" must follow world "down", give or take a little roll.
    if R[1, 2] <= 0.0 or abs(math.degrees(math.atan2(R[0, 2], R[1, 2]))) > 25.0:
        return "That would tilt the camera sideways. Check the points."
    point_pins = [pin for pin in pins if pin.kind == "point"]
    if point_pins:
        field = np.array([[_LANDMARK_BY_ID[p.ref].x, _LANDMARK_BY_ID[p.ref].y] for p in point_pins])
        depth = (R @ (_field_to_world(field) - camera.position).T).T[:, 2]
        if np.any(depth <= 0.0):
            return "Some points would be behind the camera. Check the points."
    return None


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------


def _edit_anchors(base: Camera, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ground points in view, and where the base camera shows them."""
    grid = [(item.x, item.y) for item in LANDMARKS]
    grid += [(float(x), float(y)) for x in range(0, 106, 15) for y in range(0, 69, 17)]
    world = _field_to_world(np.array(grid, dtype=np.float64))
    h = _project_h(base.params(), base, world)
    keep = h[:, 2] > 1e-6
    world, h = world[keep], h[keep]
    px = h[:, :2] / h[:, 2:3]
    inside = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    if inside.sum() < 4:
        # Nothing of the pitch in view: fall back to everything in front.
        return world, px
    return world[inside], px[inside]


def _solve_edit(pins: Sequence[Pin], base: Camera, width: int, height: int) -> Optional[Camera]:
    """Move the base camera as little as possible to meet the pins."""
    weight = (
        _ANCHOR_WEIGHT_MANY_PINS
        if constraint_count(pins) >= CONSTRAINTS_NEEDED + 1
        else _ANCHOR_WEIGHT_FEW_PINS
    )
    problem = _build_problem(pins, _edit_anchors(base, width, height), weight)
    solved = _refine(base, problem)
    return solved[0] if solved else None


def _pnp_starts(
    pins: Sequence[Pin], width: int, height: int
) -> List[Tuple[float, Camera]]:
    """Plane-based pose for a range of focal lengths, best first."""
    points = [pin for pin in pins if pin.kind == "point"]
    world = _field_to_world(
        np.array([(_LANDMARK_BY_ID[p.ref].x, _LANDMARK_BY_ID[p.ref].y) for p in points])
    )
    image = np.array([(p.x, p.y) for p in points], dtype=np.float64)
    starts: List[Tuple[float, Camera]] = []
    for focal in np.geomspace(0.35 * width, 8.0 * width, 36):
        K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
        try:
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                world, image, K, None, flags=cv2.SOLVEPNP_IPPE
            )
        except cv2.error:
            continue
        for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R = cv2.Rodrigues(rvec)[0]
            position = -R.T @ tvec.ravel()
            if position[2] > -1.0:
                continue
            error = float(np.ravel(errors)[index]) if errors is not None else 0.0
            starts.append(
                (
                    error,
                    Camera(float(focal), rvec.ravel().astype(np.float64), position, width / 2.0, height / 2.0),
                )
            )
    starts.sort(key=lambda row: row[0])
    return starts[:6]


def _look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation vector of a level camera at ``position`` looking at ``target``."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    if down[2] < 0.0:
        right, down = -right, -down
    R = np.vstack((right, down, forward))
    if np.linalg.det(R) < 0.0:
        R[0] = -R[0]
    return cv2.Rodrigues(R)[0].ravel()


def _viewpoint_starts(pins: Sequence[Pin], width: int, height: int) -> List[Camera]:
    """Typical broadcast viewpoints around the pins, for when points are scarce."""
    field: List[Tuple[float, float]] = []
    for pin in pins:
        if pin.kind == "point":
            mark = _LANDMARK_BY_ID[pin.ref]
            field.append((mark.x, mark.y))
        else:
            line = _LINE_BY_ID[pin.ref]
            field.append(((line.a[0] + line.b[0]) / 2.0, (line.a[1] + line.b[1]) / 2.0))
    centre = np.mean(field, axis=0) - np.array([FIELD_LENGTH / 2.0, FIELD_WIDTH / 2.0])
    target = np.array([centre[0], centre[1], 0.0])
    starts: List[Camera] = []
    for focal_scale in (0.9, 1.6, 2.8):
        for offset in (-35.0, -12.0, 12.0, 35.0):
            for camera_height in (15.0, 30.0):
                position = np.array([centre[0] + offset, 75.0, -camera_height])
                starts.append(
                    Camera(
                        focal_scale * width,
                        _look_at(position, target),
                        position,
                        width / 2.0,
                        height / 2.0,
                    )
                )
    return starts


def _solve_scratch(
    pins: Sequence[Pin],
    width: int,
    height: int,
    start: Optional[Camera] = None,
) -> Optional[Camera]:
    """Find a camera from pins alone."""
    problem = _build_problem(pins)
    tolerance = _MISS_TOLERANCE * width

    def _acceptable(camera: Camera) -> bool:
        return camera_problem(camera, width, pins) is None

    # A drag barely moves the answer, so the last camera is the best start.
    if start is not None:
        solved = _refine(start, problem, 60)
        if solved and _acceptable(solved[0]) and _mean_miss(solved[0], problem) <= tolerance:
            return solved[0]

    candidates: List[Camera] = []
    if sum(pin.kind == "point" for pin in pins) >= 4:
        candidates = [camera for _, camera in _pnp_starts(pins, width, height)]
    if not candidates:
        candidates = _viewpoint_starts(pins, width, height)

    best: Optional[Camera] = None
    best_miss = math.inf
    for candidate in candidates:
        solved = _refine(candidate, problem, 80)
        if not solved or not _acceptable(solved[0]):
            continue
        miss = _mean_miss(solved[0], problem)
        if miss < best_miss:
            best, best_miss = solved[0], miss
    return best


def _mean_miss(camera: Camera, problem: _Problem) -> float:
    """Average pixel miss over the pins."""
    return float(np.mean(_pin_misses(camera, problem))) if _pin_count(problem) else 0.0


def _pin_count(problem: _Problem) -> int:
    return len(problem.point_world) + len(problem.line_a)


def _pin_misses(camera: Camera, problem: _Problem) -> np.ndarray:
    """One pixel miss per pin, point pins first."""
    raw = _pin_residuals(camera.params(), camera, problem)
    count = len(problem.point_world)
    points = np.hypot(raw[: 2 * count : 2], raw[1 : 2 * count : 2])
    return np.concatenate((points, np.abs(raw[2 * count :])))


def estimate_line_error(camera: Camera, pins: Sequence[Pin], width: int, height: int) -> float:
    """How far a line position could be off, in metres, from hand-placement error.

    Propagates a 2 px placement error through the camera to the field position of
    ground points in view.  Bunched pins leave the camera loose, which shows up here
    as a large number, so it also serves as the check for points too close together.
    """
    problem = _build_problem(pins)
    values = camera.params()
    steps = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2])

    jac = np.empty((len(_pin_residuals(values, camera, problem)), 7))
    for i, step in enumerate(steps):
        delta = np.zeros(7)
        delta[i] = step
        jac[:, i] = (
            _pin_residuals(values + delta, camera, problem)
            - _pin_residuals(values - delta, camera, problem)
        ) / (2.0 * step)
    # Compare parameters on a common scale so the inverse is well behaved.
    scaled = jac * _PARAM_SCALE
    try:
        cov = np.linalg.pinv(scaled.T @ scaled + 1e-9 * np.eye(7)) * _PLACEMENT_SIGMA_PX**2
    except np.linalg.LinAlgError:
        return math.inf

    pixels = _ground_probes(camera, width, height)
    if not len(pixels):
        return math.inf
    grads = np.empty((len(pixels), 7))
    for i, step in enumerate(steps):
        delta = np.zeros(7)
        delta[i] = step
        grads[:, i] = (
            _pixels_to_field_x(values + delta, camera, pixels)
            - _pixels_to_field_x(values - delta, camera, pixels)
        ) / (2.0 * step)
    grads = grads * _PARAM_SCALE
    variance = np.einsum("ij,jk,ik->i", grads, cov, grads)
    return float(np.median(np.sqrt(np.maximum(variance, 0.0))))


def _ground_probes(camera: Camera, width: int, height: int) -> np.ndarray:
    """Image pixels in view that show the pitch."""
    columns = np.linspace(0.1 * width, 0.9 * width, 6)
    rows = np.linspace(0.3 * height, 0.95 * height, 5)
    pixels = np.array([(u, v) for u in columns for v in rows], dtype=np.float64)
    field_x = _pixels_to_field_x(camera.params(), camera, pixels)
    field_y = _pixels_to_field_y(camera.params(), camera, pixels)
    keep = (
        np.isfinite(field_x)
        & np.isfinite(field_y)
        & (field_x > -2.0)
        & (field_x < FIELD_LENGTH + 2.0)
        & (field_y > -2.0)
        & (field_y < FIELD_WIDTH + 2.0)
    )
    return pixels[keep]


def _pixel_rays(values: np.ndarray, template: Camera, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ground intersection in world metres for each pixel, plus a validity mask."""
    focal = math.exp(float(values[0]))
    R = cv2.Rodrigues(np.asarray(values[1:4], dtype=np.float64))[0]
    rays = np.column_stack(
        (
            (pixels[:, 0] - template.cx) / focal,
            (pixels[:, 1] - template.cy) / (focal * template.aspect),
            np.ones(len(pixels)),
        )
    )
    world = rays @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = -values[6] / world[:, 2]
    valid = np.isfinite(scale) & (scale > 0.0)
    return values[4:6] + scale[:, None] * world[:, :2], valid


def _pixels_to_field_x(values: np.ndarray, template: Camera, pixels: np.ndarray) -> np.ndarray:
    ground, valid = _pixel_rays(values, template, pixels)
    return np.where(valid, ground[:, 0] + FIELD_LENGTH / 2.0, np.nan)


def _pixels_to_field_y(values: np.ndarray, template: Camera, pixels: np.ndarray) -> np.ndarray:
    ground, valid = _pixel_rays(values, template, pixels)
    return np.where(valid, ground[:, 1] + FIELD_WIDTH / 2.0, np.nan)


@dataclass
class Fit:
    """How much to trust a calibration, in words the screen can show."""

    tone: str  # "ok" | "warn" | "bad" | "idle"
    label: str
    message: str
    can_continue: bool
    miss_px: Optional[float] = None
    line_error_m: Optional[float] = None
    worst: Optional[int] = None  # index of the most doubtful pin
    needed: int = 0  # how many more point pins are needed

    def to_dict(self) -> Dict[str, Any]:
        """JSON form."""
        finite_error = self.line_error_m is not None and math.isfinite(self.line_error_m)
        return {
            "tone": self.tone,
            "label": self.label,
            "message": self.message,
            "can_continue": self.can_continue,
            "miss_px": None if self.miss_px is None else round(self.miss_px, 2),
            "line_error_m": round(self.line_error_m, 3) if finite_error else None,
            "worst": self.worst,
            "needed": self.needed,
        }


@dataclass
class Solution:
    """Result of one solve."""

    camera: Optional[Camera]
    fit: Fit
    mode: str  # "edit" | "hand"


def _worst_pin(
    pins: Sequence[Pin],
    camera: Camera,
    width: int,
    height: int,
    base: Optional[Camera],
) -> int:
    """The pin whose removal helps the fit most."""
    best_index, best_cost = 0, math.inf
    for index in range(len(pins)):
        rest = [pin for i, pin in enumerate(pins) if i != index]
        if constraint_count(rest) < CONSTRAINTS_NEEDED:
            continue
        problem = _build_problem(rest)
        solved = _refine(camera, problem, 30)
        if solved and solved[1] < best_cost:
            best_index, best_cost = index, solved[1]
    if math.isinf(best_cost):
        # Too few pins to leave one out: name the largest miss.
        ordered = sorted(
            range(len(pins)), key=lambda i: (pins[i].kind != "point", i)
        )
        misses = _pin_misses(camera, _build_problem(pins))
        return ordered[int(np.argmax(misses))]
    return best_index


def _side_warning(camera: Camera, width: int, height: int, attacking_side: Optional[str]) -> Optional[str]:
    """A warning when the fitted end of the pitch contradicts the attack direction."""
    if attacking_side not in ("left", "right"):
        return None
    row = height * 0.65
    left, centre, right = _pixels_to_field_x(
        camera.params(),
        camera,
        np.array([[width * 0.35, row], [width / 2.0, row], [width * 0.65, row]]),
    )
    if not (np.isfinite(left) and np.isfinite(centre) and np.isfinite(right)) or left == right:
        return None
    # A reverse-angle camera runs pitch metres the other way on screen, so measure it.
    screen_right_sign = 1.0 if right > left else -1.0
    attack_sign = screen_right_sign if attacking_side == "right" else -screen_right_sign
    if (centre - FIELD_LENGTH / 2.0) * attack_sign < -15.0:
        return f"This shows the far end from the attack, which goes to the {attacking_side}. Check the end of the pitch."
    return None


def _assess(
    pins: Sequence[Pin],
    camera: Camera,
    width: int,
    height: int,
    mode: str,
    base: Optional[Camera],
    attacking_side: Optional[str],
) -> Fit:
    """Turn a solved camera into a status for the screen."""
    problem_text = camera_problem(camera, width, pins)
    if problem_text:
        return Fit("bad", "CHECK THE POINTS", problem_text, False)

    problem = _build_problem(pins)
    miss = _mean_miss(camera, problem)
    tolerance = max(_MISS_TOLERANCE * width, 3.0)
    count = len(pins)
    ordered = sorted(range(count), key=lambda i: (pins[i].kind != "point", i))
    plural = "point" if count == 1 else "points"

    if miss > tolerance:
        worst = _worst_pin([pins[i] for i in ordered], camera, width, height, base)
        worst_index = ordered[worst]
        name = pin_name(pins[worst_index])
        return Fit(
            "bad",
            "POINTS DISAGREE",
            f"Check: {name}.",
            False,
            miss_px=miss,
            worst=worst_index,
        )

    line_error: Optional[float] = None
    if constraint_count(pins) >= CONSTRAINTS_NEEDED:
        line_error = estimate_line_error(camera, pins, width, height)

    warning = _side_warning(camera, width, height, attacking_side)

    if mode == "hand":
        if line_error is None or line_error > LINE_ERROR_BAD_M:
            return Fit(
                "bad",
                "TOO CLOSE TOGETHER",
                "These points leave the camera too uncertain. Add one farther away, across the pitch.",
                False,
                miss_px=miss,
                line_error_m=line_error,
            )
        if warning:
            return Fit("warn", "CHECK THE END", warning, True, miss_px=miss, line_error_m=line_error)
        if constraint_count(pins) <= CONSTRAINTS_NEEDED + 1:
            return Fit(
                "warn",
                "CHECK IT",
                "This is the minimum, so nothing confirms it yet. Add one more to double-check.",
                True,
                miss_px=miss,
                line_error_m=line_error,
            )
        if line_error > LINE_ERROR_WEAK_M:
            return Fit(
                "warn",
                "WEAK FIT",
                f"The line could be about {line_error:.2f} m off. Add points farther apart.",
                True,
                miss_px=miss,
                line_error_m=line_error,
            )
        return Fit(
            "ok",
            "GOOD FIT",
            f"Line within about {line_error:.2f} m. Average miss {miss:.1f} px.",
            True,
            miss_px=miss,
            line_error_m=line_error,
        )

    if warning:
        return Fit("warn", "CHECK THE END", warning, True, miss_px=miss, line_error_m=line_error)
    return Fit(
        "ok",
        "ADJUSTED",
        f"{count} {plural} set by you. Average miss {miss:.1f} px.",
        True,
        miss_px=miss,
        line_error_m=line_error,
    )


def solve(
    pins: Sequence[Pin],
    width: int,
    height: int,
    base: Optional[Camera] = None,
    start: Optional[Camera] = None,
    attacking_side: Optional[str] = None,
) -> Solution:
    """Solve a camera from pins.

    Args:
        pins: The placements, in image pixels.
        width: Image width.
        height: Image height.
        base: An automatic camera to edit.  ``None`` solves from scratch.
        start: The previous answer, to keep a drag fast (from scratch only).
        attacking_side: Used only to warn about the wrong end of the pitch.

    Returns:
        The camera (``None`` when there isn't one yet) and its status.
    """
    if base is not None:
        if not pins:
            return Solution(
                base,
                Fit("idle", "NO CHANGES", "Automatic result.", True),
                "edit",
            )
        camera = _solve_edit(pins, base, width, height)
        if camera is None:
            return Solution(
                base,
                Fit("bad", "CHECK THE POINTS", "No camera fits these points.", False),
                "edit",
            )
        return Solution(
            camera,
            _assess(pins, camera, width, height, "edit", base, attacking_side),
            "edit",
        )

    missing = CONSTRAINTS_NEEDED - constraint_count(pins)
    if missing > 0:
        needed = math.ceil(missing / 2.0)
        return Solution(
            None,
            Fit(
                "warn",
                f"PLACE {needed} MORE",
                "Choose points far apart. A point counts twice as much as a spot on a line.",
                False,
                needed=needed,
            ),
            "hand",
        )

    camera = _solve_scratch(pins, width, height, start)
    if camera is None:
        return Solution(
            None,
            Fit(
                "bad",
                "CHECK THE POINTS",
                "No believable camera fits these points. Check that each one is on the marking it names.",
                False,
            ),
            "hand",
        )
    return Solution(
        camera,
        _assess(pins, camera, width, height, "hand", None, attacking_side),
        "hand",
    )


def to_calibration_result(
    camera: Camera,
    pins: Sequence[Pin],
    width: int,
    height: int,
    source: str,
    fit: Optional[Fit] = None,
) -> CalibrationResult:
    """Build the same result shape PnLCalib produces.

    Args:
        camera: The solved camera.
        pins: The placements behind it.
        width: Image width.
        height: Image height.
        source: ``"edited"`` for an adjusted automatic result, ``"hand"`` for placed.
        fit: The fit status, recorded for the report.
    """
    K, R, position = camera.K, camera.R, np.asarray(camera.position, dtype=np.float64)
    projection = K @ (R @ np.column_stack((np.eye(3), -position)))
    field_to_image = camera.field_to_image()

    # Hand pins take the place of PnLCalib's keypoints on the debug overlay.
    keypoints = {
        f"{pin.ref}" if pin.kind == "point" else f"{pin.ref}#{index}": {"x": pin.x, "y": pin.y}
        for index, pin in enumerate(pins)
    }
    miss = fit.miss_px if fit is not None and fit.miss_px is not None else 0.0
    result = CalibrationResult(
        homography=np.linalg.inv(field_to_image),
        homography_inv=field_to_image,
        keypoints=keypoints,
        keypoints_raw=keypoints,
        image_size=(height, width),
        model_size=(height, width),
        quality_score=max(0.0, min(1.0, 1.0 - miss / 20.0)),
        method="manual",
        camera_matrix=K,
        projection_matrix=projection,
        rotation_matrix=R,
        position=position.copy(),
    )
    result.debug_info = {
        "source": source,
        "pins": [{"kind": p.kind, "ref": p.ref, "x": p.x, "y": p.y} for p in pins],
        "reprojection_error": miss,
    }
    return result
