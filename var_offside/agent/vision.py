"""The images the vision checks send, and the detector re-run for a missed player.

Everything here draws on a copy, so the session's frame is never altered.
"""

from __future__ import annotations

import re
import zlib
from statistics import median
from typing import Collection, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from var_offside.agent.scene import BBox, Scene, role_of
from var_offside.calibration import pitch_solver as ps
from var_offside.calibration.base import CalibrationResult
from var_offside.detection import PlayerDetection

GRID_COLUMNS = 8
GRID_ROWS = 6
CROP_HEIGHT = 170
CROP_PAD = 0.12
SHEET_COLUMNS = 10
RERUN_UPSCALE = 2.0
RERUN_CONFIDENCE = 0.15
RERUN_REACH = 0.5  # How far past the named cell to look, in cells.
SAME_PERSON_OVERLAP = 0.5
MIN_HEIGHT_RATIO = 0.35

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 230, 255)
GRID_COLOR = (240, 240, 200)
ORANGE = (0, 140, 255)
CYAN = (255, 230, 0)


def _label(canvas: np.ndarray, text: str, x: int, y: int, scale: float, color=WHITE) -> None:
    """Text with a dark backing, so it reads on any part of a frame."""
    thickness = max(1, int(round(scale * 2)))
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(np.clip(x, 0, max(0, canvas.shape[1] - width - 4)))
    y = int(np.clip(y, height + 4, canvas.shape[0] - 2))
    cv2.rectangle(canvas, (x, y - height - 4), (x + width + 4, y + 2), BLACK, -1)
    cv2.putText(canvas, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _scale_for(image: np.ndarray) -> float:
    return max(0.45, image.shape[1] / 2400.0)


def number_frame(image: np.ndarray, detections: Sequence[PlayerDetection], grid: bool = False) -> np.ndarray:
    """The frame with each person boxed and numbered, and optionally the lettered grid."""
    canvas = image.copy()
    scale = _scale_for(image)
    if grid:
        _draw_grid(canvas, scale)
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), YELLOW, max(1, int(round(scale * 2))))
        _label(canvas, str(det.player_id), x1, y1 - 2, scale)
    return canvas


def box_letters(scene: Scene) -> Dict[int, str]:
    """The letter each team gets. It varies by frame, so a habit of picking "A" can't favour Team 1."""
    boxes = sorted(tuple(int(value) for value in det.bbox) for det in scene.detections)
    if zlib.crc32(repr(boxes).encode("ascii")) % 2:
        return {1: "B", 2: "A"}
    return {1: "A", 2: "B"}


def play_frame(scene: Scene, unsorted: Collection[int] = ()) -> np.ndarray:
    """The frame with teams in two colours and letters, the ball, and which side is left or right.

    People in ``unsorted``, such as a likely keeper, get a white box and no letter, so a keeper
    grouped with the wrong team can't decide which team is attacking.
    """
    canvas = scene.image.copy()
    scale = _scale_for(canvas)
    thick = max(2, int(round(scale * 3)))
    letters = box_letters(scene)
    colours = {"A": ORANGE, "B": CYAN}
    for det in scene.detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        sorted_player = role_of(det) == "player" and det.player_id not in unsorted
        letter = letters.get(det.team_id) if sorted_player else None
        colour = colours[letter] if letter else WHITE
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thick)
        _label(canvas, f"{letter} {det.player_id}" if letter else str(det.player_id), x1, y1 - 2, scale, colour)
    if scene.ball is not None:
        x, y = (int(round(v)) for v in scene.ball.center)
        cv2.circle(canvas, (x, y), max(10, int(round(16 * scale))), WHITE, thick, cv2.LINE_AA)
        _label(canvas, "BALL", x + 12, y - 12, scale)
    return np.vstack([_play_key(canvas.shape[1], scale), canvas])


def _play_key(width: int, scale: float) -> np.ndarray:
    """A strip above the frame, so the key never covers a player or the pitch."""
    size = scale * 1.4
    key = np.full((int(60 * size) + 12, width, 3), 15, dtype=np.uint8)
    baseline = key.shape[0] - 10
    _label(key, "<- LEFT", 8, baseline, size)
    _label(key, "A = orange", width // 4, baseline, size, ORANGE)
    _label(key, "B = cyan", width // 2 + width // 12, baseline, size, CYAN)
    _label(key, "RIGHT ->", width, baseline, size)
    return key


def _draw_grid(canvas: np.ndarray, scale: float) -> None:
    height, width = canvas.shape[:2]
    for column in range(1, GRID_COLUMNS):
        x = column * width // GRID_COLUMNS
        cv2.line(canvas, (x, 0), (x, height), GRID_COLOR, 1, cv2.LINE_AA)
    for row in range(1, GRID_ROWS):
        y = row * height // GRID_ROWS
        cv2.line(canvas, (0, y), (width, y), GRID_COLOR, 1, cv2.LINE_AA)
    for column in range(GRID_COLUMNS):
        for row in range(GRID_ROWS):
            text = f"{chr(ord('A') + column)}{row + 1}"
            _label(
                canvas,
                text,
                column * width // GRID_COLUMNS + 3,
                row * height // GRID_ROWS + int(22 * scale) + 6,
                scale * 0.7,
                GRID_COLOR,
            )


def parse_cell(cell: str) -> Optional[Tuple[int, int]]:
    """A cell name as (column, row), or None if it isn't on the grid."""
    match = re.fullmatch(r"\s*([A-Za-z])\s*(\d+)\s*", cell or "")
    if not match:
        return None
    column, row = ord(match.group(1).upper()) - ord("A"), int(match.group(2)) - 1
    return (column, row) if 0 <= column < GRID_COLUMNS and 0 <= row < GRID_ROWS else None


def cell_box(cell: str, width: int, height: int) -> Optional[BBox]:
    """The image rectangle of a grid cell."""
    parsed = parse_cell(cell)
    if parsed is None:
        return None
    column, row = parsed
    return (
        column * width / GRID_COLUMNS,
        row * height / GRID_ROWS,
        (column + 1) * width / GRID_COLUMNS,
        (row + 1) * height / GRID_ROWS,
    )


def cell_of(x: float, y: float, width: int, height: int) -> str:
    """The grid cell holding an image point."""
    column = int(np.clip(x * GRID_COLUMNS / width, 0, GRID_COLUMNS - 1))
    row = int(np.clip(y * GRID_ROWS / height, 0, GRID_ROWS - 1))
    return f"{chr(ord('A') + column)}{row + 1}"


def crop_player(image: np.ndarray, det: PlayerDetection) -> np.ndarray:
    """A numbered crop of one person, enlarged to a common height. No line is drawn."""
    x1, y1, x2, y2 = det.bbox
    pad_x, pad_y = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    height, width = image.shape[:2]
    left, right = int(max(0, x1 - pad_x)), int(min(width, x2 + pad_x))
    top, bottom = int(max(0, y1 - pad_y)), int(min(height, y2 + pad_y))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        crop = np.zeros((CROP_HEIGHT, CROP_HEIGHT // 2, 3), dtype=np.uint8)
    scale = CROP_HEIGHT / crop.shape[0]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    crop = crop.copy()
    _label(crop, str(det.player_id), 2, 20, 0.55)
    return crop


def _rows_of(crops: Sequence[np.ndarray]) -> List[np.ndarray]:
    gap = 6
    rows = []
    for start in range(0, len(crops), SHEET_COLUMNS):
        row: List[np.ndarray] = []
        for crop in crops[start : start + SHEET_COLUMNS]:
            row += [crop, np.full((CROP_HEIGHT, gap, 3), 30, dtype=np.uint8)]
        rows.append(np.hstack(row))
    return rows


def _stack(rows: Sequence[np.ndarray]) -> np.ndarray:
    width = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        fill = np.full((row.shape[0], width - row.shape[1], 3), 30, dtype=np.uint8)
        padded.append(np.hstack([row, fill]))
    return np.vstack(padded)


def crop_sheet(image: np.ndarray, detections: Sequence[PlayerDetection]) -> Optional[np.ndarray]:
    """Numbered crops of these people, or None when there are none."""
    crops = [crop_player(image, det) for det in detections]
    return _stack(_rows_of(crops)) if crops else None


def team_contact_sheet(scene: Scene) -> Optional[np.ndarray]:
    """Crops of outfield players in two labelled groups: group_1 is Team 1, group_2 is Team 2."""
    rows: List[np.ndarray] = []
    for team in (1, 2):
        people = [det for det in scene.detections if det.team_id == team and role_of(det) == "player"]
        if not people:
            continue
        caption = np.full((30, 300, 3), 15, dtype=np.uint8)
        _label(caption, f"group_{team}", 4, 24, 0.7)
        rows.append(caption)
        rows += _rows_of([crop_player(scene.image, det) for det in people])
    return _stack(rows) if rows else None


def keeper_candidates(scene: Scene, deepest: Sequence[int], outliers: Sequence[int]) -> List[PlayerDetection]:
    """The people worth showing the keeper check, without officials."""
    wanted = set(deepest) | set(outliers)
    wanted |= {det.player_id for det in scene.detections if det.detector_class == "goalkeeper"}
    return [det for det in scene.detections if det.player_id in wanted and role_of(det) != "referee"]


def _project_line(calibration: CalibrationResult, a: Sequence[float], b: Sequence[float]) -> List[Tuple[int, int]]:
    points = []
    for t in np.linspace(0.0, 1.0, 9):
        pixel = calibration.project_to_image((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        if pixel is not None:
            points.append((int(round(pixel[0])), int(round(pixel[1]))))
    return points


def pitch_overlay(image: np.ndarray, calibration: CalibrationResult) -> Tuple[np.ndarray, List[str]]:
    """Named pitch lines drawn over the frame, and the names shown. Knows nothing about players.

    This takes only the image and camera, so teams, the line and the verdict can't reach it.
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = _scale_for(image)
    names = []
    for line in ps.pitch_model()["lines"]:
        points = _project_line(calibration, line["a"], line["b"])
        visible = [(x, y) for x, y in points if 0 <= x < width and 0 <= y < height]
        if len(visible) < 2:
            continue
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, YELLOW, max(1, int(round(scale * 2))), cv2.LINE_AA)
        x, y = visible[len(visible) // 2]
        _label(canvas, line["id"], x, y - 4, scale * 0.8, YELLOW)
        names.append(line["id"])
    return canvas, names


def _overlap_of_first(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / area if area > 0.0 else 0.0


def rerun_detector(
    detector, image: np.ndarray, detections: Sequence[PlayerDetection], cell: str
) -> Optional[BBox]:
    """Run the real detector on an enlarged crop of a grid cell, looking for a new person.

    The model only names the cell, and any box comes from the detector. Returns the box in
    frame coordinates, or None if the detector finds nobody new there.
    """
    height, width = image.shape[:2]
    box = cell_box(cell, width, height)
    if box is None:
        return None
    reach_x = (box[2] - box[0]) * RERUN_REACH
    reach_y = (box[3] - box[1]) * RERUN_REACH
    left, top = int(max(0, box[0] - reach_x)), int(max(0, box[1] - reach_y))
    right, bottom = int(min(width, box[2] + reach_x)), int(min(height, box[3] + reach_y))
    crop = cv2.resize(
        image[top:bottom, left:right], None, fx=RERUN_UPSCALE, fy=RERUN_UPSCALE, interpolation=cv2.INTER_CUBIC
    )

    previous_ball = getattr(detector, "last_ball", None)
    try:
        found = detector.detect(crop, confidence=RERUN_CONFIDENCE, filter_by_height=False, use_pose=False)
    finally:
        detector.last_ball = previous_ball

    heights = [det.bbox[3] - det.bbox[1] for det in detections]
    smallest = MIN_HEIGHT_RATIO * median(heights) if heights else 0.0
    centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    best: Optional[BBox] = None
    best_distance = float("inf")
    for det in found:
        x1, y1, x2, y2 = det.bbox
        mapped = (
            left + x1 / RERUN_UPSCALE,
            top + y1 / RERUN_UPSCALE,
            left + x2 / RERUN_UPSCALE,
            top + y2 / RERUN_UPSCALE,
        )
        if mapped[3] - mapped[1] < smallest:
            continue
        if any(_overlap_of_first(mapped, known.bbox) > SAME_PERSON_OVERLAP for known in detections):
            continue
        foot = ((mapped[0] + mapped[2]) / 2.0, mapped[3])
        distance = (foot[0] - centre[0]) ** 2 + (foot[1] - centre[1]) ** 2
        if distance < best_distance:
            best, best_distance = mapped, distance
    return best


def box_2d_to_pixels(box_2d: Sequence[int], width: int, height: int) -> BBox:
    """Convert a [ymin, xmin, ymax, xmax] box on a 0 to 1000 scale to image (x1, y1, x2, y2)."""
    ymin, xmin, ymax, xmax = box_2d
    return (xmin / 1000.0 * width, ymin / 1000.0 * height, xmax / 1000.0 * width, ymax / 1000.0 * height)


def known_ids(scene: Scene) -> Dict[int, PlayerDetection]:
    """Everyone a model answer may name."""
    return {det.player_id: det for det in scene.detections}
