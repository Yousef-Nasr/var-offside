"""
Player markers on server-drawn images, matching the photo view in ``static/players.js``.

A ring on the ground and a label above the head, with fixed team colours: Team 1 is
always red and Team 2 always blue, whatever the kits look like.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

# BGR.  The same colours as TEAM_COLORS in static/players.js.
TEAM_BGR: Dict[int, Tuple[int, int, int]] = {1: (71, 51, 255), 2: (255, 125, 46)}
TEAM_HEX: Dict[int, str] = {1: "#ff3347", 2: "#2e7dff"}
REFEREE_BGR: Tuple[int, int, int] = (160, 168, 154)
NO_TEAM_BGR: Tuple[int, int, int] = (204, 211, 201)

_LABEL_FONT = cv2.FONT_HERSHEY_DUPLEX


def marker_color(detection) -> Tuple[int, int, int]:
    """Marker colour for one detection."""
    if getattr(detection, "is_referee", False):
        return REFEREE_BGR
    return TEAM_BGR.get(detection.team_id, NO_TEAM_BGR)


def _ring_size(bbox: Sequence[float]) -> Tuple[int, int, int, int]:
    """Centre and radii of the ring under a box."""
    x1, _, x2, y2 = bbox
    rx = max((x2 - x1) * 0.62, 9.0)
    ry = max(rx * 0.3, 3.0)
    return int(round((x1 + x2) / 2.0)), int(round(y2 - ry * 0.25)), int(round(rx)), int(round(ry))


def _dashed_arc(image, center, axes, start, end, color, thickness) -> None:
    """A dashed ellipse arc, for goalkeepers."""
    step = 26
    for angle in range(int(start), int(end), step):
        cv2.ellipse(
            image, center, axes, 0, angle, min(angle + step * 0.6, end), color, thickness, cv2.LINE_AA
        )


def _draw_ring(image: np.ndarray, detection) -> None:
    """Soft pool of colour plus a ring, stronger on the near side."""
    color = marker_color(detection)
    cx, cy, rx, ry = _ring_size(detection.bbox)
    center, axes = (cx, cy), (rx, ry)

    overlay = image.copy()
    cv2.ellipse(overlay, center, axes, 0, 0, 360, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.24, image, 0.76, 0, image)

    if detection.is_goalkeeper:
        _dashed_arc(image, center, axes, 180, 360, color, 1)
        _dashed_arc(image, center, axes, 0, 180, (6, 8, 4), 4)
        _dashed_arc(image, center, axes, 0, 180, color, 2)
        return
    cv2.ellipse(image, center, axes, 0, 180, 360, color, 1, cv2.LINE_AA)
    cv2.ellipse(image, center, axes, 0, 0, 180, (6, 8, 4), 4, cv2.LINE_AA)
    cv2.ellipse(image, center, axes, 0, 0, 180, color, 2, cv2.LINE_AA)


def _draw_label(
    image: np.ndarray,
    detection,
    number: int,
    placed: List[Tuple[int, int, int, int]],
    scale: float,
) -> None:
    """A team-colour number chip, with GK or REF beside it, above the head."""
    color = marker_color(detection)
    height_px = int(round(20 * scale))
    pad = int(round(6 * scale))
    font_scale = 0.46 * scale
    thickness = 1

    parts = [(str(number), color, (255, 255, 255))]
    if detection.is_goalkeeper:
        parts.append(("GK", (10, 12, 7), (255, 255, 255)))
    if getattr(detection, "is_referee", False):
        parts.append(("REF", (10, 12, 7), (255, 255, 255)))
    sizes = [cv2.getTextSize(text, _LABEL_FONT, font_scale, thickness)[0] for text, _, _ in parts]
    total = sum(size[0] + pad * 2 for size in sizes)

    h, w = image.shape[:2]
    x1, y1, x2, _ = detection.bbox
    cx = int(round((x1 + x2) / 2.0))
    x = int(np.clip(cx - total / 2, 2, max(w - total - 2, 2)))
    y = int(max(2, y1 - height_px - int(7 * scale)))

    def overlaps() -> bool:
        return any(
            x < rx2 and x + total > rx1 and y < ry2 and y + height_px > ry1
            for rx1, ry1, rx2, ry2 in placed
        )

    tries = 2
    while overlaps() and tries:
        y = max(2, y - height_px - 2)
        tries -= 1
    if overlaps():
        return
    placed.append((x, y, x + total, y + height_px))

    tip = int(np.clip(cx, x + pad, x + total - pad))
    size = int(round(4 * scale))
    pointer = np.array(
        [[tip - size, y + height_px - 1], [tip + size, y + height_px - 1], [tip, y + height_px + size + 1]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, pointer, color, cv2.LINE_AA)

    at = x
    for (text, fill, ink), (text_w, text_h) in zip(parts, sizes):
        width = text_w + pad * 2
        cv2.rectangle(image, (at, y), (at + width - 1, y + height_px), fill, -1)
        cv2.putText(
            image,
            text,
            (at + pad, y + (height_px + text_h) // 2),
            _LABEL_FONT,
            font_scale,
            ink,
            thickness,
            cv2.LINE_AA,
        )
        at += width


def draw_player_markers(image: np.ndarray, detections: Sequence) -> np.ndarray:
    """Return a copy of ``image`` with a ring and a label for every player."""
    output = image.copy()
    scale = max(0.8, min(output.shape[1] / 1280.0, 2.0))
    order = sorted(range(len(detections)), key=lambda i: detections[i].bbox[3])
    for index in order:
        _draw_ring(output, detections[index])
    placed: List[Tuple[int, int, int, int]] = []
    for index in order:
        _draw_label(output, detections[index], index, placed, scale)
    return output
