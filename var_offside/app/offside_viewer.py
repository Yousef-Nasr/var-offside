"""
Offside Overlay Viewer - ImageViewer with an interactive, hoverable offside line.

The offside line is projected onto the image using the calibration data so it
aligns correctly with the pitch perspective:

  1. Homography  (project_to_image)  -> field (field_x, y) for y=0..68  -> image
  2. 3D projection matrix  (PnLCalib)  -> same idea, centered 4-D coords
  3. Vanishing-point only  -> perspective line toward VP
  4. Fallback  -> vertical line at image_x
"""

import numpy as np
from PyQt5.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsTextItem,
    QGraphicsPathItem,
    QToolTip,
)
from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QWheelEvent,
    QPen,
    QBrush,
    QColor,
    QFont,
    QPainterPath,
)
from PyQt5.QtCore import Qt, QPointF, QLineF

from typing import Optional, List, Tuple

from var_offside.offside.line_calculator import get_directional_ground_point
from var_offside.visualization.line_projection import project_line


# Hover distance from the line (scene pixels)
_LINE_HOVER_DISTANCE = 14

# Colors
_OFFSIDE_COLOR = QColor(220, 40, 40, 200)
_ONSIDE_COLOR = QColor(40, 180, 40, 200)
_PLAYER_OFFSIDE_COLOR = QColor(220, 40, 40, 180)
_PLAYER_ONSIDE_COLOR = QColor(40, 180, 40, 180)


# ======================================================================
# Projection helpers
# ======================================================================


def _project_offside_line(
    field_x: float,
    image_x: float,
    image_y: Optional[float],
    image_h: int,
    image_w: int,
    calibration,
) -> List[Tuple[float, float]]:
    """
    Return a list of (x, y) **image** points that trace the offside line
    across the visible pitch, properly projected with the calibration.

    Falls back to simpler methods when calibration data is limited.

    Args:
        field_x: Offside position in field coordinates (metres).
        image_x: Offside x in image pixels (defining body-part).
        image_y: Offside y in image pixels (defining body-part), or None.
        image_h: Image height.
        image_w: Image width.
        calibration: CalibrationResult (may be None).
    """

    return project_line(field_x, image_x, image_y, image_h, image_w, calibration)


# ======================================================================
# Helper: distance from point to polyline
# ======================================================================


def _point_to_polyline_distance(px, py, pts):
    """Minimum distance from (px, py) to a polyline defined by *pts*."""
    best = float("inf")
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        dx, dy = x2 - x1, y2 - y1
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            d = ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            d = ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
        if d < best:
            best = d
    return best


# ======================================================================
# Viewer widget
# ======================================================================


class OffsideOverlayViewer(QGraphicsView):
    """
    Zoomable, pannable image viewer with a **perspective-correct**
    interactive offside-line overlay.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        # View settings
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.black)
        self.setMouseTracking(True)

        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0

        # Overlay items
        self._line_path_item: Optional[QGraphicsPathItem] = None
        self._line_points: List[Tuple[float, float]] = []
        self._player_markers: List[QGraphicsEllipseItem] = []
        self._label_items: List[QGraphicsTextItem] = []

        # Data
        self._offside_result = None
        self._calibration = None
        self._image_height = 0
        self._image_width = 0

    # ------------------------------------------------------------------
    # Image display (same API as ImageViewer)
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray):
        if image is None:
            return
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1].copy()
        else:
            rgb = image
        h, w = rgb.shape[:2]
        self._image_height = h
        self._image_width = w
        if len(rgb.shape) == 3:
            q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(rgb.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_view()

    def fit_to_view(self):
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._zoom = 1.0

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15
        if event.angleDelta().y() > 0:
            if self._zoom * factor < self._max_zoom:
                self._zoom *= factor
                self.scale(factor, factor)
        else:
            if self._zoom / factor > self._min_zoom:
                self._zoom /= factor
                self.scale(1 / factor, 1 / factor)

    def get_current_image(self) -> Optional[np.ndarray]:
        pixmap = self._pixmap_item.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        w, h = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(h * w * 3)
        arr = np.array(ptr).reshape(h, w, 3)
        return arr[:, :, ::-1].copy()

    # ------------------------------------------------------------------
    # Offside overlay
    # ------------------------------------------------------------------

    def set_offside_data(self, offside_result, calibration=None):
        """
        Display the perspective-projected offside line and player markers.

        Args:
            offside_result: ``OffsideResult`` from the pipeline.
            calibration: ``CalibrationResult`` (used for proper projection).
        """
        self._clear_overlay()
        if offside_result is None:
            return

        self._offside_result = offside_result
        self._calibration = calibration
        line = offside_result.offside_line
        is_offside = offside_result.any_offside

        # --- Compute perspective-correct line points ---
        h = self._image_height or 1080
        w = self._image_width or 1920

        image_y = line.reference_point[1]

        self._line_points = _project_offside_line(
            field_x=line.field_x,
            image_x=line.image_x,
            image_y=image_y,
            image_h=h,
            image_w=w,
            calibration=calibration,
        )

        # --- Draw polyline via QPainterPath ---
        base_color = _OFFSIDE_COLOR if is_offside else _ONSIDE_COLOR
        pen = QPen(base_color, 3, Qt.SolidLine)
        pen.setCosmetic(True)

        path = QPainterPath()
        if self._line_points:
            path.moveTo(QPointF(*self._line_points[0]))
            for pt in self._line_points[1:]:
                path.lineTo(QPointF(*pt))

        self._line_path_item = QGraphicsPathItem(path)
        self._line_path_item.setPen(pen)
        self._line_path_item.setZValue(10)
        self._scene.addItem(self._line_path_item)

        # --- Player markers ---
        for status in offside_result.player_statuses:
            det = status.player_detection
            color = _PLAYER_OFFSIDE_COLOR if status.is_offside else _PLAYER_ONSIDE_COLOR

            if status.most_advanced_part is not None:
                px, py = status.most_advanced_part.x, status.most_advanced_part.y
            else:
                px, py = get_directional_ground_point(
                    det, offside_result.attacking_side
                )

            r = 8
            ellipse = QGraphicsEllipseItem(px - r, py - r, 2 * r, 2 * r)
            ellipse.setBrush(QBrush(color))
            ellipse.setPen(QPen(QColor(255, 255, 255, 200), 2))
            ellipse.setZValue(11)
            ellipse.setData(0, status.player_idx)
            self._scene.addItem(ellipse)
            self._player_markers.append(ellipse)

            tag = "AHEAD" if status.is_offside else "BEHIND"
            label = QGraphicsTextItem(f"P{status.player_idx} {tag}")
            label.setDefaultTextColor(QColor(255, 255, 255))
            label.setFont(QFont("Monospace", 7))
            label.setPos(px + r + 2, py - r)
            label.setZValue(12)
            self._scene.addItem(label)
            self._label_items.append(label)

    def clear_offside_data(self):
        self._clear_overlay()
        self._offside_result = None

    def _clear_overlay(self):
        if self._line_path_item:
            self._scene.removeItem(self._line_path_item)
            self._line_path_item = None
        self._line_points.clear()
        for item in self._player_markers:
            self._scene.removeItem(item)
        self._player_markers.clear()
        for item in self._label_items:
            self._scene.removeItem(item)
        self._label_items.clear()
        QToolTip.hideText()

    # ------------------------------------------------------------------
    # Hover / tooltip
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._offside_result is None:
            return

        scene_pos = self.mapToScene(event.pos())

        # Player markers have priority
        for marker in self._player_markers:
            if marker.contains(marker.mapFromScene(scene_pos)):
                idx = marker.data(0)
                self._show_player_tooltip(event, idx)
                return

        # Proximity to the projected polyline
        if self._line_points:
            dist = _point_to_polyline_distance(
                scene_pos.x(),
                scene_pos.y(),
                self._line_points,
            )
            if dist < _LINE_HOVER_DISTANCE / max(self._zoom, 0.01):
                self._show_line_tooltip(event)
                return

        QToolTip.hideText()

    def _show_line_tooltip(self, event):
        result = self._offside_result
        line = result.offside_line

        parts = []
        verdict = "AHEAD OF LINE" if result.any_offside else "NO PLAYER AHEAD"
        parts.append(f"<b>{verdict}</b>")
        parts.append(f"Line X: {line.image_x:.0f} px")
        if line.field_x is not None:
            parts.append(f"Field X: {line.field_x:.1f} m")
        parts.append(f"Defined by player #{line.defining_player_idx}")
        if line.defining_body_part:
            parts.append(f"Body part: {line.defining_body_part.part_type.value}")
        else:
            parts.append("Anchor: bbox bottom-center")
        parts.append(f"Confidence: {line.confidence:.0%}")
        parts.append("")
        for s in result.player_statuses:
            tag = (
                "<font color='red'>AHEAD</font>"
                if s.is_offside
                else "<font color='lime'>BEHIND</font>"
            )
            parts.append(f"P{s.player_idx}: {tag}  {abs(s.margin_px):.0f}px")
        QToolTip.showText(event.globalPos(), "<br>".join(parts), self)

    def _show_player_tooltip(self, event, player_idx):
        result = self._offside_result
        status = None
        for s in result.player_statuses:
            if s.player_idx == player_idx:
                status = s
                break
        if status is None:
            QToolTip.hideText()
            return

        tag = "AHEAD OF LINE" if status.is_offside else "BEHIND LINE"
        color = "red" if status.is_offside else "lime"
        part_name = (
            status.most_advanced_part.part_type.value
            if status.most_advanced_part
            else "bbox bottom"
        )

        parts = [
            f"<b>Player #{status.player_idx}</b>",
            f"Status: <font color='{color}'>{tag}</font>",
            f"Margin: {abs(status.margin_px):.1f} px",
            f"Body part: {part_name}",
            f"Confidence: {status.confidence:.0%}",
        ]
        QToolTip.showText(event.globalPos(), "<br>".join(parts), self)
