"""
Player Editor Dialog - Adapted for var_offside PlayerDetection objects.

Features:
- View detections on image with team-colored bounding boxes
- Click to select a player
- Delete a misdetected player
- Reassign team for a wrongly clustered player
- Draw new bounding box to add a missed player
- Apply changes triggers pipeline rerun from offside stage
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QComboBox, QGroupBox,
    QWidget, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QColor
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

from var_offside.detection.player_detector import PlayerDetection


# ---------------------------------------------------------------------------
# Lightweight bbox wrapper used by the canvas
# ---------------------------------------------------------------------------

class _BoxAdapter:
    """Thin adapter so ImageCanvas can work with PlayerDetection.bbox tuples."""

    __slots__ = ("x1", "y1", "x2", "y2")

    def __init__(self, bbox: Tuple[float, float, float, float]):
        self.x1, self.y1, self.x2, self.y2 = bbox

    @staticmethod
    def from_detection(det: PlayerDetection) -> "_BoxAdapter":
        return _BoxAdapter(det.bbox)


# ---------------------------------------------------------------------------
# ImageCanvas (adapted from birdseye_var)
# ---------------------------------------------------------------------------

class ImageCanvas(QLabel):
    """Canvas for displaying image with player boxes; click-to-select and draw-to-add."""

    player_clicked = pyqtSignal(int)
    box_drawn = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")

        self.original_image: Optional[np.ndarray] = None
        self.players: List[Tuple[_BoxAdapter, int, Tuple]] = []
        self.selected_index = -1

        # Drawing state
        self.drawing_mode = False
        self._draw_start: Optional[Tuple[int, int]] = None
        self._draw_current: Optional[Tuple[int, int]] = None

        # Coordinate mapping
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def set_image(self, image: np.ndarray):
        self.original_image = image.copy()
        self._update_display()

    def set_players(self, players: List[Tuple[_BoxAdapter, int, Tuple]], selected: int = -1):
        self.players = players
        self.selected_index = selected
        self._update_display()

    def set_selected(self, index: int):
        self.selected_index = index
        self._update_display()

    def set_drawing_mode(self, enabled: bool):
        self.drawing_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    # -- rendering ----------------------------------------------------------

    def _update_display(self):
        if self.original_image is None:
            return
        img = self.original_image.copy()

        for i, (box, team_id, color) in enumerate(self.players):
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            thickness = 4 if i == self.selected_index else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            if i == self.selected_index:
                cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 2)
            cv2.putText(img, f"P{i+1} T{team_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self._show_cv_image(img)

    def _show_cv_image(self, img: np.ndarray):
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        ws = self.size()
        scaled = pixmap.scaled(ws, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scale = scaled.width() / w if w else 1.0
        self.offset_x = (ws.width() - scaled.width()) // 2
        self.offset_y = (ws.height() - scaled.height()) // 2
        self.setPixmap(scaled)

    def _widget_to_image(self, pos: QPoint) -> Tuple[int, int]:
        x = int((pos.x() - self.offset_x) / self.scale)
        y = int((pos.y() - self.offset_y) / self.scale)
        return x, y

    # -- mouse events -------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        ix, iy = self._widget_to_image(event.pos())
        if self.drawing_mode:
            self._draw_start = (ix, iy)
            self._draw_current = (ix, iy)
        else:
            for i, (box, _, _) in enumerate(self.players):
                if box.x1 <= ix <= box.x2 and box.y1 <= iy <= box.y2:
                    self.selected_index = i
                    self.player_clicked.emit(i)
                    self._update_display()
                    return

    def mouseMoveEvent(self, event):
        if self.drawing_mode and self._draw_start:
            self._draw_current = self._widget_to_image(event.pos())
            self._draw_temp()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self.drawing_mode and self._draw_start:
            ex, ey = self._widget_to_image(event.pos())
            x1 = min(self._draw_start[0], ex)
            y1 = min(self._draw_start[1], ey)
            x2 = max(self._draw_start[0], ex)
            y2 = max(self._draw_start[1], ey)
            if x2 - x1 > 20 and y2 - y1 > 20:
                self.box_drawn.emit(x1, y1, x2, y2)
            self._draw_start = None
            self._draw_current = None
            self._update_display()

    def _draw_temp(self):
        if self.original_image is None or not self._draw_start:
            return
        img = self.original_image.copy()
        for i, (box, _, color) in enumerate(self.players):
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if self._draw_current:
            x1 = min(self._draw_start[0], self._draw_current[0])
            y1 = min(self._draw_start[1], self._draw_current[1])
            x2 = max(self._draw_start[0], self._draw_current[0])
            y2 = max(self._draw_start[1], self._draw_current[1])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self._show_cv_image(img)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


# ---------------------------------------------------------------------------
# PlayerEditorDialog
# ---------------------------------------------------------------------------

class PlayerEditorDialog(QDialog):
    """Modal dialog for editing player detections (var_offside flavour)."""

    def __init__(
        self,
        image: np.ndarray,
        detections: List[PlayerDetection],
        team_colors: Dict[int, Tuple[int, int, int]],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Player Detection Editor")
        self.setMinimumSize(1000, 700)
        self.setModal(True)

        self.image = image.copy()
        self.detections: List[PlayerDetection] = list(detections)
        self.team_colors = dict(team_colors)
        self.selected_index = -1

        self._setup_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # -- Left: canvas + mode buttons --
        left = QWidget()
        left_lay = QVBoxLayout(left)

        mode_row = QHBoxLayout()
        self.btn_select = QPushButton("Select Mode")
        self.btn_select.setCheckable(True)
        self.btn_select.setChecked(True)
        self.btn_select.clicked.connect(lambda: self._set_mode("select"))
        self.btn_select.setStyleSheet("QPushButton:checked { background-color: #2a7d2a; color: white; }")
        mode_row.addWidget(self.btn_select)

        self.btn_draw = QPushButton("Draw New Box")
        self.btn_draw.setCheckable(True)
        self.btn_draw.clicked.connect(lambda: self._set_mode("draw"))
        self.btn_draw.setStyleSheet("QPushButton:checked { background-color: #2a5d8a; color: white; }")
        mode_row.addWidget(self.btn_draw)
        left_lay.addLayout(mode_row)

        self.canvas = ImageCanvas()
        self.canvas.player_clicked.connect(self._on_player_clicked)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        left_lay.addWidget(self.canvas)

        self.lbl_instructions = QLabel("Click on a player to select")
        self.lbl_instructions.setStyleSheet("color: #888; font-style: italic;")
        left_lay.addWidget(self.lbl_instructions)

        layout.addWidget(left, stretch=2)

        # -- Right: list + edit controls --
        right = QWidget()
        right.setFixedWidth(280)
        right_lay = QVBoxLayout(right)

        list_group = QGroupBox("Players")
        list_lay = QVBoxLayout(list_group)
        self.player_list = QListWidget()
        self.player_list.itemClicked.connect(self._on_list_clicked)
        list_lay.addWidget(self.player_list)
        right_lay.addWidget(list_group)

        edit_group = QGroupBox("Edit Selected Player")
        edit_lay = QVBoxLayout(edit_group)

        team_row = QHBoxLayout()
        team_row.addWidget(QLabel("Team:"))
        self.team_combo = QComboBox()
        self.team_combo.addItem("Team 1", 1)
        self.team_combo.addItem("Team 2", 2)
        team_row.addWidget(self.team_combo)
        self.btn_set_team = QPushButton("Apply")
        self.btn_set_team.clicked.connect(self._set_team)
        self.btn_set_team.setEnabled(False)
        team_row.addWidget(self.btn_set_team)
        edit_lay.addLayout(team_row)

        self.btn_delete = QPushButton("Delete Selected Player")
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_delete.setStyleSheet(
            "QPushButton { background-color: #8b0000; color: white; padding: 8px; }"
            "QPushButton:hover { background-color: #a50000; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        edit_lay.addWidget(self.btn_delete)
        right_lay.addWidget(edit_group)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #aaa; font-size: 11px;")
        right_lay.addWidget(self.lbl_info)

        right_lay.addStretch()

        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_apply = QPushButton("Apply Changes")
        btn_apply.clicked.connect(self.accept)
        btn_apply.setStyleSheet(
            "QPushButton { background-color: #2a7d2a; color: white; padding: 10px 20px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3a9d3a; }"
        )
        btn_row.addWidget(btn_apply)
        right_lay.addLayout(btn_row)

        layout.addWidget(right)

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str):
        if mode == "select":
            self.btn_select.setChecked(True)
            self.btn_draw.setChecked(False)
            self.canvas.set_drawing_mode(False)
            self.lbl_instructions.setText("Click on a player to select")
        else:
            self.btn_select.setChecked(False)
            self.btn_draw.setChecked(True)
            self.canvas.set_drawing_mode(True)
            self.lbl_instructions.setText("Click and drag to draw a box for new player")

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh(self):
        self.canvas.set_image(self.image)

        players = []
        for det in self.detections:
            color = self.team_colors.get(det.team_id, (128, 128, 128))
            players.append((_BoxAdapter.from_detection(det), det.team_id or 0, color))
        self.canvas.set_players(players, self.selected_index)

        self.player_list.clear()
        for i, det in enumerate(self.detections):
            tid = det.team_id or 0
            color = self.team_colors.get(tid, (128, 128, 128))
            gk_tag = " [GK]" if det.is_goalkeeper else ""
            text = f"Player {i+1} - Team {tid}{gk_tag}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, i)
            item.setBackground(QColor(color[2], color[1], color[0], 80))
            if i == self.selected_index:
                item.setSelected(True)
            self.player_list.addItem(item)

        self.lbl_info.setText(f"{len(self.detections)} players total")

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_player_clicked(self, idx: int):
        self.selected_index = idx
        self._update_selection()

    def _on_list_clicked(self, item: QListWidgetItem):
        self.selected_index = item.data(Qt.UserRole)
        self._update_selection()

    def _update_selection(self):
        has = 0 <= self.selected_index < len(self.detections)
        self.btn_delete.setEnabled(has)
        self.btn_set_team.setEnabled(has)
        if has:
            tid = self.detections[self.selected_index].team_id or 1
            idx = self.team_combo.findData(tid)
            if idx >= 0:
                self.team_combo.setCurrentIndex(idx)
        self.canvas.set_selected(self.selected_index)
        for i in range(self.player_list.count()):
            item = self.player_list.item(i)
            item.setSelected(item.data(Qt.UserRole) == self.selected_index)

    # ------------------------------------------------------------------
    # Editing actions
    # ------------------------------------------------------------------

    def _on_box_drawn(self, x1: int, y1: int, x2: int, y2: int):
        new_det = PlayerDetection(
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            confidence=1.0,
            keypoints=np.zeros((17, 3), dtype=np.float32),
            player_id=len(self.detections),
            team_id=1,
        )
        self.detections.append(new_det)
        self.selected_index = len(self.detections) - 1
        self._set_mode("select")
        self._refresh()
        self._update_selection()
        self.lbl_info.setText(f"Added Player {len(self.detections)} - set team!")

    def _set_team(self):
        if not (0 <= self.selected_index < len(self.detections)):
            return
        new_team = self.team_combo.currentData()
        self.detections[self.selected_index].team_id = new_team
        self._refresh()
        self.lbl_info.setText(f"Player {self.selected_index + 1} -> Team {new_team}")

    def _delete_selected(self):
        if self.selected_index < 0:
            return
        reply = QMessageBox.question(
            self, "Delete Player",
            f"Delete Player {self.selected_index + 1}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            del self.detections[self.selected_index]
            self.selected_index = -1
            self._refresh()
            self._update_selection()
            self.lbl_info.setText("Player deleted")

    # ------------------------------------------------------------------
    # Public result accessor
    # ------------------------------------------------------------------

    def get_edited_detections(self) -> List[PlayerDetection]:
        """Return the edited list of PlayerDetection objects."""
        return list(self.detections)
