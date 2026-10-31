"""
Image Viewer Widget - Zoomable and pannable image display.
"""

import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QPainter
from PyQt5.QtCore import Qt


class ImageViewer(QGraphicsView):
    """Zoomable, pannable image viewer using QGraphicsView."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.black)

        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0

    def set_image(self, image: np.ndarray):
        """Set the image to display."""
        if image is None:
            return

        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1].copy()
        else:
            rgb = image

        h, w = rgb.shape[:2]

        if len(rgb.shape) == 3:
            q_image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q_image = QImage(rgb.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_view()

    def fit_to_view(self):
        """Fit the image to the viewport."""
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._zoom = 1.0

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with the mouse wheel."""
        factor = 1.15

        if event.angleDelta().y() > 0:
            if self._zoom * factor < self._max_zoom:
                self._zoom *= factor
                self.scale(factor, factor)
        else:
            if self._zoom / factor > self._min_zoom:
                self._zoom /= factor
                self.scale(1 / factor, 1 / factor)

    def get_current_image(self) -> np.ndarray:
        """Return the current image as a BGR numpy array."""
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return None

        image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 3)

        arr = np.array(ptr).reshape(height, width, 3)
        return arr[:, :, ::-1].copy()
