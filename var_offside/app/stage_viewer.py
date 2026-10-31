"""
Stage Viewer - QTabWidget with ImageViewer per pipeline stage.

The "offside" tab uses a specialised OffsideOverlayViewer that adds an
interactive, hoverable offside-line overlay on top of the image.
"""

import numpy as np
from PyQt5.QtWidgets import QTabWidget
from var_offside.app.image_viewer import ImageViewer
from var_offside.app.offside_viewer import OffsideOverlayViewer


# Pipeline stage definitions: (key, display_name)
STAGES = [
    ("original", "Original"),
    ("detection", "Detection"),
    ("automation", "Automation"),
    ("teams", "Teams"),
    ("calibration", "Calibration"),
    ("offside", "Offside"),
    ("birdseye", "Bird's Eye"),
    ("result", "Result"),
]


class StageViewer(QTabWidget):
    """Tab widget showing one ImageViewer per pipeline stage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewers = {}

        for key, name in STAGES:
            if key == "offside":
                viewer = OffsideOverlayViewer()
            else:
                viewer = ImageViewer()
            self._viewers[key] = viewer
            self.addTab(viewer, name)

    @property
    def offside_viewer(self) -> OffsideOverlayViewer:
        """Direct access to the offside overlay viewer."""
        return self._viewers["offside"]

    def set_stage_image(self, stage_key: str, image: np.ndarray):
        """
        Update the image for a specific stage tab.

        Args:
            stage_key: One of the STAGES keys
            image: BGR numpy array
        """
        viewer = self._viewers.get(stage_key)
        if viewer is None:
            return
        if image is None:
            if hasattr(viewer, "_pixmap_item"):
                viewer._pixmap_item.setPixmap(viewer._pixmap_item.pixmap().__class__())
            if stage_key == "offside" and hasattr(viewer, "clear_offside_data"):
                viewer.clear_offside_data()
            return
        viewer.set_image(image)

    def get_viewer(self, stage_key: str):
        """Get the viewer widget for a given stage."""
        return self._viewers.get(stage_key)

    def switch_to_stage(self, stage_key: str):
        """Switch to the tab for a given stage."""
        for i, (key, _) in enumerate(STAGES):
            if key == stage_key:
                self.setCurrentIndex(i)
                return

    def clear_all(self):
        """Clear all stage images (except original)."""
        for key, viewer in self._viewers.items():
            if key != "original":
                if hasattr(viewer, "_pixmap_item"):
                    viewer._pixmap_item.setPixmap(
                        viewer._pixmap_item.pixmap().__class__()
                    )
            if key == "offside":
                viewer.clear_offside_data()

    def get_current_stage_image(self) -> np.ndarray:
        """Get the image from the currently visible tab."""
        idx = self.currentIndex()
        key = STAGES[idx][0]
        viewer = self._viewers[key]
        return viewer.get_current_image()
