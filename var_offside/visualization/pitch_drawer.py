"""
Pitch drawing utilities for bird's eye view visualization.
Draws a standard FIFA football pitch (105m x 68m).
"""

import numpy as np
import cv2
from typing import Tuple


class PitchDrawer:
    """Draw a 2D football pitch with standard dimensions."""
    
    # FIFA standard pitch dimensions (meters)
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0
    
    # Penalty area
    PENALTY_AREA_LENGTH = 16.5
    PENALTY_AREA_WIDTH = 40.32
    
    # Goal area
    GOAL_AREA_LENGTH = 5.5
    GOAL_AREA_WIDTH = 18.32
    
    # Center circle radius
    CENTER_CIRCLE_RADIUS = 9.15
    
    # Penalty spot distance from goal line
    PENALTY_SPOT_DISTANCE = 11.0
    
    # Goal dimensions
    GOAL_WIDTH = 7.32
    
    def __init__(
        self,
        pixels_per_meter: float = 10.0,
        margin: int = 50,
        bg_color: Tuple[int, int, int] = (34, 139, 34),  # Green
        line_color: Tuple[int, int, int] = (255, 255, 255),  # White
        line_thickness: int = 2
    ):
        """
        Initialize pitch drawer.
        
        Args:
            pixels_per_meter: Scale factor for drawing
            margin: Margin around the pitch in pixels
            bg_color: Background color (BGR)
            line_color: Line color (BGR)
            line_thickness: Line thickness in pixels
        """
        self.ppm = pixels_per_meter
        self.margin = margin
        self.bg_color = bg_color
        self.line_color = line_color
        self.line_thickness = line_thickness
        
        # Calculate canvas size
        self.canvas_width = int(self.PITCH_LENGTH * self.ppm) + 2 * margin
        self.canvas_height = int(self.PITCH_WIDTH * self.ppm) + 2 * margin
    
    def _to_pixels(self, x: float, y: float) -> Tuple[int, int]:
        """Convert field coordinates (meters) to pixel coordinates."""
        px = int(x * self.ppm) + self.margin
        py = int(y * self.ppm) + self.margin
        return px, py
    
    def _draw_line(
        self,
        img: np.ndarray,
        x1: float, y1: float,
        x2: float, y2: float
    ):
        """Draw a line using field coordinates."""
        pt1 = self._to_pixels(x1, y1)
        pt2 = self._to_pixels(x2, y2)
        cv2.line(img, pt1, pt2, self.line_color, self.line_thickness)
    
    def _draw_circle(
        self,
        img: np.ndarray,
        cx: float, cy: float,
        radius: float,
        filled: bool = False
    ):
        """Draw a circle using field coordinates."""
        center = self._to_pixels(cx, cy)
        r = int(radius * self.ppm)
        thickness = -1 if filled else self.line_thickness
        cv2.circle(img, center, r, self.line_color, thickness)
    
    def _draw_arc(
        self,
        img: np.ndarray,
        cx: float, cy: float,
        radius: float,
        start_angle: float,
        end_angle: float
    ):
        """Draw an arc using field coordinates."""
        center = self._to_pixels(cx, cy)
        axes = (int(radius * self.ppm), int(radius * self.ppm))
        cv2.ellipse(
            img, center, axes,
            0, start_angle, end_angle,
            self.line_color, self.line_thickness
        )
    
    def draw(self) -> np.ndarray:
        """
        Draw a complete football pitch.
        
        Returns:
            BGR image of the pitch
        """
        # Create canvas
        img = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        img[:] = self.bg_color
        
        L = self.PITCH_LENGTH
        W = self.PITCH_WIDTH
        
        # Outer boundary
        self._draw_line(img, 0, 0, L, 0)      # Bottom
        self._draw_line(img, L, 0, L, W)      # Right
        self._draw_line(img, L, W, 0, W)      # Top
        self._draw_line(img, 0, W, 0, 0)      # Left
        
        # Center line
        self._draw_line(img, L/2, 0, L/2, W)
        
        # Center circle
        self._draw_circle(img, L/2, W/2, self.CENTER_CIRCLE_RADIUS)
        
        # Center spot
        self._draw_circle(img, L/2, W/2, 0.3, filled=True)
        
        # Left penalty area
        pa_y1 = (W - self.PENALTY_AREA_WIDTH) / 2
        pa_y2 = pa_y1 + self.PENALTY_AREA_WIDTH
        self._draw_line(img, 0, pa_y1, self.PENALTY_AREA_LENGTH, pa_y1)
        self._draw_line(img, self.PENALTY_AREA_LENGTH, pa_y1, self.PENALTY_AREA_LENGTH, pa_y2)
        self._draw_line(img, self.PENALTY_AREA_LENGTH, pa_y2, 0, pa_y2)
        
        # Right penalty area
        self._draw_line(img, L, pa_y1, L - self.PENALTY_AREA_LENGTH, pa_y1)
        self._draw_line(img, L - self.PENALTY_AREA_LENGTH, pa_y1, L - self.PENALTY_AREA_LENGTH, pa_y2)
        self._draw_line(img, L - self.PENALTY_AREA_LENGTH, pa_y2, L, pa_y2)
        
        # Left goal area
        ga_y1 = (W - self.GOAL_AREA_WIDTH) / 2
        ga_y2 = ga_y1 + self.GOAL_AREA_WIDTH
        self._draw_line(img, 0, ga_y1, self.GOAL_AREA_LENGTH, ga_y1)
        self._draw_line(img, self.GOAL_AREA_LENGTH, ga_y1, self.GOAL_AREA_LENGTH, ga_y2)
        self._draw_line(img, self.GOAL_AREA_LENGTH, ga_y2, 0, ga_y2)
        
        # Right goal area
        self._draw_line(img, L, ga_y1, L - self.GOAL_AREA_LENGTH, ga_y1)
        self._draw_line(img, L - self.GOAL_AREA_LENGTH, ga_y1, L - self.GOAL_AREA_LENGTH, ga_y2)
        self._draw_line(img, L - self.GOAL_AREA_LENGTH, ga_y2, L, ga_y2)
        
        # Penalty spots
        self._draw_circle(img, self.PENALTY_SPOT_DISTANCE, W/2, 0.3, filled=True)
        self._draw_circle(img, L - self.PENALTY_SPOT_DISTANCE, W/2, 0.3, filled=True)
        
        # Penalty arcs
        self._draw_arc(img, self.PENALTY_SPOT_DISTANCE, W/2, self.CENTER_CIRCLE_RADIUS, -53, 53)
        self._draw_arc(img, L - self.PENALTY_SPOT_DISTANCE, W/2, self.CENTER_CIRCLE_RADIUS, 127, 233)
        
        # Corner arcs (small quarter circles)
        corner_radius = 1.0
        self._draw_arc(img, 0, 0, corner_radius, 0, 90)
        self._draw_arc(img, L, 0, corner_radius, 90, 180)
        self._draw_arc(img, L, W, corner_radius, 180, 270)
        self._draw_arc(img, 0, W, corner_radius, 270, 360)
        
        return img
    
    def field_to_pixel(self, field_x: float, field_y: float) -> Tuple[int, int]:
        """
        Convert field coordinates to pixel coordinates.
        
        Args:
            field_x: X position on field (0-105 meters)
            field_y: Y position on field (0-68 meters)
            
        Returns:
            (px, py) pixel coordinates on the pitch image
        """
        return self._to_pixels(field_x, field_y)
    
    def draw_player(
        self,
        img: np.ndarray,
        field_x: float,
        field_y: float,
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: float = 1.5,
        label: str = None
    ) -> np.ndarray:
        """
        Draw a player on the pitch.
        
        Args:
            img: Pitch image to draw on
            field_x: X position (0-105 meters)
            field_y: Y position (0-68 meters)
            color: Player color (BGR)
            radius: Player circle radius in meters
            label: Optional text label
            
        Returns:
            Image with player drawn
        """
        # Clamp to pitch boundaries
        field_x = max(0, min(field_x, self.PITCH_LENGTH))
        field_y = max(0, min(field_y, self.PITCH_WIDTH))
        
        center = self._to_pixels(field_x, field_y)
        r = int(radius * self.ppm)
        
        # Draw filled circle with border
        cv2.circle(img, center, r, color, -1)
        cv2.circle(img, center, r, (255, 255, 255), 1)
        
        if label:
            cv2.putText(
                img, label,
                (center[0] - 5, center[1] - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1
            )
        
        return img
    
    def draw_offside_line(
        self,
        img: np.ndarray,
        field_x: float,
        color: Tuple[int, int, int] = (0, 255, 255),  # Yellow
        thickness: int = 3,
        label: str = None
    ) -> np.ndarray:
        """
        Draw a vertical offside line on the pitch.
        
        Args:
            img: Pitch image to draw on
            field_x: X position on field (0-105 meters)
            color: Line color (BGR)
            thickness: Line thickness in pixels
            label: Optional text label
            
        Returns:
            Image with offside line drawn
        """
        # Clamp to pitch boundaries
        field_x = max(0, min(field_x, self.PITCH_LENGTH))
        
        # Draw vertical line from top to bottom of pitch
        pt1 = self._to_pixels(field_x, 0)
        pt2 = self._to_pixels(field_x, self.PITCH_WIDTH)
        
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        
        if label:
            # Draw label at top of line
            cv2.putText(
                img, label,
                (pt1[0] + 5, pt1[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )
        
        return img


if __name__ == "__main__":
    # Test pitch drawing
    drawer = PitchDrawer()
    pitch = drawer.draw()
    
    # Draw test players
    drawer.draw_player(pitch, 20, 34, color=(255, 0, 0))  # Blue player
    drawer.draw_player(pitch, 80, 34, color=(0, 0, 255))  # Red player
    
    cv2.imshow("Pitch", pitch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
