"""
Central configuration for the VAR Offside Detection System.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import torch

from var_offside.config.paths import resolve_weight, results_dir


@dataclass
class ModelPaths:
    """Paths to model weights (resolved to the repo root when present)."""

    yolo_pose: str = "yolo11x-pose.pt"
    yolo_detection: str = "best_tuned.pt"
    resnet: str = "resnet50"  # Will use torchvision pretrained
    pnl_keypoints: str = "PnLCalib/weights/SV_kp"
    pnl_lines: str = "PnLCalib/weights/SV_lines"

    def __post_init__(self):
        # Anchor file/dir weights to the repo root so the app works from any
        # working directory. Non-file names (e.g. "resnet50") pass through.
        for attr in ("yolo_pose", "yolo_detection", "pnl_keypoints", "pnl_lines"):
            setattr(self, attr, resolve_weight(getattr(self, attr)))


@dataclass
class DetectionConfig:
    """Player detection configuration."""

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    min_player_height: int = 30  # Minimum bounding box height in pixels
    max_players: int = 30  # Maximum players to detect


@dataclass
class PoseConfig:
    """Pose estimation configuration."""

    # COCO keypoint indices
    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16

    # Playable body parts for offside (FIFA rules)
    playable_keypoints: Tuple[int, ...] = field(
        default_factory=lambda: (
            0,  # Head (nose as proxy)
            5,
            6,  # Shoulders
            13,
            14,  # Knees
            15,
            16,  # Feet (ankles)
        )
    )

    keypoint_confidence_threshold: float = 0.3


@dataclass
class TeamConfig:
    """Team assignment configuration."""

    num_teams: int = 2
    embedding_dim: int = 2048  # ResNet50 output dimension
    umap_components: int = 3
    jersey_crop_ratio: float = 0.7  # Focus on upper body for jersey


@dataclass
class CalibrationConfig:
    """Field calibration configuration."""

    # Green field segmentation HSV ranges
    green_lower: Tuple[int, int, int] = (35, 40, 40)
    green_upper: Tuple[int, int, int] = (85, 255, 255)

    # Hough line detection
    hough_threshold: int = 100
    min_line_length: int = 100
    max_line_gap: int = 10

    # Vanishing point
    ransac_iterations: int = 1000
    inlier_threshold: float = 5.0

    # Line angle filtering (degrees)
    min_line_angle: float = 15.0
    max_line_angle: float = 89.0


@dataclass
class OffsideConfig:
    """Offside calculation configuration."""

    # Margin in pixels to consider as "on the line"
    line_margin_px: int = 3

    # Field dimensions (meters) - FIFA standard
    field_length: float = 105.0
    field_width: float = 68.0


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    # Team colors (BGR format)
    team1_color: Tuple[int, int, int] = (255, 0, 0)  # Blue
    team2_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    goalkeeper_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow

    # Offside line colors
    offside_line_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    onside_line_color: Tuple[int, int, int] = (0, 255, 0)  # Green

    line_thickness: int = 2
    font_scale: float = 0.6


@dataclass
class VAR3DConfig:
    """3D VAR video rendering configuration."""

    sam3d_checkpoint: str = "model_sam3_3d.ckpt"
    sam3d_config: str = "sam3_body3d/model_config.yaml"
    sam3d_yolo: str = "yolo11l.pt"
    sam3d_mhr: str = "sam3_body3d/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    video_duration_sec: float = 10.0
    video_fps: int = 30
    video_width: int = 1280
    video_height: int = 720
    camera_elevation_deg: float = 35.0
    camera_distance: float = 12.0
    pitch_length: float = 105.0
    pitch_width: float = 68.0


@dataclass
class Settings:
    """Main settings container."""

    models: ModelPaths = field(default_factory=ModelPaths)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    teams: TeamConfig = field(default_factory=TeamConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    offside: OffsideConfig = field(default_factory=OffsideConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    var_3d: VAR3DConfig = field(default_factory=VAR3DConfig)

    # Device configuration
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Output paths (repo-root results/ so it is consistent across launchers)
    output_dir: Path = field(default_factory=results_dir)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
