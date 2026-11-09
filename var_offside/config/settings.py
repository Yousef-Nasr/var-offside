"""
Central configuration for the VAR Offside Detection System.
"""

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple
import torch

from var_offside.config.paths import (
    first_existing,
    project_home,
    resolve_weight,
    results_dir,
)


@dataclass
class ModelPaths:
    """Paths to model weights, resolved through :mod:`var_offside.config.paths`.

    See ``docs/getting-started/models.md`` for where to download each file.
    """

    yolo_pose: str = "yolo11x-pose.pt"  # Ultralytics downloads it if missing
    yolo_detection: str = "best_tuned.pt"
    resnet: str = "resnet50"  # Will use torchvision pretrained
    pnl_keypoints: str = field(
        default_factory=lambda: first_existing("pnlcalib/SV_kp", "PnLCalib/weights/SV_kp")
    )
    pnl_lines: str = field(
        default_factory=lambda: first_existing("pnlcalib/SV_lines", "PnLCalib/weights/SV_lines")
    )

    def __post_init__(self):
        # Anchor file weights to weights/ so the app works from any working
        # directory. Non-file names (e.g. "resnet50") pass through.
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

    # Anchor for the 3D mesh method's ground point.
    #   "directional" = current behaviour (foot centroid on the bbox forward
    #                   corner).
    #   "center" = foot centroid on the bbox bottom-centre ground point, then
    #              the leading non-arm vertex is measured (avoids counting
    #              forward reach twice).
    mesh_anchor: str = "directional"


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

    # weights/sam-3d-body-dinov3/ is the Hugging Face repo facebook/sam-3d-body-dinov3
    # as downloaded. The second name in each pair is the older development layout.
    sam3d_checkpoint: str = field(
        default_factory=lambda: first_existing(
            "sam-3d-body-dinov3/model.ckpt", "model_sam3_3d.ckpt"
        )
    )
    sam3d_config: str = field(
        default_factory=lambda: first_existing(
            "sam-3d-body-dinov3/model_config.yaml", "sam3_body3d/model_config.yaml"
        )
    )
    sam3d_yolo: str = "yolo11l.pt"
    sam3d_mhr: str = field(
        default_factory=lambda: first_existing(
            "sam-3d-body-dinov3/assets/mhr_model.pt",
            "sam3_body3d/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        )
    )
    video_duration_sec: float = 10.0
    video_fps: int = 30
    video_width: int = 1280
    video_height: int = 720
    camera_elevation_deg: float = 35.0
    camera_distance: float = 12.0
    pitch_length: float = 105.0
    pitch_width: float = 68.0


@dataclass
class ServerConfig:
    """Web app server defaults. Command-line flags still win."""

    host: str = "127.0.0.1"  # 0.0.0.0 exposes the app, which has no login, to your network
    port: int = 8765
    open_browser: bool = True
    max_upload_mb: int = 500  # larger uploads get HTTP 413


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    server: ServerConfig = field(default_factory=ServerConfig)

    # "cuda", "cpu", "cuda:1", ... ("auto" in config.yaml picks CUDA when available)
    device: str = field(default_factory=_auto_device)

    # Exports go to results/ under the project home unless configured
    output_dir: Path = field(default_factory=results_dir)

    # The config file these settings were read from, if any
    config_file: Optional[Path] = None

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------

# Path settings that are resolved through weights/ when given as relative names.
_PATH_FIELDS = {
    "models": ("yolo_pose", "yolo_detection", "pnl_keypoints", "pnl_lines"),
    "var_3d": ("sam3d_checkpoint", "sam3d_config", "sam3d_yolo", "sam3d_mhr"),
}


def find_config_file() -> Optional[Path]:
    """``$VAR_OFFSIDE_CONFIG``, else ``config.yaml`` in the project home."""
    env = os.environ.get("VAR_OFFSIDE_CONFIG")
    if env:
        path = Path(env).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"VAR_OFFSIDE_CONFIG points at a missing file: {path}")
        return path
    path = project_home() / "config.yaml"
    return path if path.is_file() else None


def _coerce(current: Any, value: Any, where: str) -> Any:
    """Match the type of the default so YAML lists become tuples, ints floats."""
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current, bool) or current is None:
        return value
    if isinstance(current, float) and isinstance(value, int):
        return float(value)
    if isinstance(current, (int, float, str)) and not isinstance(value, type(current)):
        raise ValueError(
            f"{where} should be {type(current).__name__}, got {value!r}"
        )
    return value


def apply_overrides(settings: Settings, data: Mapping[str, Any], source: str = "config") -> Settings:
    """Apply a nested mapping (as read from config.yaml) onto ``settings``.

    Unknown sections or keys raise ``ValueError`` so a typo never passes silently.
    """
    sections = {f.name for f in fields(settings) if is_dataclass(getattr(settings, f.name))}
    for key, value in (data or {}).items():
        if key in sections:
            if not isinstance(value, Mapping):
                raise ValueError(f"{source}: section '{key}' must be a mapping")
            section = getattr(settings, key)
            known = {f.name for f in fields(section)}
            for name, item in value.items():
                if name not in known:
                    raise ValueError(
                        f"{source}: unknown setting '{key}.{name}'. "
                        f"Known: {', '.join(sorted(known))}"
                    )
                item = _coerce(getattr(section, name), item, f"{source}: {key}.{name}")
                if name in _PATH_FIELDS.get(key, ()):
                    item = resolve_weight(str(Path(item).expanduser()))
                setattr(section, name, item)
        elif key == "device":
            settings.device = _auto_device() if value in (None, "auto") else str(value)
        elif key == "output_dir":
            settings.output_dir = Path(value).expanduser()
            if not settings.output_dir.is_absolute():
                settings.output_dir = project_home() / settings.output_dir
            settings.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"{source}: unknown setting '{key}'")
    return settings


def load_settings(config_file: Optional[Path] = None) -> Settings:
    """Build settings from the defaults plus ``config.yaml`` when there is one."""
    settings = Settings()
    path = config_file or find_config_file()
    if path is None:
        return settings
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: the top level must be a mapping of sections")
    apply_overrides(settings, data, source=str(path))
    settings.config_file = Path(path)
    return settings


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the settings singleton (defaults + config.yaml)."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Forget the cached settings so the next ``get_settings()`` reloads them."""
    global _settings
    _settings = None
