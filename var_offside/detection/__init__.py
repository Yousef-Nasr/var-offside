"""Detection module - Player detection and pose estimation."""

from var_offside.detection.player_detector import PlayerDetector, PlayerDetection
from var_offside.detection.body_parts import BodyPartsExtractor, PlayableBodyPart
from var_offside.detection.goalkeeper_filter import GoalkeeperFilter

__all__ = [
    "PlayerDetector",
    "PlayerDetection",
    "BodyPartsExtractor", 
    "PlayableBodyPart",
    "GoalkeeperFilter",
]
