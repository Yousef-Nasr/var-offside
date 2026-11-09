"""A private copy of one frame's scene, and the changes Auto-review can test on it."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from var_offside.calibration.base import CalibrationResult
from var_offside.detection import PlayerDetection
from var_offside.web.calibration_stage import for_analysis

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Change:
    """One edit to a scene.

    ``kind`` is one of set_team, set_role, remove_player, add_player, set_side,
    set_attacking_team, set_keeper_not_in_frame, use_box_corner or trim_box_forward.
    The last two are what-if tests of a doubtful body point, never real corrections.
    """

    kind: str
    player_id: Optional[int] = None
    team: Optional[int] = None
    role: Optional[str] = None
    side: Optional[str] = None
    bbox: Optional[BBox] = None


def role_of(detection: PlayerDetection) -> str:
    """The contract's role for a detection."""
    if getattr(detection, "is_referee", False):
        return "referee"
    return "goalkeeper" if detection.is_goalkeeper else "player"


@dataclass(frozen=True)
class Scene:
    """Everything the checks read. Changes make a new scene and never touch this one."""

    detections: Tuple[PlayerDetection, ...]
    image: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0
    side: Optional[str] = None
    attacking_team: Optional[int] = None
    keeper_choice: Optional[str] = None
    calibration: Optional[CalibrationResult] = None
    ball: Any = None
    embeddings: Dict[int, np.ndarray] = field(default_factory=dict)
    next_player_id: int = 0
    revision: int = 0

    @classmethod
    def from_result(
        cls,
        result: Any,
        side: Optional[str] = None,
        calibration: Optional[CalibrationResult] = None,
        revision: int = 0,
    ) -> "Scene":
        """Snapshot a session's analysis result. A rejected camera is dropped, as in Analyze."""
        detections = copy.deepcopy(list(result.detections))
        for index, detection in enumerate(detections):
            if detection.player_id is None:
                detection.player_id = index

        embeddings = {}
        team_result = getattr(result, "team_result", None)
        rows = getattr(team_result, "embeddings", None)
        if isinstance(rows, np.ndarray) and rows.ndim == 2 and len(rows) == len(detections):
            embeddings = {
                detection.player_id: rows[index].copy()
                for index, detection in enumerate(detections)
                if np.any(rows[index])
            }

        keeper_choice = getattr(result, "keeper_choice", None)
        if keeper_choice is None and any(det.is_goalkeeper for det in detections):
            keeper_choice = "player"

        image = getattr(result, "image", None)
        height, width = image.shape[:2] if image is not None else (0, 0)
        return cls(
            detections=tuple(detections),
            image=image,
            width=width,
            height=height,
            side=side,
            attacking_team=getattr(result, "attacking_team", None),
            keeper_choice=keeper_choice,
            calibration=for_analysis(calibration or getattr(result, "calibration", None)),
            ball=getattr(result, "ball", None),
            embeddings=embeddings,
            next_player_id=max(
                [getattr(result, "next_player_id", 0), *(det.player_id + 1 for det in detections)]
            ),
            revision=revision,
        )

    def index_of(self, player_id: Optional[int]) -> Optional[int]:
        """List position of a player id, or None when that player is gone."""
        for index, detection in enumerate(self.detections):
            if detection.player_id == player_id:
                return index
        return None

    def player(self, player_id: Optional[int]) -> Optional[PlayerDetection]:
        """The detection with this id, or None."""
        index = self.index_of(player_id)
        return None if index is None else self.detections[index]

    def apply(self, change: Change) -> "Scene":
        """Return a new scene with this change.

        Raises:
            ValueError: If the player it names isn't in the scene.
        """
        detections = copy.deepcopy(list(self.detections))
        updates: Dict[str, Any] = {}

        if change.kind == "set_side":
            return replace(self, detections=tuple(detections), side=change.side)
        if change.kind == "set_attacking_team":
            return replace(self, detections=tuple(detections), attacking_team=change.team)
        if change.kind == "set_keeper_not_in_frame":
            for detection in detections:
                detection.is_goalkeeper = False
            return replace(self, detections=tuple(detections), keeper_choice="not_in_frame")
        if change.kind == "add_player":
            added = PlayerDetection(
                bbox=change.bbox,
                confidence=0.5,
                player_id=self.next_player_id,
                team_id=change.team,
            )
            return replace(
                self,
                detections=tuple(detections + [added]),
                next_player_id=self.next_player_id + 1,
            )

        target = _find(detections, change.player_id)
        if change.kind == "set_team":
            target.team_id = change.team
        elif change.kind == "set_role":
            updates["keeper_choice"] = _set_role(detections, target, change.role, self.keeper_choice)
        elif change.kind == "remove_player":
            detections.remove(target)
            if target.is_goalkeeper:
                updates["keeper_choice"] = None
        elif change.kind == "use_box_corner":
            target.keypoints = None
        elif change.kind == "trim_box_forward":
            target.keypoints = None
            target.bbox = _trimmed_box(target.bbox, self.side)
        else:
            raise ValueError(f"Unknown change: {change.kind}")
        return replace(self, detections=tuple(detections), **updates)


def _find(detections: List[PlayerDetection], player_id: Optional[int]) -> PlayerDetection:
    for detection in detections:
        if detection.player_id == player_id:
            return detection
    raise ValueError(f"No player #{player_id} in this scene")


def _set_role(
    detections: List[PlayerDetection],
    target: PlayerDetection,
    role: Optional[str],
    keeper_choice: Optional[str],
) -> Optional[str]:
    """Set one player's role, with the same rules as Review. Returns the keeper choice."""
    was_keeper = target.is_goalkeeper
    target.is_referee = role == "referee"
    target.is_goalkeeper = False
    if role == "goalkeeper":
        for detection in detections:
            detection.is_goalkeeper = False
        target.is_goalkeeper = True
        return "player"
    if role == "referee":
        target.team_id = None
    return None if was_keeper else keeper_choice


def _trimmed_box(bbox: BBox, side: Optional[str]) -> BBox:
    """Pull in the leading quarter of a box, as if an arm made it wider."""
    x1, y1, x2, y2 = bbox
    cut = (x2 - x1) * 0.25
    return (x1 + cut, y1, x2, y2) if side == "left" else (x1, y1, x2 - cut, y2)
