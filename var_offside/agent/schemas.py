"""Shapes for model answers and for the events the animation plays."""

from __future__ import annotations

from typing import Annotated, Dict, Generic, List, Literal, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Rejects unknown fields, so a model can't slip in a verdict or a distance."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Model answers
# ---------------------------------------------------------------------------

AnswerT = TypeVar("AnswerT", bound=BaseModel)


class VisionAnswer(StrictModel, Generic[AnswerT]):
    """The wrapper shared by every vision check."""

    task: str
    status: Literal["answered", "cannot_tell"]
    certainty: Literal["clear", "likely", "unclear"]
    answer: Optional[AnswerT] = None
    evidence: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _answered_needs_an_answer(self) -> "VisionAnswer[AnswerT]":
        if self.status == "answered" and self.answer is None:
            raise ValueError('status "answered" needs an answer')
        return self


class FrameCheck(StrictModel):
    """Answer for the frame check: can this shot be judged?"""

    camera_type: Literal[
        "main_side",
        "high_tactical",
        "behind_goal",
        "low_side",
        "close_up",
        "replay_or_graphic",
        "other",
    ]
    usable: Literal["yes", "no", "cannot_tell"]
    problems: List[str] = Field(default_factory=list)


class UnnumberedPerson(StrictModel):
    grid_cell: str
    partly_hidden: bool = False
    same_kit_as: Optional[int] = None


class MissedPlayersCheck(StrictModel):
    unnumbered_people: List[UnnumberedPerson] = Field(default_factory=list)


class NotAPlayer(StrictModel):
    id: int
    what: str = ""


class PersonBox(StrictModel):
    """One person, with Gemini's native box format: [ymin, xmin, ymax, xmax], each 0 to 1000."""

    box_2d: Tuple[int, int, int, int]
    kind: Literal["player", "goalkeeper", "official", "other"] = "player"

    @model_validator(mode="after")
    def _box_is_on_the_image(self) -> "PersonBox":
        ymin, xmin, ymax, xmax = self.box_2d
        if not all(0 <= value <= 1000 for value in self.box_2d):
            raise ValueError("box_2d values must be from 0 to 1000")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError("box_2d must be [ymin, xmin, ymax, xmax] with a positive size")
        return self


class PeopleBoxesCheck(StrictModel):
    """Answer for the people-detector test: a box for each person it reports."""

    people: List[PersonBox] = Field(default_factory=list)


class FalseBoxesCheck(StrictModel):
    not_players: List[NotAPlayer] = Field(default_factory=list)
    duplicates: List[Tuple[int, int]] = Field(default_factory=list)


class KitMismatch(StrictModel):
    id: int
    matches: Literal["group_1", "group_2", "neither"]


class KitsCheck(StrictModel):
    kits: Dict[Literal["group_1", "group_2"], str]
    mismatches: List[KitMismatch] = Field(default_factory=list)
    unclear_ids: List[int] = Field(default_factory=list)


class PlayCheck(StrictModel):
    """Answer for the marked-up frame: which boxes are attacking, and towards which side."""

    attacking_team: Literal["A", "B", "cannot_tell"]
    attacked_goal_side: Literal["left", "right", "cannot_tell"]
    ball_holder_id: Optional[int] = None
    reason: str = ""


# SideCheck and AttackingTeamCheck are only used by scripts/model_test.py now.
class SideEvidence(StrictModel):
    type: Literal["goal_in_view", "run_direction", "defensive_line_shape", "keeper_position"]
    side: Literal["left", "right"]


class SideCheck(StrictModel):
    attacked_goal_side: Literal["left", "right", "cannot_tell"]
    items: List[SideEvidence] = Field(default_factory=list)


class TeamEvidence(StrictModel):
    type: Literal["ball_possession", "defensive_line_shape"]
    attacking: Literal["group_1", "group_2"]


class AttackingTeamCheck(StrictModel):
    ball_holder_id: Optional[int] = None
    attacking_team: Literal["group_1", "group_2", "cannot_tell"]
    items: List[TeamEvidence] = Field(default_factory=list)


class KeeperCheck(StrictModel):
    state: Literal["in_frame", "not_in_frame", "cannot_tell"]
    keeper_id: Optional[int] = None
    kit: str = ""


class PersonRole(StrictModel):
    id: int
    role: Literal["player", "goalkeeper", "referee", "assistant_referee", "staff_or_other"]
    kit: str = ""


class OfficialsCheck(StrictModel):
    people: List[PersonRole] = Field(default_factory=list)


class KeyPlayer(StrictModel):
    id: int
    hidden_parts: List[str] = Field(default_factory=list)
    cut_off_by_frame: bool = False
    arm_reaches_furthest: bool = False
    person_behind: bool = False


class KeyPlayersCheck(StrictModel):
    players: List[KeyPlayer] = Field(default_factory=list)


class CameraLine(StrictModel):
    name: str
    fit: Literal["on", "off", "not_visible"]
    move: Optional[Literal["left", "right", "up", "down"]] = None


class CameraCheck(StrictModel):
    lines: List[CameraLine] = Field(default_factory=list)
    overall: Literal["good", "some_off", "bad", "cannot_tell"]


# ---------------------------------------------------------------------------
# Events: one for each thing the animation can show
# ---------------------------------------------------------------------------

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


class EventBase(StrictModel):
    seq: int = 0


class Correction(EventBase):
    """A change to the scene. It can be undone and may have flipped the call."""

    reason: str
    flips_call: bool = False
    undo_id: Optional[str] = None


class CheckStarted(EventBase):
    type: Literal["check_started"] = "check_started"
    check: str


class PlayerAdded(Correction):
    type: Literal["player_added"] = "player_added"
    player_id: int
    bbox: BBox
    team: Optional[Literal[1, 2]] = None


class BoxRemoved(Correction):
    type: Literal["box_removed"] = "box_removed"
    player_id: int
    bbox: BBox
    team: Optional[Literal[1, 2]] = None


class TeamChanged(Correction):
    type: Literal["team_changed"] = "team_changed"
    player_id: int
    from_team: Optional[Literal[1, 2]] = None
    to_team: Literal[1, 2]


class RoleChanged(Correction):
    type: Literal["role_changed"] = "role_changed"
    player_id: int
    role: Literal["player", "goalkeeper", "referee"]
    from_role: Literal["player", "goalkeeper", "referee"] = "player"


class SideSet(Correction):
    type: Literal["side_set"] = "side_set"
    side: Literal["left", "right"]


class AttackingTeamSet(Correction):
    type: Literal["attacking_team_set"] = "attacking_team_set"
    team: Literal[1, 2]


class KeeperNotInFrame(Correction):
    type: Literal["keeper_not_in_frame"] = "keeper_not_in_frame"


class CameraRefined(Correction):
    type: Literal["camera_refined"] = "camera_refined"
    from_lines: List[List[Point]] = Field(default_factory=list)
    to_lines: List[List[Point]] = Field(default_factory=list)


class LineMoved(EventBase):
    type: Literal["line_moved"] = "line_moved"
    from_points: Optional[List[Point]] = None  # None when the line first appears
    to_points: List[Point]
    player_id: Optional[int] = None


class CallChanged(EventBase):
    type: Literal["call_changed"] = "call_changed"
    from_call: Literal["offside", "onside", "unknown"] = Field(alias="from")
    to_call: Literal["offside", "onside", "unknown"] = Field(alias="to")
    margin_m: Optional[float] = None


class QuestionAsked(EventBase):
    type: Literal["question_asked"] = "question_asked"
    question_id: str
    text: str
    player_ids: List[int] = Field(default_factory=list)
    choices: List[str]
    bbox: Optional[BBox] = None  # The area to look at, when no player has a box yet.


class RunEnded(EventBase):
    """Only ``cleared`` earns the final burst in the animation."""

    type: Literal["run_ended"] = "run_ended"
    ending: Literal["cleared", "needs_answer", "too_close", "not_usable"]
    reason: Optional[str] = None


Event = Annotated[
    Union[
        CheckStarted,
        PlayerAdded,
        BoxRemoved,
        TeamChanged,
        RoleChanged,
        SideSet,
        AttackingTeamSet,
        KeeperNotInFrame,
        CameraRefined,
        LineMoved,
        CallChanged,
        QuestionAsked,
        RunEnded,
    ],
    Field(discriminator="type"),
]
