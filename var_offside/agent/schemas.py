"""Shapes for model answers and for the events the animation plays."""

from __future__ import annotations

from typing import Annotated, Generic, List, Literal, Optional, Tuple, TypeVar, Union

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


# ---------------------------------------------------------------------------
# Events: one for each thing the animation can show
# ---------------------------------------------------------------------------

BBox = Tuple[float, float, float, float]


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


class BoxRemoved(Correction):
    type: Literal["box_removed"] = "box_removed"
    player_id: int
    bbox: BBox


class TeamChanged(Correction):
    type: Literal["team_changed"] = "team_changed"
    player_id: int
    from_team: Optional[Literal[1, 2]] = None
    to_team: Literal[1, 2]


class RoleChanged(Correction):
    type: Literal["role_changed"] = "role_changed"
    player_id: int
    role: Literal["player", "goalkeeper", "referee"]


class SideSet(Correction):
    type: Literal["side_set"] = "side_set"
    side: Literal["left", "right"]


class AttackingTeamSet(Correction):
    type: Literal["attacking_team_set"] = "attacking_team_set"
    team: Literal[1, 2]


class CameraRefined(Correction):
    type: Literal["camera_refined"] = "camera_refined"


class LineMoved(EventBase):
    type: Literal["line_moved"] = "line_moved"
    from_image_x: float
    to_image_x: float
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
        CameraRefined,
        LineMoved,
        CallChanged,
        QuestionAsked,
        RunEnded,
    ],
    Field(discriminator="type"),
]
