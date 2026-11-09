"""What a check can report, and the kinds of evidence behind it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from var_offside.agent.scene import Change

# Kinds of evidence. Two answers count as independent only if their kinds differ,
# so the same model asked twice, or two models, is still one kind.
VISION = "vision"
DETECTOR_CLASS = "detector_class"
DETECTOR_RERUN = "detector_rerun"
EMBEDDING = "embedding"
GEOMETRY = "geometry"
CALIBRATION = "calibration"
FRAME = "frame"
BALL = "ball"
USER = "user"

# Severities.
NOTE = "note"  # For the record only.
DOUBT = "doubt"  # Only matters if the stability test shows it can change the call.
BLOCKER = "blocker"  # The frame can't clear until a person settles this.
UNUSABLE = "unusable"  # A sign that the frame can't be judged.
UNCHECKED = "unchecked"  # The check couldn't run, so it can't be treated as passed.


@dataclass(frozen=True)
class Evidence:
    kind: str
    detail: str


@dataclass(frozen=True)
class Finding:
    """One thing a check noticed.

    Attributes:
        proposal: The correction this evidence supports, if any.
        alternatives: What the stability test should try for this doubt.
    """

    check: str
    code: str
    message: str
    severity: str = NOTE
    player_ids: Tuple[int, ...] = ()
    evidence: Tuple[Evidence, ...] = ()
    proposal: Optional[Change] = None
    alternatives: Tuple[Change, ...] = ()
