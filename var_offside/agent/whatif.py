"""The real line and decision code, run on a scene or a changed copy of it.

Uses no GPU, render thread or model, so it's safe to call many times per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from var_offside.agent.scene import Change, Scene
from var_offside.offside.decision import OffsideDecision
from var_offside.offside.line_calculator import OffsideLine, OffsideLineCalculator
from var_offside.pipeline import AnalysisResult
from var_offside.web.phases import analysis_choice_error, serialize_verdict

_LINE = OffsideLineCalculator()
_DECISION = OffsideDecision()


@dataclass(frozen=True)
class Call:
    """The result of one run: the same fields the app would report."""

    verdict: str  # "offside" | "onside" | "unknown"
    margin_m: Optional[float] = None  # + = attacker past the line
    estimate: bool = False
    message: Optional[str] = None
    line: Optional[OffsideLine] = None
    line_player_id: Optional[int] = None
    offside_ids: Tuple[int, ...] = ()
    foremost_attacker_id: Optional[int] = None
    line_points: Tuple[Tuple[float, float], ...] = ()


def run_call(scene: Scene) -> Call:
    """Work out the call for a scene, following the same steps as Analyze."""
    if scene.side not in ("left", "right"):
        return Call("unknown", message="Choose the attack direction.")

    result = AnalysisResult(
        image=scene.image,
        detections=list(scene.detections),
        calibration=scene.calibration,
        attacking_team=scene.attacking_team,
        keeper_choice=scene.keeper_choice,
    )
    error = analysis_choice_error(result)
    if error is not None:
        return Call("unknown", message=error)
    result.defending_team = 3 - scene.attacking_team

    keeper_idx = next(
        (index for index, det in enumerate(scene.detections) if det.is_goalkeeper), None
    )
    result.offside_line = _LINE.calculate_offside_line(
        result.detections,
        attacking_side=scene.side,
        calibration=scene.calibration,
        exclude_goalkeeper=True,
        defending_team=result.defending_team,
        keeper_idx=keeper_idx,
    )
    if result.offside_line is None:
        return Call("unknown", message="No visible defender can set the offside line.")

    result.offside_result = _DECISION.evaluate(
        result.detections,
        result.offside_line,
        attacking_side=scene.side,
        calibration=scene.calibration,
        exclude_goalkeeper=True,
        attacking_team=scene.attacking_team,
        defending_team=result.defending_team,
    )

    payload = serialize_verdict(result)
    statuses = [s for s in result.offside_result.player_statuses if not s.in_own_half]
    foremost = max(statuses, key=lambda status: status.margin_px, default=None)
    return Call(
        verdict=payload["verdict"],
        margin_m=payload["margin_m"],
        estimate=payload["estimate"],
        message=payload["message"],
        line=result.offside_line,
        line_player_id=payload["last_defender"]["id"],
        offside_ids=tuple(payload["offside_player_ids"]),
        foremost_attacker_id=None if foremost is None else foremost.player_detection.player_id,
        line_points=tuple(tuple(point) for point in payload["line_image_points"]),
    )


def what_if(scene: Scene, *changes: Change) -> Call:
    """The call if these changes were made, leaving ``scene`` untouched."""
    return run_call(with_changes(scene, changes))


def with_changes(scene: Scene, changes: Sequence[Change]) -> Scene:
    """A new scene with each change applied in order."""
    for change in changes:
        scene = scene.apply(change)
    return scene
