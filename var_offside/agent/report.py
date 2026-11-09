"""The saved record of a run and the result's scene, written by Python with no model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from var_offside.agent.policy import Stability, safe_call
from var_offside.agent.scene import Change, Scene, role_of
from var_offside.agent.whatif import Call, with_changes

CHOICE_KINDS = frozenset({"set_side", "set_attacking_team", "set_keeper_not_in_frame"})


def players_of(scene: Optional[Scene]) -> List[Dict[str, Any]]:
    """The contract's Player list for a scene."""
    if scene is None:
        return []
    return [
        {
            "id": det.player_id,
            "bbox": [float(value) for value in det.bbox],
            "team": det.team_id if det.team_id in (1, 2) else None,
            "role": role_of(det),
            "confidence": float(det.confidence),
        }
        for det in scene.detections
    ]


def stability_of(stability: Optional[Stability]) -> Optional[Dict[str, int]]:
    if stability is None:
        return None
    return {"alternatives_tested": stability.tested, "changed_the_call": len(stability.changed)}


def too_close_verdict(verdict: Optional[Dict[str, Any]], reason: Optional[str]) -> Optional[Dict[str, Any]]:
    """No firm call: keep the line to look at, and drop the call and the margin."""
    if verdict is None:
        return None
    changed = dict(verdict)
    changed.update(verdict="unknown", margin_m=None, message=reason, offside_player_ids=[], estimate=False)
    changed["checks"] = [
        {**check, "status": "check", "detail": "Too close"} if check.get("key") == "verdict" else check
        for check in verdict.get("checks", [])
    ]
    return changed


def _call_of(call: Call) -> Dict[str, Any]:
    return {"verdict": call.verdict, "margin_m": call.margin_m, "estimate": call.estimate, "source": "python"}


def without_changes(base: Optional[Scene], changes: Sequence[Change]) -> Optional[Dict[str, Any]]:
    """The call with the three required choices made but nothing else corrected."""
    if base is None:
        return None
    keeper_marks = [item for item in changes if item.kind == "set_role" and item.role == "goalkeeper"]
    choices = [item for item in changes if item.kind in CHOICE_KINDS] + keeper_marks
    return _call_of(safe_call(with_changes(base, choices)))


def build_record(
    base: Optional[Scene],
    scene: Optional[Scene],
    decisions: Sequence[Any],
    ending: str,
    reason: Optional[str],
    revision: int,
    stage: int,
    stability: Optional[Stability],
    questions: Sequence[Dict[str, Any]],
    vision_complete: bool,
) -> Dict[str, Any]:
    """One record per run, for the report export."""
    return {
        "revision": revision,
        "review_status": ending,
        "reason": reason,
        "stage": stage,
        "vision_complete": vision_complete,
        "call": _call_of(safe_call(scene)) if scene is not None else None,
        "without_changes": without_changes(base, [item.change for item in decisions if not item.undone]),
        "stability": stability_of(stability),
        "decisions": [
            {
                "id": item.id,
                "change": item.reason,
                "flips_call": item.flips_call,
                "decided_by": item.decided_by,
                "evidence": [{"kind": entry.kind, "detail": entry.detail} for entry in item.evidence],
                "undone": item.undone,
            }
            for item in decisions
        ],
        "questions": list(questions),
    }
