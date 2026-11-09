"""Can the players who decide the call be trusted?"""

from __future__ import annotations

from typing import List

from var_offside.agent.findings import BLOCKER, DOUBT, GEOMETRY, Evidence, Finding
from var_offside.agent.scene import Change, Scene
from var_offside.agent.whatif import run_call

EDGE_TOUCH_PX = 2.0
NEAR_EDGE_RATIO = 0.05


def _point_doubt(scene: Scene, player_id: int) -> Finding:
    """What to test about one key player's body point."""
    det = scene.player(player_id)
    if det.keypoints is None:
        kind, code = "trim_box_forward", "no_body_points"
        message = f"#{player_id} has no body points, so an arm could widen the box."
    else:
        kind, code = "use_box_corner", "box_vs_body"
        message = f"#{player_id}'s box and body points could disagree."
    return Finding(
        "line",
        code,
        message,
        DOUBT,
        (player_id,),
        (Evidence(GEOMETRY, message),),
        alternatives=(Change(kind, player_id=player_id),),
    )


def _coverage(scene: Scene, player_id: int) -> List[Finding]:
    """Blockers when the line setter may not be the real one. Nothing can correct these."""
    if not scene.width or not scene.height:
        return []
    det = scene.player(player_id)
    x1, y1, x2, y2 = det.bbox
    cut_off = (
        x1 <= EDGE_TOUCH_PX
        or y1 <= EDGE_TOUCH_PX
        or x2 >= scene.width - EDGE_TOUCH_PX
        or y2 >= scene.height - EDGE_TOUCH_PX
    )
    if cut_off:
        return [
            Finding(
                "line",
                "line_setter_cut_off",
                f"#{player_id} sets the line but is cut off by the frame.",
                BLOCKER,
                (player_id,),
            )
        ]

    if scene.keeper_choice != "not_in_frame":
        return []
    # With no keeper in view, a deeper defender could be just outside the picture.
    gap = scene.width - x2 if scene.side == "right" else x1
    if gap < NEAR_EDGE_RATIO * scene.width:
        return [
            Finding(
                "line",
                "line_near_frame_edge",
                f"#{player_id} sets the line at the edge of the frame, and anyone deeper is out of view.",
                BLOCKER,
                (player_id,),
            )
        ]
    return []


def check(scene: Scene) -> List[Finding]:
    """Coverage of the line setter, and body points of the line setter and foremost attacker."""
    call = run_call(scene)
    if call.line_player_id is None:
        return []
    findings = _coverage(scene, call.line_player_id)
    findings.append(_point_doubt(scene, call.line_player_id))
    if call.foremost_attacker_id is not None:
        findings.append(_point_doubt(scene, call.foremost_attacker_id))
    return findings
