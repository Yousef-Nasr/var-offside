"""
Qt-free pipeline phases for the browser app.

The desktop application runs its two-phase pipeline inside a ``QThread``
(:mod:`var_offside.app.pipeline_worker`).  The web server cannot use Qt, so the
same logic is copied here as plain functions returning ``(AnalysisResult,
stages)`` tuples.  Nothing in this module imports Qt and the desktop app is not
modified.

Phase 1 performs detection and team classification (always on).  The keeper, the
attacking team and the attack direction are the user's choices in Review, and
nothing here guesses them.  Calibration runs on its own through
``run_calibration``, because the Calibrate step lets the user check and correct it.
Phase 2 then uses that calibration for the offside line and the verdict.
``apply_player_edits`` rebuilds detections after the user reviews the players.

The JSON helpers at the bottom of the module produce exactly the ``Phase1`` and
``Verdict`` shapes documented in ``var_offside/web/API_CONTRACT.md``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from var_offside.calibration.base import CalibrationResult
from var_offside.config import get_settings
from var_offside.detection import PlayerDetection
from var_offside.offside.line_calculator import (
    OffsideLine,
    attack_field_sign,
    project_detection_to_field,
)
from var_offside.pipeline import AnalysisResult, OffsidePipeline
from var_offside.visualization.line_projection import project_line
from var_offside.web.calibration_stage import for_analysis
from var_offside.web.overlays import TEAM_BGR, TEAM_HEX, draw_player_markers

logger = logging.getLogger(__name__)

# Stage keys the contract allows the server to expose.
STAGE_KEYS: Tuple[str, ...] = (
    "original",
    "detection",
    "teams",
    "calibration",
    "offside",
    "birdseye",
    "result",
)

# Phase-1 stage keys in display order.
PHASE1_STAGE_KEYS: Tuple[str, ...] = ("original", "detection", "teams")

# Phase-2 stage keys in display order.
PHASE2_STAGE_KEYS: Tuple[str, ...] = ("calibration", "offside", "birdseye", "result")

ESTIMATE_MESSAGE = "This is an estimate from the drawn line, with no measured margin."

_PIPELINE: Optional[OffsidePipeline] = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> OffsidePipeline:
    """Return the lazily created, process-wide :class:`OffsidePipeline`.

    Loading the pipeline is expensive (models are loaded on first use), so it is
    created once and shared by every request.  Mirrors
    :func:`var_offside.api.service.get_pipeline` but without request-specific
    keys: the web app always runs the default ``auto`` calibration on the device
    configured in :func:`var_offside.config.get_settings`.
    """
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            settings = get_settings()
            logger.info("Creating OffsidePipeline on device=%s", settings.device)
            _PIPELINE = OffsidePipeline(device=settings.device)
        return _PIPELINE


def reset_pipeline() -> None:
    """Drop the cached pipeline (used by tests and diagnostics)."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        _PIPELINE = None


# ---------------------------------------------------------------------------
# Phase 1 - detection, teams
# ---------------------------------------------------------------------------


def run_phase1(
    pipeline: OffsidePipeline, image: np.ndarray
) -> Tuple[AnalysisResult, Dict[str, np.ndarray]]:
    """Run detection and team classification for one frame.

    Team classification always runs.  Roles, the attacking team and the attack
    direction are left for the user to set in Review.

    Args:
        pipeline: Initialised offside pipeline.
        image: BGR frame to analyse.

    Returns:
        ``(result, stages)`` where ``stages`` maps stage keys to BGR images.
    """
    if pipeline is None:
        raise ValueError("No pipeline available")
    if image is None:
        raise ValueError("No image loaded")

    result = AnalysisResult()
    result.image = image
    start = time.perf_counter()
    stages: Dict[str, np.ndarray] = {"original": image.copy()}

    # -- Detection ---------------------------------------------------------
    try:
        result.detections = pipeline.player_detector.detect(image)
    except Exception as exc:  # noqa: BLE001 - surface the message to the client
        raise RuntimeError(f"Detection failed: {exc}") from exc

    stages["detection"] = pipeline._draw_detections_debug(  # noqa: SLF001
        image, result.detections, "Detections"
    )

    if not result.detections:
        result.errors.append("No players detected")
        result.processing_time_ms = (time.perf_counter() - start) * 1000.0
        return result, stages

    # -- Teams (always on) -------------------------------------------------
    try:
        result.team_result = pipeline.team_assigner.assign_teams(
            image, result.detections
        )
    except Exception as exc:  # noqa: BLE001 - keep going without teams
        logger.warning("Team assignment skipped: %s", exc)
        result.team_result = None
        result.errors.append(f"Team assignment skipped: {exc}")

    stages["teams"] = _draw_teams_stage(pipeline, image, result)

    result.processing_time_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "Phase 1 complete: %d players, teams=%s (%.0f ms)",
        len(result.detections),
        result.team_result is not None,
        result.processing_time_ms,
    )
    return result, stages


def _draw_teams_stage(
    pipeline: OffsidePipeline, image: np.ndarray, result: AnalysisResult
) -> np.ndarray:
    """Draw the review stage: the same rings and labels as the photo view."""
    if result.team_result is not None and result.team_result.detections:
        result.team_result.detections = result.detections
    return draw_player_markers(image, result.detections)


# ---------------------------------------------------------------------------
# Player edits - rebuild detections from the contract's Player list
# ---------------------------------------------------------------------------


def apply_player_edits(
    pipeline: OffsidePipeline,
    result: AnalysisResult,
    players: Sequence[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """Rebuild detections from the client's edited player list.

    Supports team switches, removals, added boxes and the goalkeeper/referee
    roles.  The teams review stage is redrawn and returned.

    Args:
        pipeline: Initialised offside pipeline.
        result: Phase-1 result whose detections are replaced.
        players: Contract ``Player`` dicts (``id`` may be ``None`` for new rows).

    Returns:
        ``{"teams": image}`` for the redrawn review stage.
    """
    if result is None or getattr(result, "image", None) is None:
        raise ValueError("Run detection before editing players")

    image = result.image
    old_detections = list(result.detections)
    edited: List[PlayerDetection] = []

    for order, payload in enumerate(players):
        bbox = _normalize_bbox(payload.get("bbox"), image.shape)
        if bbox is None:
            logger.warning("Skipping player %d: invalid bbox %r", order, payload.get("bbox"))
            continue

        base = _base_detection(old_detections, payload.get("id"))
        role = payload.get("role") or "player"
        team = payload.get("team")
        if role == "referee":
            team = None
        if team not in (1, 2):
            team = None

        confidence = payload.get("confidence")
        if confidence is None:
            confidence = base.confidence if base is not None else 1.0
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0

        detection = PlayerDetection(
            bbox=bbox,
            confidence=confidence,
            keypoints=(
                base.keypoints
                if base is not None and _same_box(base.bbox, bbox)
                else None
            ),
            player_id=order,
            team_id=team,
            is_goalkeeper=(role == "goalkeeper"),
        )
        # ``referee`` is not a field on PlayerDetection; keep it as an extra
        # attribute so it round-trips through serialization.
        detection.is_referee = role == "referee"
        edited.append(detection)

    result.detections = edited

    if result.team_result is not None:
        result.team_result.detections = edited
        embeddings = result.team_result.embeddings
        if embeddings is not None and len(embeddings) != len(edited):
            result.team_result.embeddings = None

    # Player edits invalidate any previous analysis.  The calibration stays: it
    # depends on the frame, not on the players.
    result.side_result = None
    result.offside_line = None
    result.offside_result = None
    result.visualization = None
    result.verdict_message = None

    return {"teams": _draw_teams_stage(pipeline, image, result)}


def _same_box(first: Sequence[float], second: Sequence[float]) -> bool:
    """Whether a box is still where it was, so its pose still belongs to it."""
    return all(abs(float(a) - float(b)) < 1.0 for a, b in zip(first, second))


def _base_detection(
    detections: Sequence[PlayerDetection], player_id: Any
) -> Optional[PlayerDetection]:
    """Return the previous detection referenced by ``player_id`` if any."""
    if isinstance(player_id, bool) or not isinstance(player_id, int):
        return None
    if 0 <= player_id < len(detections):
        return detections[player_id]
    return None


def _normalize_bbox(
    bbox: Any, image_shape: Tuple[int, ...]
) -> Optional[Tuple[float, float, float, float]]:
    """Coerce a client bbox to ``(x1, y1, x2, y2)`` clipped to the frame."""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
    else:
        return None

    height, width = image_shape[:2]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def run_calibration(pipeline: OffsidePipeline, image: np.ndarray) -> CalibrationResult:
    """Run the automatic pitch calibration on one frame."""
    return pipeline.calibrator.calibrate(image)


# ---------------------------------------------------------------------------
# Phase 2 - line, verdict
# ---------------------------------------------------------------------------


def analysis_choice_error(result: AnalysisResult) -> Optional[str]:
    """Return the user action needed before analysis, if any."""
    if result.attacking_team not in (1, 2):
        return "Choose the attacking team."

    keeper_choice = getattr(result, "keeper_choice", None)
    if keeper_choice not in ("player", "not_in_frame"):
        return 'Choose the keeper or "not in frame".'

    keeper_count = sum(det.is_goalkeeper for det in result.detections)
    if keeper_choice == "player" and keeper_count != 1:
        return "Choose exactly one player as the keeper."
    if keeper_choice == "not_in_frame" and keeper_count:
        return (
            'A keeper is marked, so "not in frame" conflicts with the player list. '
            "Choose the keeper or unmark them."
        )
    return None


def run_phase2(
    pipeline: OffsidePipeline,
    result: AnalysisResult,
    attacking_side: Optional[str],
) -> Tuple[AnalysisResult, Dict[str, np.ndarray]]:
    """Run the offside line and the decision.

    Uses ``result.calibration`` when it is set, and calibrates only when it isn't.

    Args:
        pipeline: Initialised offside pipeline.
        result: Phase-1 result and the reviewed team/keeper choices.
        attacking_side: ``"left"`` or ``"right"``; anything else is rejected.

    Returns:
        ``(result, stages)`` with ``calibration``, ``offside``, optional
        ``birdseye`` and ``result`` stages.

    Raises:
        ValueError: If the side is missing/invalid or there is no phase-1 result.
    """
    if attacking_side not in ("left", "right"):
        raise ValueError("attacking_side must be 'left' or 'right'")
    if result is None or getattr(result, "image", None) is None:
        raise ValueError("Run detection before analysis")

    image = result.image
    start = time.perf_counter()
    stages: Dict[str, np.ndarray] = {}
    result.offside_line = None
    result.offside_result = None
    result.visualization = None
    result.verdict_message = None
    result.side_result = None

    choice_error = analysis_choice_error(result)
    if choice_error is not None:
        result.verdict_message = choice_error
        return result, stages

    result.defending_team = 1 if result.attacking_team == 2 else 2
    result.side_result = pipeline.side_detector.detect_with_manual_override(
        result.attacking_team, attacking_side
    )
    keeper_idx = None
    if result.keeper_choice == "player":
        keeper_idx = next(
            idx for idx, det in enumerate(result.detections) if det.is_goalkeeper
        )

    # -- Calibration -------------------------------------------------------
    if result.calibration is None:
        try:
            result.calibration = run_calibration(pipeline, image)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Calibration failed: {exc}") from exc
    result.calibration = for_analysis(result.calibration)

    calib_img = pipeline.field_overlay.draw_calibration_result(
        image, result.calibration
    )
    if result.calibration is not None and result.calibration.has_3d_calibration:
        calib_img = pipeline.field_overlay.draw_projected_field(
            calib_img, result.calibration
        )
    stages["calibration"] = calib_img

    # -- Offside line + decision ------------------------------------------
    result.offside_line = pipeline.line_calculator.calculate_offside_line(
        result.detections,
        attacking_side=attacking_side,
        calibration=result.calibration,
        exclude_goalkeeper=True,
        defending_team=result.defending_team,
        keeper_idx=keeper_idx,
    )

    if result.offside_line is None:
        defending_visible = any(
            det.team_id == result.defending_team for det in result.detections
        )
        if not defending_visible:
            result.verdict_message = "No defending-team players are visible."
        elif keeper_idx is not None:
            result.verdict_message = (
                "Only the keeper is visible; at least one other defender is needed."
            )
        else:
            result.verdict_message = "No visible defender can set the offside line."
        result.errors.append("Could not calculate line player")
        result.processing_time_ms = (time.perf_counter() - start) * 1000.0
        return result, stages

    stages["offside"] = _draw_offside_stage(pipeline, image, result)

    result.offside_result = pipeline.offside_decision.evaluate(
        result.detections,
        result.offside_line,
        attacking_side=attacking_side,
        calibration=result.calibration,
        exclude_goalkeeper=True,
        attacking_team=result.attacking_team,
        defending_team=result.defending_team,
    )

    # -- Bird's eye --------------------------------------------------------
    birdseye = _draw_birdseye(result)
    if birdseye is not None:
        stages["birdseye"] = birdseye

    # -- Clean result frame ------------------------------------------------
    result.visualization = pipeline.visualizer.create_clean_result_view(
        image, result.offside_result, result.calibration
    )
    stages["result"] = result.visualization

    result.processing_time_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "Phase 2 complete: verdict=%s, margin=%s, %.0f ms",
        "offside" if result.offside_result.any_offside else "onside",
        result.offside_result.num_ahead,
        result.processing_time_ms,
    )
    return result, stages


def _draw_offside_stage(
    pipeline: OffsidePipeline, image: np.ndarray, result: AnalysisResult
) -> np.ndarray:
    """Draw the calculated line over the frame for the debug stage."""
    try:
        return pipeline._draw_offside_line_debug(  # noqa: SLF001
            image,
            result.offside_line,
            result.detections,
            result.calibration,
        )
    except Exception as exc:  # noqa: BLE001 - debug stage is optional
        logger.warning("Offside stage drawing failed: %s", exc)
        return image.copy()


def _draw_birdseye(result: AnalysisResult) -> Optional[np.ndarray]:
    """Render a bird's eye view of the pitch with players.

    Copied from :meth:`var_offside.app.pipeline_worker.PipelineWorker._draw_birdseye`.
    Returns ``None`` when the optional ``birdseye_var`` package or a usable
    field projection is unavailable.
    """
    try:
        from birdseye_var.pitch import PitchDrawer
    except ImportError:
        return None

    calibration = getattr(result, "calibration", None)
    if calibration is None or not calibration.is_valid:
        return None
    if calibration.method == "vanishing_point":
        return None

    try:
        drawer = PitchDrawer(pixels_per_meter=10.0)
        pitch = drawer.draw()

        team_colors = TEAM_BGR

        offside_result = getattr(result, "offside_result", None)
        attacking_side = (
            offside_result.attacking_side if offside_result is not None else None
        )
        offside_line = getattr(result, "offside_line", None)

        for idx, det in enumerate(result.detections):
            field_pt = None
            if attacking_side is not None:
                field_pt = project_detection_to_field(det, attacking_side, calibration)
            if field_pt is None:
                field_pt = calibration.project_to_field(det.bottom_center)
            if field_pt is None:
                continue
            fx, fy = field_pt
            if fx < -5 or fx > 110 or fy < -5 or fy > 73:
                continue
            color = team_colors.get(det.team_id, (128, 128, 128))
            label = "GK" if det.is_goalkeeper else None
            if offside_line is not None:
                if idx == offside_line.ignored_player_idx:
                    label = "IGN"
                elif idx == offside_line.defining_player_idx:
                    label = "LINE"
            drawer.draw_player(pitch, fx, fy, color=color, label=label)

        if offside_line is not None and offside_line.field_x is not None:
            is_offside = bool(
                offside_result.any_offside if offside_result is not None else False
            )
            line_color = (0, 0, 255) if is_offside else (0, 200, 0)
            drawer.draw_offside_line(
                pitch, offside_line.field_x, color=line_color, label="Offside"
            )
        return pitch
    except Exception as exc:  # noqa: BLE001 - stage is optional
        logger.warning("Bird's eye rendering skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# JSON conversion - contract shapes
# ---------------------------------------------------------------------------


def serialize_player(index: int, detection: PlayerDetection) -> Dict[str, Any]:
    """Convert one detection to the contract's ``Player`` shape."""
    role = "player"
    if getattr(detection, "is_referee", False):
        role = "referee"
    elif detection.is_goalkeeper:
        role = "goalkeeper"

    return {
        "id": index,
        "bbox": [float(value) for value in detection.bbox],
        "team": detection.team_id if detection.team_id in (1, 2) else None,
        "role": role,
        "confidence": float(detection.confidence),
    }


def serialize_players(result: AnalysisResult) -> List[Dict[str, Any]]:
    """Convert every detection to the contract's ``Player`` shape."""
    return [
        serialize_player(index, detection)
        for index, detection in enumerate(result.detections)
    ]


def serialize_phase1(
    result: AnalysisResult, stages: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Produce the contract's ``Phase1`` payload."""
    split = _team_split(result)

    checks = [
        _check(
            "players",
            "Players",
            "ok" if result.detections else "fail",
            str(len(result.detections)),
        ),
        _check(
            "teams",
            "Teams",
            "ok" if split["both_teams"] else "check",
            split["detail"],
        ),
        _check("side", "Attack direction", "check", "choose left or right"),
        _check("calibration", "Calibration", "pending", "next step"),
        _check("defender", "Last defender", "pending", "after calibration"),
        _check("verdict", "Verdict", "pending", "after calibration"),
    ]

    return {
        "players": serialize_players(result),
        "team_colors": _team_colors(result),
        "checks": checks,
        "stages": list(stages or ()),
        "time_ms": int(round(getattr(result, "processing_time_ms", 0.0) or 0.0)),
    }


def serialize_verdict(
    result: AnalysisResult, stages: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Produce the contract's ``Verdict`` payload."""
    calibration = getattr(result, "calibration", None)
    method, full_3d = _calibration_info(calibration)
    source = (getattr(calibration, "debug_info", None) or {}).get("source")
    offside_result = getattr(result, "offside_result", None)
    line = getattr(result, "offside_line", None)
    if line is None and offside_result is not None:
        line = getattr(offside_result, "offside_line", None)

    attacking_side = _attacking_side(result, offside_result)
    margin_m = _margin_m(result, offside_result, line, attacking_side)
    rejected = bool((getattr(calibration, "debug_info", None) or {}).get("rejected"))

    if offside_result is None or line is None:
        verdict = "unknown"
        offside_ids: List[int] = []
    else:
        verdict = "offside" if bool(offside_result.any_offside) else "onside"
        offside_ids = [int(index) for index in offside_result.offside_player_indices]

    estimate = verdict != "unknown" and (
        not full_3d or method == "vanishing_point"
    )
    if estimate:
        margin_m = None
    message = getattr(result, "verdict_message", None)
    if message is None and estimate:
        message = ESTIMATE_MESSAGE
        if rejected:
            message = f"The camera was rejected and not used. {ESTIMATE_MESSAGE}"

    last_defender = _last_defender(line, result)
    line_points = _line_image_points(line, result)

    split = _team_split(result)
    players_status = "ok" if result.detections else "fail"
    calibration_status = "ok" if method != "failed" else "fail"
    calibration_detail = method
    if method == "manual":
        how = "edited by hand" if source == "edited" else "placed by hand"
        calibration_detail = f"{how}, full 3D"
    elif method != "failed":
        calibration_detail = f"{method}, full 3D" if full_3d else f"{method}, 2D only"
    elif rejected:
        calibration_detail = "camera rejected, not used"
    else:
        calibration_detail = "calibration failed"
    defender_detail = "not found"
    if last_defender is not None and last_defender["id"] is not None:
        defender_detail = f"#{last_defender['id']}"
        if last_defender["part"]:
            defender_detail += f" ({last_defender['part']})"

    if verdict == "offside":
        verdict_status = "check"
        verdict_detail = "Offside (estimate)" if estimate else "Offside"
    elif verdict == "onside":
        verdict_status = "ok"
        verdict_detail = "Onside (estimate)" if estimate else "Onside"
    else:
        verdict_status = "pending"
        verdict_detail = "Unknown"

    checks = [
        _check("players", "Players", players_status, str(len(result.detections))),
        _check(
            "teams",
            "Teams",
            "ok" if split["both_teams"] else "check",
            split["detail"],
        ),
        _check(
            "side",
            "Attack direction",
            "ok" if attacking_side in ("left", "right") else "check",
            attacking_side or "not chosen",
        ),
        _check("calibration", "Calibration", calibration_status, calibration_detail),
        _check(
            "defender",
            "Last defender",
            "ok" if defender_detail != "not found" else "check",
            defender_detail,
        ),
        _check("verdict", "Verdict", verdict_status, verdict_detail),
    ]

    return {
        "verdict": verdict,
        "estimate": estimate,
        "message": message,
        "margin_m": margin_m,
        "attacking_side": attacking_side,
        "last_defender": last_defender,
        "offside_player_ids": offside_ids,
        "line_image_points": line_points,
        "line_field_x": (
            round(float(line.field_x), 3)
            if not estimate
            and line is not None
            and getattr(line, "field_x", None) is not None
            else None
        ),
        "calibration": {
            "method": method,
            "full_3d": full_3d,
            "edited": method == "manual",
        },
        "checks": checks,
        "stages": list(stages or ()),
        "time_ms": int(round(getattr(result, "processing_time_ms", 0.0) or 0.0)),
        "view3d_available": bool(
            calibration is not None
            and calibration.is_valid
            and offside_result is not None
            and getattr(result, "image", None) is not None
        ),
    }


def _check(key: str, label: str, status: str, detail: str) -> Dict[str, str]:
    """Build one contract ``Check`` entry."""
    return {"key": key, "label": label, "status": status, "detail": detail}


def _team_split(result: AnalysisResult) -> Dict[str, Any]:
    """Return the ``"10 / 8 + GK"`` summary and whether both teams have players."""
    counts = {1: 0, 2: 0}
    keepers = {1: False, 2: False}
    for detection in result.detections:
        if detection.team_id in (1, 2):
            counts[detection.team_id] += 1
            if detection.is_goalkeeper:
                keepers[detection.team_id] = True

    parts = []
    for team_id in (1, 2):
        part = str(counts[team_id])
        if keepers[team_id]:
            part += " + GK"
        parts.append(part)

    return {
        "detail": " / ".join(parts),
        "both_teams": counts[1] > 0 and counts[2] > 0,
    }


def _team_colors(result: AnalysisResult) -> Dict[str, str]:
    """Fixed team colours as ``#rrggbb``: Team 1 red, Team 2 blue."""
    return {str(team_id): color for team_id, color in TEAM_HEX.items()}


def _calibration_info(calibration: Any) -> Tuple[str, bool]:
    """Map a ``CalibrationResult`` to the contract's method/``full_3d`` pair."""
    if calibration is None or not getattr(calibration, "is_valid", False):
        return "failed", False
    full_3d = bool(getattr(calibration, "has_3d_calibration", False))
    method = getattr(calibration, "method", "") or ""
    if method == "vanishing_point":
        return "vanishing_point", full_3d
    if method == "manual":
        return "manual", full_3d
    return "pnl", full_3d


def _attacking_side(result: AnalysisResult, offside_result: Any) -> Optional[str]:
    """Return the attacking side from the offside result or side result."""
    side = getattr(offside_result, "attacking_side", None)
    if side in ("left", "right"):
        return side
    side_result = getattr(result, "side_result", None)
    team = getattr(result, "attacking_team", None)
    if side_result is not None and team in (1, 2):
        try:
            return side_result.get_attacking_side(team)
        except Exception:  # noqa: BLE001 - defensive for fake results
            return None
    return None


def _margin_m(
    result: AnalysisResult,
    offside_result: Any,
    line: Optional[OffsideLine],
    attacking_side: Optional[str],
) -> Optional[float]:
    """Return the signed metric margin (+ = attacker ahead) or ``None``.

    The contract only allows a numeric margin when a full 3D calibration is
    available.  Geometry runs without one leave the margin in pixels, which is
    not comparable across camera angles, so ``None`` is returned instead.
    """
    calibration = getattr(result, "calibration", None)
    if calibration is None or not getattr(calibration, "has_3d_calibration", False):
        return None
    if line is None or getattr(line, "field_x", None) is None:
        return None
    if getattr(calibration, "method", None) == "vanishing_point":
        return None
    if attacking_side not in ("left", "right"):
        return None
    if offside_result is None:
        return None

    sign = getattr(line, "attack_field_sign", None)
    if sign is None:
        sign = attack_field_sign(result.detections, attacking_side, calibration)

    line_field_x = float(line.field_x)
    margins: List[float] = []
    for status in offside_result.player_statuses:
        if getattr(status, "in_own_half", False):
            continue
        metric = getattr(status, "margin_m", None)
        if metric is not None:
            margins.append(float(metric))
            continue
        detection = getattr(status, "player_detection", None)
        if detection is None or sign is None:
            continue
        field_point = project_detection_to_field(
            detection, attacking_side, calibration
        )
        if field_point is None:
            continue
        margins.append((float(field_point[0]) - line_field_x) * sign)

    if not margins:
        return None
    # The furthest-forward attacker defines the reported margin.
    return round(max(margins), 3)


def _last_defender(
    line: Optional[OffsideLine], result: AnalysisResult
) -> Dict[str, Any]:
    """Return ``{"id": ..., "part": ...}`` for the line-setting defender."""
    if line is None:
        return {"id": None, "part": None}

    part = None
    body_part = getattr(line, "defining_body_part", None)
    if body_part is not None:
        # ``PlayableBodyPart`` exposes ``part_type.value`` ("left_shoulder").
        part_type = getattr(body_part, "part_type", None)
        part = getattr(part_type, "value", None) or getattr(body_part, "name", None)
        if isinstance(part, str):
            part = part.replace("_", " ").lower()

    return {"id": int(line.defining_player_idx), "part": part}


def _line_image_points(
    line: Optional[OffsideLine], result: AnalysisResult
) -> List[List[float]]:
    """Project the offside line into image space and clip it to the frame."""
    image = getattr(result, "image", None)
    if line is None or image is None:
        return []

    height, width = image.shape[:2]
    calibration = getattr(result, "calibration", None)
    reference = getattr(line, "reference_point", (line.image_x, float(height)))
    try:
        points = project_line(
            getattr(line, "field_x", None),
            float(line.image_x),
            float(reference[1]) if reference is not None else float(height),
            height,
            width,
            calibration,
        )
    except Exception as exc:  # noqa: BLE001 - the field is cosmetic
        logger.warning("Line projection failed: %s", exc)
        points = [(float(line.image_x), 0.0), (float(line.image_x), float(height))]

    if len(points) < 2:
        return []
    clipped = _clip_segment(points[0], points[-1], width, height)
    if clipped is None:
        return []
    return [[round(float(x), 2), round(float(y), 2)] for x, y in clipped]


def _clip_segment(
    first: Sequence[float],
    second: Sequence[float],
    width: int,
    height: int,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Liang-Barsky clip of one segment to the image rectangle."""
    x0, y0 = float(first[0]), float(first[1])
    x1, y1 = float(second[0]), float(second[1])
    dx = x1 - x0
    dy = y1 - y0
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))

    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0),
        (dx, max_x - x0),
        (-dy, y0),
        (dy, max_y - y0),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)

    if t0 > t1:
        return None
    return (
        (x0 + t0 * dx, y0 + t0 * dy),
        (x0 + t1 * dx, y0 + t1 * dy),
    )
