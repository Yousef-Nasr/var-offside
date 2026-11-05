"""
Service helpers for exposing the offside pipeline over HTTP.
"""

from threading import Lock
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from var_offside.offside.line_calculator import get_directional_ground_point
from var_offside.pipeline import OffsidePipeline


_PIPELINES: Dict[Tuple[str, Optional[str]], OffsidePipeline] = {}
_PIPELINE_LOCK = Lock()


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR numpy array."""
    if not image_bytes:
        raise ValueError("No image bytes were provided")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image")
    return image


def get_pipeline(
    calibration_method: str = "auto", device: Optional[str] = None
) -> OffsidePipeline:
    """Reuse heavy pipeline instances between API requests."""
    key = (calibration_method, device)
    with _PIPELINE_LOCK:
        pipeline = _PIPELINES.get(key)
        if pipeline is None:
            pipeline = OffsidePipeline(
                calibration_method=calibration_method, device=device
            )
            _PIPELINES[key] = pipeline
        return pipeline


def analyze_image_bytes(
    image_bytes: bytes,
    attacking_side: Optional[str],
    calibration_method: str = "auto",
    device: Optional[str] = None,
) -> dict:
    """Run the offside pipeline on uploaded image bytes and return JSON-safe data."""
    image = decode_image_bytes(image_bytes)
    pipeline = get_pipeline(calibration_method=calibration_method, device=device)

    result = pipeline.analyze(
        image,
        attacking_side=attacking_side,
        generate_visualization=False,
        save_result=False,
    )

    return serialize_analysis_result(result)


def serialize_analysis_result(result) -> dict:
    """Convert an AnalysisResult into a Telegram- and API-friendly payload."""
    offside_result = result.offside_result
    offside_line = result.offside_line
    calibration = result.calibration

    line_player = None
    if offside_line is not None and 0 <= offside_line.defining_player_idx < len(
        result.detections
    ):
        player = result.detections[offside_line.defining_player_idx]
        line_player = {
            "player_index": offside_line.defining_player_idx,
            "team_id": player.team_id,
            "is_goalkeeper": player.is_goalkeeper,
            "bbox": [float(v) for v in player.bbox],
            "ground_point": [float(v) for v in offside_line.reference_point],
            "line_image_x": float(offside_line.image_x),
            "line_field_x": float(offside_line.field_x)
            if offside_line.field_x is not None
            else None,
            "body_part": (
                offside_line.defining_body_part.part_type.value
                if offside_line.defining_body_part is not None
                else None
            ),
            "confidence": float(offside_line.confidence),
        }

    players = []
    ahead_players = []
    if offside_result is not None:
        for status in offside_result.player_statuses:
            player_payload = {
                "player_index": status.player_idx,
                "team_id": status.player_detection.team_id,
                "bbox": [float(v) for v in status.player_detection.bbox],
                "ground_point": [
                    float(v)
                    for v in get_directional_ground_point(
                        status.player_detection,
                        offside_result.attacking_side,
                    )
                ],
                "ahead_of_line": bool(status.is_offside),
                "margin_px": float(status.margin_px),
                "advanced_x": float(status.advanced_x),
                "body_part": (
                    status.most_advanced_part.part_type.value
                    if status.most_advanced_part is not None
                    else None
                ),
                "confidence": float(status.confidence),
            }
            players.append(player_payload)
            if status.is_offside:
                ahead_players.append(player_payload)

    return {
        "has_players_ahead_of_line": bool(result.is_offside),
        "players_detected": int(result.num_players_detected),
        "processing_time_ms": float(result.processing_time_ms),
        "errors": list(result.errors),
        "calibration": {
            "method": calibration.method if calibration is not None else None,
            "is_valid": bool(calibration.is_valid)
            if calibration is not None
            else False,
            "quality_score": float(calibration.quality_score)
            if calibration is not None
            else 0.0,
        },
        "analysis": {
            "attacking_side": offside_result.attacking_side
            if offside_result is not None
            else None,
            "attacking_team": result.attacking_team,
            "defending_team": result.defending_team,
            "ahead_of_line_count": offside_result.num_ahead
            if offside_result is not None
            else 0,
            "players_ahead_of_line": ahead_players,
            "evaluated_players": players,
            "line_player": line_player,
        },
    }
