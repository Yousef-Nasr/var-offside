"""
3D mesh-based offside method (fallback), ported from the Hugging Face
``mirrash7/VAR`` pipeline (``pipeline/geometry.py``).

The default geometry method draws the offside line from each player's
bounding-box bottom corner projected through the field homography.  This module
instead reconstructs a 3D body mesh per player (SAM 3D Body) and measures the
furthest-forward *body* point -- arms and hands excluded, since offside is not
judged on them -- in the attack direction.  The offside line is placed at the
furthest-forward defender and every attacker's margin is reported in metres.

It is heavier than the geometry method (needs SAM 3D Body meshes), so it is used
as an opt-in / automatic fallback -- e.g. when the homography projection is
unavailable or when higher fidelity is wanted.

Reference algorithm (verbatim ideas kept):
  * ``non_arm_mask``     -- label each vertex by its nearest keypoint and drop
                            arm/hand vertices.
  * furthest-forward     -- the offside-relevant point of a player is the body
                            vertex reaching furthest in the attack direction.
  * ``offside_plane_x``  -- the line sits at the furthest-forward marked
                            defender; every attacker past it is offside.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from var_offside.calibration.base import CalibrationResult
from var_offside.detection.player_detector import PlayerDetection
from var_offside.offside.line_calculator import (
    OffsideLine,
    get_directional_ground_point,
    project_detection_to_field,
)
from var_offside.offside.decision import OffsideResult, PlayerOffsideStatus

logger = logging.getLogger(__name__)

# Keypoint indices whose nearest vertices are arms/hands (MHR / SAM-3D-Body
# layout, taken verbatim from the reference pipeline).  Offside is not called on
# the arms, so these vertices are excluded when measuring the furthest point.
_ARM_KP = frozenset({7, 8, 41, 62} | set(range(21, 41)) | set(range(42, 62)))

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
_HALF_L = PITCH_LENGTH / 2.0
_HALF_W = PITCH_WIDTH / 2.0


def non_arm_mask(
    vertices: np.ndarray, keypoints_3d: Optional[np.ndarray]
) -> np.ndarray:
    """Per-vertex boolean mask: True for body vertices, False for arm/hand.

    Each vertex is labelled by its nearest keypoint.  Falls back to all-True
    (no exclusion) if keypoints are missing or malformed.
    """
    V = np.asarray(vertices)
    if keypoints_3d is None:
        return np.ones(len(V), bool)
    kp = np.asarray(keypoints_3d)
    if kp.ndim != 2 or kp.shape[0] < 15:
        return np.ones(len(V), bool)
    kp = kp[:, :3]
    arm = [i for i in _ARM_KP if i < len(kp)]
    keep = [i for i in range(len(kp)) if i not in _ARM_KP]
    if not arm or not keep:
        return np.ones(len(V), bool)
    d_arm = np.linalg.norm(V[:, None, :] - kp[arm][None], axis=2).min(1)
    d_keep = np.linalg.norm(V[:, None, :] - kp[keep][None], axis=2).min(1)
    return d_keep <= d_arm


def _forward_field_x(
    field_x_verts: np.ndarray, attack_sign: float, mask: Optional[np.ndarray] = None
) -> float:
    """Furthest-forward field X (metres) in the attack direction over verts."""
    sel = field_x_verts[mask] if (mask is not None and mask.any()) else field_x_verts
    return float(attack_sign * (attack_sign * sel).max())


class MeshOffsideCalculator:
    """3D mesh-based offside line + decision (HF ``mirrash7/VAR`` method).

    Usage::

        calc = MeshOffsideCalculator(device="cuda")
        line, result = calc.compute(image, detections, "right", calibration,
                                    defending_team=2, attacking_team=1)

    Returns ``(None, None)`` when it cannot run (no field projection, no
    recoverable meshes, or no defender to anchor the line).
    """

    def __init__(
        self,
        device: str = "cuda",
        margin_tolerance_m: float = 0.0,
        too_close_m: float = 0.30,
    ):
        self.device = device
        # Any positive margin flags offside (matches the reference pipeline);
        # raise this to require a minimum metric margin before flagging.
        self.margin_tolerance_m = margin_tolerance_m
        self.too_close_m = too_close_m

    # -- player selection ---------------------------------------------------

    @staticmethod
    def _select_defenders_attackers(
        detections: List[PlayerDetection],
        attacking_team: Optional[int],
        defending_team: Optional[int],
        exclude_goalkeeper: bool,
        line_defender_idx: Optional[int],
    ) -> Tuple[List[int], List[int]]:
        """Split detections into marked defenders and evaluated attackers."""

        def is_gk(i: int) -> bool:
            return exclude_goalkeeper and detections[i].is_goalkeeper

        if defending_team in (1, 2):
            defenders = [
                i
                for i, d in enumerate(detections)
                if d.team_id == defending_team and not is_gk(i)
            ]
            if attacking_team in (1, 2):
                attackers = [
                    i for i, d in enumerate(detections) if d.team_id == attacking_team
                ]
            else:
                attackers = [
                    i
                    for i, d in enumerate(detections)
                    if d.team_id != defending_team and not is_gk(i)
                ]
            return defenders, attackers

        # No team context: anchor on the geometry line-defining player and treat
        # everyone else as an attacker to evaluate.
        if line_defender_idx is not None and 0 <= line_defender_idx < len(detections):
            defenders = [line_defender_idx]
            attackers = [
                i
                for i in range(len(detections))
                if i != line_defender_idx and not is_gk(i)
            ]
            return defenders, attackers

        return [], []

    # -- placement ----------------------------------------------------------

    def _place_meshes_world(
        self,
        meshes: list,
        detections: List[PlayerDetection],
        attacking_side: str,
        calibration: CalibrationResult,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Return ``{player_idx: (field_x_verts, non_arm_mask)}`` in metres.

        Each mesh is oriented upright and grounded (shared rotation from the
        renderer) and planted at its calibrated foot field position, so vertex
        X in centred world coords maps to absolute field X = X + 52.5.
        """
        from var_offside.visualization.var_3d import VAR3DRenderer

        rot = VAR3DRenderer._compute_upright_rotation(meshes)
        placed: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for pm in meshes:
            idx = pm.player_idx
            if idx >= len(detections):
                continue
            det = detections[idx]
            fp = project_detection_to_field(det, attacking_side, calibration)
            if fp is None:
                continue
            wx = float(np.clip(fp[0] - _HALF_L, -_HALF_L, _HALF_L))
            wy = float(np.clip(fp[1] - _HALF_W, -_HALF_W, _HALF_W))

            v_world = VAR3DRenderer._orient_and_plant_mesh(pm, rot, wx, wy)
            field_x_verts = v_world[:, 0] + _HALF_L  # absolute field X (metres)
            mask = non_arm_mask(pm.vertices, pm.keypoints_3d)
            placed[idx] = (field_x_verts, mask)

        return placed

    # -- main entry ---------------------------------------------------------

    def compute(
        self,
        image,
        detections: List[PlayerDetection],
        attacking_side: str,
        calibration: Optional[CalibrationResult],
        exclude_goalkeeper: bool = True,
        attacking_team: Optional[int] = None,
        defending_team: Optional[int] = None,
        line_defender_idx: Optional[int] = None,
    ) -> Tuple[Optional[OffsideLine], Optional[OffsideResult]]:
        """Compute the offside line and decision from 3D player meshes."""
        if calibration is None or not calibration.is_valid:
            logger.info("Mesh offside: no valid calibration -- cannot run")
            return None, None
        if attacking_side not in ("left", "right"):
            logger.info("Mesh offside: attacking_side must be left/right")
            return None, None
        if not detections:
            return None, None

        # Require field projection (bbox foot -> field metres) to anchor players.
        # The vanishing-point calibrator deliberately returns None here, so the
        # mesh method cannot run on it.
        if calibration.method == "vanishing_point":
            logger.info("Mesh offside: field projection unavailable (VP calibration)")
            return None, None

        defenders, attackers = self._select_defenders_attackers(
            detections,
            attacking_team,
            defending_team,
            exclude_goalkeeper,
            line_defender_idx,
        )
        if not defenders:
            logger.info("Mesh offside: no defenders to anchor the line")
            return None, None

        want = sorted(set(defenders) | set(attackers))
        bboxes = {i: tuple(float(v) for v in detections[i].bbox) for i in want}

        meshes = self._recover_meshes(image, bboxes)
        if not meshes:
            logger.info("Mesh offside: no meshes recovered")
            return None, None

        placed = self._place_meshes_world(
            meshes, detections, attacking_side, calibration
        )
        if not placed:
            logger.info("Mesh offside: no meshes could be placed on the pitch")
            return None, None

        attack_sign = 1.0 if attacking_side == "right" else -1.0

        # Offside line = furthest-forward marked defender (arms excluded).
        def_fwd: Dict[int, float] = {}
        for i in defenders:
            if i in placed:
                fx, mask = placed[i]
                def_fwd[i] = _forward_field_x(fx, attack_sign, mask)
        if not def_fwd:
            logger.info("Mesh offside: no defender meshes placed")
            return None, None

        line_defender = max(def_fwd, key=lambda i: attack_sign * def_fwd[i])
        line_field_x = def_fwd[line_defender]

        offside_line = self._build_line(
            line_field_x, line_defender, detections, attacking_side, calibration
        )

        result = OffsideResult(
            offside_line=offside_line,
            attacking_side=attacking_side,
            attacking_team=attacking_team,
            defending_team=defending_team,
        )

        for i in attackers:
            if i not in placed:
                continue
            fx, mask = placed[i]
            fwd = _forward_field_x(fx, attack_sign, mask)
            margin_m = attack_sign * (fwd - line_field_x)
            is_off = margin_m > self.margin_tolerance_m
            result.player_statuses.append(
                PlayerOffsideStatus(
                    player_idx=i,
                    player_detection=detections[i],
                    is_offside=is_off,
                    most_advanced_part=None,
                    advanced_x=float(fwd),
                    # Keep metric ordering usable via margin_px (metres here);
                    # margin_m is the authoritative value for this method.
                    margin_px=float(margin_m),
                    margin_m=float(margin_m),
                    confidence=float(detections[i].confidence),
                )
            )
            if is_off:
                result.offside_player_indices.append(i)

        result.any_offside = len(result.offside_player_indices) > 0
        logger.info(
            "Mesh offside: line at field_x=%.2fm (defender #%d), %d/%d attackers offside",
            line_field_x,
            line_defender,
            len(result.offside_player_indices),
            len(result.player_statuses),
        )
        return offside_line, result

    # -- helpers ------------------------------------------------------------

    def _recover_meshes(self, image, bboxes: Dict[int, Tuple]) -> list:
        """Reconstruct SAM 3D Body meshes for the requested players."""
        from var_offside.config import get_settings
        from var_offside.visualization.var_3d import MeshRecovery

        cfg = get_settings().var_3d
        img_w = None
        if isinstance(image, np.ndarray):
            img_w = int(image.shape[1])
        try:
            recovery = MeshRecovery(
                checkpoint_path=cfg.sam3d_checkpoint,
                config_path=cfg.sam3d_config,
                yolo_path=cfg.sam3d_yolo,
                mhr_path=cfg.sam3d_mhr,
                device=self.device,
            )
            if img_w is None:
                # image is a path; use max bbox extent as a width hint
                img_w = int(max((b[2] for b in bboxes.values()), default=1920))
            return recovery.recover(image, bboxes, image_width=img_w)
        except Exception as exc:  # noqa: BLE001 - surface as "unavailable"
            logger.warning("Mesh offside: mesh recovery failed: %s", exc, exc_info=True)
            return []

    @staticmethod
    def _build_line(
        line_field_x: float,
        defining_idx: int,
        detections: List[PlayerDetection],
        attacking_side: str,
        calibration: CalibrationResult,
    ) -> OffsideLine:
        """Assemble an OffsideLine anchored at the metric line position."""
        reference_point = get_directional_ground_point(
            detections[defining_idx], attacking_side
        )
        image_x = float(reference_point[0])
        # Prefer projecting the metric line back to the image for a stable anchor.
        img_pt = calibration.project_to_image((line_field_x, _HALF_W))
        if img_pt is not None and np.isfinite(img_pt[0]):
            image_x = float(img_pt[0])
            reference_point = (image_x, float(reference_point[1]))

        return OffsideLine(
            field_x=float(line_field_x),
            image_x=image_x,
            reference_point=reference_point,
            defining_player_idx=defining_idx,
            defining_body_part=None,
            confidence=float(detections[defining_idx].confidence),
        )
