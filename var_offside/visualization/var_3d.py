"""
3D VAR-style offside visualization.

Reconstructs key players as 3D meshes using SAM 3D Body, places them on a
pitch plane with an offside plane, and renders a rotating-camera video.

Can run standalone:
    python -m var_offside.visualization.var_3d --image path/to/frame.jpg
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Suppress harmless "No OpenGL_accelerate module loaded" warning
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class RenderCancelled(Exception):
    """Raised when a running 3D render is asked to stop."""


class _CancellableWriter:
    """Wrap a ``cv2.VideoWriter`` and stop before writing a frame on cancel."""

    def __init__(self, writer, should_stop: Optional[Callable[[], bool]] = None):
        self._writer = writer
        self._should_stop = should_stop

    def write(self, frame: np.ndarray) -> None:
        if self._should_stop is not None and self._should_stop():
            raise RenderCancelled("3D render cancelled")
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

# ---------------------------------------------------------------------------
# SAM 3D Body helpers (avoid `from utils import ...` â€” collides with other
# packages named "utils" on sys.path)
# ---------------------------------------------------------------------------


def _get_setup_function() -> Callable[..., Any]:
    """Return the SAM 3D Body setup function.

    It uses the Fast-SAM-3D-Body clone in ``third_party/`` when present (about
    10x faster) and falls back to the original ``sam-3d-body`` clone.
    """
    from var_offside.visualization.sam3d_setup import setup_sam3d_estimator

    return setup_sam3d_estimator


# ---------------------------------------------------------------------------
# Lazy-import helpers (so the module can be imported even without trimesh /
# pyrender / SAM-3D-Body installed)
# ---------------------------------------------------------------------------


def _require_trimesh():
    try:
        import trimesh

        return trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for 3D VAR rendering.  pip install trimesh"
        )


def _require_pyrender():
    # pyrender needs an OpenGL context.  On headless Linux use EGL;
    # on Windows the native context works.
    if sys.platform != "win32" and "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    try:
        import pyrender

        return pyrender
    except ImportError:
        raise ImportError(
            "pyrender is required for 3D VAR rendering.  pip install pyrender"
        )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PlayerMesh:
    """3D mesh data for a single player."""

    player_idx: int
    team_id: Optional[int]
    is_offside: bool
    vertices: np.ndarray  # (V, 3) mesh vertices in camera space
    faces: np.ndarray  # (F, 3) triangle indices
    cam_translation: np.ndarray  # (3,) camera-space translation
    focal_length: float
    bbox: Tuple[float, float, float, float]
    image_center_x: float  # player center-x in the image (pixels)
    image_width: int  # image width (for normalising position)
    # Authoritative field position (metres, absolute 0-105 x 0-68) for this
    # player's ground point, computed the same way as the offside decision.
    # When set, the renderer plants the mesh here instead of re-projecting the
    # bbox centre (which disagreed with the 2D verdict).
    field_pos: Optional[Tuple[float, float]] = None
    # Optional 3D keypoints (V, 3) in the same local frame as ``vertices``.
    # Used by the mesh-based offside method to exclude arm/hand vertices.
    keypoints_3d: Optional[np.ndarray] = None


# PnLCalib's world is centred on the pitch with +Z pointing *down* (the
# crossbar sits at z=-2.44), while the 3D scene is Z-up.  A 180° turn about X
# maps one onto the other without mirroring: scene = (x, -y, -z).  Flipping Z
# alone would be a reflection and render the broadcast view mirrored.
_PNL_TO_SCENE = np.diag([1.0, -1.0, -1.0])


def _field_to_scene(field_x: float, field_y: float) -> Tuple[float, float]:
    """Absolute field metres (0-105, 0-68) -> centred Z-up scene (x, y)."""
    return field_x - 52.5, 34.0 - field_y


def _slerp_rotation(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two 3x3 rotation matrices."""
    from scipy.spatial.transform import Rotation, Slerp

    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([r0, r1])))
    return slerp([float(t)]).as_matrix()[0]


_PROXY_BODY_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


def _estimate_focal_length(
    image_width: int,
    image_height: int,
    calibration=None,
) -> float:
    """Estimate a usable focal length for proxy geometry."""
    if (
        calibration is not None
        and getattr(calibration, "camera_matrix", None) is not None
    ):
        K = np.asarray(calibration.camera_matrix, dtype=np.float64)
        if K.shape == (3, 3):
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            if np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0:
                return (fx + fy) * 0.5

    return float(max(image_width, image_height) * 1.35)


def _build_proxy_player_mesh(
    player_idx: int,
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, ...],
    calibration=None,
) -> PlayerMesh:
    """Create a simple 3D body proxy from a 2D player box."""
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]

    bbox_w = max(x2 - x1, 1.0)
    bbox_h = max(y2 - y1, 1.0)
    focal_length = _estimate_focal_length(image_width, image_height, calibration)

    body_height = 1.8
    body_width = float(np.clip((bbox_w / bbox_h) * body_height, 0.35, 0.9))
    body_depth = max(body_width * 0.45, 0.2)

    depth = max(focal_length * body_height / bbox_h, 2.5)
    centre_x = (x1 + x2) * 0.5
    foot_y = y2
    tx = (centre_x - image_width * 0.5) * depth / focal_length
    ty = (foot_y - image_height * 0.5) * depth / focal_length

    half_w = body_width * 0.5
    half_d = body_depth * 0.5
    vertices = np.array(
        [
            [-half_w, -body_height, -half_d],
            [half_w, -body_height, -half_d],
            [half_w, 0.0, -half_d],
            [-half_w, 0.0, -half_d],
            [-half_w, -body_height, half_d],
            [half_w, -body_height, half_d],
            [half_w, 0.0, half_d],
            [-half_w, 0.0, half_d],
        ],
        dtype=np.float32,
    )

    return PlayerMesh(
        player_idx=player_idx,
        team_id=None,
        is_offside=False,
        vertices=vertices,
        faces=_PROXY_BODY_FACES,
        cam_translation=np.array([tx, ty, depth], dtype=np.float32),
        focal_length=focal_length,
        bbox=(x1, y1, x2, y2),
        image_center_x=centre_x,
        image_width=image_width,
    )


def _merge_with_proxy_meshes(
    recovered_meshes: List[PlayerMesh],
    player_bboxes: Dict[int, Tuple[float, float, float, float]],
    image_shape: Tuple[int, ...],
    calibration=None,
) -> List[PlayerMesh]:
    """Fill any mesh-recovery gaps with simple proxy player bodies."""
    mesh_by_idx = {pm.player_idx: pm for pm in recovered_meshes}
    missing_indices = []

    for player_idx, bbox in player_bboxes.items():
        if player_idx in mesh_by_idx:
            continue
        mesh_by_idx[player_idx] = _build_proxy_player_mesh(
            player_idx,
            bbox,
            image_shape,
            calibration=calibration,
        )
        missing_indices.append(player_idx)

    if missing_indices:
        logger.warning(
            "Using proxy 3D bodies for players without SAM meshes: %s",
            missing_indices,
        )

    return [
        mesh_by_idx[player_idx]
        for player_idx in player_bboxes
        if player_idx in mesh_by_idx
    ]


def _annotate_player_meshes(player_meshes: List[PlayerMesh], result) -> None:
    """
    Attach team id, offside flag and authoritative field position to each
    recovered mesh.

    The field position is computed with the *same* projection the offside
    decision uses (``project_detection_to_field`` on the direction-aligned
    ground point), so the 3D scene places players exactly where the 2D
    verdict says they are.
    """
    from var_offside.offside.line_calculator import project_detection_to_field

    offside_res = result.offside_result
    attacking_side = getattr(offside_res, "attacking_side", None)
    statuses = {st.player_idx: st for st in offside_res.player_statuses}

    for pm in player_meshes:
        if pm.player_idx >= len(result.detections):
            continue
        det = result.detections[pm.player_idx]
        pm.team_id = det.team_id
        st = statuses.get(pm.player_idx)
        if st is not None:
            pm.is_offside = st.is_offside
        if attacking_side in ("left", "right"):
            fp = project_detection_to_field(det, attacking_side, result.calibration)
            if fp is not None:
                pm.field_pos = (float(fp[0]), float(fp[1]))


# ---------------------------------------------------------------------------
# MeshRecovery â€” wraps SAM 3D Body / Fast-SAM-3D-Body
# ---------------------------------------------------------------------------

# Module-level cache of loaded estimators.  Loading the ~2 GB SAM-3D-Body
# checkpoint takes tens of seconds and a couple of GB of VRAM, so we keep one
# per (checkpoint, device) and reuse it across every render in the process
# (e.g. when the GUI generates several replays in one session).
_ESTIMATOR_CACHE: Dict[Tuple[str, Optional[str], str, str], Any] = {}


# ---------------------------------------------------------------------------
# Reusable calibrated-pitch construction (shared by the cinematic video and
# the live web view, so both show exactly the same pitch)
# ---------------------------------------------------------------------------


def build_pitch_meshes(
    trimesh,
    visible_region: Tuple[float, float, float, float],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> List[Any]:
    """
    Build the Z-up pitch plane and its white markings as trimesh objects.

    Shared by :func:`add_pitch_surface_and_markings` (video scene, one node
    per marking) and the live web view (which merges them into one mesh for
    speed).  World frame: pitch centre at the origin, ``Z`` up, markings at
    ``z=0.01``.

    Args:
        trimesh: The imported ``trimesh`` module.
        visible_region: ``(x_min, x_max, y_min, y_max)`` in absolute field
            metres (0-105, 0-68); penalty markings outside it are skipped.
        pitch_length: Pitch length in metres (goal to goal).
        pitch_width: Pitch width in metres (touchline to touchline).

    Returns:
        List of ``trimesh.Trimesh`` with face colours set.
    """
    meshes: List[Any] = []
    half_l = pitch_length / 2.0
    half_w = pitch_width / 2.0
    vis_x_min, vis_x_max, _, _ = visible_region

    # Pitch surface (full pitch at Z=0)
    pitch_verts = np.array(
        [
            [-half_l, -half_w, 0],
            [half_l, -half_w, 0],
            [half_l, half_w, 0],
            [-half_l, half_w, 0],
        ],
        dtype=np.float64,
    )
    pitch_faces = np.array([[0, 1, 2], [0, 2, 3]])
    pitch_color = np.array([38, 96, 44, 255], dtype=np.uint8)  # grass green (RGBA)
    pm_tri = trimesh.Trimesh(vertices=pitch_verts, faces=pitch_faces, process=False)
    pm_tri.visual.face_colors = np.tile(pitch_color, (2, 1))
    meshes.append(pm_tri)

    # Pitch markings (lines at Z=0.01)
    lw = 0.15
    white = np.array([220, 220, 240, 255], dtype=np.uint8)

    def quad_line(pts, color=white):
        v = np.array(pts, dtype=np.float64)
        f = np.array([[0, 1, 2], [0, 2, 3]])
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        m.visual.face_colors = np.tile(color, (2, 1))
        return m

    z_l = 0.01  # line height
    # Sidelines & goal lines
    for sign in (-1, 1):
        y = sign * half_w
        meshes.append(
            quad_line(
                [
                    [-half_l, y - lw, z_l],
                    [half_l, y - lw, z_l],
                    [half_l, y + lw, z_l],
                    [-half_l, y + lw, z_l],
                ]
            )
        )
        x = sign * half_l
        meshes.append(
            quad_line(
                [
                    [x - lw, -half_w, z_l],
                    [x + lw, -half_w, z_l],
                    [x + lw, half_w, z_l],
                    [x - lw, half_w, z_l],
                ]
            )
        )

    # Centre line
    meshes.append(
        quad_line(
            [
                [-lw, -half_w, z_l],
                [lw, -half_w, z_l],
                [lw, half_w, z_l],
                [-lw, half_w, z_l],
            ]
        )
    )

    # Penalty area lines (in 0-105 coords then centred)
    penalty_lines = [
        # Left penalty area
        [0, 13.84, 16.5, 13.84],
        [16.5, 13.84, 16.5, 54.16],
        [0, 54.16, 16.5, 54.16],
        # Right penalty area
        [88.5, 13.84, 105, 13.84],
        [88.5, 13.84, 88.5, 54.16],
        [88.5, 54.16, 105, 54.16],
        # Left goal area
        [0, 24.84, 5.5, 24.84],
        [5.5, 24.84, 5.5, 43.16],
        [0, 43.16, 5.5, 43.16],
        # Right goal area
        [99.5, 24.84, 105, 24.84],
        [99.5, 24.84, 99.5, 43.16],
        [99.5, 43.16, 105, 43.16],
    ]
    for lx1, ly1, lx2, ly2 in penalty_lines:
        # Check if in or near visible region
        if max(lx1, lx2) < vis_x_min - 10 or min(lx1, lx2) > vis_x_max + 10:
            continue
        cx1, cy1 = lx1 - half_l, ly1 - half_w
        cx2, cy2 = lx2 - half_l, ly2 - half_w
        # Determine line orientation
        if abs(cx1 - cx2) < 0.1:  # vertical
            meshes.append(
                quad_line(
                    [
                        [cx1 - lw, min(cy1, cy2), z_l],
                        [cx1 + lw, min(cy1, cy2), z_l],
                        [cx1 + lw, max(cy1, cy2), z_l],
                        [cx1 - lw, max(cy1, cy2), z_l],
                    ]
                )
            )
        else:  # horizontal
            meshes.append(
                quad_line(
                    [
                        [min(cx1, cx2), cy1 - lw, z_l],
                        [max(cx1, cx2), cy1 - lw, z_l],
                        [max(cx1, cx2), cy1 + lw, z_l],
                        [min(cx1, cx2), cy1 + lw, z_l],
                    ]
                )
            )
    return meshes


def build_pitch_stripe_meshes(
    trimesh,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> List[Any]:
    """
    Build the alternating grass mow-stripe quads as trimesh objects.

    Args:
        trimesh: The imported ``trimesh`` module.
        pitch_length: Pitch length in metres.
        pitch_width: Pitch width in metres.

    Returns:
        List of ``trimesh.Trimesh`` with face colours set.
    """
    half_l = pitch_length / 2.0
    half_w = pitch_width / 2.0
    stripe_count = 10
    stripe_w = (2 * half_l) / stripe_count
    dark = np.array([34, 88, 40, 255], dtype=np.uint8)  # darker grass (RGBA)
    light = np.array([46, 112, 52, 255], dtype=np.uint8)  # lighter grass (RGBA)
    z_s = 0.005
    meshes: List[Any] = []
    for i in range(stripe_count):
        color = light if i % 2 == 0 else dark
        x0 = -half_l + i * stripe_w
        x1 = x0 + stripe_w
        verts = np.array(
            [
                [x0, -half_w, z_s],
                [x1, -half_w, z_s],
                [x1, half_w, z_s],
                [x0, half_w, z_s],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        m.visual.face_colors = np.tile(color, (2, 1))
        meshes.append(m)
    return meshes


def add_pitch_surface_and_markings(
    scene,
    trimesh,
    pyrender,
    visible_region: Tuple[float, float, float, float],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    merge: bool = False,
) -> None:
    """
    Add the Z-up pitch plane and its white markings to a pyrender scene.

    This is the reusable form of the pitch construction used by
    :meth:`VAR3DRenderer._build_scene_calibrated`: the same grass colour,
    line width, colours and penalty-area geometry, so live views and the
    cinematic video look identical.

    Args:
        scene: ``pyrender.Scene`` to add the geometry to.
        trimesh: The imported ``trimesh`` module.
        pyrender: The imported ``pyrender`` module.
        visible_region: ``(x_min, x_max, y_min, y_max)`` in absolute field
            metres (0-105, 0-68); penalty markings outside it are skipped.
        pitch_length: Pitch length in metres (goal to goal).
        pitch_width: Pitch width in metres (touchline to touchline).
        merge: When ``True`` all markings become one mesh node (faster live
            rendering); the default keeps one node per marking (video path).
    """
    meshes = build_pitch_meshes(
        trimesh, visible_region, pitch_length, pitch_width
    )
    if merge:
        combined = trimesh.util.concatenate(meshes)
        scene.add(pyrender.Mesh.from_trimesh(combined, smooth=False))
        return
    for m in meshes:
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))


def add_pitch_stripes(
    scene,
    trimesh,
    pyrender,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> None:
    """Add alternating grass stripe quads on top of the pitch surface."""
    for m in build_pitch_stripe_meshes(trimesh, pitch_length, pitch_width):
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))


def compute_visible_pitch_region(
    calibration,
    image_width: int,
    image_height: int,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Tuple[float, float, float, float]:
    """
    Field rectangle (absolute metres) visible in a broadcast frame.

    Projects the frame's bottom corners and a near-horizon row through the
    calibration homography and returns the padded, clamped field bounds.

    Args:
        calibration: ``CalibrationResult`` with ``project_to_field``.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        pitch_length: Pitch length in metres.
        pitch_width: Pitch width in metres.

    Returns:
        ``(x_min, x_max, y_min, y_max)`` in absolute field metres.
    """
    corners_img = [
        (0, image_height),  # bottom-left
        (image_width, image_height),  # bottom-right
        (0, image_height * 0.3),  # upper-left (~horizon)
        (image_width, image_height * 0.3),  # upper-right (~horizon)
    ]

    field_corners = []
    for cx, cy in corners_img:
        fp = calibration.project_to_field((cx, cy))
        if fp is not None:
            field_corners.append(fp)

    if field_corners:
        fc = np.array(field_corners)
        vis_x_min = max(0.0, fc[:, 0].min() - 5)
        vis_x_max = min(pitch_length, fc[:, 0].max() + 5)
        vis_y_min = max(0.0, fc[:, 1].min() - 5)
        vis_y_max = min(pitch_width, fc[:, 1].max() + 5)
        logger.info(
            "  Visible pitch region: x=[%.0f, %.0f] y=[%.0f, %.0f]",
            vis_x_min,
            vis_x_max,
            vis_y_min,
            vis_y_max,
        )
    else:
        # Fallback: show full pitch
        vis_x_min, vis_x_max = 0, pitch_length
        vis_y_min, vis_y_max = 0, pitch_width
        logger.warning("  Could not determine visible region - using full pitch")

    return (float(vis_x_min), float(vis_x_max), float(vis_y_min), float(vis_y_max))


class MeshRecovery:
    """
    Run SAM 3D Body (or Fast-SAM-3D-Body) on an image and return
    per-player meshes for the requested player indices.

    Key change vs. original: bboxes are passed **directly** to the
    estimator via ``process_one_image(bboxes=...)``, eliminating the
    unreliable IoU-matching step.
    """

    def __init__(
        self,
        checkpoint_path: str = "sam-3d-body-dinov3/model.ckpt",
        config_path: str = "sam-3d-body-dinov3/model_config.yaml",
        yolo_path: str = "yolo11l.pt",
        mhr_path: str = "sam-3d-body-dinov3/assets/mhr_model.pt",
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.yolo_path = yolo_path
        self.mhr_path = mhr_path
        self.device = device
        self._estimator = None
        self._faces = None

    def _resolve_paths(self) -> Tuple[str, Optional[str], Optional[str], str]:
        """Resolve weight paths through ``weights/``; verify the checkpoint."""
        from var_offside.config.paths import resolve_weight, weights_dir

        hint = (
            "Download it with `hf download facebook/sam-3d-body-dinov3 "
            f"--local-dir {weights_dir() / 'sam-3d-body-dinov3'}` "
            "(see docs/getting-started/models.md)."
        )

        ckpt = resolve_weight(self.checkpoint_path) or ""
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"SAM 3D Body checkpoint not found: {self.checkpoint_path!r}. {hint}"
            )

        cfg = resolve_weight(self.config_path) if self.config_path else None
        if not (cfg and os.path.isfile(cfg)):
            # The Hugging Face download keeps model_config.yaml next to model.ckpt.
            cfg = os.path.join(os.path.dirname(ckpt), "model_config.yaml")

        yolo = resolve_weight(self.yolo_path) if self.yolo_path else None

        mhr = resolve_weight(self.mhr_path) if self.mhr_path else self.mhr_path
        if mhr and not os.path.isfile(mhr):
            raise FileNotFoundError(
                f"SAM 3D Body MHR model not found: {self.mhr_path!r}. {hint}"
            )

        return (
            ckpt,
            cfg if cfg and os.path.isfile(cfg) else None,
            yolo if yolo and os.path.isfile(yolo) else None,
            mhr,
        )

    # Lazy init so import is fast; cached so the checkpoint loads only once.
    def _ensure_loaded(self):
        if self._estimator is not None:
            return

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is required overall
            raise RuntimeError(
                "SAM 3D Body needs PyTorch with CUDA; torch is not importable."
            ) from exc

        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                f"SAM 3D Body needs a CUDA GPU (device={self.device!r}); "
                "CUDA is not available or a non-CUDA device was requested. "
                "Use a CUDA device or the 2D/proxy fallback."
            )

        ckpt, cfg, yolo, mhr = self._resolve_paths()
        cache_key = (ckpt, cfg, mhr, self.device)
        cached = _ESTIMATOR_CACHE.get(cache_key)
        if cached is not None:
            logger.debug("MeshRecovery: reusing cached estimator for %s", cache_key)
            self._estimator = cached
            self._faces = cached.faces
            return

        setup_fn = _get_setup_function()
        logger.info("MeshRecovery: loading SAM-3D-Body checkpoint %s", ckpt)
        self._estimator = setup_fn(
            checkpoint_path=ckpt,
            config_path=cfg,
            mhr_path=mhr,
            yolo_path=yolo,
            device=self.device,
        )
        self._faces = self._estimator.faces
        _ESTIMATOR_CACHE[cache_key] = self._estimator

    @staticmethod
    def unload_all():
        """Drop cached estimators and free GPU memory."""
        _ESTIMATOR_CACHE.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch may be absent
            pass

    @property
    def faces(self) -> np.ndarray:
        self._ensure_loaded()
        return self._faces

    def recover(
        self,
        image: Union[str, np.ndarray],
        player_bboxes: Dict[int, Tuple[float, float, float, float]],
        image_width: int,
        bbox_thr: float = 0.3,
    ) -> List[PlayerMesh]:
        """
        Run mesh recovery on the image for specific players.

        The player bboxes are passed **directly** to
        ``process_one_image(bboxes=...)`` â€” one mesh output per bbox,
        in the same order.  No IoU matching needed.

        Args:
            image: Path to the image file, **or** a BGR ``np.ndarray`` (as
                produced by ``cv2.imread`` / the GUI).  Arrays are converted
                to the RGB layout the estimator expects.
            player_bboxes: {player_idx: (x1,y1,x2,y2)} for players we want.
            image_width: Width of the image in pixels.
            bbox_thr: Detection confidence threshold (unused when bboxes
                      are passed directly, kept for API compat).

        Returns:
            List of PlayerMesh (one per successfully recovered player).
        """
        self._ensure_loaded()

        # The estimator accepts a path (loaded as BGR) or an ndarray (assumed
        # RGB).  Convert an incoming BGR frame so colours/orientation match.
        if isinstance(image, np.ndarray):
            estimator_input: Any = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            src_desc = f"<ndarray {image.shape}>"
        else:
            estimator_input = image
            src_desc = str(image)

        # Build the bboxes array in a deterministic order
        ordered_indices = list(player_bboxes.keys())
        bbox_array = np.array(
            [player_bboxes[idx] for idx in ordered_indices],
            dtype=np.float32,
        ).reshape(-1, 4)

        logger.info(
            "Recovering meshes for %d player(s): indices %s",
            len(ordered_indices),
            ordered_indices,
        )
        logger.debug("Bboxes:\n%s", bbox_array)

        # Pass bboxes directly â€” no YOLO re-detection, no IoU matching
        outputs = self._estimator.process_one_image(
            estimator_input,
            bboxes=bbox_array,
            bbox_thr=bbox_thr,
        )

        if not outputs:
            logger.warning("SAM 3D Body returned no outputs for %s", src_desc)
            return []

        logger.info(
            "SAM 3D Body returned %d output(s) for %d bbox(es)",
            len(outputs),
            len(ordered_indices),
        )

        matched: List[PlayerMesh] = []
        for i, out in enumerate(outputs):
            if i >= len(ordered_indices):
                break
            pidx = ordered_indices[i]
            bbox = player_bboxes[pidx]
            cx = (bbox[0] + bbox[2]) / 2.0

            verts = out["pred_vertices"]
            cam_t = out["pred_cam_t"]

            # Sanity check
            if verts is None or np.any(np.isnan(verts)):
                logger.warning("Player %d: invalid vertices (NaN), skipping", pidx)
                continue
            if cam_t is None or np.any(np.isnan(cam_t)):
                logger.warning("Player %d: invalid cam_t (NaN), skipping", pidx)
                continue

            pm = PlayerMesh(
                player_idx=pidx,
                team_id=None,
                is_offside=False,
                vertices=verts,
                faces=self._faces,
                cam_translation=cam_t,
                focal_length=float(out["focal_length"]),
                bbox=bbox,
                image_center_x=cx,
                image_width=image_width,
                keypoints_3d=out.get("pred_keypoints_3d"),
            )
            matched.append(pm)
            logger.info(
                "  Player %d: verts=%s, cam_t=%s",
                pidx,
                verts.shape,
                cam_t,
            )

        logger.info("Successfully recovered %d mesh(es)", len(matched))
        return matched


# ---------------------------------------------------------------------------
# VAR3DRenderer â€” multi-phase scene building & video output
# ---------------------------------------------------------------------------


class VAR3DRenderer:
    """
    Build a broadcast-style VAR 3D replay video:
        Phase 1 â€” Hold the annotated 2D image
        Phase 2 â€” Crossfade into 3D pitch scene
        Phase 3 â€” Camera sweeps from high angle down to near-perpendicular view
    """

    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    # 3D jersey colours (RGB 0-1) — vivid blue vs red per team
    TEAM1_COLOR = (0.12, 0.30, 0.95)  # blue jersey (team 1)
    TEAM2_COLOR = (0.90, 0.12, 0.12)  # red jersey (team 2)
    HIGHLIGHT_COLOR = (1.0, 0.85, 0.0)  # amber accent for offside player

    # 2D overlay / bird's-eye marker colours (BGR, matching the app settings)
    TEAM1_BGR = (255, 40, 40)  # blue
    TEAM2_BGR = (40, 40, 255)  # red
    GK_BGR = (0, 220, 255)  # amber/yellow for goalkeepers

    def __init__(
        self,
        video_width: int = 1280,
        video_height: int = 720,
        duration_sec: float = 10.0,
        fps: int = 30,
        camera_elevation_deg: float = 35.0,
        camera_distance: float = 12.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.video_width = video_width
        self.video_height = video_height
        self.duration_sec = duration_sec
        self.fps = fps
        self.camera_elevation_deg = camera_elevation_deg
        self.camera_distance = camera_distance
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    # ---- public API -------------------------------------------------------

    def render(
        self,
        player_meshes: List[PlayerMesh],
        offside_x_norm: float,
        is_offside: bool,
        output_path: str,
        original_image: Optional[np.ndarray] = None,
        annotated_image: Optional[np.ndarray] = None,
        calibration=None,
        offside_field_x: Optional[float] = None,
        attacking_side: Optional[str] = None,
        attacker_idx: Optional[int] = None,
        defender_idx: Optional[int] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> str:
        """
        Render a multi-phase 3D VAR video.

        Phases:
            1. Original image with 3D mesh overlay on players (3s)
            2. Crossfade transition to 3D pitch scene (1s)
            3. Camera sweeps from high angle down to near-perpendicular
            4. Low bird's-eye over the key players and the wall
            5. Side-on view exactly along the offside line
            6. Freeze with the animated OFFSIDE / ONSIDE verdict
        """
        trimesh = _require_trimesh()
        pyrender = _require_pyrender()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        raw_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.video_width, self.video_height),
        )
        if not raw_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")
        writer = _CancellableWriter(raw_writer, should_stop)

        renderer = None
        try:
            # --- Phase durations ---
            has_image = original_image is not None or annotated_image is not None
            overlay_hold_sec = 3.0 if has_image else 0.0
            transition_sec = 1.0 if has_image else 0.0
            sweep_sec = max(self.duration_sec - overlay_hold_sec - transition_sec, 3.0)

            overlay_frames = int(overlay_hold_sec * self.fps)
            transition_frames = int(transition_sec * self.fps)
            sweep_frames = int(sweep_sec * self.fps)

            # ---- Phase 1: Original image with 3D mesh overlay ----
            frame_overlay = None
            if has_image:
                base_img = original_image if original_image is not None else annotated_image
                if base_img is not None:
                    frame_overlay = self._render_mesh_on_image(
                        base_img,
                        player_meshes,
                        trimesh,
                        pyrender,
                        is_offside,
                    )
                    frame_overlay = self._resize_frame(frame_overlay)

                logger.info(
                    "Phase 1: Mesh overlay on original image (%d frames)", overlay_frames
                )
                if frame_overlay is not None:
                    for _ in range(overlay_frames):
                        writer.write(frame_overlay)

            # ==================================================================
            #  Build 3D pitch scene
            #  Three calibration levels:
            #    A) Full PnL (K + R + C) â†' real-world pitch, calibrated camera
            #    B) Homography-only â†' real-world pitch, estimated camera
            #    C) No calibration â†' SAM camera-space fallback
            # ==================================================================

            has_full_3d = (
                calibration is not None
                and calibration.has_3d_calibration
                and calibration.camera_matrix is not None
                and calibration.rotation_matrix is not None
                and calibration.position is not None
            )
            has_homography = (
                calibration is not None
                and calibration.homography is not None
                and not has_full_3d
                and getattr(calibration, "method", None) != "vanishing_point"
            )
            z_up = has_full_3d or has_homography

            fl = player_meshes[0].focal_length
            img_w = player_meshes[0].image_width

            scene = pyrender.Scene(
                bg_color=np.array([0.03, 0.07, 0.04, 1.0]),  # dark stadium green
                ambient_light=np.array([0.32, 0.34, 0.32]),
            )

            if has_full_3d:
                logger.info("Using PnL 3D calibration (K/R/C) for pitch reconstruction")
                cam_base_pose, cam_obj, focus_positions = self._build_scene_calibrated(
                    scene,
                    trimesh,
                    pyrender,
                    player_meshes,
                    calibration,
                    offside_x_norm,
                    is_offside,
                    full_3d=True,
                    offside_field_x=offside_field_x,
                    attacking_side=attacking_side,
                )
            elif has_homography:
                logger.info("Using homography for pitch reconstruction (no K/R/C)")
                cam_base_pose, cam_obj, focus_positions = self._build_scene_calibrated(
                    scene,
                    trimesh,
                    pyrender,
                    player_meshes,
                    calibration,
                    offside_x_norm,
                    is_offside,
                    full_3d=False,
                    offside_field_x=offside_field_x,
                    attacking_side=attacking_side,
                )
            else:
                logger.info("No calibration â€” using SAM camera-space placement")
                cam_base_pose, cam_obj, focus_positions = self._build_scene_cam_space(
                    scene,
                    trimesh,
                    pyrender,
                    player_meshes,
                    offside_x_norm,
                    is_offside,
                    fl,
                    img_w,
                )

            # Grass mow-stripes so the pitch reads as a green grass rectangle.
            if z_up:
                self._add_pitch_stripes(scene, trimesh, pyrender)

            # The calibrated broadcast camera (full PnL only): the 3D scene seen
            # from it lines up with the original frame, so the crossfade starts
            # there and then flies to the cinematic framing.
            broadcast_pose = cam_base_pose.copy() if has_full_3d else None
            broadcast_yfov = None
            if has_full_3d:
                K = np.asarray(calibration.camera_matrix, dtype=np.float64)
                img_size = getattr(calibration, "image_size", None)
                img_h = int(img_size[0]) if img_size and img_size[0] else int(K[1, 2] * 2)
                s = min(self.video_width / img_w, self.video_height / max(img_h, 1))
                broadcast_yfov = 2.0 * math.atan((self.video_height / 2.0) / (K[1, 1] * s))

            cine_target = None
            # Cinematic framing: view the offside wall from the attacking side,
            # looking down the goal direction, so the wall is clearly visible.
            if z_up and attacking_side in ("left", "right"):
                cine = self._cinematic_camera_pose(
                    offside_field_x,
                    offside_x_norm,
                    img_w,
                    focus_positions,
                    attacking_side,
                    calibration,
                    pyrender,
                )
                if cine is not None:
                    cam_base_pose, cam_obj, cine_target = cine

            cam_node = scene.add(cam_obj, pose=cam_base_pose)
            renderer = pyrender.OffscreenRenderer(self.video_width, self.video_height)

            def render_3d_frame(cam_pose: np.ndarray) -> np.ndarray:
                scene.set_pose(cam_node, cam_pose)
                color, _ = renderer.render(scene)
                frame_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                self._add_var_overlay(frame_bgr, is_offside)
                return frame_bgr

            # Focus centre: orbit the point the cinematic camera frames, so the
            # sweep does not jump away from the wall framing on its first frame.
            if cine_target is not None:
                centre = cine_target
            elif focus_positions:
                centre = np.mean(focus_positions, axis=0)
            else:
                centre = np.array([0.0, 0.0, 0.0])

            fly_in = (
                broadcast_pose is not None
                and broadcast_yfov is not None
                and cine_target is not None
            )
            cine_yfov = cam_obj.yfov if fly_in else None

            # ---- Phase 2: Crossfade from overlay into the 3D pitch ----
            if transition_frames > 0 and has_image:
                logger.info("Phase 2: Crossfade transition (%d frames)", transition_frames)
                fallback_image = (
                    annotated_image if annotated_image is not None else original_image
                )
                src = frame_overlay
                if src is None and fallback_image is not None:
                    src = self._resize_frame(fallback_image)

                for i in range(transition_frames):
                    t = i / max(transition_frames - 1, 1)
                    t_smooth = 0.5 - 0.5 * math.cos(t * math.pi)

                    if fly_in:
                        cam_obj.yfov = broadcast_yfov
                        frame_3d = render_3d_frame(broadcast_pose)
                    else:
                        sweep_pose = self._compute_sweep_pose(
                            cam_base_pose, centre, z_up, 0.0, t_smooth * 0.15
                        )
                        frame_3d = render_3d_frame(sweep_pose)

                    if src is not None:
                        alpha = 0.5 - 0.5 * math.cos(t * math.pi)
                        blended = cv2.addWeighted(src, 1.0 - alpha, frame_3d, alpha, 0)
                        writer.write(blended)
                    else:
                        writer.write(frame_3d)

            # ---- Phase 3: Camera sweep (broadcast â†' near-overhead) ----
            logger.info("Phase 3: Camera sweep (%d frames)", sweep_frames)

            # Fly from the broadcast camera down to the cinematic close-up.
            fly_frames = 0
            if fly_in:
                fly_frames = int(sweep_frames * 0.4)
                sweep_frames -= fly_frames
                r0, r1 = broadcast_pose[:3, :3], cam_base_pose[:3, :3]
                p0, p1 = broadcast_pose[:3, 3], cam_base_pose[:3, 3]
                for i in range(fly_frames):
                    t = i / max(fly_frames - 1, 1)
                    t_smooth = 0.5 - 0.5 * math.cos(t * math.pi)
                    pose = np.eye(4)
                    pose[:3, :3] = _slerp_rotation(r0, r1, t_smooth)
                    pose[:3, 3] = (1.0 - t_smooth) * p0 + t_smooth * p1
                    cam_obj.yfov = (1.0 - t_smooth) * broadcast_yfov + t_smooth * cine_yfov
                    writer.write(render_3d_frame(pose))
                cam_obj.yfov = cine_yfov

            last_pose = cam_base_pose
            last_frame = None
            for i in range(sweep_frames):
                t = i / max(sweep_frames - 1, 1)
                t_smooth = 0.5 - 0.5 * math.cos(t * math.pi)

                # Sweep from 0Â° (start angle) toward a near-overhead view
                # The lateral orbit is subtle (~15Â°), the main motion is pitching down
                orbit_frac = t_smooth * 0.15
                down_frac = t_smooth * 0.55

                pose = self._compute_sweep_pose(
                    cam_base_pose, centre, z_up, orbit_frac, down_frac
                )
                last_pose = pose
                last_frame = render_3d_frame(pose)
                writer.write(last_frame)

            # ---- Phases 4-6: bird's-eye -> side-on along the line -> verdict ----
            finale_frames = 0
            if cine_target is not None and z_up:
                finale_frames = self._render_finale(
                    writer,
                    scene,
                    trimesh,
                    pyrender,
                    render_3d_frame,
                    cam_obj,
                    last_pose=last_pose,
                    target=cine_target,
                    focus_positions=focus_positions,
                    calibration=calibration,
                    attacking_side=attacking_side,
                    is_offside=is_offside,
                    attacker_idx=attacker_idx,
                    defender_idx=defender_idx,
                )
            elif last_frame is not None:
                # No full ending available (e.g. no attacking side): still
                # reveal the verdict over the last rendered 3D frame.
                assets = self._build_verdict_assets(
                    is_offside, self._verdict_caption(is_offside, None)
                )
                n = int(2.8 * self.fps)
                for i in range(n):
                    frame = self._draw_verdict_reveal(
                        last_frame, assets, i / self.fps
                    )
                    writer.write(frame)
                    finale_frames += 1

            total = (
                overlay_frames
                + transition_frames
                + fly_frames
                + sweep_frames
                + finale_frames
            )
            logger.info(
                "3D VAR video saved â†' %s  (%d frames, %.1fs)",
                output_path,
                total,
                total / self.fps,
            )
            return output_path
        except RenderCancelled:
            # Close the file first: Windows will not delete an open file.
            writer.release()
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise
        finally:
            writer.release()
            if renderer is not None:
                renderer.delete()

    # ------------------------------------------------------------------
    # Finale: bird's-eye, side-on line view, verdict reveal
    # ------------------------------------------------------------------

    @staticmethod
    def _broadcast_side(calibration) -> float:
        """Scene-y sign of the broadcast camera's touchline (-1 = main camera)."""
        pos = getattr(calibration, "position", None)
        if pos is not None:
            cam_y = float(
                (_PNL_TO_SCENE @ np.asarray(pos, dtype=np.float64).reshape(3))[1]
            )
            if abs(cam_y) > 1e-3:
                return math.copysign(1.0, cam_y)
        return -1.0

    @staticmethod
    def _look_at_pose(cam_pos: np.ndarray, look_at: np.ndarray) -> np.ndarray:
        """Z-up look-at pose in pyrender convention (camera looks down -Z)."""
        forward = look_at - cam_pos
        forward /= np.linalg.norm(forward) + 1e-8
        right = np.cross(forward, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up / (np.linalg.norm(up) + 1e-8)
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos
        return pose

    def _lead_point(
        self, v_world: np.ndarray, pm: PlayerMesh, attacking_side: str
    ) -> Tuple[float, float]:
        """Leading x toward the attacked goal (arms excluded) and foot y."""
        mask = np.ones(len(v_world), bool)
        try:
            from var_offside.offside.mesh_offside import non_arm_mask

            if len(pm.vertices) == len(v_world):
                mask = non_arm_mask(pm.vertices, pm.keypoints_3d)
        except Exception:  # noqa: BLE001 - fall back to the whole body
            pass
        body = v_world[mask] if np.any(mask) else v_world
        lead_x = float(body[:, 0].min() if attacking_side == "left" else body[:, 0].max())
        low = body[body[:, 2] <= np.percentile(body[:, 2], 8.0)]
        foot_y = float(low[:, 1].mean()) if len(low) else float(body[:, 1].mean())
        return lead_x, foot_y

    def _verdict_caption(self, is_offside: bool, gap_m: Optional[float]) -> str:
        if is_offside:
            if gap_m is not None and gap_m > 0.005:
                return f"ATTACKER AHEAD BY {gap_m:.2f} M  -  LINE FROM 2ND-LAST DEFENDER"
            return "ATTACKER IN AN OFFSIDE POSITION"
        if gap_m is not None and gap_m < -0.005:
            return f"ATTACKER BEHIND THE LINE BY {-gap_m:.2f} M  -  PLAY ON"
        if gap_m is not None and abs(gap_m) <= 0.005:
            return "ATTACKER LEVEL WITH THE LINE  -  PLAY ON"
        return "ATTACKER ONSIDE  -  PLAY ON"

    def _build_verdict_assets(self, is_offside: bool, caption: str) -> Dict[str, Any]:
        """Pre-render the static layers of the verdict reveal (BGR floats)."""
        W, H = self.video_width, self.video_height
        if is_offside:
            stops = [(18, 15, 143), (43, 38, 224), (62, 75, 255)]
            glow = np.array([40, 50, 255], np.float32)
        else:
            stops = [(43, 90, 11), (83, 166, 31), (122, 212, 57)]
            glow = np.array([120, 230, 40], np.float32)

        # Horizontal 3-stop gradient slab
        xs = np.linspace(0.0, 1.0, W, dtype=np.float32)[:, None]
        c0, c1, c2 = (np.array(c, np.float32) for c in stops)
        left = c0 + (c1 - c0) * np.clip(xs / 0.4, 0, 1)
        grad = np.where(xs < 0.4, left, c1 + (c2 - c1) * np.clip((xs - 0.4) / 0.6, 0, 1))
        y0, y1 = int(H * 0.58), int(H * 0.80)
        slab = np.zeros((H, W, 3), np.float32)
        slab[y0:y1] = grad[None, :, :]
        slab_mask = np.zeros((H, W), np.float32)
        slab_mask[y0:y1] = 1.0
        halo = cv2.GaussianBlur(slab_mask, (0, 0), sigmaX=H * 0.04)

        # Radial flash
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W * 0.5) / W) ** 2 + ((yy - H * 0.6) / H) ** 2)
        flash = np.clip(1.0 - r / 0.7, 0.0, 1.0) ** 1.5

        word = self._render_text_rgba(
            "OFFSIDE" if is_offside else "ONSIDE", int(H * 0.13), spacing=0.25
        )
        sub = self._render_text_rgba(
            caption, int(H * 0.032), spacing=0.04,
            fonts=("arialbd.ttf", "DejaVuSans-Bold.ttf"),
        )
        return {
            "slab": slab, "slab_mask": slab_mask, "halo": halo, "glow": glow,
            "flash": flash, "word": word, "sub": sub,
            "slab_rows": (y0, y1),
        }

    @staticmethod
    def _render_text_rgba(
        text: str,
        px: int,
        spacing: float = 0.0,
        fonts: Tuple[str, ...] = ("impact.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"),
    ) -> np.ndarray:
        """White text with a soft glow as a float RGBA array (alpha 0-1)."""
        try:
            from PIL import Image, ImageDraw, ImageFont

            font = None
            for name in fonts:
                try:
                    font = ImageFont.truetype(name, px)
                    break
                except OSError:
                    continue
            if font is None:
                raise OSError("no TrueType font")
            gap = int(px * spacing)
            widths = [font.getbbox(ch)[2] for ch in text]
            w = sum(widths) + gap * max(len(text) - 1, 0) + px
            h = int(px * 1.5)
            img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(img)
            x = px // 2
            for ch, cw in zip(text, widths):
                draw.text((x, int(px * 0.15)), ch, fill=255, font=font)
                x += cw + gap
            alpha = np.asarray(img, np.float32) / 255.0
        except Exception:  # noqa: BLE001 - PIL/font missing: OpenCV fallback
            scale = px / 30.0
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
            canvas = np.zeros((int(th * 2), tw + px), np.uint8)
            cv2.putText(canvas, text, (px // 2, int(th * 1.4)),
                        cv2.FONT_HERSHEY_DUPLEX, scale, 255, 2, cv2.LINE_AA)
            alpha = canvas.astype(np.float32) / 255.0
        glow = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(px * 0.12, 1.0)) * 0.6
        rgba = np.zeros(alpha.shape + (4,), np.float32)
        rgba[..., :3] = 255.0
        rgba[..., 3] = np.clip(np.maximum(alpha, glow), 0.0, 1.0)
        return rgba

    @staticmethod
    def _paste_rgba(dst: np.ndarray, rgba: np.ndarray, cx: float, cy: float,
                    scale: float, alpha: float) -> None:
        """Alpha-blend a float RGBA sprite centred at (cx, cy), in place."""
        if alpha <= 0.0:
            return
        if abs(scale - 1.0) > 1e-3:
            rgba = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h, w = rgba.shape[:2]
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        H, W = dst.shape[:2]
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(W, x0 + w), min(H, y0 + h)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        sp = rgba[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
        a = sp[..., 3:4] * alpha
        region = dst[dy0:dy1, dx0:dx1]
        region[:] = region * (1.0 - a) + sp[..., :3] * a

    def _draw_verdict_reveal(self, frame: np.ndarray, assets: Dict[str, Any],
                             t: float) -> np.ndarray:
        """Compose the verdict animation at ``t`` seconds into the reveal."""
        def ease_out(x):
            x = float(np.clip(x, 0.0, 1.0))
            return 1.0 - (1.0 - x) ** 3

        def ease_back(x):
            x = float(np.clip(x, 0.0, 1.0))
            c = 1.70158
            return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2

        out = frame.astype(np.float32)
        H, W = out.shape[:2]
        glow = assets["glow"]

        # 1. Flash: quick peak then settles to a faint tint
        fa = (t / 0.16) if t < 0.16 else max(0.25, 1.0 - (t - 0.16) / 0.9)
        fa *= 0.55
        f = assets["flash"][..., None] * fa
        out = out * (1.0 - f) + glow * f

        # 2. Banner slab wipes in from the left, with a halo
        p = ease_out((t - 0.25) / 0.45)
        if p > 0:
            xcut = int(W * p)
            m = assets["slab_mask"].copy()
            m[:, xcut:] = 0.0
            halo = assets["halo"].copy()
            halo[:, min(W, xcut + int(H * 0.08)):] = 0.0
            h3 = (halo * 0.55)[..., None]
            out = out * (1.0 - h3) + glow * h3
            m3 = m[..., None] * 0.92
            out = out * (1.0 - m3) + assets["slab"] * m3

        # 3. Word drops in from 1.6x scale with overshoot
        y0, y1 = assets["slab_rows"]
        q = (t - 0.55) / 0.6
        if q > 0:
            scale = 1.6 - 0.6 * ease_back(q)
            self._paste_rgba(out, assets["word"], W * 0.5, (y0 + y1) * 0.5,
                             scale, float(np.clip(q * 1.6, 0.0, 1.0)))

        # 4. Caption fades in under the banner, on a dark strip for contrast
        s = (t - 1.05) / 0.4
        if s > 0:
            a = float(np.clip(s, 0.0, 1.0))
            sub = assets["sub"]
            cy = y1 + H * 0.06
            sh, sw = sub.shape[0], sub.shape[1]
            sy0, sy1 = int(cy - sh * 0.45), int(cy + sh * 0.45)
            sx0, sx1 = max(0, int(W / 2 - sw / 2)), min(W, int(W / 2 + sw / 2))
            out[sy0:sy1, sx0:sx1] *= 1.0 - 0.6 * a
            self._paste_rgba(out, sub, W * 0.5, cy, 1.0, a)
        return np.clip(out, 0, 255).astype(np.uint8)

    # Non-arm MHR70 keypoints -> body-part label for the line-setting point.
    _PART_NAMES = {
        0: "HEAD", 1: "HEAD", 2: "HEAD", 3: "HEAD", 4: "HEAD", 69: "NECK",
        5: "LEFT SHOULDER", 6: "RIGHT SHOULDER",
        67: "LEFT SHOULDER", 68: "RIGHT SHOULDER",
        9: "LEFT HIP", 10: "RIGHT HIP", 11: "LEFT KNEE", 12: "RIGHT KNEE",
        13: "LEFT ANKLE", 14: "RIGHT ANKLE",
        15: "LEFT FOOT", 16: "LEFT FOOT", 17: "LEFT HEEL",
        18: "RIGHT FOOT", 19: "RIGHT FOOT", 20: "RIGHT HEEL",
    }

    def _line_setting_part(
        self,
        v_world: np.ndarray,
        pm: PlayerMesh,
        attacking_side: str,
        trimesh,
        pyrender,
    ) -> Optional[Dict[str, Any]]:
        """Glowing overlay for the defender's body part that sets the line.

        The part is the non-arm vertex furthest toward the attacked goal; the
        overlay copies the body surface within ~15 cm of it, lifted slightly
        along the surface normal so it sits on top of the body.
        """
        mask = np.ones(len(v_world), bool)
        try:
            from var_offside.offside.mesh_offside import non_arm_mask

            if len(pm.vertices) == len(v_world):
                mask = non_arm_mask(pm.vertices, pm.keypoints_3d)
        except Exception:  # noqa: BLE001 - fall back to the whole body
            pass
        idxs = np.nonzero(mask)[0]
        if len(idxs) == 0:
            return None
        xs = v_world[idxs, 0]
        k = int(idxs[np.argmin(xs) if attacking_side == "left" else np.argmax(xs)])
        lead = v_world[k]

        name = "BODY"
        kp = pm.keypoints_3d
        if kp is not None and len(pm.vertices) == len(v_world):
            kp = np.asarray(kp, dtype=np.float64)
            ids = [i for i in self._PART_NAMES if i < len(kp)]
            if ids:
                d = np.linalg.norm(kp[ids] - np.asarray(pm.vertices[k], np.float64), axis=1)
                name = self._PART_NAMES[ids[int(np.argmin(d))]]

        faces = np.asarray(pm.faces)
        for radius in (0.20, 0.30):
            region = mask & (np.linalg.norm(v_world - lead, axis=1) < radius)
            fm = region[faces].all(axis=1)
            if fm.sum() >= 4:
                break
        patch = trimesh.Trimesh(vertices=v_world, faces=faces[fm], process=False)
        normals = np.asarray(patch.vertex_normals)
        centre = v_world[mask].mean(axis=0)
        used = np.unique(faces[fm])
        if len(used) and np.mean(np.sum(normals[used] * (v_world[used] - centre), axis=1)) < 0:
            normals = -normals  # make sure the lift points out of the body
        lifted = trimesh.Trimesh(
            vertices=v_world + normals * 0.012, faces=faces[fm], process=False
        )
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 0.96, 0.72, 1.0],
            emissiveFactor=[1.0, 0.92, 0.55],
            doubleSided=True,
        )
        meshes = [pyrender.Mesh.from_trimesh(lifted, material=material, smooth=False)]
        dot = trimesh.creation.icosphere(subdivisions=2, radius=0.06)
        dot.apply_translation(lead)
        meshes.append(pyrender.Mesh.from_trimesh(dot, material=material, smooth=True))
        return {"point": lead, "name": name, "meshes": meshes}

    def _project_point(self, cam_obj, pose: np.ndarray, p: np.ndarray):
        """Scene point -> output pixel for a pyrender camera at ``pose``."""
        P = cam_obj.get_projection_matrix(self.video_width, self.video_height)
        clip = P @ np.linalg.inv(pose) @ np.append(p, 1.0)
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        return ((ndc[0] + 1) * 0.5 * self.video_width, (1 - ndc[1]) * 0.5 * self.video_height)

    def _draw_part_callout(
        self, frame: np.ndarray, anchor, title: str, detail: str, alpha: float
    ) -> np.ndarray:
        """Leader line + label box pointing at the line-setting part."""
        if anchor is None or alpha <= 0:
            return frame
        W, H = self.video_width, self.video_height
        ax, ay = float(anchor[0]), float(anchor[1])
        t_img = self._render_text_rgba(title, int(H * 0.034), 0.08, ("arialbd.ttf", "DejaVuSans-Bold.ttf"))
        d_img = self._render_text_rgba(detail, int(H * 0.028), 0.04, ("arialbd.ttf", "DejaVuSans-Bold.ttf"))
        bw = max(t_img.shape[1], d_img.shape[1]) + 20
        bh = t_img.shape[0] + d_img.shape[0]
        right = ax < W * 0.62
        bx = ax + W * 0.08 if right else ax - W * 0.08 - bw
        by = max(H * 0.10, ay - H * 0.22)
        bx = float(np.clip(bx, 10, W - bw - 10))
        out = frame.astype(np.float32)
        x0, y0, x1, y1 = int(bx), int(by), int(bx + bw), int(by + bh)
        out[y0:y1, x0:x1] *= 1.0 - 0.7 * alpha
        line_end = (x0 if right else x1, int(by + bh * 0.5))
        lay = out.copy()
        cv2.line(lay, (int(ax), int(ay)), line_end, (255, 255, 255), 3, cv2.LINE_AA)
        out = out * (1 - alpha) + lay * alpha
        self._paste_rgba(out, t_img, x0 + t_img.shape[1] / 2 + 6, y0 + t_img.shape[0] / 2, 1.0, alpha)
        self._paste_rgba(out, d_img, x0 + d_img.shape[1] / 2 + 6,
                         y0 + t_img.shape[0] + d_img.shape[0] / 2 - 4, 1.0, alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _render_finale(
        self,
        writer,
        scene,
        trimesh,
        pyrender,
        render_3d_frame: Callable[[np.ndarray], np.ndarray],
        cam_obj,
        last_pose: np.ndarray,
        target: np.ndarray,
        focus_positions: List[np.ndarray],
        calibration,
        attacking_side: Optional[str],
        is_offside: bool,
        attacker_idx: Optional[int] = None,
        defender_idx: Optional[int] = None,
    ) -> int:
        """Bird's-eye, then side-on along the offside line, then the verdict.

        Returns the number of frames written.
        """
        fps = self.fps
        half_w = self.PITCH_WIDTH / 2.0
        offside_cx = float(target[0])
        side = self._broadcast_side(calibration)
        aspect = self.video_width / max(self.video_height, 1)
        yfov = float(cam_obj.yfov)
        hfov = 2.0 * math.atan(math.tan(yfov / 2.0) * aspect)
        verdict_rgb = (255, 45, 40) if is_offside else (40, 230, 110)

        if focus_positions:
            pts = np.array(focus_positions, dtype=np.float64)[:, :2]
        else:
            pts = np.array([[offside_cx, float(target[1])]])
        xs = np.append(pts[:, 0], offside_cx)
        ys = pts[:, 1]
        cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())

        # --- Bird's-eye: low overhead framed on the players + wall ----------
        # Margin keeps the players clear of the HUD bar at the top.
        ext_x = max(float(np.ptp(xs)), 6.0) + 9.0
        ext_y = max(float(np.ptp(ys)), 6.0) + 9.0
        eye_h = max(ext_x / (2 * math.tan(hfov / 2)), ext_y / (2 * math.tan(yfov / 2)))
        eye_h = float(np.clip(eye_h, 15.0, 34.0))
        bird = np.eye(4)
        bird[:3, 0] = [-side, 0.0, 0.0]  # broadcast left/right preserved
        bird[:3, 1] = [0.0, -side, 0.0]  # far touchline at the top
        bird[:3, 2] = [0.0, 0.0, 1.0]  # looking straight down
        bird[:3, 3] = [cx, cy, eye_h]

        # --- Side-on: on the line, beyond the nearest player, looking along it
        near_y = ys.min() if side < 0 else ys.max()
        cam_y = float(np.clip(near_y + side * 12.0, -half_w - 12.0, half_w + 12.0))
        side_pose = self._look_at_pose(
            np.array([offside_cx, cam_y, 2.6]), np.array([offside_cx, cy, 0.9])
        )

        # --- Gap bar + verdict highlights ----------------------------------
        planted = getattr(self, "_planted_meshes", {}) or {}
        gap_m = None
        glow_meshes = []
        att = None
        if attacker_idx is not None and attacker_idx in planted:
            att = planted[attacker_idx]
        else:
            cands = [v for k, v in planted.items() if k != defender_idx]
            off = [v for v in cands if v[1].is_offside]
            att = (off or cands or [None])[0]
        if att is not None and attacking_side in ("left", "right"):
            v_att, pm_att = att
            lead_x, foot_y = self._lead_point(v_att, pm_att, attacking_side)
            gap_m = (offside_cx - lead_x) if attacking_side == "left" else (lead_x - offside_cx)
            x0, x1 = sorted((lead_x, offside_cx))
            if x1 - x0 > 0.01:
                bw = 0.14
                bar = trimesh.Trimesh(
                    vertices=[[x0, foot_y - bw, 0.05], [x1, foot_y - bw, 0.05],
                              [x1, foot_y + bw, 0.05], [x0, foot_y + bw, 0.05]],
                    faces=[[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]],
                    process=False,
                )
                bar.visual.face_colors = np.tile([255, 220, 30, 255], (4, 1))
                scene.add(pyrender.Mesh.from_trimesh(bar, smooth=False))
            # Inflated translucent shell around the attacker for the pulse
            c = v_att.mean(axis=0)
            shell = trimesh.Trimesh(
                vertices=c + (v_att - c) * 1.05,
                faces=pm_att.faces,
                vertex_colors=np.tile(list(verdict_rgb) + [110], (len(v_att), 1)),
                process=False,
            )
            glow_meshes.append(pyrender.Mesh.from_trimesh(shell, smooth=True))

        wall = trimesh.Trimesh(
            vertices=[[offside_cx, -half_w, 0], [offside_cx, half_w, 0],
                      [offside_cx, half_w, 3.0], [offside_cx, -half_w, 3.0]],
            faces=[[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]],
            process=False,
        )
        wall.visual.face_colors = np.tile(list(verdict_rgb) + [120], (4, 1))
        glow_meshes.append(pyrender.Mesh.from_trimesh(wall, smooth=False))
        for gm in glow_meshes:
            gm.is_visible = False
            scene.add(gm)

        written = 0

        def move(p_from: np.ndarray, p_to: np.ndarray, seconds: float) -> None:
            nonlocal written
            n = max(int(seconds * fps), 1)
            for i in range(n):
                t = i / max(n - 1, 1)
                ts = 0.5 - 0.5 * math.cos(t * math.pi)
                pose = np.eye(4)
                pose[:3, :3] = _slerp_rotation(p_from[:3, :3], p_to[:3, :3], ts)
                pose[:3, 3] = (1 - ts) * p_from[:3, 3] + ts * p_to[:3, 3]
                writer.write(render_3d_frame(pose))
                written += 1

        def hold(pose: np.ndarray, seconds: float) -> None:
            nonlocal written
            frame = render_3d_frame(pose)
            for _ in range(max(int(seconds * fps), 1)):
                writer.write(frame)
                written += 1

        logger.info(
            "Finale: bird's-eye at %.1f m, side-on at y=%.1f on line x=%.1f, gap=%s",
            eye_h, cam_y, offside_cx,
            "n/a" if gap_m is None else f"{gap_m:+.2f} m",
        )
        # --- Line-setting body part on the defender --------------------------
        part = None
        if (
            defender_idx is not None
            and defender_idx in planted
            and attacking_side in ("left", "right")
        ):
            try:
                part = self._line_setting_part(
                    *planted[defender_idx], attacking_side, trimesh, pyrender
                )
            except Exception as exc:  # noqa: BLE001 - the flash is optional
                logger.warning("Line-setting part highlight skipped: %s", exc)
        part_meshes = part["meshes"] if part else []
        for pmh in part_meshes:
            pmh.is_visible = False
            scene.add(pmh)

        def show_part(on: bool) -> None:
            for pmh in part_meshes:
                pmh.is_visible = on

        move(last_pose, bird, 1.6)
        hold(bird, 0.7)
        move(bird, side_pose, 2.0)
        hold(side_pose, 0.4)

        if part is not None:
            # Slide along the line (still exactly side-on) up to the part,
            # pulse it slowly with a label, then pull back for the verdict.
            pt = part["point"]
            # ~6.5 m back keeps the whole player in frame with the part
            # clearly visible; aiming at mid-body avoids cutting the head.
            close = self._look_at_pose(
                np.array([offside_cx, pt[1] + side * 6.5, 1.1]),
                np.array([pt[0], pt[1], 0.8]),
            )
            move(side_pose, close, 1.0)
            dark = render_3d_frame(close)
            show_part(True)
            glow = render_3d_frame(close)
            anchor = self._project_point(cam_obj, close, pt)
            title = "LINE SET HERE"
            num = f"#{defender_idx}" if defender_idx is not None else ""
            detail = f"DEFENDER {num}  -  {part['name']}".replace("  ", " ")
            n = int(2.8 * fps)
            for i in range(n):
                t = i / fps
                # Slow, soft pulse (1 s period): dark -> lit -> dark -> lit ...
                # reaching full glow at 2.5 s and holding it.
                level = 1.0 if t >= 2.5 else 0.5 - 0.5 * math.cos(2 * math.pi * t)
                frame = cv2.addWeighted(dark, 1.0 - level, glow, level, 0)
                frame = self._draw_part_callout(
                    frame, anchor, title, detail, float(np.clip(t / 0.5, 0, 1))
                )
                writer.write(frame)
                written += 1
            move(close, side_pose, 0.8)
            logger.info("Line-setting part: %s at %s", part["name"], np.round(pt, 2))

        # Verdict: freeze on the side-on view; pulse the glow layers.  The
        # line-setting part stays steadily lit throughout.
        base = render_3d_frame(side_pose)
        for gm in glow_meshes:
            gm.is_visible = True
        lit = render_3d_frame(side_pose)
        for gm in glow_meshes:
            gm.is_visible = False

        assets = self._build_verdict_assets(is_offside, self._verdict_caption(is_offside, gap_m))
        n = int(2.8 * fps)
        for i in range(n):
            t = i / fps
            pulse = 0.0 if t < 0.15 else 0.55 + 0.45 * math.sin((t - 0.15) * 2 * math.pi * 1.6)
            frame = cv2.addWeighted(base, 1.0 - pulse, lit, pulse, 0)
            writer.write(self._draw_verdict_reveal(frame, assets, t))
            written += 1
        return written

    def render_debug(
        self,
        player_meshes: List[PlayerMesh],
        offside_x_norm: float,
        is_offside: bool,
        output_path: str,
        original_image: Optional[np.ndarray] = None,
        calibration=None,
        detections=None,
        offside_field_x: Optional[float] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> str:
        """
        Render a 2D debug overlay video on the original broadcast image.

        Phases:
            1. Original image (2s hold)
            2. Fade in: bird's-eye pitch surface + projected lines + keypoints (3s)
            3. Fade in: 3D player meshes + offside line (2s)
            4. Hold complete overlay (3s)

        Everything is projected onto the original image. No 3D scene.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        raw_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.video_width, self.video_height),
        )
        if not raw_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")
        writer = _CancellableWriter(raw_writer, should_stop)

        try:
            base_img, pitch_overlay, full_overlay = self._compose_debug_layers(
                player_meshes,
                offside_x_norm,
                is_offside,
                original_image=original_image,
                calibration=calibration,
                detections=detections,
                expand_to_full_pitch=False,
                offside_field_x=offside_field_x,
            )

            def write_frame(frame: np.ndarray):
                writer.write(self._resize_frame(frame))

            p1 = int(2.0 * self.fps)
            p2 = int(3.0 * self.fps)
            p3 = int(2.0 * self.fps)
            p4 = int(3.0 * self.fps)

            # Phase 1: original image
            logger.info("Debug Phase 1: Original image (%d frames)", p1)
            for _ in range(p1):
                write_frame(base_img)

            # Phase 2: crossfade original -> pitch surface + lines + keypoints
            logger.info("Debug Phase 2: Pitch overlay fade-in (%d frames)", p2)
            for i in range(p2):
                t = i / max(p2 - 1, 1)
                alpha = 0.5 - 0.5 * math.cos(t * math.pi)
                blended = cv2.addWeighted(base_img, 1.0 - alpha, pitch_overlay, alpha, 0)
                write_frame(blended)

            # Phase 3: crossfade pitch lines -> full overlay (meshes + line)
            logger.info("Debug Phase 3: Meshes + line fade-in (%d frames)", p3)
            for i in range(p3):
                t = i / max(p3 - 1, 1)
                alpha = 0.5 - 0.5 * math.cos(t * math.pi)
                blended = cv2.addWeighted(
                    pitch_overlay, 1.0 - alpha, full_overlay, alpha, 0
                )
                write_frame(blended)

            # Phase 4: hold
            logger.info("Debug Phase 4: Hold complete overlay (%d frames)", p4)
            for _ in range(p4):
                write_frame(full_overlay)

            total = p1 + p2 + p3 + p4
            logger.info(
                "Debug overlay video saved -> %s  (%d frames, %.1fs)",
                output_path,
                total,
                total / self.fps,
            )
            return output_path
        except RenderCancelled:
            # Close the file first: Windows will not delete an open file.
            writer.release()
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise
        finally:
            writer.release()

    def render_debug_image(
        self,
        player_meshes: List[PlayerMesh],
        offside_x_norm: float,
        is_offside: bool,
        output_path: str,
        original_image: Optional[np.ndarray] = None,
        calibration=None,
        detections=None,
        offside_field_x: Optional[float] = None,
    ) -> str:
        """Render a zoomed-out debug still that includes the full projected pitch."""
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        _, _, full_overlay = self._compose_debug_layers(
            player_meshes,
            offside_x_norm,
            is_offside,
            original_image=original_image,
            calibration=calibration,
            detections=detections,
            expand_to_full_pitch=True,
            offside_field_x=offside_field_x,
        )
        cv2.imwrite(output_path, full_overlay)
        logger.info("Debug overlay image saved -> %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Zoom-out VAR video (3D bird's-eye via pyrender)
    # ------------------------------------------------------------------

    def render_zoomout_video(
        self,
        player_meshes: List[PlayerMesh],
        offside_x_norm: float,
        is_offside: bool,
        output_path: str,
        original_image: Optional[np.ndarray] = None,
        calibration=None,
        detections=None,
        show_offside_line: bool = True,
        offside_field_x: Optional[float] = None,
        attacking_side: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Render a zoom-out VAR video: broadcast overlay -> 3D bird's-eye pitch."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        raw_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.video_width, self.video_height),
        )
        if not raw_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")
        writer = _CancellableWriter(raw_writer, should_stop)

        try:
            base_img, pitch_overlay, full_overlay = self._compose_debug_layers(
                player_meshes,
                offside_x_norm,
                is_offside,
                original_image=original_image,
                calibration=calibration,
                detections=detections,
                expand_to_full_pitch=False,
                offside_field_x=offside_field_x,
            )

            logger.debug("Zoomout: debug layers composed, rendering 3D bird's-eye")
            birds_eye_frames = self._render_birdseye_3d_frames(
                player_meshes,
                offside_x_norm,
                is_offside,
                calibration,
                offside_field_x=offside_field_x,
                attacking_side=attacking_side,
                should_stop=should_stop,
            )
            if birds_eye_frames is not None and len(birds_eye_frames) > 0:
                birds_eye_final = birds_eye_frames[-1]
                use_3d = True
                logger.debug("Zoomout: got %d 3D frames", len(birds_eye_frames))
            else:
                logger.info("Zoomout: 3D bird's-eye unavailable, using 2D fallback")
                birds_eye_final = self._draw_birdseye_pitch_fallback(
                    player_meshes,
                    offside_x_norm,
                    is_offside,
                    calibration,
                )
                use_3d = False

            p1 = int(2.0 * self.fps)
            p2 = int(2.0 * self.fps)
            p3 = int(3.0 * self.fps)
            p4 = int(3.0 * self.fps)

            full_overlay_resized = self._resize_frame(full_overlay)
            base_img_resized = self._resize_frame(base_img)

            def write_frame(frame: np.ndarray):
                writer.write(self._resize_frame(frame))

            logger.info("Zoomout Phase 1: Original image (%d frames)", p1)
            for _ in range(p1):
                write_frame(base_img)

            logger.info("Zoomout Phase 2: Overlay crossfade (%d frames)", p2)
            for i in range(p2):
                t = i / max(p2 - 1, 1)
                alpha = 0.5 - 0.5 * math.cos(t * math.pi)
                blended = cv2.addWeighted(
                    base_img_resized, 1.0 - alpha, full_overlay_resized, alpha, 0
                )
                write_frame(blended)

            logger.info("Zoomout Phase 3: Transition to bird's-eye (%d frames)", p3)
            if use_3d and birds_eye_frames is not None and len(birds_eye_frames) >= p3:
                for i in range(p3):
                    t = i / max(p3 - 1, 1)
                    alpha = 0.5 - 0.5 * math.cos(t * math.pi)
                    idx_3d = min(
                        int(t * (len(birds_eye_frames) - 1)), len(birds_eye_frames) - 1
                    )
                    frame_3d = birds_eye_frames[idx_3d]
                    blended = cv2.addWeighted(
                        full_overlay_resized, 1.0 - alpha, frame_3d, alpha, 0
                    )
                    write_frame(blended)
            else:
                birdseye_matched = cv2.resize(
                    birds_eye_final,
                    (full_overlay_resized.shape[1], full_overlay_resized.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                for i in range(p3):
                    t = i / max(p3 - 1, 1)
                    alpha = 0.5 - 0.5 * math.cos(t * math.pi)
                    blended = cv2.addWeighted(
                        full_overlay_resized, 1.0 - alpha, birdseye_matched, alpha, 0
                    )
                    write_frame(blended)

            logger.info("Zoomout Phase 4: Hold bird's-eye (%d frames)", p4)
            hold_frame = self._resize_frame(birds_eye_final)
            for _ in range(p4):
                writer.write(hold_frame)

            total = p1 + p2 + p3 + p4
            logger.info(
                "Zoomout VAR video saved -> %s  (%d frames, %.1fs)",
                output_path,
                total,
                total / self.fps,
            )
            return output_path
        except RenderCancelled:
            # Close the file first: Windows will not delete an open file.
            writer.release()
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise
        finally:
            writer.release()

    def _render_birdseye_3d_frames(
        self,
        player_meshes: List[PlayerMesh],
        offside_x_norm: float,
        is_offside: bool,
        calibration,
        offside_field_x: Optional[float] = None,
        attacking_side: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Optional[List[np.ndarray]]:
        """Render 3D bird's-eye pitch frames via pyrender.

        Returns a list of BGR frames sweeping from broadcast angle
        to overhead, or None if pyrender is unavailable.
        """
        try:
            trimesh_mod = _require_trimesh()
            pyrender_mod = _require_pyrender()
        except ImportError:
            logger.warning("pyrender/trimesh not available for 3D bird's-eye")
            return None

        if calibration is None:
            logger.warning("No calibration -- cannot render 3D bird's-eye")
            return None

        has_full_3d = (
            calibration.has_3d_calibration
            and calibration.camera_matrix is not None
            and calibration.rotation_matrix is not None
            and calibration.position is not None
        )
        if not has_full_3d and getattr(calibration, "method", None) == "vanishing_point":
            # The VP calibrator's homography is only a rough placeholder; a
            # 3D reconstruction from it is meaningless.  Use the 2D fallback.
            logger.info(
                "Vanishing-point calibration has no 3D camera; "
                "using 2D bird's-eye fallback"
            )
            return None

        renderer = None
        try:
            half_l = self.PITCH_LENGTH / 2.0
            half_w = self.PITCH_WIDTH / 2.0
            scene = pyrender_mod.Scene(
                bg_color=np.array([0.03, 0.07, 0.04, 1.0]),  # dark stadium green
                ambient_light=np.array([0.32, 0.34, 0.32]),
            )

            cam_base_pose, cam_obj, focus_positions = self._build_scene_calibrated(
                scene,
                trimesh_mod,
                pyrender_mod,
                player_meshes,
                calibration,
                offside_x_norm,
                is_offside,
                full_3d=has_full_3d,
                offside_field_x=offside_field_x,
                attacking_side=attacking_side,
            )

            # Add grass stripes for visual depth
            self._add_pitch_stripes(scene, trimesh_mod, pyrender_mod)

            # Offscreen renderer
            renderer = pyrender_mod.OffscreenRenderer(
                self.video_width, self.video_height
            )

            if focus_positions:
                centre = np.mean(focus_positions, axis=0)
            else:
                centre = np.array([0.0, 0.0, 0.0])

            cam_start_pos = cam_base_pose[:3, 3].copy()
            # Height at which the key players (plus margin) fill the frame
            # with the broadcast intrinsics; a fixed 70 m with a telephoto lens
            # left them as specks.
            if len(focus_positions) > 1:
                fp_arr = np.array(focus_positions, dtype=np.float64)
                extent = float(np.ptp(fp_arr[:, :2], axis=0).max())
            else:
                extent = 0.0
            extent = max(extent, 12.0) * 1.8
            fy = float(getattr(cam_obj, "fy", 0.0) or 0.0)
            if fy > 0:
                eye_height = float(
                    np.clip(extent * fy / self.video_height, 25.0, 90.0)
                )
            else:
                eye_height = 70.0

            num_frames = int(3.0 * self.fps)
            frames = []

            # Add camera node to scene
            cam_node = scene.add(cam_obj, pose=cam_base_pose)

            # Overhead camera: straight down, but yawed to match the broadcast
            # heading so the bird's-eye keeps the SAME left/right/near-far
            # orientation as the calibrated view (pitch orientation is derived
            # from the calibration, not a fixed compass direction).
            broadcast_look = -cam_base_pose[:3, 2].astype(np.float64)
            f_xy = broadcast_look[:2].copy()
            norm = np.linalg.norm(f_xy)
            if norm > 1e-6:
                f_xy /= norm
            else:
                f_xy = np.array([0.0, 1.0])
            # screen-up = "into the field" from the broadcast camera
            up_screen = np.array([f_xy[0], f_xy[1], 0.0])
            right_screen = np.array([f_xy[1], -f_xy[0], 0.0])
            over_pose = np.eye(4, dtype=np.float64)
            over_pose[:3, 3] = [centre[0], centre[1], eye_height]
            over_pose[:3, 0] = right_screen
            over_pose[:3, 1] = up_screen
            over_pose[:3, 2] = [0.0, 0.0, 1.0]  # looking straight down (-Z)

            logger.info(
                "Rendering %d bird's-eye frames (broadcast -> overhead)...",
                num_frames,
            )

            start_rot = cam_base_pose[:3, :3].astype(np.float64)
            end_rot = over_pose[:3, :3].astype(np.float64)

            for i in range(num_frames):
                if should_stop is not None and should_stop():
                    raise RenderCancelled("3D render cancelled")
                t = i / max(num_frames - 1, 1)
                t_smooth = 0.5 - 0.5 * math.cos(t * math.pi)

                # Interpolate position
                pose = np.eye(4, dtype=np.float64)
                pose[:3, 3] = (1.0 - t_smooth) * cam_start_pos + t_smooth * over_pose[
                    :3, 3
                ]

                # Interpolate orientation on the rotation manifold.  (A linear
                # blend + QR re-orthogonalisation flips axis signs and renders
                # the camera rolled 180 degrees or mirrored.)
                pose[:3, :3] = _slerp_rotation(start_rot, end_rot, t_smooth)

                scene.set_pose(cam_node, pose)
                color, _ = renderer.render(scene)
                frame_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                self._add_var_overlay(frame_bgr, is_offside)
                frames.append(frame_bgr)

                if (i + 1) % 15 == 0 or i == num_frames - 1:
                    logger.info(
                        "  Bird's-eye frame %d/%d rendered",
                        i + 1,
                        num_frames,
                    )

            logger.info("Bird's-eye 3D rendering complete (%d frames)", len(frames))
            return frames

        except RenderCancelled:
            raise
        except Exception as exc:
            logger.warning("3D bird's-eye rendering failed: %s", exc, exc_info=True)
            return None
        finally:
            if renderer is not None:
                renderer.delete()

    @staticmethod
    def _add_pitch_stripes(scene, trimesh, pyrender):
        """Add alternating grass stripe quads on top of the pitch surface."""
        half_l = 52.5
        half_w = 34.0
        stripe_count = 10
        stripe_w = (2 * half_l) / stripe_count
        dark = np.array([34, 88, 40, 255], dtype=np.uint8)  # darker grass (RGBA)
        light = np.array([46, 112, 52, 255], dtype=np.uint8)  # lighter grass (RGBA)
        z_s = 0.005
        for i in range(stripe_count):
            color = light if i % 2 == 0 else dark
            x0 = -half_l + i * stripe_w
            x1 = x0 + stripe_w
            verts = np.array(
                [
                    [x0, -half_w, z_s],
                    [x1, -half_w, z_s],
                    [x1, half_w, z_s],
                    [x0, half_w, z_s],
                ],
                dtype=np.float64,
            )
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            m.visual.face_colors = np.tile(color, (2, 1))
            scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))

    def _field_to_px(self, fx: float, fy: float, ppm: float, margin: float):
        """Field metres (0-105, 0-68) -> top-down pixel coordinates."""
        return int((fx + margin) * ppm), int((fy + margin) * ppm)

    def _draw_birdseye_pitch(
        self, w: int, h: int, ppm: float, margin: float
    ) -> np.ndarray:
        """Draw a flat top-down pitch (green with white markings)."""
        img = np.full((h, w, 3), (30, 90, 35), dtype=np.uint8)  # BGR grass green

        # Faint mow stripes for depth
        stripe_count = 10
        stripe_w = int((self.PITCH_LENGTH * ppm) / stripe_count)
        for i in range(stripe_count):
            if i % 2 == 0:
                x0 = int(margin * ppm) + i * stripe_w
                cv2.rectangle(
                    img, (x0, 0), (x0 + stripe_w, h), (34, 100, 40), -1
                )

        white = (235, 235, 235)
        L, W = self.PITCH_LENGTH, self.PITCH_WIDTH

        def line(p1, p2):
            cv2.line(
                img,
                self._field_to_px(*p1, ppm, margin),
                self._field_to_px(*p2, ppm, margin),
                white,
                2,
                cv2.LINE_AA,
            )

        # Outline + halfway line
        line((0, 0), (L, 0))
        line((L, 0), (L, W))
        line((L, W), (0, W))
        line((0, W), (0, 0))
        line((L / 2, 0), (L / 2, W))

        # Centre circle + spot
        cx, cy = self._field_to_px(L / 2, W / 2, ppm, margin)
        cv2.circle(img, (cx, cy), int(9.15 * ppm), white, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, white, -1, cv2.LINE_AA)

        # Penalty + goal areas (both ends)
        for gx, sign in ((0, 1), (L, -1)):
            # penalty area 16.5m deep, 40.32m wide (13.84..54.16)
            line((gx, 13.84), (gx + sign * 16.5, 13.84))
            line((gx + sign * 16.5, 13.84), (gx + sign * 16.5, 54.16))
            line((gx + sign * 16.5, 54.16), (gx, 54.16))
            # goal area 5.5m deep (24.84..43.16)
            line((gx, 24.84), (gx + sign * 5.5, 24.84))
            line((gx + sign * 5.5, 24.84), (gx + sign * 5.5, 43.16))
            line((gx + sign * 5.5, 43.16), (gx, 43.16))
        return img

    def _draw_birdseye_players(
        self,
        img: np.ndarray,
        player_meshes: List[PlayerMesh],
        offside_x_norm: float,
        is_offside: bool,
        calibration,
        _unused_a=None,
        _unused_b=None,
        ppm: float = 10.0,
        margin: float = 5.0,
    ) -> np.ndarray:
        """Draw team-coloured player dots + the offside line on the top-down pitch."""
        # Offside line (metres): sample the image column through calibration.
        offside_field_x = None
        if calibration is not None and player_meshes:
            img_w = player_meshes[0].image_width
            xs = []
            for sy in np.linspace(100, 600, 5):
                fp = calibration.project_to_field((offside_x_norm * img_w, sy))
                if fp is not None:
                    xs.append(fp[0])
            if xs:
                offside_field_x = float(np.median(xs))

        if offside_field_x is not None:
            ox = int((offside_field_x + margin) * ppm)
            line_color = (60, 60, 255) if is_offside else (60, 220, 60)
            cv2.line(img, (ox, 0), (ox, img.shape[0]), line_color, 2, cv2.LINE_AA)

        # Players
        for pm in player_meshes:
            if pm.field_pos is not None:
                fx, fy = pm.field_pos
            elif calibration is not None:
                bx = (pm.bbox[0] + pm.bbox[2]) / 2.0
                fp = calibration.project_to_field((bx, pm.bbox[3]))
                if fp is None:
                    continue
                fx, fy = fp
            else:
                continue
            px, py = self._field_to_px(fx, fy, ppm, margin)
            color = self.TEAM1_BGR if pm.team_id == 1 else self.TEAM2_BGR
            cv2.circle(img, (px, py), max(5, int(ppm * 0.5)), color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), max(5, int(ppm * 0.5)), (255, 255, 255), 1,
                       cv2.LINE_AA)
            if pm.is_offside:
                cv2.circle(img, (px, py), max(8, int(ppm * 0.8)), self.GK_BGR, 2,
                           cv2.LINE_AA)
        return img

    def _draw_birdseye_pitch_fallback(
        self,
        player_meshes: List[PlayerMesh],
        offside_x_norm: float,
        is_offside: bool,
        calibration,
    ) -> np.ndarray:
        """2D bird's-eye fallback when pyrender is unavailable."""
        ppm = max(8, min(14, int(min(self.video_width, self.video_height) / 68)))
        margin = 5.0
        w = int((self.PITCH_LENGTH + 2 * margin) * ppm)
        h = int((self.PITCH_WIDTH + 2 * margin) * ppm)
        birdseye = self._draw_birdseye_pitch(w, h, ppm, margin)
        birdseye = self._draw_birdseye_players(
            birdseye,
            player_meshes,
            offside_x_norm,
            is_offside,
            calibration,
            None,
            None,
            ppm,
            margin,
        )
        return birdseye

    # ---- Debug overlay helpers -------------------------------------------

    def _compose_debug_layers(
        self,
        player_meshes: List[PlayerMesh],
        offside_x_norm: float,
        is_offside: bool,
        original_image: Optional[np.ndarray] = None,
        calibration=None,
        detections=None,
        expand_to_full_pitch: bool = False,
        offside_field_x: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the base, pitch-only, and full debug overlays."""
        source_image = (
            original_image.copy()
            if original_image is not None
            else np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        )
        source_shape = (int(source_image.shape[0]), int(source_image.shape[1]))
        base_img, image_offset, image_scale = self._build_debug_canvas(
            source_image,
            calibration,
            expand_to_full_pitch=expand_to_full_pitch,
        )

        pitch_overlay = self._draw_debug_pitch_surface(
            base_img.copy(),
            calibration,
            image_offset=image_offset,
            image_scale=image_scale,
        )
        pitch_overlay = self._draw_debug_pitch_lines(
            pitch_overlay,
            calibration,
            image_offset=image_offset,
            image_scale=image_scale,
        )
        pitch_overlay = self._draw_debug_keypoints(
            pitch_overlay,
            calibration,
            image_offset=image_offset,
            image_scale=image_scale,
        )

        # trimesh/pyrender are only needed by the OpenGL paths; the mesh
        # overlay itself is pure OpenCV, so a missing GL stack must not block
        # the debug view.
        try:
            trimesh = _require_trimesh()
        except ImportError:
            trimesh = None
        try:
            pyrender = _require_pyrender()
        except ImportError:
            pyrender = None
        mesh_img = self._render_mesh_on_image(
            pitch_overlay.copy(),
            player_meshes,
            trimesh,
            pyrender,
            is_offside,
            projection_size=source_shape,
            image_offset=image_offset,
            image_scale=image_scale,
        )
        full_overlay = self._draw_debug_offside_line(
            mesh_img,
            offside_x_norm,
            is_offside,
            calibration,
            source_image_shape=source_shape,
            image_offset=image_offset,
            image_scale=image_scale,
            offside_field_x=offside_field_x,
        )
        if detections is not None:
            full_overlay = self._draw_detection_labels(
                full_overlay,
                detections,
                calibration,
                image_offset=image_offset,
                image_scale=image_scale,
            )

        return base_img, pitch_overlay, full_overlay

    def _build_debug_canvas(
        self,
        source_image: np.ndarray,
        calibration,
        expand_to_full_pitch: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """Create either the original image canvas or an expanded full-pitch canvas."""
        if not expand_to_full_pitch or calibration is None:
            return source_image.copy(), (0, 0), 1.0

        src_h, src_w = source_image.shape[:2]
        points = [
            (0.0, 0.0),
            (float(src_w), 0.0),
            (float(src_w), float(src_h)),
            (0.0, float(src_h)),
        ]

        pitch_corners = [
            (0.0, 0.0),
            (self.PITCH_LENGTH, 0.0),
            (self.PITCH_LENGTH, self.PITCH_WIDTH),
            (0.0, self.PITCH_WIDTH),
        ]
        for field_point in pitch_corners:
            image_point = self._project_field_to_image_float(calibration, field_point)
            if image_point is not None:
                points.append(image_point)

        if len(points) <= 4:
            return source_image.copy(), (0, 0), 1.0

        xs = [p[0] for p in points if np.isfinite(p[0])]
        ys = [p[1] for p in points if np.isfinite(p[1])]
        if not xs or not ys:
            return source_image.copy(), (0, 0), 1.0

        padding = 80
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        offset_x = float(padding - min_x)
        offset_y = float(padding - min_y)
        canvas_w = float(max(max_x, float(src_w)) + offset_x + padding)
        canvas_h = float(max(max_y, float(src_h)) + offset_y + padding)

        if canvas_w <= src_w and canvas_h <= src_h:
            return source_image.copy(), (0, 0), 1.0

        max_canvas_dim = 3200.0
        max_canvas_pixels = 8_000_000.0
        scale = min(
            1.0,
            max_canvas_dim / max(canvas_w, 1.0),
            max_canvas_dim / max(canvas_h, 1.0),
            math.sqrt(max_canvas_pixels / max(canvas_w * canvas_h, 1.0)),
        )

        canvas_w_i = max(1, int(math.ceil(canvas_w * scale)))
        canvas_h_i = max(1, int(math.ceil(canvas_h * scale)))
        offset_x_i = int(round(offset_x * scale))
        offset_y_i = int(round(offset_y * scale))
        scaled_src_w = max(1, int(round(src_w * scale)))
        scaled_src_h = max(1, int(round(src_h * scale)))

        if scale < 1.0:
            placed_image = cv2.resize(
                source_image,
                (scaled_src_w, scaled_src_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            placed_image = source_image

        canvas = np.zeros((canvas_h_i, canvas_w_i, 3), dtype=np.uint8)
        canvas[
            offset_y_i : offset_y_i + placed_image.shape[0],
            offset_x_i : offset_x_i + placed_image.shape[1],
        ] = placed_image
        return canvas, (offset_x_i, offset_y_i), scale

    @staticmethod
    def _project_field_to_image_float(
        calibration,
        field_point: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """Project a field point to original-image pixel coordinates as floats."""
        if calibration is None:
            return None
        if calibration.projection_matrix is not None:
            P = calibration.projection_matrix
            fx, fy = field_point
            p = np.array([fx - 52.5, fy - 34.0, 0.0, 1.0], dtype=np.float64)
            proj = P @ p
            if proj[2] <= 0:
                return None
            proj /= proj[2]
            if not np.isfinite(proj[0]) or not np.isfinite(proj[1]):
                return None
            return (float(proj[0]), float(proj[1]))
        if calibration.homography_inv is not None:
            pt = calibration.project_to_image(field_point)
            if pt is None:
                return None
            if not np.isfinite(pt[0]) or not np.isfinite(pt[1]):
                return None
            return (float(pt[0]), float(pt[1]))
        return None

    def _build_pitch_birds_eye_texture(
        self,
        pixels_per_meter: int = 14,
    ) -> Tuple[np.ndarray, int]:
        """Build a top-down pitch texture using the real field dimensions."""
        width_px = max(4, int(round(self.PITCH_LENGTH * pixels_per_meter)))
        height_px = max(4, int(round(self.PITCH_WIDTH * pixels_per_meter)))
        texture = np.zeros((height_px, width_px, 3), dtype=np.uint8)

        stripe_count = 10
        dark_blue = np.array([130, 70, 18], dtype=np.uint8)
        light_blue = np.array([180, 110, 28], dtype=np.uint8)

        for stripe_idx in range(stripe_count):
            x0 = int(round(stripe_idx * width_px / stripe_count))
            x1 = int(round((stripe_idx + 1) * width_px / stripe_count))
            color = light_blue if stripe_idx % 2 == 0 else dark_blue
            texture[:, x0:x1] = color

        return texture, pixels_per_meter

    def _draw_debug_pitch_surface(
        self,
        image: np.ndarray,
        calibration,
        image_offset: Tuple[int, int] = (0, 0),
        image_scale: float = 1.0,
    ) -> np.ndarray:
        """Project a filled bird's-eye pitch surface onto the original image."""
        if calibration is None:
            return image

        output = image.copy()
        img_h, img_w = output.shape[:2]
        offset_x, offset_y = image_offset

        if calibration.homography_inv is not None:
            pitch_texture, pixels_per_meter = self._build_pitch_birds_eye_texture()
            tex_to_field = np.array(
                [
                    [1.0 / pixels_per_meter, 0.0, 0.0],
                    [0.0, 1.0 / pixels_per_meter, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            translation = np.array(
                [
                    [float(image_scale), 0.0, float(offset_x)],
                    [0.0, float(image_scale), float(offset_y)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            warp_matrix = translation @ calibration.homography_inv @ tex_to_field
            warped_pitch = cv2.warpPerspective(
                pitch_texture,
                warp_matrix,
                (img_w, img_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            mask = cv2.cvtColor(warped_pitch, cv2.COLOR_BGR2GRAY) > 0
            if np.any(mask):
                blended = cv2.addWeighted(output, 0.25, warped_pitch, 0.75, 0)
                output[mask] = blended[mask]
                return output

        overlay = output.copy()
        stripe_count = 10
        stripe_colors = [
            (180, 110, 28),
            (130, 70, 18),
        ]
        for stripe_idx in range(stripe_count):
            x0 = self.PITCH_LENGTH * stripe_idx / stripe_count
            x1 = self.PITCH_LENGTH * (stripe_idx + 1) / stripe_count
            projected = []
            for field_point in (
                (x0, 0.0),
                (x1, 0.0),
                (x1, self.PITCH_WIDTH),
                (x0, self.PITCH_WIDTH),
            ):
                image_point = self._project_field_to_image(
                    calibration,
                    field_point,
                    image_offset=image_offset,
                    image_scale=image_scale,
                )
                if image_point is None:
                    projected = []
                    break
                projected.append(image_point)

            if len(projected) == 4:
                cv2.fillConvexPoly(
                    overlay,
                    np.array(projected, dtype=np.int32),
                    stripe_colors[stripe_idx % 2],
                )

        return cv2.addWeighted(overlay, 0.75, output, 0.25, 0)

    @staticmethod
    def _project_field_to_image(
        calibration,
        field_point: Tuple[float, float],
        image_offset: Tuple[int, int] = (0, 0),
        image_scale: float = 1.0,
    ) -> Optional[Tuple[int, int]]:
        """Project a field point (metres, 0-105 x 0-68) to image pixel."""
        if calibration is None:
            return None
        offset_x, offset_y = image_offset
        if calibration.projection_matrix is not None:
            P = calibration.projection_matrix
            fx, fy = field_point
            p = np.array([fx - 52.5, fy - 34.0, 0.0, 1.0])
            proj = P @ p
            if proj[2] <= 0:
                return None
            proj /= proj[2]
            return (
                int(proj[0] * image_scale + offset_x),
                int(proj[1] * image_scale + offset_y),
            )
        if calibration.homography_inv is not None:
            pt = calibration.project_to_image(field_point)
            if pt is not None:
                return (
                    int(pt[0] * image_scale + offset_x),
                    int(pt[1] * image_scale + offset_y),
                )
        return None

    def _draw_debug_pitch_lines(
        self,
        image: np.ndarray,
        calibration,
        image_offset: Tuple[int, int] = (0, 0),
        image_scale: float = 1.0,
    ) -> np.ndarray:
        """Draw projected pitch lines on the image using calibration."""
        output = image.copy()

        if calibration is None:
            cv2.putText(
                output,
                "NO CALIBRATION",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
            return output

        cyan = (255, 255, 0)
        yellow = (0, 255, 255)
        line_color = cyan
        step = 3.0

        lines_coords = [
            (0, 0, 105, 0),
            (105, 0, 105, 68),
            (105, 68, 0, 68),
            (0, 68, 0, 0),
            (52.5, 0, 52.5, 68),
            (0, 13.84, 16.5, 13.84),
            (16.5, 13.84, 16.5, 54.16),
            (16.5, 54.16, 0, 54.16),
            (88.5, 13.84, 105, 13.84),
            (88.5, 13.84, 88.5, 54.16),
            (88.5, 54.16, 105, 54.16),
            (0, 24.84, 5.5, 24.84),
            (5.5, 24.84, 5.5, 43.16),
            (5.5, 43.16, 0, 43.16),
            (99.5, 24.84, 105, 24.84),
            (99.5, 24.84, 99.5, 43.16),
            (99.5, 43.16, 105, 43.16),
        ]

        for x1, y1, x2, y2 in lines_coords:
            pts = []
            n_seg = max(int(math.hypot(x2 - x1, y2 - y1) / step), 2)
            for t_val in np.linspace(0, 1, n_seg):
                fx = x1 + t_val * (x2 - x1)
                fy = y1 + t_val * (y2 - y1)
                ip = self._project_field_to_image(
                    calibration,
                    (fx, fy),
                    image_offset=image_offset,
                    image_scale=image_scale,
                )
                if ip is not None:
                    pts.append(ip)
            for j in range(len(pts) - 1):
                cv2.line(output, pts[j], pts[j + 1], line_color, 2)

        # Center circle
        circle_pts = []
        for ang in np.linspace(0, 2 * np.pi, 72):
            fx = 52.5 + 9.15 * math.cos(ang)
            fy = 34.0 + 9.15 * math.sin(ang)
            ip = self._project_field_to_image(
                calibration,
                (fx, fy),
                image_offset=image_offset,
                image_scale=image_scale,
            )
            if ip is not None:
                circle_pts.append(ip)
        if len(circle_pts) > 2:
            cv2.polylines(output, [np.array(circle_pts, np.int32)], True, line_color, 2)

        for spot_x, spot_y in [(11, 34), (94, 34), (52.5, 34)]:
            ip = self._project_field_to_image(
                calibration,
                (spot_x, spot_y),
                image_offset=image_offset,
                image_scale=image_scale,
            )
            if ip is not None:
                cv2.circle(output, ip, 5, yellow, -1)

        # Info text
        method = calibration.method if calibration else "none"
        quality = calibration.quality_score if calibration else 0.0
        has_3d = calibration.has_3d_calibration if calibration else False
        info_color = (
            (0, 255, 0)
            if quality > 0.7
            else (0, 255, 255)
            if quality > 0.4
            else (0, 0, 255)
        )
        cv2.putText(
            output,
            f"Calib: {method}  q={quality:.2f}  3D={'YES' if has_3d else 'NO'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            info_color,
            2,
        )
        return output

    @staticmethod
    def _draw_debug_keypoints(
        image: np.ndarray,
        calibration,
        image_offset: Tuple[int, int] = (0, 0),
        image_scale: float = 1.0,
    ) -> np.ndarray:
        """Draw PnLCalib detected keypoints on the image."""
        if (
            calibration is None
            or not hasattr(calibration, "keypoints")
            or not calibration.keypoints
        ):
            return image
        output = image.copy()
        kp = (
            calibration.keypoints_raw
            or calibration.debug_info.get("keypoints_raw")
            or calibration.keypoints
        )
        img_h, img_w = output.shape[:2]
        orig_h, orig_w = (
            calibration.image_size
            or calibration.debug_info.get("image_size")
            or (img_h, img_w)
        )
        model_h, model_w = (
            calibration.model_size
            or calibration.debug_info.get("model_size")
            or (img_h, img_w)
        )
        if isinstance(kp, dict):
            offset_x, offset_y = image_offset
            for name, val in kp.items():
                if isinstance(val, dict):
                    x = val.get("x", val.get("x_1", 0))
                    y = val.get("y", val.get("y_1", 0))
                elif isinstance(val, (list, tuple)) and len(val) >= 2:
                    x, y = val[0], val[1]
                else:
                    continue

                # Normalized keypoints are first expanded to the model grid.
                if 0.0 <= x <= 1.5 and 0.0 <= y <= 1.5:
                    x *= float(model_w)
                    y *= float(model_h)

                # Raw/model-grid keypoints must then be mapped back to the
                # original image coordinates used by the overlays.
                if model_w > 0 and model_h > 0:
                    x *= float(orig_w) / float(model_w)
                    y *= float(orig_h) / float(model_h)

                ix = int(x * image_scale + offset_x)
                iy = int(y * image_scale + offset_y)
                cv2.circle(output, (ix, iy), 6, (0, 255, 0), -1)
                cv2.circle(output, (ix, iy), 6, (255, 255, 255), 1)
                cv2.putText(
                    output,
                    str(name)[:6],
                    (ix + 8, iy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                )
        n_kp = len(kp) if isinstance(kp, dict) else 0
        cv2.putText(
            output,
            f"Keypoints: {n_kp}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        return output

    def _draw_debug_offside_line(
        self,
        image: np.ndarray,
        offside_x_norm: float,
        is_offside: bool,
        calibration,
        source_image_shape: Optional[Tuple[int, int]] = None,
        image_offset: Tuple[int, int] = (0, 0),
        image_scale: float = 1.0,
        offside_field_x: Optional[float] = None,
    ) -> np.ndarray:
        """Draw the offside line on the image."""
        output = image.copy()
        h, w = output.shape[:2]
        source_h, source_w = source_image_shape or output.shape[:2]
        offset_x, offset_y = image_offset
        offside_pixel_src = offside_x_norm * source_w
        offside_pixel = offside_pixel_src * image_scale + offset_x
        line_color = (0, 0, 255) if is_offside else (0, 220, 0)

        def _extend_and_clip(
            x0: float, y0: float, x1: float, y1: float
        ) -> bool:
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < 1e-6:
                return False
            ux, uy = dx / length, dy / length
            ext = math.hypot(w, h) * 4.0 + 1.0
            pt1 = (int(round(x0 - ux * ext)), int(round(y0 - uy * ext)))
            pt2 = (int(round(x0 + ux * ext)), int(round(y0 + uy * ext)))
            ok, c1, c2 = cv2.clipLine((0, 0, w, h), pt1, pt2)
            if not ok:
                return False
            cv2.line(output, c1, c2, line_color, 3)
            return True

        drawn = False
        if (
            offside_field_x is not None
            and calibration is not None
            and calibration.projection_matrix is not None
        ):
            # Metric offside line projected with the full camera: one line
            # through the two field points (x, 0) and (x, pitch_width).
            p0 = self._project_field_to_image_float(
                calibration, (float(offside_field_x), 0.0)
            )
            p1 = self._project_field_to_image_float(
                calibration, (float(offside_field_x), self.PITCH_WIDTH)
            )
            if p0 is not None and p1 is not None:
                drawn = _extend_and_clip(
                    p0[0] * image_scale + offset_x,
                    p0[1] * image_scale + offset_y,
                    p1[0] * image_scale + offset_x,
                    p1[1] * image_scale + offset_y,
                )

        if (
            not drawn
            and calibration is not None
            and calibration.vanishing_point is not None
        ):
            # Single perspective line through the VP and the offside reference
            # point at the bottom of the image (replaces the old two-segment V).
            vx, vy = calibration.vanishing_point
            vx = vx * image_scale + offset_x
            vy = vy * image_scale + offset_y
            drawn = _extend_and_clip(vx, vy, offside_pixel, float(h))

        if not drawn:
            cv2.line(
                output, (int(offside_pixel), 0), (int(offside_pixel), h), line_color, 3
            )

        if calibration is not None:
            fp = calibration.project_to_field((offside_pixel_src, source_h * 0.7))
            if fp is not None:
                cv2.putText(
                    output,
                    f"Offside line: x={fp[0]:.1f}m",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    line_color,
                    2,
                )

        verdict = "OFFSIDE" if is_offside else "ONSIDE"
        vc = (0, 0, 255) if is_offside else (0, 220, 0)
        cv2.putText(
            output, verdict, (w - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, vc, 3
        )
        cv2.putText(
            output,
            "DEBUG OVERLAY",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        return output

    @staticmethod
    def _draw_detection_labels(
        image: np.ndarray,
        detections,
        calibration,
        image_offset: Tuple[int, int] = (0, 0),
        image_scale: float = 1.0,
    ) -> np.ndarray:
        """Draw detection bbox labels with field positions on the image."""
        output = image.copy()
        if detections is None:
            return output
        offset_x, offset_y = image_offset
        for det in detections:
            bgr = (
                VAR3DRenderer.TEAM1_BGR
                if det.team_id == 1
                else VAR3DRenderer.TEAM2_BGR
            )
            if hasattr(det, "is_goalkeeper") and det.is_goalkeeper:
                bgr = VAR3DRenderer.GK_BGR
            bx1, by1, bx2, by2 = [int(v) for v in det.bbox]
            draw_x1 = int(bx1 * image_scale + offset_x)
            draw_y1 = int(by1 * image_scale + offset_y)
            draw_x2 = int(bx2 * image_scale + offset_x)
            draw_y2 = int(by2 * image_scale + offset_y)
            cv2.rectangle(output, (draw_x1, draw_y1), (draw_x2, draw_y2), bgr, 2)
            foot_x = (bx1 + bx2) // 2
            foot_y = by2
            if calibration is not None:
                fp = calibration.project_to_field((foot_x, foot_y))
                if fp is not None:
                    label = f"({fp[0]:.1f},{fp[1]:.1f}m)"
                    cv2.circle(
                        output,
                        (
                            int(foot_x * image_scale + offset_x),
                            int(foot_y * image_scale + offset_y),
                        ),
                        5,
                        bgr,
                        -1,
                    )
                    cv2.putText(
                        output,
                        label,
                        (
                            int(foot_x * image_scale + offset_x + 8),
                            int(foot_y * image_scale + offset_y - 5),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        bgr,
                        1,
                    )
        return output

    # ---- Scene builders ---------------------------------------------------

    @staticmethod
    def _compute_upright_rotation(
        player_meshes: List["PlayerMesh"], calibration=None
    ) -> np.ndarray:
        """Shared camera->world rotation whose +Z is the players' true 'up'.

        With a full PnL calibration the broadcast camera rotation is known, and
        SAM's camera shares its image axes, so bodies get their true heading
        on the pitch from ``R``; the mesh geometry then only removes residual
        lean.  Without it, the heading assumes camera-right = pitch +X.

        Following the reference VAR pipeline, the vertical axis is derived from
        the mesh geometry (feet->head vector, averaged over the players) rather
        than assuming a fixed camera orientation.  A SAM camera is rarely
        perfectly level, so a fixed rotation leaves bodies leaning and only a
        corner of a foot touches the pitch.  Aligning the measured body-up with
        world +Z makes players stand vertically with their feet flat on z=0.

        Returns a 3x3 matrix ``R`` such that ``v_world = v_cam @ R`` puts the
        vertical axis on +Z.
        """
        # Proxy boxes are perfectly camera-aligned and would bias the average
        # body-up of real SAM meshes, so only use them when nothing else exists.
        sam_meshes = [pm for pm in player_meshes if len(pm.vertices) > 8]
        sources = sam_meshes or player_meshes

        R = getattr(calibration, "rotation_matrix", None)
        if R is not None:
            # v_scene = M @ v_cam  (CV camera axes -> Z-up scene)
            M = _PNL_TO_SCENE @ np.asarray(R, dtype=np.float64).T
            ups_s: List[np.ndarray] = []
            for pm in sources:
                v = (
                    np.asarray(pm.vertices, dtype=np.float64)
                    + np.asarray(pm.cam_translation, dtype=np.float64).reshape(1, 3)
                ) @ M.T
                z = v[:, 2]
                feet = v[z <= np.percentile(z, 10.0)]
                head = v[z >= np.percentile(z, 90.0)]
                if len(feet) and len(head):
                    u = head.mean(axis=0) - feet.mean(axis=0)
                    if np.linalg.norm(u) > 1e-6:
                        ups_s.append(u / np.linalg.norm(u))
            corr = np.eye(3)
            if ups_s:
                u = np.mean(ups_s, axis=0)
                u /= np.linalg.norm(u) + 1e-8
                axis = np.cross(u, [0.0, 0.0, 1.0])
                s = np.linalg.norm(axis)
                angle = math.atan2(s, float(u[2]))
                # Only straighten modest lean; a large angle means SAM's
                # estimate disagrees with the calibration, so trust R.
                if s > 1e-8 and angle < math.radians(35.0):
                    k = axis / s
                    Kx = np.array(
                        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
                    )
                    corr = np.eye(3) + math.sin(angle) * Kx + (
                        1 - math.cos(angle)
                    ) * (Kx @ Kx)
            return (corr @ M).T

        ups: List[np.ndarray] = []
        for pm in sources:
            v = np.asarray(pm.vertices, dtype=np.float64) + np.asarray(
                pm.cam_translation, dtype=np.float64
            ).reshape(1, 3)
            y = v[:, 1]  # SAM camera space: +Y points down
            feet = v[y >= np.percentile(y, 90.0)]
            head = v[y <= np.percentile(y, 10.0)]
            if len(feet) and len(head):
                u = head.mean(axis=0) - feet.mean(axis=0)
                nrm = np.linalg.norm(u)
                if nrm > 1e-6:
                    ups.append(u / nrm)

        if ups:
            ez = np.mean(ups, axis=0)
            nrm = np.linalg.norm(ez)
            ez = ez / nrm if nrm > 1e-6 else np.array([0.0, -1.0, 0.0])
        else:
            # Fallback: SAM +Y is down, so up is -Y.
            ez = np.array([0.0, -1.0, 0.0])

        fwd = np.array([0.0, 0.0, 1.0])  # SAM camera forward (into the scene)
        ex = np.cross(fwd, ez)
        if np.linalg.norm(ex) < 1e-6:
            ex = np.cross(np.array([1.0, 0.0, 0.0]), ez)
        ex /= np.linalg.norm(ex) + 1e-8
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey) + 1e-8
        # Columns are the world basis expressed in camera coords, so
        # ``v_cam @ R`` projects onto (ex, ey, ez) with +Z = up.
        return np.stack([ex, ey, ez], axis=1)

    @staticmethod
    def _orient_and_plant_mesh(
        pm: "PlayerMesh",
        rot_cam_to_world: np.ndarray,
        wx: float,
        wy: float,
        target_height: float = 1.80,
        attacking_side: Optional[str] = None,
    ) -> np.ndarray:
        """
        Turn a SAM camera-space mesh into a world-space, upright, grounded
        body planted at field position ``(wx, wy)``.

        ``rot_cam_to_world`` is the shared geometry-derived rotation from
        :meth:`_compute_upright_rotation` that makes the body vertical.

        Improvements over the previous force-to-1.8m approach:
          * Uses the measured body-up axis so bodies stand vertically and their
            feet sit flat on the ground (not just one toe touching).
          * Grounds on a robust low percentile of vertices (noise-tolerant).
          * Normalises height only *gently* (clamped scale) so a crouching or
            leaping player keeps their posture instead of being stretched.
          * Plants the **feet** (lowest vertices) at ``(wx, wy)`` rather than
            the body centroid, so players stand on their field position.
          * With ``attacking_side`` given, ``wx`` is the body's leading edge
            toward the attacked goal (the directional ground point used by the
            offside decision), so that edge -- not the foot centroid -- is
            placed on it and the wall touches the defender instead of cutting
            through them.
        """
        v_cam = np.asarray(pm.vertices, dtype=np.float64) + np.asarray(
            pm.cam_translation, dtype=np.float64
        ).reshape(1, 3)
        v_world = v_cam @ rot_cam_to_world

        # Ground: lowest ~2% of vertices to z=0 (robust to stray verts)
        v_world[:, 2] -= np.percentile(v_world[:, 2], 2.0)

        # Gentle height normalisation (keep posture)
        height = np.percentile(v_world[:, 2], 98.0)
        if height > 1e-3:
            scale = float(np.clip(target_height / height, 0.80, 1.25))
            v_world *= scale

        # Re-snap feet to the ground plane after scaling
        v_world[:, 2] -= v_world[:, 2].min()

        # Plant the feet at the field position
        top = max(float(v_world[:, 2].max()), 1e-3)
        foot_mask = v_world[:, 2] <= max(0.12, 0.08 * top)
        if not np.any(foot_mask):
            foot_mask = v_world[:, 2] <= np.percentile(v_world[:, 2], 5.0)
        foot_xy = v_world[foot_mask, :2].mean(axis=0)
        if attacking_side == "right":
            v_world[:, 0] += wx - v_world[:, 0].max()
        elif attacking_side == "left":
            v_world[:, 0] += wx - v_world[:, 0].min()
        else:
            v_world[:, 0] += wx - foot_xy[0]
        v_world[:, 1] += wy - foot_xy[1]
        return v_world

    def _build_scene_calibrated(
        self,
        scene,
        trimesh,
        pyrender,
        player_meshes: List[PlayerMesh],
        calibration,
        offside_x_norm: float,
        is_offside: bool,
        full_3d: bool = True,
        offside_field_x: Optional[float] = None,
        attacking_side: Optional[str] = None,
    ):
        """
        Build pitch scene using calibration (homography + optional 3D).

        When *full_3d* is True we have K, R, C from PnLCalib and can
        reconstruct the broadcast camera precisely.  When False we only
        have a homography â†' estimate a plausible oblique camera from it.

        ``offside_field_x`` is the authoritative offside line position in
        absolute field metres (0-105) taken from the offside decision. When
        provided it is used directly instead of re-projecting an image column,
        keeping the 3D wall consistent with the 2D verdict.

        World frame: origin at pitch centre (-52.5..52.5, -34..34), Z=up.
        This is PnL's world turned 180° about X (see ``_PNL_TO_SCENE``), so
        scene y = 34 - field_y; pitch markings are symmetric in y.
        """
        img_w = player_meshes[0].image_width
        img_h_est = 720  # default; overridden by real image size / K below

        # Prefer the true image height recorded by the calibrator over a
        # value guessed from the principal point.
        img_size = getattr(calibration, "image_size", None)
        if img_size and len(img_size) == 2 and img_size[0]:
            img_h_est = int(img_size[0])

        half_l = self.PITCH_LENGTH / 2.0
        half_w = self.PITCH_WIDTH / 2.0

        K = R = C = None
        if calibration.camera_matrix is not None:
            K = calibration.camera_matrix
            if not (img_size and img_size[0]):
                img_h_est = int(K[1, 2] * 2)
        if calibration.rotation_matrix is not None:
            R = calibration.rotation_matrix
        if calibration.position is not None:
            C = np.array(calibration.position, dtype=np.float64).reshape(3)

        # =============================================================
        # 1. Determine visible pitch region from bird's eye view
        # =============================================================
        # Project image corners through homography â†' field coords (0-105, 0-68)
        corners_img = [
            (0, img_h_est),  # bottom-left
            (img_w, img_h_est),  # bottom-right
            (0, img_h_est * 0.3),  # upper-left (~horizon)
            (img_w, img_h_est * 0.3),  # upper-right (~horizon)
        ]

        field_corners = []
        for cx, cy in corners_img:
            fp = calibration.project_to_field((cx, cy))
            if fp is not None:
                field_corners.append(fp)

        if field_corners:
            fc = np.array(field_corners)
            vis_x_min = max(0.0, fc[:, 0].min() - 5)
            vis_x_max = min(self.PITCH_LENGTH, fc[:, 0].max() + 5)
            vis_y_min = max(0.0, fc[:, 1].min() - 5)
            vis_y_max = min(self.PITCH_WIDTH, fc[:, 1].max() + 5)
            logger.info(
                "  Visible pitch region: x=[%.0f, %.0f] y=[%.0f, %.0f]",
                vis_x_min,
                vis_x_max,
                vis_y_min,
                vis_y_max,
            )
        else:
            # Fallback: show full pitch
            vis_x_min, vis_x_max = 0, self.PITCH_LENGTH
            vis_y_min, vis_y_max = 0, self.PITCH_WIDTH
            logger.warning("  Could not determine visible region â€” using full pitch")

        # Convert to centred coords for 3D scene

        # =============================================================
        # 2. Pitch surface (full pitch at Z=0)
        # =============================================================
        pitch_verts = np.array(
            [
                [-half_l, -half_w, 0],
                [half_l, -half_w, 0],
                [half_l, half_w, 0],
                [-half_l, half_w, 0],
            ],
            dtype=np.float64,
        )
        pitch_faces = np.array([[0, 1, 2], [0, 2, 3]])
        pitch_color = np.array([38, 96, 44, 255], dtype=np.uint8)  # grass green (RGBA)
        pm_tri = trimesh.Trimesh(vertices=pitch_verts, faces=pitch_faces, process=False)
        pm_tri.visual.face_colors = np.tile(pitch_color, (2, 1))
        scene.add(pyrender.Mesh.from_trimesh(pm_tri, smooth=False))

        # =============================================================
        # 3. Pitch markings (lines at Z=0.01)
        # =============================================================
        lw = 0.15
        white = np.array([220, 220, 240, 255], dtype=np.uint8)

        def quad_line(pts, color=white):
            v = np.array(pts, dtype=np.float64)
            f = np.array([[0, 1, 2], [0, 2, 3]])
            m = trimesh.Trimesh(vertices=v, faces=f, process=False)
            m.visual.face_colors = np.tile(color, (2, 1))
            return m

        z_l = 0.01  # line height
        # Sidelines & goal lines
        for sign in (-1, 1):
            y = sign * half_w
            scene.add(
                pyrender.Mesh.from_trimesh(
                    quad_line(
                        [
                            [-half_l, y - lw, z_l],
                            [half_l, y - lw, z_l],
                            [half_l, y + lw, z_l],
                            [-half_l, y + lw, z_l],
                        ]
                    ),
                    smooth=False,
                )
            )
            x = sign * half_l
            scene.add(
                pyrender.Mesh.from_trimesh(
                    quad_line(
                        [
                            [x - lw, -half_w, z_l],
                            [x + lw, -half_w, z_l],
                            [x + lw, half_w, z_l],
                            [x - lw, half_w, z_l],
                        ]
                    ),
                    smooth=False,
                )
            )

        # Centre line
        scene.add(
            pyrender.Mesh.from_trimesh(
                quad_line(
                    [
                        [-lw, -half_w, z_l],
                        [lw, -half_w, z_l],
                        [lw, half_w, z_l],
                        [-lw, half_w, z_l],
                    ]
                ),
                smooth=False,
            )
        )

        # Penalty area lines (in 0-105 coords then centred)
        penalty_lines = [
            # Left penalty area
            [0, 13.84, 16.5, 13.84],
            [16.5, 13.84, 16.5, 54.16],
            [0, 54.16, 16.5, 54.16],
            # Right penalty area
            [88.5, 13.84, 105, 13.84],
            [88.5, 13.84, 88.5, 54.16],
            [88.5, 54.16, 105, 54.16],
            # Left goal area
            [0, 24.84, 5.5, 24.84],
            [5.5, 24.84, 5.5, 43.16],
            [0, 43.16, 5.5, 43.16],
            # Right goal area
            [99.5, 24.84, 105, 24.84],
            [99.5, 24.84, 99.5, 43.16],
            [99.5, 43.16, 105, 43.16],
        ]
        for lx1, ly1, lx2, ly2 in penalty_lines:
            # Check if in or near visible region
            if max(lx1, lx2) < vis_x_min - 10 or min(lx1, lx2) > vis_x_max + 10:
                continue
            cx1, cy1 = lx1 - half_l, ly1 - half_w
            cx2, cy2 = lx2 - half_l, ly2 - half_w
            # Determine line orientation
            if abs(cx1 - cx2) < 0.1:  # vertical
                scene.add(
                    pyrender.Mesh.from_trimesh(
                        quad_line(
                            [
                                [cx1 - lw, min(cy1, cy2), z_l],
                                [cx1 + lw, min(cy1, cy2), z_l],
                                [cx1 + lw, max(cy1, cy2), z_l],
                                [cx1 - lw, max(cy1, cy2), z_l],
                            ]
                        ),
                        smooth=False,
                    )
                )
            else:  # horizontal
                scene.add(
                    pyrender.Mesh.from_trimesh(
                        quad_line(
                            [
                                [min(cx1, cx2), cy1 - lw, z_l],
                                [max(cx1, cx2), cy1 - lw, z_l],
                                [max(cx1, cx2), cy1 + lw, z_l],
                                [min(cx1, cx2), cy1 + lw, z_l],
                            ]
                        ),
                        smooth=False,
                    )
                )

        # =============================================================
        # 4. Offside line position (metres, absolute 0-105)
        # =============================================================
        if offside_field_x is not None and np.isfinite(offside_field_x):
            # Authoritative value from the offside decision â€” keeps the 3D
            # wall aligned with the 2D verdict.
            offside_field_x_abs = float(offside_field_x)
        else:
            # Fallback: back-project the image column at multiple heights.
            # (The offside line runs through the vanishing point, so a single
            # sample would be biased â€” take the median of several.)
            offside_pixel = offside_x_norm * img_w
            offside_field_xs = []
            for sample_y in np.linspace(img_h_est * 0.4, img_h_est * 0.9, 5):
                fp = calibration.project_to_field((offside_pixel, sample_y))
                if fp is not None:
                    offside_field_xs.append(fp[0])
            if offside_field_xs:
                offside_field_x_abs = float(np.median(offside_field_xs))
            else:
                offside_field_x_abs = offside_x_norm * self.PITCH_LENGTH

        offside_field_x_abs = float(np.clip(offside_field_x_abs, 0.0, self.PITCH_LENGTH))
        offside_cx = offside_field_x_abs - half_l  # centred
        logger.info(
            "  Offside line at field_x=%.1f (abs) / %.1f (centred)",
            offside_field_x_abs,
            offside_cx,
        )

        # Offside line on ground (bright red/green)
        ol_lw = 0.2
        ol_color = (
            np.array([255, 60, 60, 220], dtype=np.uint8)
            if is_offside
            else np.array([60, 255, 60, 220], dtype=np.uint8)
        )
        scene.add(
            pyrender.Mesh.from_trimesh(
                quad_line(
                    [
                        [offside_cx - ol_lw, -half_w, 0.02],
                        [offside_cx + ol_lw, -half_w, 0.02],
                        [offside_cx + ol_lw, half_w, 0.02],
                        [offside_cx - ol_lw, half_w, 0.02],
                    ],
                    color=ol_color,
                ),
                smooth=False,
            )
        )

        # Offside vertical plane (translucent)
        plane_h = 3.0
        op_color = np.array(
            [255, 40, 40, 70] if is_offside else [40, 255, 40, 70], dtype=np.uint8
        )
        op_verts = np.array(
            [
                [offside_cx, -half_w, 0],
                [offside_cx, half_w, 0],
                [offside_cx, half_w, plane_h],
                [offside_cx, -half_w, plane_h],
            ],
            dtype=np.float64,
        )
        # Both windings so the wall is visible from either side (pyrender
        # back-face culls, and the camera can be on either side of the wall).
        op_tri = trimesh.Trimesh(
            vertices=op_verts,
            faces=np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]]),
            process=False,
        )
        op_tri.visual.face_colors = np.tile(op_color, (4, 1))
        scene.add(pyrender.Mesh.from_trimesh(op_tri, smooth=False))

        # =============================================================
        # 5. Place player meshes at calibrated field positions
        # =============================================================
        # Shared "up" derived from the mesh geometry so every body stands
        # vertically and its feet sit flat on the pitch (see reference pipeline).
        rot_cam_to_world = self._compute_upright_rotation(
            player_meshes, calibration if full_3d else None
        )

        focus_positions: List[np.ndarray] = []
        # Planted world-space bodies, reused by the finale (gap bar, glow).
        self._planted_meshes: Dict[int, Tuple[np.ndarray, PlayerMesh]] = {}
        for pm in player_meshes:
            # --- Field position (metres, centred at pitch origin) ---------
            # Prefer the authoritative position from the offside decision so
            # the 3D placement matches the 2D verdict exactly.
            plant_side = attacking_side
            if pm.field_pos is not None and all(np.isfinite(v) for v in pm.field_pos):
                wx, wy = _field_to_scene(float(pm.field_pos[0]), float(pm.field_pos[1]))
            else:
                # Bbox bottom-centre is not the directional edge point.
                plant_side = None
                bbox = pm.bbox
                foot_x = (bbox[0] + bbox[2]) / 2.0
                foot_y = bbox[3]
                field_pt = calibration.project_to_field((foot_x, foot_y))
                if field_pt is not None:
                    wx, wy = _field_to_scene(float(field_pt[0]), float(field_pt[1]))
                else:
                    wx = (pm.image_center_x / pm.image_width - 0.5) * self.PITCH_LENGTH
                    wy = 0.0
                    logger.warning(
                        "  Player %d: project_to_field failed, fallback x=%.1f",
                        pm.player_idx,
                        wx,
                    )
            wx = float(np.clip(wx, -half_l, half_l))
            wy = float(np.clip(wy, -half_w, half_w))
            logger.info(
                "  Player %d placed at field (%.1f, %.1f)",
                pm.player_idx,
                wx + half_l,
                half_w - wy,
            )

            # SAM body -> upright, grounded, planted at (wx, wy)
            v_world = self._orient_and_plant_mesh(
                pm, rot_cam_to_world, wx, wy, attacking_side=plant_side
            )
            self._planted_meshes[pm.player_idx] = (v_world, pm)

            # Colour
            if pm.is_offside:
                rgb = self.HIGHLIGHT_COLOR
            elif pm.team_id == 1:
                rgb = self.TEAM1_COLOR
            else:
                rgb = self.TEAM2_COLOR

            color_uint8 = [int(c * 255) for c in rgb] + [255]
            vert_colors = np.tile(color_uint8, (v_world.shape[0], 1)).astype(np.uint8)
            mesh_tri = trimesh.Trimesh(
                vertices=v_world,
                faces=pm.faces,
                vertex_colors=vert_colors,
                process=False,
            )
            scene.add(pyrender.Mesh.from_trimesh(mesh_tri, smooth=True))
            focus_positions.append(np.array([wx, wy, 0.9]))

        # =============================================================
        # 6. Lights
        # =============================================================
        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        dl_pose = np.eye(4)
        dl_pose[:3, 3] = [0, 0, 50]
        scene.add(dl, pose=dl_pose)

        for lpos in [[30, 20, 40], [-30, -20, 40], [0, 0, 60]]:
            pl = pyrender.PointLight(color=[1.0, 0.98, 0.95], intensity=300.0)
            pp = np.eye(4)
            pp[:3, 3] = lpos
            scene.add(pl, pose=pp)

        # =============================================================
        # 7. Camera from calibration
        # =============================================================
        if full_3d and K is not None and R is not None and C is not None:
            flip = np.diag([1.0, -1.0, -1.0])

            # PnL world convention: x_cam = R @ (X_world - C)
            #   â†' camera-to-world rotation is R^T  (world axes in camera cols)
            # pyrender convention: OpenGL camera looks down -Z, Y up.
            # We need to flip the Y and Z axes of R^T to go from
            # computer-vision convention (Y-down, Z-forward) to OpenGL
            # (Y-up, Z-backward).
            # PnL's world is Z-down; rotate it into the Z-up scene so the
            # camera ends up above the pitch (not 15 m under it, mirrored).
            cam_rot_world = _PNL_TO_SCENE @ R.T @ flip
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = cam_rot_world
            cam_pose[:3, 3] = _PNL_TO_SCENE @ C

            # Scale intrinsics uniformly + letterbox offset, exactly like
            # _resize_frame, so the 3D render lines up with the broadcast
            # frame during the crossfade.
            s = min(self.video_width / img_w, self.video_height / max(img_h_est, 1))
            x_off = (self.video_width - int(img_w * s)) // 2
            y_off = (self.video_height - int(img_h_est * s)) // 2

            cam_obj = pyrender.IntrinsicsCamera(
                fx=K[0, 0] * s,
                fy=K[1, 1] * s,
                cx=K[0, 2] * s + x_off,
                cy=K[1, 2] * s + y_off,
                znear=1.0,
                zfar=500.0,
            )
            logger.info("  Camera from K/R/C: scene pos=%s", cam_pose[:3, 3])
        else:
            # Homography-only or fallback: estimate an oblique camera
            # using vanishing-point geometry or a sensible default.
            cam_height = 35.0
            azimuth = math.radians(25)
            elevation = math.radians(35)

            if calibration is not None and calibration.vanishing_point is not None:
                vx, vy = calibration.vanishing_point
                azimuth = math.atan2(
                    vy - img_h_est / 2.0,
                    vx - img_w / 2.0 + 1e-8,
                )
                horiz_offset = math.sqrt(
                    (vx - img_w / 2.0) ** 2 + (vy - img_h_est / 2.0) ** 2
                )
                fl_est = (
                    max(K[0, 0], K[1, 1])
                    if K is not None
                    else max(img_w, img_h_est) * 1.35
                )
                elevation = (
                    math.atan2(horiz_offset, fl_est) if fl_est > 0 else math.radians(35)
                )
                elevation = min(max(elevation, math.radians(15)), math.radians(65))

            vis_cx = (vis_x_min + vis_x_max) / 2.0 - half_l
            vis_cy = (vis_y_min + vis_y_max) / 2.0 - half_w
            cam_x = vis_cx + cam_height * math.cos(elevation) * math.cos(azimuth)
            cam_y = vis_cy + cam_height * math.cos(elevation) * math.sin(azimuth)
            cam_z = cam_height * math.sin(elevation)

            cam_pose = np.eye(4)
            cam_pos = np.array([cam_x, cam_y, cam_z])
            forward = np.array([vis_cx, vis_cy, 0.0]) - cam_pos
            forward /= np.linalg.norm(forward) + 1e-8
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            right /= np.linalg.norm(right) + 1e-8
            up = np.cross(right, forward)
            up /= np.linalg.norm(up) + 1e-8
            cam_pose[:3, 0] = right
            cam_pose[:3, 1] = up
            cam_pose[:3, 2] = -forward
            cam_pose[:3, 3] = cam_pos

            cam_obj = pyrender.PerspectiveCamera(yfov=np.radians(60.0))
            logger.info(
                "  Oblique camera (homography) at (%.1f, %.1f, %.1f)",
                cam_x,
                cam_y,
                cam_z,
            )

        return cam_pose, cam_obj, focus_positions

    def _cinematic_camera_pose(
        self,
        offside_field_x: Optional[float],
        offside_x_norm: float,
        img_w: float,
        focus_positions: List[np.ndarray],
        attacking_side: str,
        calibration,
        pyrender,
    ):
        """Frame the offside wall from the attacking side, down the goal direction.

        The camera sits behind the offside wall on the attacking half and looks
        toward the goal being attacked, so the translucent wall and the last
        defender / most-advanced attacker read clearly.  It stays on the same
        touchline as the broadcast camera (so left/right match the 2D frame
        and players are seen from the front) and its distance adapts to the
        spread of the key players so their 3D bodies fill the frame.
        Returns ``(cam_pose, cam_obj, target)`` or ``None`` if the wall
        position is unknown.
        """
        half_l = self.PITCH_LENGTH / 2.0

        # Same authoritative offside line the wall is built from, in centred
        # coords, so the framing is consistent with every other stage.
        if offside_field_x is not None and np.isfinite(offside_field_x):
            offside_abs = float(offside_field_x)
        elif calibration is not None:
            xs = []
            for sy in np.linspace(100, 600, 5):
                fp = calibration.project_to_field((offside_x_norm * img_w, sy))
                if fp is not None:
                    xs.append(fp[0])
            if not xs:
                return None
            offside_abs = float(np.median(xs))
        else:
            return None
        offside_cx = float(np.clip(offside_abs, 0.0, self.PITCH_LENGTH)) - half_l

        yfov = np.radians(40.0)
        aspect = self.video_width / max(self.video_height, 1)
        hfov = 2.0 * math.atan(math.tan(yfov / 2.0) * aspect)

        if focus_positions:
            fp = np.array(focus_positions, dtype=np.float64)
            target_y = float(np.clip(fp[:, 1].mean(), -30.0, 30.0))
            xs = np.append(fp[:, 0], offside_cx)
            spread = max(float(np.ptp(xs)), float(np.ptp(fp[:, 1])), 4.0)
        else:
            target_y = 0.0
            spread = 12.0
        target = np.array([offside_cx, target_y, 0.9])

        # goal_dir points from the wall toward the goal being attacked.
        goal_dir = 1.0 if attacking_side == "right" else -1.0

        # Stay on the broadcast camera's touchline (scene y sign of the
        # calibrated camera); default to the main-camera side (-y).
        side = -1.0
        pos = getattr(calibration, "position", None)
        if pos is not None:
            cam_y = float((_PNL_TO_SCENE @ np.asarray(pos, dtype=np.float64).reshape(3))[1])
            if abs(cam_y) > 1e-3:
                side = math.copysign(1.0, cam_y)

        # Distance so the key players + wall span ~60% of the frame width.
        dist = float(np.clip(spread / (2.0 * math.tan(hfov / 2.0) * 0.6), 10.0, 45.0))
        elevation = math.radians(24.0)
        h_dir = np.array([-goal_dir * 0.75, side * 0.66])
        h_dir /= np.linalg.norm(h_dir)
        cam_pos = np.array(
            [
                target[0] + dist * math.cos(elevation) * h_dir[0],
                target[1] + dist * math.cos(elevation) * h_dir[1],
                target[2] + dist * math.sin(elevation),
            ]
        )

        forward = target - cam_pos
        forward /= np.linalg.norm(forward) + 1e-8
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-8

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos

        cam_obj = pyrender.PerspectiveCamera(
            yfov=yfov, aspectRatio=aspect, znear=0.3, zfar=500.0
        )
        logger.info(
            "  Cinematic camera: attacking_side=%s, wall_cx=%.1f, dist=%.1f, pos=%s",
            attacking_side,
            offside_cx,
            dist,
            np.round(cam_pos, 1),
        )
        return pose, cam_obj, target

    def _build_scene_cam_space(
        self,
        scene,
        trimesh,
        pyrender,
        player_meshes: List[PlayerMesh],
        offside_x_norm: float,
        is_offside: bool,
        fl: float,
        img_w: float,
    ):
        """
        Fallback: build scene in SAM camera space (no calibration).

        Returns:
            (cam_pose, cam_object, focus_positions)
        """
        # Convert vertices to pyrender space (flip Y, Z)
        player_verts_pr = []
        all_feet_y = []
        for pm in player_meshes:
            v_cam = pm.vertices + pm.cam_translation
            v_pr = v_cam.copy()
            v_pr[:, 1] *= -1
            v_pr[:, 2] *= -1
            player_verts_pr.append((v_pr, pm))
            all_feet_y.append(v_pr[:, 1].min())

        ground_y = min(all_feet_y) - 0.05
        avg_z_sam = np.mean([pm.cam_translation[2] for pm in player_meshes])
        offside_pixel_x = offside_x_norm * img_w
        cx_img = img_w / 2.0
        offside_x = (offside_pixel_x - cx_img) * avg_z_sam / fl

        logger.info(
            "  Fallback scene: ground_y=%.2f, offside_x=%.2f", ground_y, offside_x
        )

        # Ground plane
        ext = 50.0
        gp_verts = np.array(
            [
                [-ext, ground_y, 5],
                [ext, ground_y, 5],
                [ext, ground_y, -100],
                [-ext, ground_y, -100],
            ],
            dtype=np.float64,
        )
        gp_mesh = trimesh.Trimesh(
            vertices=gp_verts,
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )
        gp_mesh.visual.face_colors = np.tile(
            np.array([25, 45, 120, 255], dtype=np.uint8),
            (2, 1),
        )
        scene.add(pyrender.Mesh.from_trimesh(gp_mesh, smooth=False))

        # Offside plane
        plane_h = 4.0
        op_color = np.array(
            [255, 40, 40, 90] if is_offside else [40, 255, 40, 90], dtype=np.uint8
        )
        op_verts = np.array(
            [
                [offside_x, ground_y, 5],
                [offside_x, ground_y, -100],
                [offside_x, ground_y + plane_h, -100],
                [offside_x, ground_y + plane_h, 5],
            ],
            dtype=np.float64,
        )
        op_mesh = trimesh.Trimesh(
            vertices=op_verts,
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )
        op_mesh.visual.face_colors = np.tile(op_color, (2, 1))
        scene.add(pyrender.Mesh.from_trimesh(op_mesh, smooth=False))

        # Offside line on ground
        lw = 0.08
        ol_verts = np.array(
            [
                [offside_x - lw, ground_y + 0.01, 5],
                [offside_x + lw, ground_y + 0.01, 5],
                [offside_x + lw, ground_y + 0.01, -100],
                [offside_x - lw, ground_y + 0.01, -100],
            ],
            dtype=np.float64,
        )
        ol_mesh = trimesh.Trimesh(
            vertices=ol_verts,
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )
        ol_mesh.visual.face_colors = np.tile(
            np.array([255, 60, 60, 220], dtype=np.uint8),
            (2, 1),
        )
        scene.add(pyrender.Mesh.from_trimesh(ol_mesh, smooth=False))

        # Player meshes
        focus_positions = []
        for v_pr, pm in player_verts_pr:
            # Ground each player individually so its feet touch the pitch plane
            # (otherwise players above the global lowest foot float in the air).
            v_pr = v_pr.copy()
            v_pr[:, 1] += ground_y - v_pr[:, 1].min()
            if pm.is_offside:
                rgb = self.HIGHLIGHT_COLOR
            elif pm.team_id == 1:
                rgb = self.TEAM1_COLOR
            else:
                rgb = self.TEAM2_COLOR
            color_uint8 = [int(c * 255) for c in rgb] + [255]
            vert_colors = np.tile(color_uint8, (v_pr.shape[0], 1)).astype(np.uint8)
            mesh_tri = trimesh.Trimesh(
                vertices=v_pr,
                faces=pm.faces,
                vertex_colors=vert_colors,
                process=False,
            )
            scene.add(pyrender.Mesh.from_trimesh(mesh_tri, smooth=True))
            focus_positions.append(v_pr.mean(axis=0))

        # Lights
        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        dl_pose = np.eye(4)
        dl_pose[:3, 3] = [0, 5, 0]
        scene.add(dl, pose=dl_pose)
        for lpos in [[5, 3, -10], [-5, 3, -10], [0, 5, -20]]:
            pl = pyrender.PointLight(color=[1.0, 0.98, 0.95], intensity=40.0)
            pp = np.eye(4)
            pp[:3, 3] = lpos
            scene.add(pl, pose=pp)

        # Camera
        cam_pose = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        cam_obj = pyrender.IntrinsicsCamera(
            fx=fl,
            fy=fl,
            cx=self.video_width / 2.0,
            cy=self.video_height / 2.0,
            znear=0.1,
            zfar=200.0,
        )

        return cam_pose, cam_obj, focus_positions

    # ---- mesh-on-image overlay -------------------------------------------

    def _render_mesh_on_image(
        self,
        image: np.ndarray,
        player_meshes: List[PlayerMesh],
        trimesh,
        pyrender,
        is_offside: bool,
        projection_size: Optional[Tuple[int, int]] = None,
        image_offset: Tuple[int, int] = (0, 0),
        image_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Render 3D meshes overlaid on the original image.

        Uses direct 2D projection (no OpenGL) with the SAM 3D Body
        focal_length and cam_translation to project mesh triangles
        onto the image.  Fast and fully Windows-compatible.
        """
        h, w = image.shape[:2]
        proj_h, proj_w = projection_size or (h, w)
        offset_x, offset_y = image_offset

        # Create an overlay layer for the meshes
        overlay = image.copy()

        logger.info("Rendering mesh overlay on original image (%dx%d)", w, h)

        for pm in player_meshes:
            try:
                # Choose colour (BGR for OpenCV)
                if pm.is_offside:
                    rgb = self.HIGHLIGHT_COLOR
                elif pm.team_id == 1:
                    rgb = self.TEAM1_COLOR
                else:
                    rgb = self.TEAM2_COLOR
                bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

                # --- Project vertices to 2D ---
                # SAM 3D Body: camera at origin, Y-down, Z-forward
                verts_3d = pm.vertices + pm.cam_translation  # (V, 3)
                fl = pm.focal_length
                cx = proj_w / 2.0
                cy = proj_h / 2.0

                # Perspective projection:  x_2d = fx * X/Z + cx
                z = verts_3d[:, 2]
                valid = z > 0.1  # behind camera check
                x_2d = np.full(len(verts_3d), -1.0)
                y_2d = np.full(len(verts_3d), -1.0)
                x_2d[valid] = fl * verts_3d[valid, 0] / z[valid] + cx
                y_2d[valid] = fl * verts_3d[valid, 1] / z[valid] + cy

                # Align mesh feet with bbox bottom.
                # SAM cam_translation centres the body (roughly hip level),
                # so the projected mesh floats above the player's feet.
                # We shift the entire projection so that the lowest projected
                # vertex (feet) lands at the bbox bottom (y2).
                bbox_y2 = pm.bbox[3]
                if valid.any():
                    mesh_feet_y = y_2d[valid].max()  # max y = lowest point (Y-down)
                    y_shift = bbox_y2 - mesh_feet_y
                    y_2d[valid] += y_shift

                x_2d[valid] = x_2d[valid] * image_scale + offset_x
                y_2d[valid] = y_2d[valid] * image_scale + offset_y

                # Compute face depth for painter's algorithm (back-to-front)
                faces = pm.faces
                face_z = z[faces].mean(axis=1)
                face_order = np.argsort(-face_z)  # far to near

                # Simple shading: use face normal dot product with light direction
                v0 = verts_3d[faces[:, 0]]
                v1 = verts_3d[faces[:, 1]]
                v2 = verts_3d[faces[:, 2]]
                normals = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
                norm_len[norm_len < 1e-8] = 1e-8
                normals = normals / norm_len

                # Light from camera direction (0, 0, 1) + slight top-down (0, -0.3, 0.7)
                light_dir = np.array([0.0, -0.3, 0.7])
                light_dir /= np.linalg.norm(light_dir)
                intensity = np.abs(normals @ light_dir)  # 0..1
                # Ambient + diffuse
                intensity = np.clip(0.3 + 0.7 * intensity, 0, 1)

                # Draw faces in depth order (painter's algorithm)
                count_drawn = 0
                for fi in face_order:
                    f = faces[fi]
                    # Skip if any vertex is behind camera
                    if not (valid[f[0]] and valid[f[1]] and valid[f[2]]):
                        continue

                    pts = np.array(
                        [
                            [int(x_2d[f[0]]), int(y_2d[f[0]])],
                            [int(x_2d[f[1]]), int(y_2d[f[1]])],
                            [int(x_2d[f[2]]), int(y_2d[f[2]])],
                        ]
                    )

                    # Skip faces outside image bounds
                    if (
                        pts[:, 0].max() < 0
                        or pts[:, 0].min() >= w
                        or pts[:, 1].max() < 0
                        or pts[:, 1].min() >= h
                    ):
                        continue

                    # Shade the face color
                    shade = intensity[fi]
                    face_color = (
                        int(bgr[0] * shade),
                        int(bgr[1] * shade),
                        int(bgr[2] * shade),
                    )
                    cv2.fillPoly(overlay, [pts], face_color)
                    count_drawn += 1

                logger.info(
                    "  Player %d: projected %d/%d faces",
                    pm.player_idx,
                    count_drawn,
                    len(faces),
                )

            except Exception as exc:
                logger.warning(
                    "  Could not overlay mesh for player %d: %s",
                    pm.player_idx,
                    exc,
                    exc_info=True,
                )

        # Blend overlay onto original (semi-transparent mesh)
        alpha = 0.75
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        logger.info("  Mesh overlay composited successfully")

        # Add "VAR 3D REPLAY" overlay
        self._add_var_overlay(result, is_offside)
        return result

    # ---- overlay helpers --------------------------------------------------

    def _resize_frame(self, image: np.ndarray) -> np.ndarray:
        """Resize an image to the video dimensions, letterboxing if needed."""
        h, w = image.shape[:2]
        target_w, target_h = self.video_width, self.video_height
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Letterbox
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        return canvas

    @staticmethod
    def _add_var_overlay(frame: np.ndarray, is_offside: bool):
        """Draw a broadcast-style banner + verdict pill on a rendered frame."""
        w = frame.shape[1]

        # Semi-transparent banner at top
        banner_h = 52
        banner = frame[:banner_h, :, :].copy()
        tint = np.full_like(banner, (10, 10, 30))
        cv2.addWeighted(tint, 0.78, banner, 0.22, 0, banner)
        frame[:banner_h, :, :] = banner

        # Coloured accent line under the banner (verdict-tinted)
        accent = (60, 60, 255) if is_offside else (60, 210, 60)
        cv2.line(frame, (0, banner_h), (w, banner_h), accent, 3, cv2.LINE_AA)

        # "VAR 3D REPLAY" title
        cv2.putText(
            frame,
            "VAR 3D REPLAY",
            (20, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Verdict pill (filled rounded-ish rectangle with white text)
        verdict = "OFFSIDE" if is_offside else "ONSIDE"
        pill_color = (40, 40, 220) if is_offside else (40, 180, 40)
        font, fscale, fthick = cv2.FONT_HERSHEY_DUPLEX, 0.85, 1
        (tw, th), _ = cv2.getTextSize(verdict, font, fscale, fthick)
        pad_x, pad_y = 18, 10
        x2 = w - 16
        x1 = x2 - (tw + 2 * pad_x)
        cy = banner_h // 2
        y1 = cy - (th // 2 + pad_y)
        y2 = cy + (th // 2 + pad_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), pill_color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            verdict,
            (x1 + pad_x, cy + th // 2),
            font,
            fscale,
            (255, 255, 255),
            fthick,
            cv2.LINE_AA,
        )

    def _compute_sweep_pose(
        self,
        cam_base_pose: np.ndarray,
        centre: np.ndarray,
        z_up: bool,
        orbit_frac: float,
        down_frac: float,
    ) -> np.ndarray:
        """Compute a camera pose for the sweep animation.

        Starts from *cam_base_pose* and applies:
          - A lateral orbit (orbit_frac: 0â†'0.15 gives ~15Â° swing)
          - A downward pitch toward near-overhead (down_frac: 0â†'1)

        In Z-up mode (pitch scene) the orbit is in the XY ground plane and
        the camera lifts toward directly above the focus.  In Y-up mode
        (SAM cam-space) the orbit is in the XZ ground plane.
        """
        cam_pos_start = cam_base_pose[:3, 3].copy()
        orbit_d = np.linalg.norm(cam_pos_start - centre)
        if orbit_d < 1.0:
            orbit_d = 40.0

        if z_up:
            world_up = np.array([0.0, 0.0, 1.0])
            dir_start = cam_pos_start - centre
            start_angle = math.atan2(dir_start[1], dir_start[0])

            orbit_angle = orbit_frac * math.pi * 0.5
            new_angle = start_angle + orbit_angle

            start_elevation = math.atan2(
                dir_start[2],
                math.sqrt(dir_start[0] ** 2 + dir_start[1] ** 2) + 1e-8,
            )
            target_elevation = math.radians(75)
            elevation = start_elevation + down_frac * (
                target_elevation - start_elevation
            )

            horiz_dist = orbit_d * math.cos(elevation)
            cam_x = centre[0] + horiz_dist * math.cos(new_angle)
            cam_y = centre[1] + horiz_dist * math.sin(new_angle)
            cam_z = centre[2] + orbit_d * math.sin(elevation)
        else:
            world_up = np.array([0.0, 1.0, 0.0])
            dir_start = cam_pos_start - centre
            dir_xz = np.array([dir_start[0], 0, dir_start[2]])
            sa = math.atan2(dir_xz[0], dir_xz[2])

            orbit_angle = orbit_frac * math.pi * 0.5
            new_angle = sa + orbit_angle

            start_elev = math.atan2(
                dir_start[1],
                math.sqrt(dir_start[0] ** 2 + dir_start[2] ** 2) + 1e-8,
            )
            target_elev = math.radians(70)
            elevation = start_elev + down_frac * (target_elev - start_elev)

            horiz_d = orbit_d * math.cos(elevation)
            cam_x = centre[0] + horiz_d * math.sin(new_angle)
            cam_y = centre[1] + orbit_d * math.sin(elevation)
            cam_z = centre[2] + horiz_d * math.cos(new_angle)

        cam_pos = np.array([cam_x, cam_y, cam_z])
        forward = centre - cam_pos
        forward /= np.linalg.norm(forward) + 1e-8

        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([1.0, 0.0, 0.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-8

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos
        return pose


# ---------------------------------------------------------------------------
# High-level convenience: from AnalysisResult to video
# ---------------------------------------------------------------------------


def render_var_3d_video(
    image_path: str,
    analysis_result,
    output_path: str = "results/offside_3d.mp4",
    device: str = "cuda",
    extra_player_bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Optional[str]:
    """
    End-to-end: take an AnalysisResult, recover key player meshes, render
    the multi-phase 3D VAR video.

    Args:
        image_path: Path to the original image.
        analysis_result: An ``AnalysisResult`` from ``OffsidePipeline``.
        output_path: Where to write the MP4 file.
        device: Torch device for SAM 3D Body.
        extra_player_bboxes: Optional dict ``{player_idx: (x1,y1,x2,y2)}``
            of additional players to include in the 3D scene.  When *None*
            only the last defender and the most-offside attacker are shown.

    Phases:
        1.  Original image with 3D mesh overlay on players (3s)
        2.  Crossfade transition into the 3D pitch scene (1s)
        3.  Camera sweep down to near-perpendicular view of players
    """
    from var_offside.config import get_settings

    settings = get_settings()
    cfg = settings.var_3d

    result = analysis_result
    if result.offside_result is None:
        logger.warning("No offside result â€” skipping 3D video.")
        return None

    source_image = result.image
    if source_image is None:
        source_image = cv2.imread(image_path)
        if source_image is None:
            logger.warning("Could not load image for 3D video: %s", image_path)
            return None

    offside_res = result.offside_result
    line = offside_res.offside_line
    h, w = source_image.shape[:2]

    # ---- Prepare Phase-1 annotated image ----
    annotated_image = None
    try:
        from var_offside.visualization.offside_viz import OffsideVisualizer

        viz = OffsideVisualizer()
        annotated_image = viz.visualize_offside_result(
            source_image,
            offside_res,
            calibration=result.calibration,
        )
    except Exception as exc:
        logger.warning("Could not generate annotated image for Phase 1: %s", exc)

    # ---- Identify key players ----
    key_indices: Dict[int, Tuple[float, float, float, float]] = {}

    # Default: last defender + most-offside attacker only
    def_idx = line.defining_player_idx
    if def_idx < len(result.detections):
        key_indices[def_idx] = result.detections[def_idx].bbox

    most_adv = offside_res.most_offside_player
    if most_adv is None and offside_res.player_statuses:
        most_adv = max(offside_res.player_statuses, key=lambda s: s.margin_px)
    if most_adv is not None:
        aidx = most_adv.player_idx
        if aidx < len(result.detections):
            key_indices[aidx] = result.detections[aidx].bbox

    # Merge caller-supplied bboxes (for future use)
    if extra_player_bboxes:
        key_indices.update(extra_player_bboxes)

    if not key_indices:
        logger.warning("No key players identified â€” skipping 3D video.")
        return None

    logger.info(
        "Key players for 3D: %s",
        {k: tuple(round(v, 1) for v in bbox) for k, bbox in key_indices.items()},
    )

    # ---- Mesh recovery ----
    player_meshes: List[PlayerMesh] = []
    try:
        recovery = MeshRecovery(
            checkpoint_path=cfg.sam3d_checkpoint,
            config_path=cfg.sam3d_config,
            yolo_path=cfg.sam3d_yolo,
            mhr_path=cfg.sam3d_mhr,
            device=device,
        )
        player_meshes = recovery.recover(
            image_path,
            key_indices,
            image_width=w,
        )
    except Exception as exc:
        logger.warning(
            "SAM 3D Body mesh recovery failed, falling back to proxy bodies: %s",
            exc,
            exc_info=True,
        )

    player_meshes = _merge_with_proxy_meshes(
        player_meshes,
        key_indices,
        source_image.shape,
        calibration=result.calibration,
    )

    if not player_meshes:
        logger.warning("Could not build any player meshes for 3D video.")
        return None

    logger.info("Got %d mesh(es) â€” proceeding to render", len(player_meshes))

    # Annotate team, offside flag and authoritative field position
    _annotate_player_meshes(player_meshes, result)

    # Offside line normalised x (fallback) + authoritative field metres
    offside_x_norm = line.image_x / w
    offside_field_x = line.field_x
    attacking_side = getattr(offside_res, "attacking_side", None)

    # ---- Render video ----
    renderer = VAR3DRenderer(
        video_width=cfg.video_width,
        video_height=cfg.video_height,
        duration_sec=cfg.video_duration_sec,
        fps=cfg.video_fps,
        camera_elevation_deg=cfg.camera_elevation_deg,
        camera_distance=cfg.camera_distance,
        pitch_length=cfg.pitch_length,
        pitch_width=cfg.pitch_width,
    )

    try:
        return renderer.render(
            player_meshes,
            offside_x_norm,
            offside_res.any_offside,
            output_path,
            original_image=source_image,
            annotated_image=annotated_image,
            calibration=result.calibration,
            offside_field_x=offside_field_x,
            attacking_side=attacking_side,
            attacker_idx=most_adv.player_idx if most_adv is not None else None,
            defender_idx=def_idx,
            should_stop=should_stop,
        )
    except RenderCancelled:
        raise
    except Exception as exc:
        logger.error("3D VAR rendering failed: %s", exc, exc_info=True)
        return None


def render_debug_3d_video(
    image_path: str,
    analysis_result,
    output_path: str = "results/debug_overlay.mp4",
    device: str = "cuda",
    extra_player_bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Optional[str]:
    """
    Render a debug overlay video on the original broadcast image.

    Shows PnLCalib-projected pitch lines and keypoints on the original image,
    then overlays the recovered 3D player meshes in-place, followed by the
    perspective offside line. This is intended as a calibration/debug view.

    Phases:
        1.  Original image (2s)
        2.  Fade in: projected pitch lines and keypoints from calibration (3s)
        3.  Fade in: player meshes + offside line (2s)
        4.  Hold complete overlay (3s)
    """
    from var_offside.config import get_settings

    settings = get_settings()
    cfg = settings.var_3d

    result = analysis_result
    if result.offside_result is None:
        logger.warning("No offside result — skipping debug overlay.")
        return None

    source_image = result.image
    if source_image is None:
        source_image = cv2.imread(image_path)
        if source_image is None:
            logger.warning("Could not load image for debug overlay: %s", image_path)
            return None

    offside_res = result.offside_result
    line = offside_res.offside_line
    h, w = source_image.shape[:2]
    offside_x_norm = line.image_x / w

    key_indices: Dict[int, Tuple[float, float, float, float]] = {}

    def_idx = line.defining_player_idx
    if def_idx < len(result.detections):
        key_indices[def_idx] = result.detections[def_idx].bbox

    most_adv = offside_res.most_offside_player
    if most_adv is None and offside_res.player_statuses:
        most_adv = max(offside_res.player_statuses, key=lambda s: s.margin_px)
    if most_adv is not None:
        aidx = most_adv.player_idx
        if aidx < len(result.detections):
            key_indices[aidx] = result.detections[aidx].bbox

    if extra_player_bboxes:
        key_indices.update(extra_player_bboxes)

    if not key_indices:
        logger.warning("No key players identified — skipping debug overlay.")
        return None

    player_meshes: List[PlayerMesh] = []
    try:
        recovery = MeshRecovery(
            checkpoint_path=cfg.sam3d_checkpoint,
            config_path=cfg.sam3d_config,
            yolo_path=cfg.sam3d_yolo,
            mhr_path=cfg.sam3d_mhr,
            device=device,
        )
        player_meshes = recovery.recover(
            image_path,
            key_indices,
            image_width=w,
        )
    except Exception as exc:
        logger.warning(
            "SAM 3D Body mesh recovery failed, falling back to proxy bodies: %s",
            exc,
            exc_info=True,
        )

    player_meshes = _merge_with_proxy_meshes(
        player_meshes,
        key_indices,
        source_image.shape,
        calibration=result.calibration,
    )

    if not player_meshes:
        logger.warning("Could not build any player meshes for debug overlay.")
        return None

    _annotate_player_meshes(player_meshes, result)

    renderer = VAR3DRenderer(
        video_width=cfg.video_width,
        video_height=cfg.video_height,
        duration_sec=9.0,
        fps=cfg.video_fps,
        pitch_length=cfg.pitch_length,
        pitch_width=cfg.pitch_width,
    )

    try:
        output_path_obj = Path(output_path)
        debug_image_path = str(
            output_path_obj.with_name(f"{output_path_obj.stem}_zoomout.png")
        )
        result.debug_3d_image_path = renderer.render_debug_image(
            player_meshes,
            offside_x_norm,
            offside_res.any_offside,
            debug_image_path,
            original_image=source_image,
            calibration=result.calibration,
            detections=result.detections,
            offside_field_x=line.field_x,
        )
        result.debug_3d_video_path = renderer.render_debug(
            player_meshes,
            offside_x_norm,
            offside_res.any_offside,
            output_path,
            original_image=source_image,
            calibration=result.calibration,
            detections=result.detections,
            offside_field_x=line.field_x,
            should_stop=should_stop,
        )
        return result.debug_3d_video_path
    except RenderCancelled:
        raise
    except Exception as exc:
        logger.error("Debug overlay rendering failed: %s", exc, exc_info=True)
        return None


def render_zoomout_3d_video(
    image_path: str,
    analysis_result,
    output_path: str = "results/zoomout_var.mp4",
    device: str = "cuda",
    extra_player_bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    show_offside_line: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Optional[str]:
    """
    Render a zoom-out VAR video with pitch overlay and 3D player meshes.

    Phases (~10s):
        1. Original image (2s)
        2. Crossfade: pitch + keypoints + meshes (+ offside line in 2nd half) (2s)
        3. Linear zoom-out to full-pitch view (3s)
        4. Hold full-pitch view (3s)

    Args:
        image_path: Path to the original image.
        analysis_result: An ``AnalysisResult`` from ``OffsidePipeline``.
        output_path: Where to write the MP4 file.
        device: Torch device for SAM 3D Body.
        extra_player_bboxes: Optional dict of additional player bboxes.
        show_offside_line: Whether to show the offside line overlay.
    """
    from var_offside.config import get_settings

    import sys

    sys.stdout.flush()

    settings = get_settings()
    cfg = settings.var_3d

    result = analysis_result
    if result.offside_result is None:
        logger.warning("No offside result — skipping zoomout video.")
        return None

    source_image = result.image
    if source_image is None:
        source_image = cv2.imread(image_path)
        if source_image is None:
            logger.warning("Could not load image for zoomout video: %s", image_path)
            return None

    offside_res = result.offside_result
    line = offside_res.offside_line
    h, w = source_image.shape[:2]
    offside_x_norm = line.image_x / w

    key_indices: Dict[int, Tuple[float, float, float, float]] = {}

    def_idx = line.defining_player_idx
    if def_idx < len(result.detections):
        key_indices[def_idx] = result.detections[def_idx].bbox

    most_adv = offside_res.most_offside_player
    if most_adv is None and offside_res.player_statuses:
        most_adv = max(offside_res.player_statuses, key=lambda s: s.margin_px)
    if most_adv is not None:
        aidx = most_adv.player_idx
        if aidx < len(result.detections):
            key_indices[aidx] = result.detections[aidx].bbox

    if extra_player_bboxes:
        key_indices.update(extra_player_bboxes)

    if not key_indices:
        logger.warning("No key players identified — skipping zoomout video.")
        return None

    player_meshes: List[PlayerMesh] = []
    try:
        recovery = MeshRecovery(
            checkpoint_path=cfg.sam3d_checkpoint,
            config_path=cfg.sam3d_config,
            yolo_path=cfg.sam3d_yolo,
            mhr_path=cfg.sam3d_mhr,
            device=device,
        )
        player_meshes = recovery.recover(
            image_path,
            key_indices,
            image_width=w,
        )
    except Exception as exc:
        logger.warning(
            "SAM 3D Body mesh recovery failed, falling back to proxy bodies: %s",
            exc,
            exc_info=True,
        )

    player_meshes = _merge_with_proxy_meshes(
        player_meshes,
        key_indices,
        source_image.shape,
        calibration=result.calibration,
    )

    if not player_meshes:
        logger.warning("Could not build any player meshes for zoomout video.")
        return None

    _annotate_player_meshes(player_meshes, result)

    renderer = VAR3DRenderer(
        video_width=cfg.video_width,
        video_height=cfg.video_height,
        duration_sec=10.0,
        fps=cfg.video_fps,
        pitch_length=cfg.pitch_length,
        pitch_width=cfg.pitch_width,
    )

    try:
        return renderer.render_zoomout_video(
            player_meshes,
            offside_x_norm,
            offside_res.any_offside,
            output_path,
            original_image=source_image,
            calibration=result.calibration,
            detections=result.detections,
            show_offside_line=show_offside_line,
            offside_field_x=line.field_x,
            attacking_side=getattr(offside_res, "attacking_side", None),
            should_stop=should_stop,
        )
    except RenderCancelled:
        raise
    except Exception as exc:
        logger.error("Zoomout VAR video rendering failed: %s", exc, exc_info=True)
        print(f"ERROR: Zoomout VAR video rendering failed: {exc}")
        import traceback

        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def _main():
    """
    Standalone entry point.

    Usage:
        python -m var_offside.visualization.var_3d \\
            --image images/offside.jpg \\
            --output results/offside_3d.mp4 \\
            --attacking-team 1 --attacking-side right
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="3D VAR Offside Video Generator (standalone)",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output",
        default="results/offside_3d.mp4",
        help="Output video path (default: results/offside_3d.mp4).",
    )
    parser.add_argument(
        "--attacking-team",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which team is attacking (1 or 2).",
    )
    parser.add_argument(
        "--attacking-side",
        default="left",
        choices=["left", "right"],
        help="Direction of attack.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (cuda / cpu).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Video duration in seconds (default: 10s for multi-phase).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate debug video (static broadcast camera, no orbit).",
    )
    parser.add_argument(
        "--zoomout",
        action="store_true",
        help="Generate zoom-out VAR video (pitch overlay + zoom to full pitch).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Run the full offside pipeline first
    from var_offside.pipeline import OffsidePipeline

    pipeline = OffsidePipeline(device=args.device)

    print(f"Running offside analysis on {args.image} ...")
    analysis = pipeline.analyze(
        args.image,
        attacking_team=args.attacking_team,
        attacking_side=args.attacking_side,
        generate_visualization=False,
        verbose=True,
    )

    if analysis.errors:
        print(f"Pipeline errors: {analysis.errors}")

    if analysis.offside_result is None:
        print("Could not compute offside result. Exiting.")
        sys.exit(1)

    summary = pipeline.get_summary(analysis)
    print(summary)

    # Override duration/fps from CLI
    from var_offside.config import get_settings

    settings = get_settings()
    settings.var_3d.video_duration_sec = args.duration
    settings.var_3d.video_fps = args.fps

    if args.zoomout:
        output = (
            args.output.replace(".mp4", "_zoomout.mp4")
            if args.output.endswith(".mp4")
            else args.output
        )
        print(f"\nGenerating ZOOM-OUT VAR video -> {output} ...")
        result_path = render_zoomout_3d_video(
            image_path=args.image,
            analysis_result=analysis,
            output_path=output,
            device=args.device,
        )
    elif args.debug:
        output = (
            args.output.replace(".mp4", "_debug.mp4")
            if args.output.endswith(".mp4")
            else args.output
        )
        print(f"\nGenerating DEBUG 3D video -> {output} ...")
        result_path = render_debug_3d_video(
            image_path=args.image,
            analysis_result=analysis,
            output_path=output,
            device=args.device,
        )
    else:
        print(f"\nGenerating 3D VAR video -> {args.output} ...")
        result_path = render_var_3d_video(
            image_path=args.image,
            analysis_result=analysis,
            output_path=args.output,
            device=args.device,
        )

    if result_path:
        print(f"Done! Video saved to: {result_path}")
    else:
        print("3D video generation failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    _main()
