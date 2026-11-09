"""
Load a SAM 3D Body estimator for the 3D replay.

SAM 3D Body (Meta, SAM License) is not bundled. Clone it into ``third_party/``:

* ``third_party/Fast-SAM-3D-Body`` (MIT, optional, about 10x faster), or
* ``third_party/sam-3d-body`` (the original release).

The fast clone is used when present. Checkpoints come from the Hugging Face
repo ``facebook/sam-3d-body-dinov3``; see ``docs/getting-started/models.md``.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Optional

import numpy as np

from var_offside.config.paths import third_party_dir

logger = logging.getLogger(__name__)

# The upstream renderer defaults to EGL, which only exists on Linux.
if sys.platform != "win32" and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"


class YOLOHumanDetector:
    """Ultralytics person detector in the interface SAM 3D Body expects."""

    def __init__(self, model_path: str = "yolo11l.pt", device: str = "cuda"):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.model.to(device)

    def run_human_detection(
        self,
        img: np.ndarray,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        default_to_full_image: bool = False,
    ) -> np.ndarray:
        """Return person boxes as an ``(N, 4)`` float array of ``x1, y1, x2, y2``."""
        results = self.model(img, conf=bbox_thr, iou=nms_thr, verbose=False)[0]
        boxes = [
            box.xyxy[0].cpu().numpy()
            for box in results.boxes
            if int(box.cls) == det_cat_id
        ]
        if not boxes and default_to_full_image:
            h, w = img.shape[:2]
            boxes.append(np.array([0, 0, w, h], dtype=np.float32))
        return np.array(boxes, dtype=np.float32).reshape(-1, 4) if boxes else np.array([])


def _add_to_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def _human_detector(yolo_path: Optional[str], device: str):
    if yolo_path and os.path.exists(yolo_path):
        logger.info("Loading YOLO person detector from %s", yolo_path)
        return YOLOHumanDetector(model_path=yolo_path, device=device)
    return None  # boxes are passed in directly


def _setup_original(
    checkpoint_path: str,
    config_path: Optional[str],
    mhr_path: str,
    yolo_path: Optional[str],
    device: str,
):
    import torch

    root = third_party_dir("sam-3d-body", "sam3_body3d/sam-3d-body")
    if not root.is_dir():
        raise FileNotFoundError(
            f"SAM 3D Body code not found at {root}. Clone "
            "https://github.com/facebookresearch/sam-3d-body into third_party/ "
            "(see docs/getting-started/models.md)."
        )
    _add_to_path(str(root))

    from sam_3d_body import SAM3DBodyEstimator
    from sam_3d_body.models.meta_arch import SAM3DBody
    from sam_3d_body.utils.checkpoint import load_state_dict
    from sam_3d_body.utils.config import get_config

    if not config_path or not os.path.isfile(config_path):
        config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "model_config.yaml not found next to the SAM 3D Body checkpoint."
        )

    model_cfg = get_config(config_path)
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()

    model = SAM3DBody(model_cfg)
    # The upstream checkpoint stores config objects, so weights_only can't be used.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_state_dict(model, checkpoint.get("state_dict", checkpoint), strict=False)
    model = model.to(device).eval()

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=_human_detector(yolo_path, device),
        human_segmentor=None,
        fov_estimator=None,
    )


def _setup_fast(
    fast_root: str,
    checkpoint_path: str,
    config_path: Optional[str],
    mhr_path: str,
    yolo_path: Optional[str],
    device: str,
):
    import torch

    _add_to_path(fast_root)

    # An optimisation hint that older PyTorch builds don't support.
    try:
        torch.compiler.cudagraph_mark_step_begin()
    except (ImportError, AttributeError, TypeError):
        torch.compiler.cudagraph_mark_step_begin = lambda: None

    from sam_3d_body import SAM3DBodyEstimator as FastEstimator
    from sam_3d_body import load_sam_3d_body as fast_load

    # load_sam_3d_body expects model_config.yaml next to the checkpoint.
    dst_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if config_path and os.path.isfile(config_path) and not os.path.isfile(dst_cfg):
        shutil.copy2(config_path, dst_cfg)

    model, model_cfg = fast_load(
        checkpoint_path=checkpoint_path, device=device, mhr_path=mhr_path
    )
    return FastEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=_human_detector(yolo_path, device),
        human_segmentor=None,
        fov_estimator=None,
    )


def setup_sam3d_estimator(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    mhr_path: str = "",
    yolo_path: Optional[str] = None,
    device: Optional[str] = "cuda",
):
    """Build a SAM 3D Body estimator, preferring the Fast-SAM-3D-Body clone."""
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    fast_root = third_party_dir("Fast-SAM-3D-Body")
    if fast_root.is_dir():
        logger.info("Loading SAM 3D Body with Fast-SAM-3D-Body from %s", fast_root)
        return _setup_fast(
            str(fast_root), checkpoint_path, config_path, mhr_path, yolo_path, device
        )
    logger.info("Fast-SAM-3D-Body not found; loading the original SAM 3D Body")
    return _setup_original(checkpoint_path, config_path, mhr_path, yolo_path, device)
