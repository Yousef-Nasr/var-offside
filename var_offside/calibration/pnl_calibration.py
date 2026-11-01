"""
PnLCalib-based field calibration.
Uses deep learning for keypoint and line detection.
Provides camera parameters and homography for projecting world->image coordinates.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from var_offside.calibration.base import BaseCalibration, CalibrationResult
from var_offside.config import get_settings


class PnLCalibration(BaseCalibration):
    """
    Field calibration using PnLCalib deep learning approach.

    Uses pretrained models to detect:
    1. Field keypoints (corners, penalty spots, etc.)
    2. Field line extremities

    Then computes camera calibration via PnP + refinement.
    """

    def __init__(
        self,
        weights_keypoints: Optional[str] = None,
        weights_lines: Optional[str] = None,
        use_pnl_refine: bool = True,
        kp_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
    ):
        """
        Initialize PnLCalib.

        Args:
            weights_keypoints: Path to keypoint detection weights
            weights_lines: Path to line detection weights
            use_pnl_refine: Use PnL refinement module for better accuracy
            kp_threshold: Keypoint detection threshold
            line_threshold: Line detection threshold
        """
        settings = get_settings()

        # Determine PnLCalib path
        self.pnl_path = Path("PnLCalib")

        self.weights_kp = weights_keypoints or str(self.pnl_path / "weights" / "SV_kp")
        self.weights_lines = weights_lines or str(
            self.pnl_path / "weights" / "SV_lines"
        )
        self.use_pnl_refine = use_pnl_refine
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold
        self.device = settings.device

        # Models will be loaded lazily
        self._kp_model = None
        self._line_model = None
        self._cam_calibrator = None
        self._transform = None

        # Check if PnLCalib is available
        self._pnl_available = self._check_pnl_available()

    def _check_pnl_available(self) -> bool:
        """Check if PnLCalib module is available."""
        if self.pnl_path.exists():
            # Add PnLCalib to path
            pnl_str = str(self.pnl_path.absolute())
            if pnl_str not in sys.path:
                sys.path.insert(0, pnl_str)

            # Check if weights exist
            kp_exists = Path(self.weights_kp).exists()
            lines_exists = Path(self.weights_lines).exists()

            if not kp_exists:
                print(f"[PnLCalib] Keypoint weights not found: {self.weights_kp}")
            if not lines_exists:
                print(f"[PnLCalib] Line weights not found: {self.weights_lines}")

            return kp_exists and lines_exists
        return False

    def _load_models(self):
        """Load PnLCalib models."""
        if not self._pnl_available:
            raise RuntimeError("PnLCalib module not found or weights missing.")

        try:
            import yaml
            from model.cls_hrnet import get_cls_net
            from model.cls_hrnet_l import get_cls_net as get_cls_net_l
            from utils.utils_calib import FramebyFrameCalib

            # Load keypoint model config and weights
            cfg_path = self.pnl_path / "config" / "hrnetv2_w48.yaml"
            cfg = yaml.safe_load(open(cfg_path, "r"))

            loaded_state = torch.load(self.weights_kp, map_location=self.device)
            self._kp_model = get_cls_net(cfg)
            self._kp_model.load_state_dict(loaded_state)
            self._kp_model.to(self.device)
            self._kp_model.eval()

            # Load line model config and weights
            cfg_l_path = self.pnl_path / "config" / "hrnetv2_w48_l.yaml"
            cfg_l = yaml.safe_load(open(cfg_l_path, "r"))

            loaded_state_l = torch.load(self.weights_lines, map_location=self.device)
            self._line_model = get_cls_net_l(cfg_l)
            self._line_model.load_state_dict(loaded_state_l)
            self._line_model.to(self.device)
            self._line_model.eval()

            # Transform for resizing
            self._transform = T.Resize((540, 960))

            print("[PnLCalib] Models loaded successfully")

        except ImportError as e:
            raise RuntimeError(f"Failed to import PnLCalib modules: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load PnLCalib models: {e}")

    def get_name(self) -> str:
        return "pnl"

    def calibrate(self, image: np.ndarray) -> CalibrationResult:
        """
        Calibrate using PnLCalib.

        Args:
            image: BGR input image

        Returns:
            CalibrationResult with homography and camera parameters
        """
        result = CalibrationResult(method=self.get_name())

        if not self._pnl_available:
            result.debug_info["error"] = "PnLCalib not available"
            return result

        try:
            # Load models if not loaded
            if self._kp_model is None:
                self._load_models()

            # Import utilities
            from utils.utils_calib import FramebyFrameCalib
            from utils.utils_heatmap import (
                get_keypoints_from_heatmap_batch_maxpool,
                get_keypoints_from_heatmap_batch_maxpool_l,
                complete_keypoints,
                coords_to_dict,
            )

            h_orig, w_orig = image.shape[:2]

            # Initialize frame-by-frame calibrator
            cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)

            # Preprocess image
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = TF.to_tensor(frame_pil).float().unsqueeze(0)

            # Resize if needed
            if frame_tensor.size()[-1] != 960:
                frame_tensor = self._transform(frame_tensor)

            frame_tensor = frame_tensor.to(self.device)
            _, _, h, w = frame_tensor.size()

            # Run inference
            with torch.no_grad():
                heatmaps_kp = self._kp_model(frame_tensor)
                heatmaps_line = self._line_model(frame_tensor)

            # Extract keypoints and lines
            kp_coords = get_keypoints_from_heatmap_batch_maxpool(
                heatmaps_kp[:, :-1, :, :]
            )
            line_coords = get_keypoints_from_heatmap_batch_maxpool_l(
                heatmaps_line[:, :-1, :, :]
            )

            kp_dict = coords_to_dict(kp_coords, threshold=self.kp_threshold)
            lines_dict = coords_to_dict(line_coords, threshold=self.line_threshold)

            # Keep raw image-space keypoints for debug overlays, then normalize
            # for calibration solving.
            kp_dict_raw, lines_dict_raw = complete_keypoints(
                kp_dict[0],
                lines_dict[0],
                w=w,
                h=h,
                normalize=False,
            )
            kp_dict, lines_dict = complete_keypoints(
                kp_dict[0],
                lines_dict[0],
                w=w,
                h=h,
                normalize=True,
            )

            # Update calibrator
            cam.update(kp_dict, lines_dict)

            # Run heuristic voting to find best calibration
            final_params = cam.heuristic_voting(refine_lines=self.use_pnl_refine)

            if final_params is not None:
                # Extract camera parameters
                cam_params = final_params["cam_params"]

                # Build projection matrix
                x_focal = cam_params["x_focal_length"]
                y_focal = cam_params["y_focal_length"]
                principal = np.array(cam_params["principal_point"])
                position = np.array(cam_params["position_meters"])
                rotation = np.array(cam_params["rotation_matrix"])

                # Compute projection matrix P = K @ [R | t]
                It = np.eye(4)[:-1]
                It[:, -1] = -position
                K = np.array(
                    [[x_focal, 0, principal[0]], [0, y_focal, principal[1]], [0, 0, 1]]
                )
                P = K @ (rotation @ It)

                # Store results
                result.camera_matrix = K
                result.projection_matrix = P
                result.rotation_matrix = rotation
                result.position = position

                # Compute a true ground-plane homography from K/R/C instead of
                # trusting the calibrator's internal (centred, Z-down) matrix.
                # For ground points (Z=0) the camera projection reduces to
                #   H_c = K @ [R[:,0], R[:,1], -R @ C]
                # in PnL's centred world; compose with the absolute-field
                # translation so H_field2img takes field metres (0-105, 0-68).
                if K is not None and rotation is not None and position is not None:
                    try:
                        H_c = K @ np.column_stack(
                            [
                                rotation[:, 0],
                                rotation[:, 1],
                                -(rotation @ position),
                            ]
                        )
                        T = np.array(
                            [
                                [1.0, 0.0, -52.5],
                                [0.0, 1.0, -34.0],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float64,
                        )
                        H_field2img = H_c @ T
                        if abs(H_field2img[2, 2]) > 1e-10:
                            H_field2img = H_field2img / H_field2img[2, 2]
                            # base.py: homography maps image -> field (absolute
                            # metres), homography_inv maps field -> image.
                            result.homography_inv = H_field2img
                            result.homography = np.linalg.inv(H_field2img)
                    except np.linalg.LinAlgError:
                        pass

                # Store keypoints
                result.keypoints = kp_dict
                result.keypoints_raw = kp_dict_raw
                result.image_size = (h_orig, w_orig)
                result.model_size = (h, w)

                # Store field lines for visualization
                result.field_lines = self._convert_lines_to_list(lines_dict)

                # Quality score based on reprojection error
                rep_err = final_params.get("rep_err", 10.0)
                result.quality_score = max(0, min(1, 1.0 - (rep_err / 20.0)))

                # Debug info
                result.debug_info = {
                    "num_keypoints": len(kp_dict),
                    "num_lines": len(lines_dict),
                    "reprojection_error": rep_err,
                    "mode": final_params.get("mode", "unknown"),
                    "pan_degrees": cam_params.get("pan_degrees"),
                    "tilt_degrees": cam_params.get("tilt_degrees"),
                    "roll_degrees": cam_params.get("roll_degrees"),
                    "keypoints_raw": kp_dict_raw,
                    "image_size": (h_orig, w_orig),
                    "model_size": (h, w),
                }

            else:
                result.debug_info["error"] = (
                    "Calibration failed - no valid parameters found"
                )

        except Exception as e:
            result.debug_info["error"] = str(e)
            import traceback

            traceback.print_exc()

        return result

    def _convert_lines_to_list(self, lines_dict: Dict) -> list:
        """Convert lines dictionary to list of line tuples."""
        lines = []
        for key, value in lines_dict.items():
            x1, y1 = value["x_1"], value["y_1"]
            x2, y2 = value["x_2"], value["y_2"]
            lines.append((x1, y1, x2, y2))
        return lines

    def project_point_to_image(
        self, world_point: Tuple[float, float, float], result: CalibrationResult
    ) -> Optional[Tuple[float, float]]:
        """
        Project a 3D world point to 2D image coordinates.

        Args:
            world_point: (x, y, z) in field coordinates (meters)
            result: CalibrationResult from calibrate()

        Returns:
            (x, y) image coordinates or None if projection fails
        """
        if result.projection_matrix is None:
            return None

        # Field coordinates: origin at center, x along length, y along width
        # Adjust for PnLCalib coordinate system (centered at 52.5, 34)
        x, y, z = world_point
        x_adj = x - 52.5
        y_adj = y - 34.0

        point_h = np.array([x_adj, y_adj, z, 1.0])
        proj = result.projection_matrix @ point_h

        if abs(proj[2]) < 1e-10:
            return None

        proj /= proj[2]
        return (float(proj[0]), float(proj[1]))

    def draw_field_overlay(
        self,
        image: np.ndarray,
        result: CalibrationResult,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw projected field lines on image.

        Args:
            image: Input image
            result: CalibrationResult
            color: Line color (BGR)
            thickness: Line thickness

        Returns:
            Image with field overlay
        """
        output = image.copy()

        if result.projection_matrix is None:
            return output

        # Standard field line coordinates (in meters)
        # Using PnLCalib's adjusted coordinates (centered)
        lines_coords = [
            [[0.0, 54.16, 0.0], [16.5, 54.16, 0.0]],
            [[16.5, 13.84, 0.0], [16.5, 54.16, 0.0]],
            [[16.5, 13.84, 0.0], [0.0, 13.84, 0.0]],
            [[88.5, 54.16, 0.0], [105.0, 54.16, 0.0]],
            [[88.5, 13.84, 0.0], [88.5, 54.16, 0.0]],
            [[88.5, 13.84, 0.0], [105.0, 13.84, 0.0]],
            [[52.5, 0.0, 0.0], [52.5, 68, 0.0]],
            [[0.0, 68.0, 0.0], [105.0, 68.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 68.0, 0.0]],
            [[105.0, 0.0, 0.0], [105.0, 68.0, 0.0]],
            [[0.0, 0.0, 0.0], [105.0, 0.0, 0.0]],
            [[0.0, 43.16, 0.0], [5.5, 43.16, 0.0]],
            [[5.5, 43.16, 0.0], [5.5, 24.84, 0.0]],
            [[5.5, 24.84, 0.0], [0.0, 24.84, 0.0]],
            [[99.5, 43.16, 0.0], [105.0, 43.16, 0.0]],
            [[99.5, 43.16, 0.0], [99.5, 24.84, 0.0]],
            [[99.5, 24.84, 0.0], [105.0, 24.84, 0.0]],
        ]

        P = result.projection_matrix

        for line in lines_coords:
            w1, w2 = line[0], line[1]

            # Adjust to centered coordinates
            p1 = np.array([w1[0] - 52.5, w1[1] - 34, w1[2], 1.0])
            p2 = np.array([w2[0] - 52.5, w2[1] - 34, w2[2], 1.0])

            i1 = P @ p1
            i2 = P @ p2

            if abs(i1[2]) < 1e-10 or abs(i2[2]) < 1e-10:
                continue

            i1 /= i1[2]
            i2 /= i2[2]

            pt1 = (int(i1[0]), int(i1[1]))
            pt2 = (int(i2[0]), int(i2[1]))

            cv2.line(output, pt1, pt2, color, thickness)

        return output
