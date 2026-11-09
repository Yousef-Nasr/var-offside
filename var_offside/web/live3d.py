"""
Live 3D offside scene for the browser app.

Builds a fast stand-in pitch scene from an :class:`AnalysisResult`, upgrades
the key players to SAM 3D Body meshes on demand, and renders orbit cameras
and cinematic replays as BGR frames for the WebSocket view.

The scene reuses the geometry, colours and helpers of
:mod:`var_offside.visualization.var_3d` so the live view and the exported
cinematic video show the same pitch, players and offside wall.

Threading: pyrender / OpenGL contexts are thread-bound, so every
:class:`LiveScene` call must happen on the server's one render thread.  The
exception is :meth:`LiveScene.recover_bodies`, which touches no OpenGL state.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from var_offside.web.overlays import TEAM_BGR
from var_offside.visualization.var_3d import (
    _PNL_TO_SCENE,
    VAR3DRenderer,
    MeshRecovery,
    PlayerMesh,
    _annotate_player_meshes,
    _estimate_focal_length,
    _field_to_scene,
    _merge_with_proxy_meshes,
    _require_pyrender,
    _require_trimesh,
    _slerp_rotation,
    build_pitch_meshes,
    build_pitch_stripe_meshes,
    compute_visible_pitch_region,
)

logger = logging.getLogger(__name__)

#: Default option values for :meth:`LiveScene.render` / :meth:`LiveScene.replay`.
DEFAULT_OPTIONS: Dict[str, bool] = {
    "line_plane": True,
    "photo": False,
    "labels": True,
    "pulse": True,
    "key_only": False,
}


def preload_body_model_imports() -> None:
    """Import the body model's heavy dependency on the render thread.

    Importing ``pytorch_lightning`` probes optional packages, one of which
    touches OpenGL through pyglet.  On Windows that fails once a GL window exists
    or from another thread, so run this on the render thread before any scene is
    built.  Importing it again is free.
    """
    try:
        import pytorch_lightning  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - recovery reports its own failure later
        logger.debug("Body model import preload skipped: %s", exc)


class LiveScene:
    """
    Real-time 3D view of one analysed offside situation.

    Parameters
    ----------
    result:
        ``AnalysisResult`` from ``OffsidePipeline.analyze`` (with full pitch
        calibration for a 3D view).
    image_bgr:
        The analysed frame (BGR ``np.ndarray``), used for the optional photo
        backdrop and as SAM 3D Body input.
    device:
        Torch device for SAM 3D Body mesh recovery (``"cuda"`` / ``"cpu"``).
    on_status:
        Optional ``callback(stage, message)``.  Stages follow the web API
        contract: ``loading``, ``proxy``, ``meshes``, ``ready``, ``error``.

    Notes
    -----
    Call :meth:`build_proxy` first (fast stand-in scene), then
    :meth:`upgrade_meshes` (slow SAM 3D Body pass), then :meth:`render` /
    :meth:`preset` / :meth:`replay` on the same thread.  :meth:`close` frees
    the OpenGL context and GPU memory.
    """

    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0
    DEFAULT_FOV = 40.0
    #: Lens for the orbit presets: long enough that players don't splay apart.
    PRESET_FOV = 24.0
    MIN_DISTANCE = 2.0
    MAX_DISTANCE = 300.0

    #: Slow pulse on the line-setting body part (seconds per cycle).
    PULSE_PERIOD = 2.6

    TEAM1_RGB = VAR3DRenderer.TEAM1_COLOR
    TEAM2_RGB = VAR3DRenderer.TEAM2_COLOR
    NEUTRAL_RGB = (0.72, 0.72, 0.72)
    #: Offside line and wall: ice blue reads clearly on grass in every verdict.
    LINE_RGB = (56, 214, 255)


    def __init__(
        self,
        result,
        image_bgr: np.ndarray,
        device: str = "cuda",
        on_status: Optional[Callable[[str, str], None]] = None,
    ):
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            raise ValueError("LiveScene needs the analysed frame as a BGR array")
        if getattr(result, "offside_result", None) is None:
            raise ValueError("LiveScene needs an offside result")

        self.result = result
        self.image_bgr = image_bgr
        self.device = device
        self._on_status = on_status if callable(on_status) else None

        self.width = int(image_bgr.shape[1])
        self.height = int(image_bgr.shape[0])
        self.calibration = getattr(result, "calibration", None)
        self.offside_result = getattr(result, "offside_result", None)

        # Scene objects (created by build_proxy)
        self._viz = VAR3DRenderer(
            video_width=self.width,
            video_height=self.height,
            pitch_length=self.PITCH_LENGTH,
            pitch_width=self.PITCH_WIDTH,
        )
        self._scene = None
        self._pyrender = None
        self._trimesh = None
        self._renderer = None
        self._camera_node = None
        self._camera_obj = None
        self._wall_mesh = None

        self._render_width = 1280
        self._render_height = 720
        self._closed = False
        self._meshes_ready = False

        # Player state
        self._player_field: Dict[int, Tuple[float, float]] = {}
        self._player_scene: Dict[int, Tuple[float, float]] = {}
        self._proxy_meshes: Dict[int, Any] = {}
        self._mesh_nodes: Dict[int, Any] = {}
        self._planted_meshes: Dict[int, Tuple[np.ndarray, PlayerMesh]] = {}
        self._focus_positions: List[np.ndarray] = []
        self._label_points: Dict[int, np.ndarray] = {}
        self._label_colors: Dict[int, Tuple[int, int, int]] = {}

        # Line-setting part highlight
        self._part_meshes: List[Any] = []
        self._part_point: Optional[np.ndarray] = None
        self._part_name: Optional[str] = None

        # Offside line / verdict state (filled by _prepare)
        self._attacking_side = ""
        self._defender_idx: Optional[int] = None
        self._offside_field_x: Optional[float] = None
        self._offside_cx = 0.0
        self._is_offside = False

        self._last_camera: Optional[Dict[str, Any]] = None

        # Kit colours sampled from the frame, per team id (BGR)
        self._shadow_meshes: Dict[int, Any] = {}
        self._ring_meshes: Dict[int, List[Any]] = {}
        self._key_only = False
        self._key_players: set = set()

        # Gap bar + verdict glow + replay caches
        self._gap_node = None
        self._gap_mesh = None
        self._glow_meshes: List[Any] = []
        self._glow_nodes: List[Any] = []
        self._mix_cache: Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray, Any]] = {}
        self._verdict_assets: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._sky_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._label_sprites: Dict[Tuple[str, Tuple[int, int, int], int], np.ndarray] = {}
        self._broadcast_cache: Optional[Dict[str, Any]] = None

    @property
    def has_pulse(self) -> bool:
        """True once the defender's line-setting body part is known."""
        return self._part_point is not None

    @property
    def part_name(self) -> Optional[str]:
        """Name of the defender's line-setting body part, when known."""
        return self._part_name

    # ------------------------------------------------------------------
    # Scene building
    # ------------------------------------------------------------------

    def build_proxy(self) -> None:
        """
        Build the fast stand-in scene: pitch, offside line, proxy players.

        No SAM 3D Body inference runs here, so this is quick enough for the
        "proxy" frame the web view shows while body meshes load.

        Raises:
            ValueError: if the result has no full pitch calibration.
        """
        if self._closed:
            raise RuntimeError("LiveScene is closed")
        if self._scene is not None:
            return

        if not self._has_full_calibration():
            raise ValueError("3D view needs full pitch calibration")

        self._notify("loading", "")
        self._prepare()
        # Prime torch's CUDA kernels *before* the OpenGL context exists.  On
        # some Windows drivers the first cuBLAS/cuDNN inference after GL init
        # can kill the process once SAM body meshes are drawn; the analysis
        # pipeline normally does this implicitly, but live-only use needs it.
        self._prime_cuda()
        preload_body_model_imports()

        trimesh = _require_trimesh()
        pyrender = _require_pyrender()
        self._trimesh = trimesh
        self._pyrender = pyrender

        scene = pyrender.Scene(
            bg_color=np.array([0.04, 0.06, 0.09, 1.0]),
            ambient_light=np.array([0.26, 0.27, 0.28]),
        )

        img_h = self._calibration_image_height()
        visible = compute_visible_pitch_region(
            self.calibration,
            self.width,
            img_h,
            self.PITCH_LENGTH,
            self.PITCH_WIDTH,
        )
        # One merged mesh for the whole pitch keeps the live draw call count
        # low (the video path keeps one node per marking).
        pitch_parts = build_pitch_meshes(
            trimesh, visible, self.PITCH_LENGTH, self.PITCH_WIDTH
        )
        pitch_parts += build_pitch_stripe_meshes(
            trimesh, self.PITCH_LENGTH, self.PITCH_WIDTH
        )
        scene.add(
            pyrender.Mesh.from_trimesh(
                trimesh.util.concatenate(pitch_parts), smooth=False
            )
        )
        self._add_surround(scene)
        self._add_offside_line(scene)
        self._add_proxy_players(scene)
        self._scene = scene
        self._build_gap_bar()
        self._add_lights(scene)
        self._add_camera(scene)

        self._scene = scene
        # One offscreen renderer for the whole view; render() only resizes it.
        self._renderer = pyrender.OffscreenRenderer(
            self._render_width, self._render_height
        )
        self._notify("proxy", "")
        logger.info(
            "LiveScene proxy built: %d stand-in players, line at x=%.1f",
            len(self._proxy_meshes),
            self._offside_cx,
        )

    def upgrade_meshes(
        self, should_stop: Optional[Callable[[], bool]] = None
    ) -> None:
        """
        Recover SAM 3D Body meshes and plant them, on the calling thread.

        The web view uses :meth:`recover_bodies` and :meth:`plant_bodies`
        separately so the slow inference doesn't hold up the render thread.
        """
        if self._closed or self._meshes_ready:
            return
        if self._scene is None:
            self.build_proxy()
        self.plant_bodies(self.recover_bodies(should_stop), should_stop)

    def recover_bodies(
        self, should_stop: Optional[Callable[[], bool]] = None
    ) -> Optional[Tuple[List[PlayerMesh], np.ndarray]]:
        """
        Run SAM 3D Body for the players that define the line.

        Touches no OpenGL state, so it can run on any thread.  The result is
        kept on the analysis result, so reconnecting doesn't repeat inference.
        Never raises for recovery problems: it reports ``proxy`` and returns
        ``None``.

        Args:
            should_stop: Optional predicate; ``True`` abandons the recovery.

        Returns:
            ``(player_meshes, rot_cam_to_world)`` for :meth:`plant_bodies`.
        """
        if self._closed or self._meshes_ready:
            return None
        key_indices = self._mesh_player_indices()
        if not key_indices:
            self._notify("proxy", "Body meshes unavailable: no key players")
            return None
        if self._stopped(should_stop):
            return None

        cache_key = tuple(sorted(key_indices.items()))
        cached = getattr(self.result, "_live_bodies", None)
        # Placement depends on the calibration, which the user can change.
        if (
            cached is not None
            and cached[0] == cache_key
            and cached[2] is self.calibration
        ):
            return cached[1]

        # Body poses don't depend on the calibration, so an edit needn't redo them.
        raw = getattr(self.result, "_live_raw_bodies", None)
        if raw is not None and raw[0] == cache_key:
            recovered = copy.deepcopy(raw[1])
        else:
            self._notify("meshes", f"Body meshes 0/{len(key_indices)}")
            try:
                from var_offside.config import get_settings

                cfg = get_settings().var_3d
                recovery = MeshRecovery(
                    checkpoint_path=cfg.sam3d_checkpoint,
                    config_path=cfg.sam3d_config,
                    yolo_path=cfg.sam3d_yolo,
                    mhr_path=cfg.sam3d_mhr,
                    device=self.device,
                )
                recovered = recovery.recover(
                    self.image_bgr,
                    key_indices,
                    image_width=self.width,
                )
            except Exception as exc:  # noqa: BLE001 - never crash the view
                reason = str(exc) or exc.__class__.__name__
                logger.warning(
                    "SAM 3D Body mesh recovery failed, keeping stand-ins: %s",
                    exc,
                    exc_info=True,
                )
                self._notify("proxy", f"Body meshes unavailable: {reason}")
                return None

            if self._stopped(should_stop):
                return None
            if not recovered:
                logger.warning("SAM 3D Body returned no meshes; keeping stand-ins")
                self._notify("proxy", "Body meshes unavailable: no meshes recovered")
                return None
            self.result._live_raw_bodies = (cache_key, copy.deepcopy(recovered))

        recovered = _merge_with_proxy_meshes(
            recovered,
            key_indices,
            self.image_bgr.shape,
            calibration=self.calibration,
        )
        _annotate_player_meshes(recovered, self.result)
        rot_cam_to_world = VAR3DRenderer._compute_upright_rotation(
            recovered, self.calibration
        )
        bodies = (recovered, rot_cam_to_world)
        self.result._live_bodies = (cache_key, bodies, self.calibration)
        return bodies

    def plant_bodies(
        self,
        bodies: Optional[Tuple[List[PlayerMesh], np.ndarray]],
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Swap stand-ins for recovered bodies.  Call on the render thread."""
        if self._closed or self._meshes_ready or bodies is None:
            return
        if self._scene is None:
            self.build_proxy()

        recovered, rot_cam_to_world = bodies
        added = 0
        for i, pm in enumerate(recovered, start=1):
            if self._stopped(should_stop):
                return
            # Proxy bodies have exactly 8 vertices; anything richer is a SAM mesh.
            if self._can_plant(pm):
                self._plant_recovered_mesh(pm, rot_cam_to_world)
                added += 1
            self._notify("meshes", f"Body meshes {i}/{len(recovered)}")

        if added == 0:
            logger.warning("No usable SAM meshes; keeping stand-ins")
            self._notify("proxy", "Body meshes unavailable: no usable meshes")
            return

        self._meshes_ready = True
        self._build_part_highlight()
        self._build_gap_bar()
        self._notify("ready", "")
        logger.info("LiveScene upgraded %d player(s) to real body meshes", added)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    def preset(self, name: str) -> Dict[str, Any]:
        """
        Return a named orbit camera in scene metres (Y up).

        Args:
            name: ``"broadcast"``, ``"birdseye"``, ``"line"`` or ``"free"``.

        Returns:
            ``{"target": [x, y, z], "yaw", "pitch", "distance", "fov"}``

        Raises:
            ValueError: for an unknown preset name.
        """
        if self._closed:
            raise RuntimeError("LiveScene is closed")
        if self._scene is None:
            self.build_proxy()

        if name == "broadcast":
            return dict(self._preset_broadcast())
        if name == "birdseye":
            return self._preset_birdseye()
        if name == "line":
            return self._preset_line()
        if name == "free":
            return self._preset_free()
        raise ValueError(f"Unknown camera preset: {name!r}")

    def _preset_broadcast(self) -> Dict[str, Any]:
        """Original calibrated camera as an orbit around its ground target."""
        if self._broadcast_cache is not None:
            return self._broadcast_cache
        cal = self.calibration
        cam_s = _PNL_TO_SCENE @ np.asarray(cal.position, dtype=np.float64).reshape(3)

        target = np.array([0.0, 0.0, 0.0])
        fp = cal.project_to_field((self.width / 2.0, self.height / 2.0))
        if fp is not None and np.all(np.isfinite(fp)):
            sx, sy = _field_to_scene(float(fp[0]), float(fp[1]))
            target = np.array([sx, sy, 0.0])

        self._broadcast_cache = self._orbit_camera(
            cam_s, target, self._broadcast_fov()
        )
        return self._broadcast_cache

    def _preset_birdseye(self) -> Dict[str, Any]:
        """High-but-not-very-high view above the offside-line area."""
        cx, cy = self._line_area_centre()
        xs, ys = self._focus_spread()
        aspect = max(self.width, 1) / max(self.height, 1)
        yfov = math.radians(self.PRESET_FOV)
        hfov = 2.0 * math.atan(math.tan(yfov / 2.0) * aspect)
        ext_x = max(float(np.ptp(xs)), 6.0) + 9.0
        ext_y = max(float(np.ptp(ys)), 6.0) + 9.0
        eye_h = max(
            ext_x / (2.0 * math.tan(hfov / 2.0)),
            ext_y / (2.0 * math.tan(yfov / 2.0)),
        )
        eye_h = float(np.clip(eye_h, 30.0, 90.0))

        pitch = 70.0
        target = np.array([cx, cy, 0.9])
        distance = eye_h / max(math.sin(math.radians(pitch)), 1e-3)
        # Stay on the broadcast touchline so left/right match the video.
        side = self._broadcast_side()
        yaw = 0.0 if side < 0 else 180.0
        return self._camera_dict(target, yaw, pitch, distance, self.PRESET_FOV)

    def _preset_line(self) -> Dict[str, Any]:
        """Side-on view along the offside line at player height."""
        _, cy = self._line_area_centre()
        cx = self._offside_cx
        side = self._broadcast_side()
        _, ys = self._focus_spread()
        near_y = float(np.min(ys)) if side < 0 else float(np.max(ys))
        half_w = self.PITCH_WIDTH / 2.0
        cam_y = float(np.clip(near_y + side * 22.0, -half_w - 30.0, half_w + 30.0))
        cam_s = np.array([cx, cam_y, 2.6])
        target_s = np.array([cx, cy, 0.9])
        return self._orbit_camera(cam_s, target_s, self.PRESET_FOV)

    def _preset_free(self) -> Dict[str, Any]:
        """Pleasant three-quarter view of the players around the line."""
        cx, cy = self._line_area_centre()
        xs, ys = self._focus_spread()
        side = self._broadcast_side()
        attacking_side = self._attacking_side or "right"
        goal_dir = 1.0 if attacking_side == "right" else -1.0

        aspect = max(self.width, 1) / max(self.height, 1)
        yfov = math.radians(self.PRESET_FOV)
        hfov = 2.0 * math.atan(math.tan(yfov / 2.0) * aspect)
        spread = max(float(np.ptp(xs)), float(np.ptp(ys)), 4.0)
        distance = float(
            np.clip(spread / (2.0 * math.tan(hfov / 2.0) * 0.6), 18.0, 90.0)
        )

        elevation = math.radians(26.0)
        h_dir = np.array([-goal_dir * 0.75, side * 0.66])
        h_dir /= np.linalg.norm(h_dir) + 1e-8
        target = np.array([cx, cy, 0.9])
        cam_s = target + np.array(
            [
                distance * math.cos(elevation) * h_dir[0],
                distance * math.cos(elevation) * h_dir[1],
                distance * math.sin(elevation),
            ]
        )
        return self._orbit_camera(cam_s, target, self.PRESET_FOV)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        camera: dict,
        width: int,
        height: int,
        options: dict,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Render one frame from an orbit camera.

        Args:
            camera: Camera dict (``target``, ``yaw``, ``pitch``, ``distance``,
                ``fov``); missing fields fall back to the last camera used.
            width: Output width in pixels.
            height: Output height in pixels.
            options: ``line_plane``, ``photo``, ``labels``, ``pulse`` flags.
            t: Seconds since the view opened (drives the slow pulse).

        Returns:
            BGR ``uint8`` frame of exactly ``width`` x ``height``.
        """
        if self._closed:
            raise RuntimeError("LiveScene is closed")
        if self._scene is None:
            self.build_proxy()

        opts = self._options(options)
        cam = self._normalise_camera(camera)
        pose, fov = self._pose_from_camera(cam)
        return self._render_pose(pose, fov, width, height, opts, t, camera=cam)

    def replay(
        self, width: int, height: int, options: dict, fps: int = 24
    ) -> Iterator[np.ndarray]:
        """
        Yield every frame of the cinematic replay (see :meth:`replay_steps`).

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            options: Render options; only ``line_plane`` applies to a replay.
            fps: Frames per second of the replay.

        Yields:
            BGR ``uint8`` frames of exactly ``width`` x ``height``.
        """
        for step in self.replay_steps(fps):
            yield self.render_step(step, width, height, options)

    def replay_steps(self, fps: int = 24, kind: str = "replay") -> List[Dict[str, Any]]:
        """
        Build the cinematic timeline, one step per frame.

        The story matches the exported video: the photo dissolves into the 3D
        pitch, the camera rises to the bird's-eye view, comes down side-on along
        the line, closes in on the defender's line-setting part with a slow
        pulse and callout, pulls back, and reveals the verdict.  ``kind``
        ``"birdseye"`` stops at the bird's-eye view and reveals the verdict there.

        Steps are plain dicts for :meth:`render_step`, so a live viewer can skip
        frames to hold real speed and an export can render all of them.
        """
        if self._closed:
            raise RuntimeError("LiveScene is closed")
        if self._scene is None:
            self.build_proxy()

        fps = max(int(fps), 1)
        self._mix_cache.clear()
        self._build_glow()

        broadcast = self._pose_from_camera(self._preset_broadcast())
        bird = self._pose_from_camera(self._preset_birdseye())
        line = self._pose_from_camera(self._preset_line())
        steps: List[Dict[str, Any]] = []

        def hold(view, seconds: float, **extra) -> None:
            for _ in range(max(int(seconds * fps), 1)):
                steps.append({"pose": view[0], "fov": view[1], **extra})

        def move(view_a, view_b, seconds: float, **extra) -> None:
            n = max(int(seconds * fps), 1)
            for i in range(n):
                ts = 0.5 - 0.5 * math.cos(math.pi * i / max(n - 1, 1))
                pose = np.eye(4)
                pose[:3, :3] = _slerp_rotation(view_a[0][:3, :3], view_b[0][:3, :3], ts)
                pose[:3, 3] = (1.0 - ts) * view_a[0][:3, 3] + ts * view_b[0][:3, 3]
                steps.append(
                    {"pose": pose, "fov": (1.0 - ts) * view_a[1] + ts * view_b[1], **extra}
                )

        def verdict(view, seconds: float) -> None:
            n = max(int(seconds * fps), 1)
            for i in range(n):
                t = i / fps
                level = 0.0 if t < 0.15 else 0.55 + 0.45 * math.sin((t - 0.15) * 2 * math.pi * 0.8)
                steps.append(
                    {
                        "pose": view[0],
                        "fov": view[1],
                        "part_lit": True,
                        "mix": "verdict",
                        "level": level,
                        "verdict_t": t,
                    }
                )

        hold(broadcast, 0.9, photo=1.0)
        n = max(int(0.9 * fps), 1)
        for i in range(n):
            steps.append(
                {
                    "pose": broadcast[0],
                    "fov": broadcast[1],
                    "photo": 1.0 - (i + 1) / n,
                }
            )

        if kind == "birdseye":
            move(broadcast, bird, 2.2)
            hold(bird, 1.5)
            verdict(bird, 3.0)
            return steps

        move(broadcast, bird, 1.6)
        hold(bird, 0.7)
        move(bird, line, 2.0)
        hold(line, 0.4)

        close = self._part_close_view()
        if close is not None:
            move(line, close, 1.0)
            n = max(int(2.8 * fps), 1)
            for i in range(n):
                t = i / fps
                # Slow, soft pulse (one cycle per ~1.4 s), then holds fully lit.
                level = 1.0 if t >= 2.1 else 0.5 - 0.5 * math.cos(2 * math.pi * t / 1.4)
                steps.append(
                    {
                        "pose": close[0],
                        "fov": close[1],
                        "mix": "part",
                        "level": level,
                        "callout": float(np.clip(t / 0.5, 0.0, 1.0)),
                    }
                )
            move(close, line, 0.8, part_lit=True)

        verdict(line, 3.0)
        return steps

    def render_step(
        self,
        step: Dict[str, Any],
        width: int,
        height: int,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        """Render one timeline step from :meth:`replay_steps`."""
        if self._closed:
            raise RuntimeError("LiveScene is closed")
        width = max(int(width), 2)
        height = max(int(height), 2)
        opts = self._options(options)
        opts.update(labels=False, photo=False, pulse=False)
        opts["part_visible"] = bool(step.get("part_lit", False))
        pose, fov = step["pose"], step["fov"]

        mix = step.get("mix")
        anchor = None
        if mix:
            dark, lit, anchor = self._mix_frames(mix, pose, fov, width, height, opts)
            level = float(np.clip(step.get("level", 0.0), 0.0, 1.0))
            frame = cv2.addWeighted(dark, 1.0 - level, lit, level, 0)
        else:
            frame = self._render_pose(pose, fov, width, height, opts, 0.0)

        photo = float(step.get("photo", 0.0))
        if photo > 0.0:
            frame = self._blend_photo(frame, photo)

        self._viz.video_width = width
        self._viz.video_height = height
        if step.get("callout"):
            name = self._part_name or "BODY PART"
            detail = f"DEFENDER #{self._defender_idx} - {name}"
            frame = self._viz._draw_part_callout(
                frame, anchor, "LINE SET HERE", detail, float(step["callout"])
            )
        if step.get("verdict_t") is not None:
            frame = self._viz._draw_verdict_reveal(
                frame, self._verdict_layers(width, height), float(step["verdict_t"])
            )
        return frame

    def _mix_frames(
        self,
        mix: str,
        pose: np.ndarray,
        fov: float,
        width: int,
        height: int,
        opts: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Two cached renders of a held shot, for cheap per-frame pulsing."""
        key = (mix, width, height)
        cached = self._mix_cache.get(key)
        if cached is not None:
            return cached

        dark_opts = dict(opts)
        lit_opts = dict(opts)
        if mix == "part":
            dark_opts["part_visible"] = False
            lit_opts["part_visible"] = True
        else:
            dark_opts["part_visible"] = lit_opts["part_visible"] = True
            lit_opts["glow"] = True
        dark = self._render_pose(pose, fov, width, height, dark_opts, 0.0)
        lit = self._render_pose(pose, fov, width, height, lit_opts, 0.0)
        anchor = None
        if self._part_point is not None:
            anchor = self._project(pose, self._part_point, width, height)
        self._mix_cache[key] = (dark, lit, anchor)
        return self._mix_cache[key]

    def _verdict_layers(self, width: int, height: int) -> Dict[str, Any]:
        """Verdict reveal layers for this frame size, built once."""
        key = (width, height)
        if key not in self._verdict_assets:
            caption = self._viz._verdict_caption(self._is_offside, self._gap_metres())
            self._verdict_assets[key] = self._viz._build_verdict_assets(
                self._is_offside, caption
            )
        return self._verdict_assets[key]

    def _part_close_view(self) -> Optional[Tuple[np.ndarray, float]]:
        """Close-up on the line-setting part, still exactly side-on to the line."""
        if self._part_point is None:
            return None
        pt = self._part_point
        side = self._broadcast_side()
        pose = VAR3DRenderer._look_at_pose(
            np.array([self._offside_cx, pt[1] + side * 8.0, 1.1]),
            np.array([float(pt[0]), float(pt[1]), 0.8]),
        )
        return pose, math.radians(self.PRESET_FOV)

    def close(self) -> None:
        """Delete the offscreen renderer and free GPU memory where possible."""
        if self._closed:
            return
        self._closed = True
        renderer, self._renderer = self._renderer, None
        if renderer is not None:
            try:
                renderer.delete()
            except Exception as exc:  # noqa: BLE001 - closing must not raise
                logger.warning("Could not delete offscreen renderer: %s", exc)
        self._scene = None
        self._camera_node = None
        self._camera_obj = None
        self._wall_mesh = None
        self._proxy_meshes.clear()
        self._shadow_meshes.clear()
        self._ring_meshes.clear()
        self._mesh_nodes.clear()
        self._planted_meshes.clear()
        self._part_meshes.clear()
        self._glow_meshes.clear()
        self._glow_nodes.clear()
        self._gap_node = None
        self._gap_mesh = None
        self._mix_cache.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch may be absent
            pass

    # ------------------------------------------------------------------
    # Preparation: calibration, line, players
    # ------------------------------------------------------------------

    def _has_full_calibration(self) -> bool:
        cal = self.calibration
        return bool(
            cal is not None
            and getattr(cal, "has_3d_calibration", False)
            and cal.camera_matrix is not None
            and cal.rotation_matrix is not None
            and cal.position is not None
        )

    def _prime_cuda(self) -> None:
        """Run a tiny CUDA matmul + conv before any OpenGL context is made.

        On some Windows drivers (AMD iGPU + CUDA in one process) the first
        cuBLAS/cuDNN inference *after* an OpenGL context exists can terminate
        the process when SAM body meshes are later drawn.  The web flow runs
        the analysis pipeline first, which primes CUDA implicitly; this keeps
        a stand-alone ``LiveScene`` equally safe.  Failures are ignored.
        """
        if not str(self.device).startswith("cuda"):
            return
        try:
            import torch
            import torch.nn.functional as F

            if not torch.cuda.is_available():
                return
            a = torch.ones((16, 16), device=self.device)
            _ = a @ a
            x = torch.ones((1, 3, 8, 8), device=self.device)
            k = torch.ones((2, 3, 3, 3), device=self.device)
            _ = F.conv2d(x, k, padding=1)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - a live view without CUDA works
            logger.debug("CUDA pre-init skipped: %s", exc)

    def _calibration_image_height(self) -> int:
        cal = self.calibration
        size = getattr(cal, "image_size", None)
        if size and len(size) == 2 and size[0]:
            return int(size[0])
        K = getattr(cal, "camera_matrix", None)
        if K is not None:
            return max(1, int(round(float(K[1, 2]) * 2.0)))
        return self.height

    def _prepare(self) -> None:
        """Resolve the line, attacking side and each player's field position."""
        from var_offside.offside.line_calculator import project_detection_to_field

        offside = self.offside_result
        line = getattr(offside, "offside_line", None)

        self._is_offside = bool(getattr(offside, "any_offside", False))
        self._attacking_side = getattr(offside, "attacking_side", "") or ""
        self._defender_idx = getattr(line, "defining_player_idx", None)

        field_x = getattr(line, "field_x", None)
        if field_x is not None and np.isfinite(field_x):
            self._offside_field_x = float(field_x)
        elif line is not None and self.calibration is not None:
            image_x = float(getattr(line, "image_x", self.width / 2.0))
            samples = []
            for sample_y in np.linspace(self.height * 0.35, self.height * 0.95, 5):
                fp = self.calibration.project_to_field((image_x, sample_y))
                if fp is not None:
                    samples.append(float(fp[0]))
            if samples:
                self._offside_field_x = float(np.median(samples))

        if self._offside_field_x is not None:
            self._offside_cx = float(
                np.clip(self._offside_field_x, 0.0, self.PITCH_LENGTH)
                - self.PITCH_LENGTH / 2.0
            )

        n_det = len(self.result.detections)
        all_positions: List[np.ndarray] = []
        for idx, det in enumerate(self.result.detections):
            fp = None
            if self._attacking_side in ("left", "right"):
                try:
                    fp = project_detection_to_field(
                        det, self._attacking_side, self.calibration
                    )
                except Exception:  # noqa: BLE001 - fall back to bbox foot
                    fp = None
            if fp is None:
                foot_x = (det.bbox[0] + det.bbox[2]) / 2.0
                fp = self.calibration.project_to_field((foot_x, det.bbox[3]))
            if fp is None or not np.all(np.isfinite(fp)):
                continue
            self._player_field[idx] = (float(fp[0]), float(fp[1]))

            wx, wy = _field_to_scene(float(fp[0]), float(fp[1]))
            wx = float(np.clip(wx, -self.PITCH_LENGTH / 2.0, self.PITCH_LENGTH / 2.0))
            wy = float(np.clip(wy, -self.PITCH_WIDTH / 2.0, self.PITCH_WIDTH / 2.0))
            self._player_scene[idx] = (wx, wy)
            all_positions.append(np.array([wx, wy, 0.9]))
            self._label_points[idx] = np.array([wx, wy, 1.95])

            if det.team_id in TEAM_BGR:
                self._label_colors[idx] = TEAM_BGR[det.team_id]
            else:
                self._label_colors[idx] = (200, 200, 200)

        # Frame on the key players around the line (defender + attackers on or
        # near it), not on the whole pitch, so presets stay tight and readable.
        self._key_players = set(self._mesh_player_indices())
        framing = [idx for idx in self._key_players if idx in self._player_scene]
        if framing:
            self._focus_positions = [
                np.array([*self._player_scene[idx], 0.9]) for idx in framing
            ]
        else:
            self._focus_positions = all_positions

        if not self._focus_positions:
            logger.warning("LiveScene: no player field positions could be resolved")

        logger.info(
            "LiveScene prepared: %d/%d players placed, line=%s, side=%s",
            len(self._player_scene),
            n_det,
            "n/a" if self._offside_field_x is None else f"{self._offside_field_x:.1f}",
            self._attacking_side or "unknown",
        )

    def _add_surround(self, scene) -> None:
        """Apron, advertising boards and stands, so the pitch isn't in a void.

        Every panel faces inward; pyrender culls back faces, so cameras outside
        the ground (like the broadcast camera) look straight through them.
        """
        half_l = self.PITCH_LENGTH / 2.0
        half_w = self.PITCH_WIDTH / 2.0
        verts: List[List[float]] = []
        faces: List[List[int]] = []
        colours: List[List[int]] = []

        def panel(a, b, c, d, rgb) -> None:
            base = len(verts)
            verts.extend([list(a), list(b), list(c), list(d)])
            faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
            colours.extend([list(rgb) + [255]] * 2)

        # Apron: a ground plane under and around the pitch.
        m = 60.0
        panel(
            (-half_l - m, -half_w - m, -0.04),
            (half_l + m, -half_w - m, -0.04),
            (half_l + m, half_w + m, -0.04),
            (-half_l - m, half_w + m, -0.04),
            (20, 58, 30),
        )

        def side(p0, p1, out) -> None:
            """Boards + tiers rising outward from the edge p0-p1; out = unit outward."""
            p0, p1, out = (np.array(v, dtype=np.float64) for v in (p0, p1, out))

            def at(p, dist, z):
                q = p + out * dist
                return (float(q[0]), float(q[1]), float(z))

            # Winding gives an inward (toward the pitch) face normal.
            def wall(d0, z0, d1, z1, rgb):
                panel(at(p0, d0, z0), at(p1, d0, z0), at(p1, d1, z1), at(p0, d1, z1), rgb)

            wall(5.0, 0.0, 5.0, 0.95, (28, 44, 74))
            wall(5.0, 0.95, 5.0, 1.05, (150, 160, 168))
            wall(5.0, 1.05, 9.0, 1.05, (12, 17, 22))
            tiers = 8
            for i in range(tiers):
                d0 = 9.0 + i * 3.4
                z0 = 1.05 + i * 2.2
                shade = (17, 22, 28) if i % 2 == 0 else (23, 29, 37)
                wall(d0, z0, d0 + 3.4, z0 + 2.2, shade)

        ext_l, ext_w = half_l + 42.0, half_w + 42.0
        side((-ext_l, half_w, 0), (ext_l, half_w, 0), (0, 1, 0))
        side((ext_l, -half_w, 0), (-ext_l, -half_w, 0), (0, -1, 0))
        side((half_l, ext_w, 0), (half_l, -ext_w, 0), (1, 0, 0))
        side((-half_l, -ext_w, 0), (-half_l, ext_w, 0), (-1, 0, 0))

        mesh = self._trimesh.Trimesh(
            vertices=np.array(verts, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            process=False,
        )
        mesh.visual.face_colors = np.array(colours, dtype=np.uint8)
        scene.add(self._pyrender.Mesh.from_trimesh(mesh, smooth=False))

    def _add_offside_line(self, scene) -> None:
        """Add the ground line and the translucent offside wall."""
        trimesh = self._trimesh
        pyrender = self._pyrender

        if self._offside_field_x is None:
            return

        half_w = self.PITCH_WIDTH / 2.0
        line_rgb = wall_rgb = self.LINE_RGB

        def quad(verts, rgba, faces):
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh.visual.face_colors = np.tile(rgba, (len(faces), 1))
            return mesh

        lw = 0.07
        line_verts = np.array(
            [
                [self._offside_cx - lw, -half_w, 0.02],
                [self._offside_cx + lw, -half_w, 0.02],
                [self._offside_cx + lw, half_w, 0.02],
                [self._offside_cx - lw, half_w, 0.02],
            ],
            dtype=np.float64,
        )
        scene.add(
            pyrender.Mesh.from_trimesh(
                quad(line_verts, list(line_rgb) + [220], [[0, 1, 2], [0, 2, 3]]),
                smooth=False,
            )
        )

        plane_h = 3.0
        wall_verts = np.array(
            [
                [self._offside_cx, -half_w, 0],
                [self._offside_cx, half_w, 0],
                [self._offside_cx, half_w, plane_h],
                [self._offside_cx, -half_w, plane_h],
            ],
            dtype=np.float64,
        )
        wall = quad(
            wall_verts,
            list(wall_rgb) + [78],
            [[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]],
        )
        self._wall_mesh = pyrender.Mesh.from_trimesh(wall, smooth=False)
        scene.add(self._wall_mesh)

    def _add_proxy_players(self, scene) -> None:
        """Add a team-coloured stand-in, a soft shadow and a ground ring per player."""
        pyrender = self._pyrender
        trimesh = self._trimesh
        ring = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
        unit = np.stack([np.cos(ring), np.sin(ring)], axis=1)
        for idx in sorted(self._player_scene):
            det = self.result.detections[idx]
            rgb = self._team_rgb(det)
            mesh = self._make_standin_mesh(rgb)
            pose = np.eye(4)
            pose[:3, 3] = [self._player_scene[idx][0], self._player_scene[idx][1], 0.0]
            node = scene.add(mesh, pose=pose)
            self._proxy_meshes[idx] = mesh
            self._mesh_nodes[idx] = node

            centre = np.array(self._player_scene[idx])
            rim = centre + unit * np.array([0.62, 0.42])
            verts = np.column_stack(
                [
                    np.vstack([centre, rim]),
                    np.full(len(ring) + 1, 0.035),
                ]
            )
            fan = np.array(
                [[0, 1 + i, 1 + (i + 1) % len(ring)] for i in range(len(ring))]
            )
            shadow = trimesh.Trimesh(vertices=verts, faces=fan, process=False)
            shadow.visual.face_colors = np.tile(
                [4, 12, 8, 105], (len(shadow.faces), 1)
            ).astype(np.uint8)
            shadow_mesh = pyrender.Mesh.from_trimesh(shadow, smooth=False)
            scene.add(shadow_mesh)
            self._shadow_meshes[idx] = shadow_mesh
            self._add_ground_ring(scene, idx, det, rgb, centre)

    def _add_ground_ring(self, scene, idx, det, rgb, centre: np.ndarray) -> None:
        """Team-colour ring and glow on the pitch, as in the photo view."""
        rgba = [int(c * 255) for c in rgb]
        meshes = [
            self._ground_disc(centre, 0.74, 0.042, rgba + [62]),
            self._ground_annulus(
                centre,
                0.72,
                0.83,
                0.05,
                rgba + [255],
                dashed=bool(det.is_goalkeeper),
            ),
        ]
        if idx == self._defender_idx:
            meshes.append(
                self._ground_annulus(centre, 0.98, 1.06, 0.05, list(self.LINE_RGB) + [255])
            )
        elif self._player_is_offside(idx):
            meshes.append(
                self._ground_annulus(centre, 0.98, 1.06, 0.05, [255, 255, 255, 255])
            )
        for mesh in meshes:
            scene.add(mesh)
        self._ring_meshes[idx] = meshes

    def _player_is_offside(self, idx: int) -> bool:
        """Whether the verdict calls this player offside."""
        indices = getattr(self.offside_result, "offside_player_indices", None) or ()
        return idx in indices

    def _ground_disc(self, centre: np.ndarray, radius: float, z: float, rgba: List[int]):
        """A flat translucent disc on the pitch."""
        angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
        rim = centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        verts = np.column_stack(
            [np.vstack([centre, rim]), np.full(len(angles) + 1, z)]
        )
        faces = np.array(
            [[0, 1 + i, 1 + (i + 1) % len(angles)] for i in range(len(angles))]
        )
        disc = self._trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        disc.visual.face_colors = np.tile(rgba, (len(faces), 1)).astype(np.uint8)
        return self._pyrender.Mesh.from_trimesh(disc, smooth=False)

    def _ground_annulus(
        self,
        centre: np.ndarray,
        inner: float,
        outer: float,
        z: float,
        rgba: List[int],
        dashed: bool = False,
    ):
        """A flat ring on the pitch; dashed marks a goalkeeper."""
        count = 48
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        unit = np.column_stack([np.cos(angles), np.sin(angles)])
        verts = np.column_stack(
            [
                np.vstack([centre + inner * unit, centre + outer * unit]),
                np.full(2 * count, z),
            ]
        )
        faces = []
        for i in range(count):
            if dashed and (i // 4) % 2:
                continue
            j = (i + 1) % count
            faces.append([i, count + i, count + j])
            faces.append([i, count + j, j])
        ring = self._trimesh.Trimesh(
            vertices=verts, faces=np.array(faces), process=False
        )
        ring.visual.face_colors = np.tile(rgba, (len(faces), 1)).astype(np.uint8)
        return self._pyrender.Mesh.from_trimesh(ring, smooth=False)

    def _add_lights(self, scene) -> None:
        """Floodlight-style key and fill, with no blow-out on the grass."""
        pyrender = self._pyrender
        for direction, intensity in (
            (np.array([0.25, -0.45, 1.0]), 1.15),
            (np.array([-0.5, 0.55, 0.65]), 0.45),
        ):
            pose = VAR3DRenderer._look_at_pose(direction * 60.0, np.zeros(3))
            scene.add(
                pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=intensity),
                pose=pose,
            )

    def _add_camera(self, scene) -> None:
        """Create the perspective camera node, starting on the broadcast view."""
        pyrender = self._pyrender
        cam_obj = pyrender.PerspectiveCamera(
            yfov=math.radians(self.DEFAULT_FOV),
            aspectRatio=self.width / max(self.height, 1),
            znear=0.1,
            zfar=500.0,
        )
        try:
            camera = self._preset_broadcast()
            pose, fov = self._pose_from_camera(camera)
            cam_obj.yfov = fov
            self._last_camera = camera
        except Exception as exc:  # noqa: BLE001 - a fallback view is better
            logger.warning("Broadcast camera unavailable, using free view: %s", exc)
            pose = VAR3DRenderer._look_at_pose(
                np.array([0.0, -25.0, 15.0]), np.array([0.0, 0.0, 0.0])
            )
        self._camera_obj = cam_obj
        self._camera_node = scene.add(cam_obj, pose=pose)

    # ------------------------------------------------------------------
    # Stand-in bodies
    # ------------------------------------------------------------------

    def _make_standin_mesh(self, rgb: Tuple[float, float, float]):
        """Low-poly mannequin in team colours (one pyrender mesh)."""
        trimesh = self._trimesh
        kit = [int(c * 255) for c in rgb] + [255]
        shorts = [int(c * 150) for c in rgb] + [255]
        skin = [214, 164, 132, 255]
        boot = [24, 24, 28, 255]

        def limb(radius, length, x, y, z, colour):
            part = trimesh.creation.capsule(height=length, radius=radius, count=[6, 8])
            part.apply_translation(-part.bounds.mean(axis=0) + np.array([x, y, z]))
            return part, colour

        head = trimesh.creation.icosphere(subdivisions=1, radius=0.115)
        head.apply_translation([0.0, 0.0, 1.635])
        parts = [
            limb(0.155, 0.32, 0.0, 0.0, 1.17, kit),  # torso
            limb(0.145, 0.12, 0.0, 0.0, 0.88, shorts),  # hips
            limb(0.052, 0.46, 0.0, 0.25, 1.06, kit),  # arms
            limb(0.052, 0.46, 0.0, -0.25, 1.06, kit),
            limb(0.078, 0.34, 0.0, 0.095, 0.62, skin),  # thighs
            limb(0.078, 0.34, 0.0, -0.095, 0.62, skin),
            limb(0.066, 0.3, 0.0, 0.095, 0.27, kit),  # socks
            limb(0.066, 0.3, 0.0, -0.095, 0.27, kit),
            limb(0.058, 0.14, 0.06, 0.095, 0.06, boot),  # boots
            limb(0.058, 0.14, 0.06, -0.095, 0.06, boot),
            (head, skin),
        ]
        verts, faces, colours = [], [], []
        base = 0
        for part, colour in parts:
            verts.append(part.vertices)
            faces.append(part.faces + base)
            colours.append(np.tile(colour, (len(part.vertices), 1)))
            base += len(part.vertices)
        shape = trimesh.Trimesh(
            vertices=np.vstack(verts),
            faces=np.vstack(faces),
            vertex_colors=np.vstack(colours).astype(np.uint8),
            process=False,
        )
        return self._pyrender.Mesh.from_trimesh(shape, smooth=True)

    def _team_rgb(self, det) -> Tuple[float, float, float]:
        """Team colour for scene geometry.

        The renderer's gamma step lifts dark channels, which turns a strong red
        pink, so the colour is converted to linear light first.
        """
        bgr = TEAM_BGR.get(det.team_id)
        if bgr is not None:
            return tuple((bgr[i] / 255.0) ** 2.2 for i in (2, 1, 0))
        return self.NEUTRAL_RGB

    # ------------------------------------------------------------------
    # Mesh upgrade helpers
    # ------------------------------------------------------------------

    def _mesh_player_indices(self) -> Dict[int, Tuple[float, float, float, float]]:
        """Players to recover meshes for (defender + attackers on the line).

        Mirrors the cinematic video's selection (last defender plus the most
        offside attacker) and adds the attackers nearest the line -- measured
        in field metres, ahead or behind -- so the line view frames the same
        players the verdict talks about.  At most six bodies are recovered.
        """
        offside = self.offside_result
        statuses = list(getattr(offside, "player_statuses", []) or [])
        detections = self.result.detections
        chosen: List[int] = []
        line_x = self._offside_field_x

        def add(idx: Optional[int]) -> None:
            if idx is None or idx in chosen:
                return
            if 0 <= idx < len(detections):
                chosen.append(int(idx))

        add(self._defender_idx)
        add(self._attacker_idx())

        # Most offside attacker first (same as render_var_3d_video).
        ordered = sorted(
            statuses,
            key=lambda s: float(getattr(s, "margin_px", 0.0)),
            reverse=True,
        )
        for status in ordered:
            if getattr(status, "is_offside", False):
                add(status.player_idx)

        # Attackers closest to the line in field metres.
        nearby = []
        for status in statuses:
            fp = self._player_field.get(status.player_idx)
            if fp is None or line_x is None:
                continue
            nearby.append((abs(fp[0] - line_x), status))
        nearby.sort(key=lambda item: item[0])
        for distance, status in nearby:
            if distance > 6.0:
                break
            add(status.player_idx)

        # Fallback: if nothing was near the line, keep the top attackers.
        if len(chosen) <= 1:
            for status in ordered:
                add(status.player_idx)

        chosen = chosen[:6]
        return {idx: tuple(float(v) for v in detections[idx].bbox) for idx in chosen}

    def _can_plant(self, pm: PlayerMesh) -> bool:
        return bool(
            pm.player_idx in self._proxy_meshes
            and pm.field_pos is not None
            and np.all(np.isfinite(pm.field_pos))
            and len(pm.vertices) > 8
        )

    def _plant_recovered_mesh(
        self, pm: PlayerMesh, rot_cam_to_world: np.ndarray
    ) -> None:
        """Hide a stand-in and add the recovered body at its field position."""
        trimesh = self._trimesh
        pyrender = self._pyrender
        idx = pm.player_idx
        fx, fy = float(pm.field_pos[0]), float(pm.field_pos[1])
        wx, wy = _field_to_scene(fx, fy)
        wx = float(np.clip(wx, -self.PITCH_LENGTH / 2.0, self.PITCH_LENGTH / 2.0))
        wy = float(np.clip(wy, -self.PITCH_WIDTH / 2.0, self.PITCH_WIDTH / 2.0))

        v_world = VAR3DRenderer._orient_and_plant_mesh(
            pm,
            rot_cam_to_world,
            wx,
            wy,
            attacking_side=self._attacking_side or None,
        )
        rgb = self._team_rgb(self.result.detections[idx])
        rgba = [int(c * 255) for c in rgb] + [255]
        mesh_tri = trimesh.Trimesh(
            vertices=v_world,
            faces=pm.faces,
            vertex_colors=np.tile(rgba, (v_world.shape[0], 1)).astype(np.uint8),
            process=False,
        )
        mesh = pyrender.Mesh.from_trimesh(mesh_tri, smooth=True)
        node = self._scene.add(mesh)
        self._mesh_nodes[idx] = node

        old = self._proxy_meshes.get(idx)
        if old is not None:
            old.is_visible = False

        self._planted_meshes[idx] = (v_world, pm)

        # Label anchor: centre of the top of the head.
        top = float(v_world[:, 2].max())
        head = v_world[v_world[:, 2] >= top - 0.12]
        if len(head):
            centre = head[:, :2].mean(axis=0)
            self._label_points[idx] = np.array(
                [float(centre[0]), float(centre[1]), top + 0.10]
            )

    def _build_part_highlight(self) -> None:
        """Glowing 3D overlay + anchor for the defender's line-setting part."""
        defender_idx = self._defender_idx
        planted = self._planted_meshes.get(defender_idx) if defender_idx is not None else None
        if planted is None or self._attacking_side not in ("left", "right"):
            return
        try:
            v_world, pm = planted
            part = self._viz._line_setting_part(
                v_world,
                pm,
                self._attacking_side,
                self._trimesh,
                self._pyrender,
            )
        except Exception as exc:  # noqa: BLE001 - the highlight is optional
            logger.warning("Line-setting part highlight skipped: %s", exc)
            return
        if not part:
            return
        self._part_point = np.asarray(part["point"], dtype=np.float64)
        self._part_name = part["name"]
        for mesh in part["meshes"]:
            mesh.is_visible = True
            self._scene.add(mesh)
            self._part_meshes.append(mesh)
        logger.info(
            "Line-setting part: %s at %s", self._part_name, np.round(self._part_point, 2)
        )

    def _attacker_idx(self) -> Optional[int]:
        """The attacker the verdict talks about (most offside, else nearest)."""
        offside = self.offside_result
        most = getattr(offside, "most_offside_player", None)
        att_idx = getattr(most, "player_idx", None)
        if att_idx is None:
            statuses = list(getattr(offside, "player_statuses", []) or [])
            if statuses:
                att_idx = max(
                    statuses,
                    key=lambda s: float(getattr(s, "margin_px", 0.0)),
                ).player_idx
        return att_idx

    def _build_gap_bar(self) -> None:
        """Yellow ground bar from the attacker's leading point to the line."""
        if self._scene is None or self._offside_field_x is None:
            return
        if self._gap_node is not None:
            self._scene.remove_node(self._gap_node)
            self._gap_node = None
            self._gap_mesh = None

        att_idx = self._attacker_idx()
        if att_idx is None:
            return
        planted = self._planted_meshes.get(att_idx)
        if planted is not None and self._attacking_side in ("left", "right"):
            lead_x, foot_y = self._viz._lead_point(*planted, self._attacking_side)
        elif att_idx in self._player_scene:
            lead_x, foot_y = self._player_scene[att_idx]
        else:
            return

        x0, x1 = sorted((float(lead_x), float(self._offside_cx)))
        if x1 - x0 <= 0.01:
            return
        bw = 0.14
        bar = self._trimesh.Trimesh(
            vertices=[
                [x0, foot_y - bw, 0.05],
                [x1, foot_y - bw, 0.05],
                [x1, foot_y + bw, 0.05],
                [x0, foot_y + bw, 0.05],
            ],
            faces=[[0, 1, 2], [0, 2, 3]],
            process=False,
        )
        bar.visual.face_colors = np.tile([255, 214, 40, 255], (2, 1))
        self._gap_mesh = self._pyrender.Mesh.from_trimesh(bar, smooth=False)
        self._gap_node = self._scene.add(self._gap_mesh)

    def _build_glow(self) -> None:
        """Verdict glow layers (brighter wall + shell on the attacker), hidden."""
        for node in self._glow_nodes:
            self._scene.remove_node(node)
        self._glow_nodes = []
        self._glow_meshes = []
        if self._offside_field_x is None:
            return

        half_w = self.PITCH_WIDTH / 2.0
        rgb = [255, 45, 40] if self._is_offside else [40, 230, 110]
        cx = self._offside_cx
        wall = self._trimesh.Trimesh(
            vertices=[
                [cx, -half_w, 0],
                [cx, half_w, 0],
                [cx, half_w, 3.0],
                [cx, -half_w, 3.0],
            ],
            faces=[[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]],
            process=False,
        )
        wall_rgb = rgb if self._is_offside else list(self.LINE_RGB)
        wall.visual.face_colors = np.tile(wall_rgb + [120], (4, 1)).astype(np.uint8)
        layers = [self._pyrender.Mesh.from_trimesh(wall, smooth=False)]

        planted = self._planted_meshes.get(self._attacker_idx())
        if planted is not None:
            v_att, pm_att = planted
            centre = v_att.mean(axis=0)
            shell = self._trimesh.Trimesh(
                vertices=centre + (v_att - centre) * 1.05,
                faces=pm_att.faces,
                vertex_colors=np.tile(rgb + [110], (len(v_att), 1)).astype(np.uint8),
                process=False,
            )
            layers.append(self._pyrender.Mesh.from_trimesh(shell, smooth=True))

        for mesh in layers:
            mesh.is_visible = False
            self._glow_meshes.append(mesh)
            self._glow_nodes.append(self._scene.add(mesh))

    # ------------------------------------------------------------------
    # Camera maths
    # ------------------------------------------------------------------

    @staticmethod
    def _scene_to_yup(p: np.ndarray) -> np.ndarray:
        """Scene (X, Y, Z-up) -> orbit (Y-up) coordinates."""
        return np.array([float(p[0]), float(p[2]), -float(p[1])])

    @staticmethod
    def _yup_to_scene(p: np.ndarray) -> np.ndarray:
        """Orbit (Y-up) coordinates -> scene (X, Y, Z-up)."""
        return np.array([float(p[0]), -float(p[2]), float(p[1])])

    def _camera_dict(
        self,
        target_scene: np.ndarray,
        yaw: float,
        pitch: float,
        distance: float,
        fov: float,
    ) -> Dict[str, Any]:
        target = self._scene_to_yup(target_scene)
        return {
            "target": [float(target[0]), float(target[1]), float(target[2])],
            "yaw": float(yaw),
            "pitch": float(np.clip(pitch, 3.0, 89.0)),
            "distance": float(np.clip(distance, self.MIN_DISTANCE, self.MAX_DISTANCE)),
            "fov": float(np.clip(fov, 10.0, 120.0)),
        }

    def _orbit_camera(
        self, cam_scene: np.ndarray, target_scene: np.ndarray, fov: float
    ) -> Dict[str, Any]:
        """Express a scene-space camera position as an orbit around a target."""
        target = self._scene_to_yup(target_scene)
        offset = self._scene_to_yup(cam_scene) - target
        distance = float(np.linalg.norm(offset))
        if distance < 1e-3:
            distance = 10.0
        pitch = math.degrees(
            math.asin(float(np.clip(offset[1] / distance, -1.0, 1.0)))
        )
        yaw = math.degrees(math.atan2(float(offset[0]), float(offset[2])))
        return {
            "target": [float(target[0]), float(target[1]), float(target[2])],
            "yaw": yaw,
            "pitch": float(np.clip(pitch, 3.0, 89.0)),
            "distance": float(np.clip(distance, self.MIN_DISTANCE, self.MAX_DISTANCE)),
            "fov": float(np.clip(fov, 10.0, 120.0)),
        }

    @staticmethod
    def _camera_position(camera: Dict[str, Any]) -> np.ndarray:
        """Camera position in Y-up orbit space for a normalised camera dict."""
        target = np.asarray(camera["target"], dtype=np.float64)
        yaw = math.radians(float(camera["yaw"]))
        pitch = math.radians(float(camera["pitch"]))
        distance = float(camera["distance"])
        return target + distance * np.array(
            [
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
                math.cos(pitch) * math.cos(yaw),
            ]
        )

    def _pose_from_camera(
        self, camera: Dict[str, Any]
    ) -> Tuple[np.ndarray, float]:
        """Orbit camera dict -> (pyrender camera pose, vertical fov)."""
        target = self._yup_to_scene(np.asarray(camera["target"], dtype=np.float64))
        cam = self._yup_to_scene(self._camera_position(camera))
        pose = VAR3DRenderer._look_at_pose(cam, target)
        return pose, math.radians(float(camera["fov"]))

    def _normalise_camera(self, camera: Optional[dict]) -> Dict[str, Any]:
        """Fill missing camera fields from the last camera used."""
        base = self._last_camera
        if base is None:
            base = self._preset_broadcast()
        cam = dict(base)
        if isinstance(camera, dict):
            for key in ("target", "yaw", "pitch", "distance", "fov"):
                if camera.get(key) is not None:
                    cam[key] = camera[key]

        target = cam.get("target")
        try:
            target = [float(v) for v in target][:3]
        except (TypeError, ValueError):
            target = [0.0, 0.0, 0.0]
        if len(target) != 3:
            target = [0.0, 0.0, 0.0]
        try:
            yaw = float(cam.get("yaw", 0.0))
            pitch = float(np.clip(float(cam.get("pitch", 20.0)), 3.0, 89.0))
            distance = float(
                np.clip(
                    float(cam.get("distance", 20.0)),
                    self.MIN_DISTANCE,
                    self.MAX_DISTANCE,
                )
            )
            fov = float(np.clip(float(cam.get("fov", self.DEFAULT_FOV)), 10.0, 120.0))
        except (TypeError, ValueError):
            yaw, pitch, distance, fov = 0.0, 20.0, 20.0, self.DEFAULT_FOV

        camera_out = {
            "target": target,
            "yaw": yaw,
            "pitch": pitch,
            "distance": distance,
            "fov": fov,
        }
        self._last_camera = camera_out
        return camera_out

    def _broadcast_fov(self) -> float:
        """Vertical field of view (degrees) of the calibrated broadcast camera."""
        K = getattr(self.calibration, "camera_matrix", None)
        img_h = self._calibration_image_height()
        if K is not None:
            fy = float(K[1, 1])
            if np.isfinite(fy) and fy > 0:
                return math.degrees(2.0 * math.atan((img_h / 2.0) / fy))
        focal = _estimate_focal_length(self.width, self.height, self.calibration)
        return math.degrees(2.0 * math.atan((img_h / 2.0) / max(focal, 1e-3)))

    def _broadcast_side(self) -> float:
        """Scene-y sign of the broadcast camera's touchline (-1 = main camera)."""
        return VAR3DRenderer._broadcast_side(self.calibration)

    def _line_area_centre(self) -> Tuple[float, float]:
        """Centre (scene x, y) of the players and the offside wall."""
        xs = [p[0] for p in self._focus_positions] + [self._offside_cx]
        ys = [p[1] for p in self._focus_positions]
        cx = 0.5 * (float(np.min(xs)) + float(np.max(xs)))
        cy = 0.5 * (float(np.min(ys)) + float(np.max(ys))) if ys else 0.0
        return cx, cy

    def _focus_spread(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._focus_positions:
            pts = np.array(self._focus_positions, dtype=np.float64)
            return np.append(pts[:, 0], self._offside_cx), pts[:, 1]
        return np.array([self._offside_cx]), np.array([0.0])

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def _render_pose(
        self,
        pose: np.ndarray,
        fov: float,
        width: int,
        height: int,
        options: Dict[str, bool],
        t: float,
        camera: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render the current scene from an explicit pose and add overlays."""
        width = max(int(width), 2)
        height = max(int(height), 2)

        if self._renderer is None:
            self._renderer = self._pyrender.OffscreenRenderer(width, height)
        else:
            self._renderer.viewport_width = width
            self._renderer.viewport_height = height
        self._render_width = width
        self._render_height = height

        self._apply_visibility(options)
        self._camera_obj.aspectRatio = width / float(height)
        self._camera_obj.yfov = float(fov)
        self._scene.set_pose(self._camera_node, pose)

        color, depth = self._renderer.render(self._scene)
        frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        sky = depth <= 0.0
        if sky.any():
            frame[sky] = self._sky(width, height)[sky]
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        if options.get("photo") and self._near_broadcast(camera):
            frame = self._blend_photo(frame)

        if options.get("labels"):
            frame = self._draw_labels(frame, pose, width, height)
        if options.get("pulse"):
            frame = self._draw_pulse(frame, pose, width, height, t)
        return frame

    def _sky(self, width: int, height: int) -> np.ndarray:
        """Evening-sky backdrop (BGR), brighter toward the horizon; cached per size."""
        key = (width, height)
        if key not in self._sky_cache:
            top = np.array([26, 15, 10], dtype=np.float32)
            bottom = np.array([78, 58, 44], dtype=np.float32)
            ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None] ** 1.4
            column = top + (bottom - top) * ramp
            self._sky_cache[key] = np.tile(column[:, None, :], (1, width, 1)).astype(
                np.uint8
            )
        return self._sky_cache[key]

    def _apply_visibility(self, options: Dict[str, bool]) -> None:
        """Toggle scene geometry instead of rebuilding it."""
        show_line = bool(options.get("line_plane", True))
        if self._wall_mesh is not None:
            self._wall_mesh.is_visible = show_line
        if self._gap_mesh is not None:
            self._gap_mesh.is_visible = show_line
        show_part = bool(options.get("part_visible", options.get("pulse", True)))
        for mesh in self._part_meshes:
            mesh.is_visible = show_part
        show_glow = bool(options.get("glow", False))
        for mesh in self._glow_meshes:
            mesh.is_visible = show_glow

        # Everyone, or only the last defender and the attacker.
        self._key_only = bool(options.get("key_only", False))
        pair = {self._defender_idx, self._attacker_idx()}
        for idx, proxy in self._proxy_meshes.items():
            keep = not self._key_only or idx in pair
            planted = idx in self._planted_meshes
            proxy.is_visible = keep and not planted
            if planted:
                self._mesh_nodes[idx].mesh.is_visible = keep
            shadow = self._shadow_meshes.get(idx)
            if shadow is not None:
                shadow.is_visible = keep
            for ring in self._ring_meshes.get(idx, ()):
                ring.is_visible = keep

    def _near_broadcast(self, camera: Optional[Dict[str, Any]]) -> bool:
        """True when the camera is ~2 degrees / 5% from the broadcast preset."""
        if camera is None:
            return False
        try:
            broadcast = self._preset_broadcast()
        except Exception:  # noqa: BLE001 - photo is optional
            return False
        dyaw = abs((float(camera["yaw"]) - float(broadcast["yaw"]) + 180.0) % 360.0 - 180.0)
        dpitch = abs(float(camera["pitch"]) - float(broadcast["pitch"]))
        dist_b = max(float(broadcast["distance"]), 1e-3)
        ddist = abs(float(camera["distance"]) - dist_b) / dist_b
        return math.hypot(dyaw, dpitch) <= 2.0 and ddist <= 0.05

    def _blend_photo(self, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Blend the frame over a broadcast render, scaled by height to line up."""
        h, w = frame.shape[:2]
        scale = h / max(self.height, 1)
        new_w = max(int(round(self.width * scale)), 1)
        photo = cv2.resize(self.image_bgr, (new_w, h), interpolation=cv2.INTER_AREA)
        if new_w >= w:
            x0 = (new_w - w) // 2
            return cv2.addWeighted(frame, 1.0 - alpha, photo[:, x0 : x0 + w], alpha, 0)
        x0 = (w - new_w) // 2
        out = frame.copy()
        out[:, x0 : x0 + new_w] = cv2.addWeighted(
            frame[:, x0 : x0 + new_w], 1.0 - alpha, photo, alpha, 0
        )
        return out

    def _project(
        self, frame_pose: np.ndarray, point: np.ndarray, width: int, height: int
    ) -> Optional[Tuple[float, float]]:
        """Scene point -> pixel for the current camera object."""
        proj = self._camera_obj.get_projection_matrix(int(width), int(height))
        try:
            view = np.linalg.inv(frame_pose)
        except np.linalg.LinAlgError:
            return None
        clip = proj @ view @ np.append(np.asarray(point, dtype=np.float64), 1.0)
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        if not np.all(np.isfinite(ndc)):
            return None
        return (
            (float(ndc[0]) + 1.0) * 0.5 * width,
            (1.0 - float(ndc[1])) * 0.5 * height,
        )

    def _draw_labels(
        self, frame: np.ndarray, pose: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Tags above heads: full for key players, a small number for the rest."""
        px = max(11, int(round(height / 720.0 * 15.0)))
        placed: List[Tuple[int, int, int, int]] = []
        items = []
        pair = {self._defender_idx, self._attacker_idx()}
        for idx, point in self._label_points.items():
            if self._key_only and idx not in pair:
                continue
            screen = self._project(pose, point, width, height)
            if screen is None:
                continue
            x, y = screen
            if x < 0 or x > width or y < 0 or y > height:
                continue
            items.append((idx in self._key_players, y, x, idx))

        # Key players claim their spot first; other tags give way or drop out.
        for is_key, y, x, idx in sorted(items, key=lambda it: (not it[0], -it[1])):
            sprite = self._label_sprite(
                self._label_parts(idx),
                self._label_colors.get(idx, (200, 200, 200)),
                px if is_key else max(10, px - 2),
            )
            sh, sw = sprite.shape[:2]
            bx = int(np.clip(x - sw / 2.0, 2, max(width - sw - 2, 3)))
            by = int(np.clip(y - sh - 3, 2, max(height - sh - 2, 3)))
            for _ in range(4 if is_key else 2):
                if not any(
                    bx < px1 and bx + sw > px0 and by < py1 and by + sh > py0
                    for px0, py0, px1, py1 in placed
                ):
                    break
                by = max(by - sh - 2, 2)
            else:
                if not is_key:
                    continue
            placed.append((bx, by, bx + sw, by + sh))
            region = frame[by : by + sh, bx : bx + sw]
            a = sprite[..., 3:4]
            region[:] = region * (1.0 - a) + sprite[..., :3] * a
        return frame

    def _label_sprite(
        self,
        parts: Tuple[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]], ...],
        color: Tuple[int, int, int],
        px: int,
    ) -> np.ndarray:
        """Number chip, any role chips and a pointer to the head (BGR + alpha floats)."""
        key = (parts, tuple(color), px)
        if key in self._label_sprites:
            return self._label_sprites[key]

        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype("arialbd.ttf", px)
        except OSError:
            font = ImageFont.load_default()
        pad = max(4, px // 3)
        plate_h = px + 2 * max(3, px // 4)
        tip = max(4, px // 3)
        boxes = [font.getbbox(text) for text, _, _ in parts]
        widths = [box[2] - box[0] + 2 * pad for box in boxes]
        w = sum(widths)
        # Channels are written as B, G, R to match the frame.
        plate = Image.new("RGBA", (w, plate_h), (20, 16, 14, 255))
        draw = ImageDraw.Draw(plate)
        at = 0
        for (text, fill, ink), box, width in zip(parts, boxes, widths):
            draw.rectangle((at, 0, at + width, plate_h), fill=tuple(fill) + (255,))
            draw.text(
                (at + pad - box[0], (plate_h - (box[3] - box[1])) / 2 - box[1]),
                text,
                font=font,
                fill=tuple(ink) + (255,),
            )
            at += width
        mask = Image.new("L", (w, plate_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, w - 1, plate_h - 1), radius=max(3, px // 4), fill=255
        )
        img = Image.new("RGBA", (w, plate_h + tip), (0, 0, 0, 0))
        img.paste(plate, (0, 0), mask)
        middle = w / 2.0
        ImageDraw.Draw(img).polygon(
            [
                (middle - tip, plate_h - 1),
                (middle + tip, plate_h - 1),
                (middle, plate_h + tip - 1),
            ],
            fill=tuple(int(c) for c in color) + (255,),
        )
        sprite = np.asarray(img, dtype=np.float32).copy()
        sprite[..., 3] /= 255.0
        self._label_sprites[key] = sprite
        return sprite

    def _label_parts(
        self, idx: int
    ) -> Tuple[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]], ...]:
        """Label chips as (text, fill BGR, text BGR): the number, then any role."""
        white = (255, 255, 255)
        plate = (20, 16, 14)
        color = tuple(int(c) for c in self._label_colors.get(idx, (200, 200, 200)))
        parts = [(str(idx), color, white)]
        det = self.result.detections[idx]
        if det.is_goalkeeper:
            parts.append(("GK", plate, white))
        if getattr(det, "is_referee", False):
            parts.append(("REF", plate, white))
        if idx == self._defender_idx:
            line_bgr = tuple(int(c) for c in reversed(self.LINE_RGB))
            parts.append(("LAST DEFENDER", line_bgr, (26, 20, 4)))
            return tuple(parts)
        if self._player_is_offside(idx):
            parts.append(("OFFSIDE", TEAM_BGR[1], white))
        fp = self._player_field.get(idx)
        sign = self._attack_sign()
        if (
            idx in self._key_players
            and fp is not None
            and self._offside_field_x is not None
            and sign is not None
        ):
            # Same sign convention as the decision, so a reverse-angle camera
            # doesn't show a negative distance for an offside attacker, and the
            # same rounding as the panel, so both show the same digits.
            distance = round((fp[0] - self._offside_field_x) * sign, 3)  # as the panel rounds it
            parts.append((f"{distance:+.2f} m", plate, white))
        return tuple(parts)

    def _draw_pulse(
        self,
        frame: np.ndarray,
        pose: np.ndarray,
        width: int,
        height: int,
        t: float,
    ) -> np.ndarray:
        """Slow soft glow on the defender's line-setting body part."""
        if self._part_point is None:
            return frame
        screen = self._project(pose, self._part_point, width, height)
        if screen is None:
            return frame
        x, y = screen
        if x < 0 or x > width or y < 0 or y > height:
            return frame

        phase = 2.0 * math.pi * (t % self.PULSE_PERIOD) / self.PULSE_PERIOD
        level = 0.5 - 0.5 * math.cos(phase)  # slow 0..1, never a fast flash
        # Keep the glow about one body part wide at any zoom.
        top = self._project(pose, self._part_point + np.array([0.0, 0.0, 0.3]), width, height)
        span = abs(top[1] - y) if top is not None else height / 40.0
        base = float(np.clip(span, 6.0, height * 0.09))
        radius = int(base * (1.0 + 0.35 * level))
        overlay = frame.copy()
        cv2.circle(overlay, (int(x), int(y)), radius, (90, 235, 255), -1, cv2.LINE_AA)
        alpha = 0.18 + 0.42 * level
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.circle(
            frame,
            (int(x), int(y)),
            max(2, int(base * 0.22)),
            (60, 220, 255),
            -1,
            cv2.LINE_AA,
        )
        return frame

    # ------------------------------------------------------------------
    # Verdict helpers
    # ------------------------------------------------------------------

    def _gap_metres(self) -> Optional[float]:
        """The call's margin in metres (+ = attacker ahead), as the verdict panel shows it.

        The body mesh is planted by its whole silhouette and only drives the
        picture (gap bar, glow).  Measuring the arm-free mesh again gave a
        second number for the same call (1.17 m against the panel's 1.26 m).
        ``None`` when there is no full-3D margin, exactly like the panel.
        """
        # Imported here: phases pulls in the whole pipeline.
        from var_offside.web.phases import verdict_margin_m

        return verdict_margin_m(self.result)

    def _attack_sign(self) -> Optional[int]:
        """+1 or -1: the direction of the attacked goal along pitch x, as the decision uses it."""
        line = getattr(self.offside_result, "offside_line", None)
        sign = getattr(line, "attack_field_sign", None)
        if sign is None and self._attacking_side in ("left", "right"):
            from var_offside.offside.line_calculator import attack_field_sign

            sign = attack_field_sign(
                self.result.detections, self._attacking_side, self.calibration
            )
        return sign

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def _options(options: Optional[dict]) -> Dict[str, bool]:
        merged = dict(DEFAULT_OPTIONS)
        if options:
            for key in DEFAULT_OPTIONS:
                if key in options and options[key] is not None:
                    merged[key] = bool(options[key])
        return merged

    def _notify(self, stage: str, message: str) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(stage, message)
        except Exception as exc:  # noqa: BLE001 - never break rendering
            logger.warning("LiveScene status callback failed: %s", exc)

    @staticmethod
    def _stopped(should_stop: Optional[Callable[[], bool]]) -> bool:
        if should_stop is None:
            return False
        try:
            return bool(should_stop())
        except Exception:  # noqa: BLE001 - treat a broken predicate as stop
            return True
