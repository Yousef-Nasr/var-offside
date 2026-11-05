"""
Smoke test for the live 3D web view (``var_offside.web.live3d.LiveScene``).

Runs the normal offside pipeline on one frame, builds the stand-in scene,
renders every camera preset, times :meth:`LiveScene.render`, writes one second
of the cinematic replay, and optionally upgrades the key players to SAM 3D
Body meshes and renders the presets again.

Usage:
    python scripts/smoke_live3d.py --image images/test7.png --no-meshes
    python scripts/smoke_live3d.py --image images/test7.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from var_offside.pipeline import OffsidePipeline  # noqa: E402
from var_offside.web.live3d import LiveScene  # noqa: E402

RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
RENDER_OPTIONS: Dict[str, bool] = {
    "line_plane": True,
    "photo": False,
    "labels": True,
    "pulse": True,
}
PRESETS = ("broadcast", "birdseye", "line", "free")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Smoke test the live 3D scene")
    parser.add_argument("--image", required=True, help="Path to the input frame")
    parser.add_argument(
        "--attacking-side",
        default="right",
        choices=["left", "right"],
        help="Attacking direction for the pipeline (default: right)",
    )
    parser.add_argument(
        "--out",
        default="results/live3d_smoke",
        help="Output directory (default: results/live3d_smoke)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for SAM 3D Body (default: cuda)",
    )
    parser.add_argument(
        "--no-meshes",
        action="store_true",
        help="Skip the slow SAM 3D Body mesh upgrade",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    return parser.parse_args()


def on_status(stage: str, message: str) -> None:
    """Print LiveScene status updates."""
    print(f"[status] {stage}: {message}" if message else f"[status] {stage}")


def render_presets(
    scene: LiveScene, out_dir: Path, suffix: str, paths: List[Path]
) -> None:
    """Render every preset to a PNG and remember the file paths."""
    for name in PRESETS:
        camera = scene.preset(name)
        frame = scene.render(
            camera, RENDER_WIDTH, RENDER_HEIGHT, RENDER_OPTIONS, t=0.0
        )
        path = out_dir / f"{name}{suffix}.png"
        cv2.imwrite(str(path), frame)
        paths.append(path)
        print(f"  {name:<9} -> {path}  ({frame.shape[1]}x{frame.shape[0]})")


def time_render(scene: LiveScene, frames: int = 20) -> float:
    """Time ``render`` on the free preset, return the mean in milliseconds."""
    camera = scene.preset("free")
    # Warm up (first frame may compile GL programs).
    scene.render(camera, RENDER_WIDTH, RENDER_HEIGHT, RENDER_OPTIONS, t=0.0)
    started = time.perf_counter()
    for i in range(frames):
        scene.render(
            camera,
            RENDER_WIDTH,
            RENDER_HEIGHT,
            RENDER_OPTIONS,
            t=i / 24.0,
        )
    elapsed = time.perf_counter() - started
    mean_ms = elapsed * 1000.0 / max(frames, 1)
    print(
        f"  render(): {mean_ms:.1f} ms/frame mean over {frames} frames "
        f"at {RENDER_WIDTH}x{RENDER_HEIGHT}"
    )
    return mean_ms


def write_replay(scene: LiveScene, path: Path, fps: int = 24) -> None:
    """Write one second of the cinematic replay to an MP4."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(path), fourcc, fps, (RENDER_WIDTH, RENDER_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    written = 0
    try:
        for frame in scene.replay(
            RENDER_WIDTH, RENDER_HEIGHT, RENDER_OPTIONS, fps=fps
        ):
            writer.write(frame)
            written += 1
            if written >= fps:
                break
    finally:
        writer.release()
    print(f"  replay ({written} frames) -> {path}")


def main() -> int:
    """Run the smoke test."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    print(f"Analysing {args.image} (attacking side: {args.attacking_side}) ...")
    pipeline = OffsidePipeline()
    result = pipeline.analyze(
        args.image,
        attacking_side=args.attacking_side,
        enable_team_classification=True,
        generate_visualization=False,
        verbose=args.verbose,
    )

    if result.errors:
        print(f"Pipeline warnings: {result.errors}")
    if result.offside_result is None:
        print("No offside result - cannot build the 3D view.")
        return 1

    image_bgr = result.image
    if image_bgr is None:
        image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"Could not read {args.image}")
        return 1

    line = result.offside_result.offside_line
    print(
        f"Detections: {len(result.detections)}  "
        f"calibration: {result.calibration.method} "
        f"(full 3D: {result.calibration.has_3d_calibration})  "
        f"line field_x: {line.field_x}  "
        f"verdict: {'OFFSIDE' if result.offside_result.any_offside else 'ONSIDE'}"
    )

    scene = LiveScene(
        result, image_bgr, device=args.device, on_status=on_status
    )
    try:
        print("Building proxy scene ...")
        scene.build_proxy()

        print("Rendering presets (proxy) ...")
        render_presets(scene, out_dir, "", paths)

        print("Timing render() ...")
        time_render(scene)

        print("Writing 1 s replay ...")
        replay_path = out_dir / "replay_1s.mp4"
        write_replay(scene, replay_path)
        paths.append(replay_path)

        if not args.no_meshes:
            print("Upgrading to SAM 3D Body meshes ...")
            scene.upgrade_meshes()
            print("Rendering presets (body meshes) ...")
            render_presets(scene, out_dir, "_meshes", paths)
    finally:
        scene.close()

    print("\nWrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
