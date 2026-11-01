"""
Compare the two 3D mesh offside anchors on a folder of images.

For each image the pipeline runs twice with ``offside_method="mesh_3d"``:
``directional`` (current behaviour) and ``center`` (bbox bottom-centre ground
point, then the leading non-arm vertex).  Prints the verdict and the largest
forward margin for both, and flags rows whose verdict flipped.

    python scripts/compare_offside_anchor.py [folder] [attacking_side]

The folder defaults to ``images/`` and the attacking side to ``left``.
This is slow (SAM 3D Body + two runs per image); use ``--limit`` to sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_folder(arg: str) -> Path:
    path = Path(arg)
    if path.is_absolute() or path.is_dir():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.is_dir() else path


def _summarise(result) -> Tuple[str, Optional[float]]:
    """Return (verdict, largest signed margin in metres) for one analysis."""
    offside = getattr(result, "offside_result", None)
    if offside is None:
        return "n/a", None
    margins = [
        status.margin_m
        for status in offside.player_statuses
        if status.margin_m is not None
    ]
    margin = max(margins) if margins else None
    return ("OFFSIDE" if offside.any_offside else "ONSIDE"), margin


def _fmt_margin(margin: Optional[float]) -> str:
    return "n/a" if margin is None else f"{margin:+.2f}m"


def main() -> int:
    from var_offside.config import get_settings
    from var_offside.pipeline import OffsidePipeline

    parser = argparse.ArgumentParser(
        description="Compare directional vs centre mesh-offside anchors."
    )
    parser.add_argument("folder", nargs="?", default="images/", help="Image folder.")
    parser.add_argument(
        "attacking_side",
        nargs="?",
        default="left",
        choices=["left", "right"],
        help="Direction of attack.",
    )
    parser.add_argument(
        "--attacking-team",
        type=int,
        default=None,
        choices=[1, 2],
        help="Optional attacking team id.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (defaults to settings.device).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N images (0 = all).",
    )
    args = parser.parse_args()

    folder = _resolve_folder(args.folder)
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 1

    images = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"No images found in {folder}")
        return 1

    settings = get_settings()
    pipeline = OffsidePipeline(device=args.device)

    header = (
        f"{'image':<32} {'verdict_directional':<20} {'margin':>9}  "
        f"{'verdict_center':<20} {'margin':>9}  FLIPPED"
    )
    print(header)
    print("-" * len(header))

    flipped_count = 0
    for image_path in images:
        runs = {}
        for anchor in ("directional", "center"):
            # The mesh calculator follows settings at call time, so one
            # pipeline instance can be reused for both runs.
            settings.offside.mesh_anchor = anchor
            analysis = pipeline.analyze(
                str(image_path),
                attacking_side=args.attacking_side,
                attacking_team=args.attacking_team,
                offside_method="mesh_3d",
                generate_visualization=False,
                verbose=False,
            )
            runs[anchor] = _summarise(analysis)

        v_dir, m_dir = runs["directional"]
        v_ctr, m_ctr = runs["center"]
        flipped = v_dir != v_ctr and v_dir != "n/a" and v_ctr != "n/a"
        if flipped:
            flipped_count += 1
        print(
            f"{image_path.name:<32} {v_dir:<20} {_fmt_margin(m_dir):>9}  "
            f"{v_ctr:<20} {_fmt_margin(m_ctr):>9}  {'FLIPPED' if flipped else ''}"
        )

    print("-" * len(header))
    print(f"{len(images)} image(s), {flipped_count} flipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
