"""
Command-line offside check on one frame.

Usage:
    var-offside --image frame.png --attacking-side left --attacking-team 1
    var-offside --image frame.png --attacking-side right --attacking-team 2 --method vanishing_point
    python -m var_offside.demo ...   (same thing)

Teams are clustered by kit colour into team 1 and team 2. Run once with
``--debug`` to see which team got which number, then pass the attackers'
number with ``--attacking-team``.
"""

import argparse
import os

import cv2
from pathlib import Path

from var_offside.pipeline import OffsidePipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VAR Line Analysis Demo")

    parser.add_argument(
        "--image", "-i", type=str, required=True, help="Path to input image"
    )

    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="auto",
        choices=["auto", "pnl", "vanishing_point", "manual"],
        help="Calibration method (default: auto)",
    )

    parser.add_argument(
        "--attacking-side",
        type=str,
        choices=["left", "right"],
        required=True,
        help="Direction the attacking team is playing towards in the image.",
    )

    parser.add_argument(
        "--attacking-team",
        type=int,
        choices=[1, 2],
        required=True,
        help="Which kit cluster is attacking (1 or 2).",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config.yaml (default: config.yaml in the project folder)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output image path (default: results/<input>_offside.jpg)",
    )

    parser.add_argument(
        "--no-display", action="store_true", help="Don't display result (just save)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Save debug images for each pipeline stage",
    )

    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_args()

    if args.config:
        os.environ["VAR_OFFSIDE_CONFIG"] = os.path.abspath(args.config)

    # Check input exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    print(f"\n{'=' * 50}")
    print("VAR LINE ANALYSIS SYSTEM")
    print(f"{'=' * 50}\n")

    print(f"Loading image: {args.image}")
    print(f"Calibration method: {args.method}")

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = OffsidePipeline(calibration_method=args.method)

    # Run analysis
    print("Running line analysis...")

    result = pipeline.analyze(
        args.image,
        attacking_side=args.attacking_side,
        attacking_team=args.attacking_team,
        generate_visualization=True,
        save_result=True,
        output_path=args.output,
        verbose=args.verbose,
        debug=args.debug,
    )

    # Print summary
    print(pipeline.get_summary(result))

    # Verbose output
    if args.verbose and result.offside_result:
        print("\nDetailed player analysis:")
        for status in result.offside_result.player_statuses:
            status_str = "ahead of line" if status.is_offside else "behind line"
            print(
                f"  Player {status.player_idx}: {status_str} by {abs(status.margin_px):.1f}px"
            )

    # Display result
    if not args.no_display and result.visualization is not None:
        print("\nDisplaying result (press any key to close)...")

        # Resize if too large
        h, w = result.visualization.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            display = cv2.resize(result.visualization, None, fx=scale, fy=scale)
        else:
            display = result.visualization

        cv2.imshow("VAR Line Analysis", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0 if not result.errors else 1


if __name__ == "__main__":
    exit(main())
