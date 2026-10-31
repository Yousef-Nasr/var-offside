"""Standalone CLI for VLM-based football image analysis."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from var_offside.automation.vlm import (
    DEFAULT_FOOTBALL_ANALYSIS_PROMPT,
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterVLMAnalyzer,
    draw_vlm_boxes,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run standalone requests + Pydantic football image analysis"
    )
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENROUTER_MODEL,
        help="OpenRouter model id",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional OpenRouter API key override",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt override",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Load prompt text from a file",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry count",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="Display debug bounding boxes with matplotlib",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional file path to save the JSON response",
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help="Disable OpenRouter reasoning mode",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def _load_prompt(args: argparse.Namespace) -> str:
    start_time = time.perf_counter()
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        logging.getLogger(__name__).debug(
            "Loaded prompt file: path=%s chars=%s elapsed_ms=%.1f",
            args.prompt_file,
            len(prompt),
            (time.perf_counter() - start_time) * 1000.0,
        )
        return prompt
    if args.prompt:
        prompt = args.prompt.strip()
        logging.getLogger(__name__).debug(
            "Loaded inline prompt: chars=%s elapsed_ms=%.1f",
            len(prompt),
            (time.perf_counter() - start_time) * 1000.0,
        )
        return prompt
    prompt = DEFAULT_FOOTBALL_ANALYSIS_PROMPT
    logging.getLogger(__name__).debug(
        "Loaded default prompt: chars=%s elapsed_ms=%.1f",
        len(prompt),
        (time.perf_counter() - start_time) * 1000.0,
    )
    return prompt


def main() -> int:
    """Run the standalone image-analysis CLI."""
    cli_start = time.perf_counter()
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        return 1

    prompt = _load_prompt(args)
    logger.info("Starting standalone VLM analysis for image=%s", image_path)
    analyzer = OpenRouterVLMAnalyzer(
        api_key=args.api_key,
        model_id=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        reasoning_enabled=not args.disable_reasoning,
    )

    analysis_start = time.perf_counter()
    result = analyzer.analyze_image(image_path=image_path, prompt=prompt)
    logger.info(
        "Standalone analysis completed: elapsed_ms=%.1f",
        (time.perf_counter() - analysis_start) * 1000.0,
    )

    json_output = result.model_dump_json(indent=2)
    print(json_output)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json_output + "\n", encoding="utf-8")
        logger.info("Saved JSON output to %s", output_path)

    if args.show_boxes:
        logger.info("Rendering debug bounding boxes")
        render_start = time.perf_counter()
        draw_vlm_boxes(image_path=image_path, response_json=result, show=True)
        logger.info(
            "Rendered debug view: elapsed_ms=%.1f",
            (time.perf_counter() - render_start) * 1000.0,
        )

    logger.info(
        "Standalone CLI finished: total_elapsed_ms=%.1f",
        (time.perf_counter() - cli_start) * 1000.0,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
