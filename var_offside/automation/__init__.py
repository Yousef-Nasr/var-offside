"""Automation helpers backed by VLMs."""

from var_offside.automation.vlm import (
    DEFAULT_FOOTBALL_ANALYSIS_PROMPT,
    DEFAULT_OPENROUTER_MODEL,
    FootballImageAnalysis,
    GoalkeeperDetection,
    NonPlayerDetection,
    OpenRouterVLMAnalyzer,
    analyze_football_frame,
    analyze_football_image,
    call_vlm_with_retry,
    coerce_analysis_response,
    draw_vlm_boxes,
)

__all__ = [
    "DEFAULT_FOOTBALL_ANALYSIS_PROMPT",
    "DEFAULT_OPENROUTER_MODEL",
    "FootballImageAnalysis",
    "GoalkeeperDetection",
    "NonPlayerDetection",
    "OpenRouterVLMAnalyzer",
    "analyze_football_frame",
    "analyze_football_image",
    "call_vlm_with_retry",
    "coerce_analysis_response",
    "draw_vlm_boxes",
]
