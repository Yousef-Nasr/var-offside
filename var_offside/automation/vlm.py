"""Requests-based football image analysis with Pydantic validation."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Literal

import cv2
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
load_dotenv()
load_dotenv(_PACKAGE_ROOT / ".env")

logger = logging.getLogger(__name__)


def _elapsed_ms(start_time: float) -> float:
    """Return elapsed milliseconds from a perf counter start."""
    return (time.perf_counter() - start_time) * 1000.0


DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL_ID", "google/gemma-4-31b-it")
DEFAULT_FOOTBALL_ANALYSIS_PROMPT = """
Analyze this football image.

Detect:
- Goalkeeper (if exists)
- Any person who is NOT a field player (e.g., referee, staff)
- Goal side in the image (left or right)

Rules:
- Ignore normal field players.
- Goal side must ALWAYS be returned (even if no goalkeeper).
- Use normalized bbox [x_min, y_min, x_max, y_max].

Return ONLY JSON:

{
  "goal_side": "left" | "right",
  "goalkeeper": {
    "bbox": [float, float, float, float],
    "confidence": float
  } | null,
  "non_players": [
    {
      "bbox": [float, float, float, float],
      "type": "referee" | "staff" | "unknown",
      "confidence": float
    }
  ]
}
""".strip()


class GoalkeeperDetection(BaseModel):
    """Structured goalkeeper detection."""

    model_config = ConfigDict(extra="forbid")

    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: list[float]) -> list[float]:
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly four values")
        if any(value < 0.0 or value > 1.0 for value in bbox):
            raise ValueError("bbox values must be normalized between 0 and 1")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x_max > x_min and y_max > y_min")
        return bbox


class NonPlayerDetection(BaseModel):
    """Structured non-player detection."""

    model_config = ConfigDict(extra="forbid")

    bbox: list[float] = Field(min_length=4, max_length=4)
    type: Literal["referee", "staff", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: list[float]) -> list[float]:
        return GoalkeeperDetection.validate_bbox(bbox)


class FootballImageAnalysis(BaseModel):
    """Validated VLM response for one football image."""

    model_config = ConfigDict(extra="forbid")

    goal_side: Literal["left", "right"]
    goalkeeper: GoalkeeperDetection | None = None
    non_players: list[NonPlayerDetection] = Field(default_factory=list)


def image_file_to_data_url(image_path: str | Path) -> str:
    """Convert a local image file into a data URL for multimodal prompts."""
    start_time = time.perf_counter()
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/png"

    logger.debug("Encoding image as data URL: path=%s mime_type=%s", path, mime_type)

    with path.open("rb") as image_file:
        image_bytes = image_file.read()
        base64_data = base64.b64encode(image_bytes).decode("utf-8")

    logger.debug(
        "Encoded image as data URL: path=%s bytes=%s elapsed_ms=%.1f",
        path,
        len(image_bytes),
        _elapsed_ms(start_time),
    )

    return f"data:{mime_type};base64,{base64_data}"


def image_array_to_data_url(image: Any, image_format: str = ".png") -> str:
    """Convert an in-memory OpenCV image into a data URL."""
    start_time = time.perf_counter()

    if image is None:
        raise ValueError("image is required")

    success, encoded = cv2.imencode(image_format, image)
    if not success:
        raise ValueError("Could not encode image for VLM analysis")

    mime_type = mimetypes.types_map.get(image_format.lower(), "image/png")
    base64_data = base64.b64encode(encoded.tobytes()).decode("utf-8")

    logger.debug(
        "Encoded ndarray as data URL: format=%s bytes=%s elapsed_ms=%.1f",
        image_format,
        encoded.size,
        _elapsed_ms(start_time),
    )

    return f"data:{mime_type};base64,{base64_data}"


def coerce_analysis_response(
    response_json: str | dict[str, Any] | FootballImageAnalysis,
) -> FootballImageAnalysis:
    """Normalize supported response payloads into the Pydantic model."""
    if isinstance(response_json, FootballImageAnalysis):
        return response_json
    if isinstance(response_json, str):
        return _validate_analysis_dict(json.loads(_strip_code_fences(response_json)))
    return _validate_analysis_dict(response_json)


class OpenRouterVLMAnalyzer:
    """Analyze football images through OpenRouter using direct HTTP requests."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = DEFAULT_OPENROUTER_MODEL,
        timeout: int = 120,
        max_retries: int = 5,
        retry_delay_seconds: float = 2.0,
        reasoning_enabled: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.timeout = timeout
        self.reasoning_enabled = reasoning_enabled

        logger.info(
            "Initialized OpenRouterVLMAnalyzer model=%s timeout=%s reasoning_enabled=%s",
            self.model_id,
            self.timeout,
            self.reasoning_enabled,
        )

    def analyze_image(
        self,
        image_path: str | Path,
        prompt: str = DEFAULT_FOOTBALL_ANALYSIS_PROMPT,
    ) -> FootballImageAnalysis:
        """Call the VLM and validate the JSON response with Pydantic."""
        analysis_start = time.perf_counter()

        dimensions_start = time.perf_counter()
        image_width, image_height = _get_image_dimensions(image_path)
        logger.debug(
            "Loaded image dimensions: image_path=%s width=%s height=%s elapsed_ms=%.1f",
            image_path,
            image_width,
            image_height,
            _elapsed_ms(dimensions_start),
        )

        image_data_url = image_file_to_data_url(image_path)

        return self._analyze_image_data(
            image_data_url=image_data_url,
            image_width=image_width,
            image_height=image_height,
            image_source=str(image_path),
            prompt=prompt,
            analysis_start=analysis_start,
        )

    def analyze_array(
        self,
        image: Any,
        prompt: str = DEFAULT_FOOTBALL_ANALYSIS_PROMPT,
    ) -> FootballImageAnalysis:
        """Call the VLM for an in-memory OpenCV image."""
        analysis_start = time.perf_counter()

        if image is None or not hasattr(image, "shape"):
            raise ValueError("image must be a valid OpenCV image array")

        image_height, image_width = image.shape[:2]
        image_data_url = image_array_to_data_url(image)

        return self._analyze_image_data(
            image_data_url=image_data_url,
            image_width=image_width,
            image_height=image_height,
            image_source="ndarray",
            prompt=prompt,
            analysis_start=analysis_start,
        )

    def _analyze_image_data(
        self,
        image_data_url: str,
        image_width: int | None,
        image_height: int | None,
        image_source: str,
        prompt: str,
        analysis_start: float,
    ) -> FootballImageAnalysis:
        """Send one prepared image payload to the VLM and validate the response."""

        payload_start = time.perf_counter()
        payload = self._build_request_payload(
            prompt=prompt, image_data_url=image_data_url
        )
        logger.debug(
            "Prepared multimodal request: image_source=%s prompt_chars=%s image_size=%sx%s payload_chars=%s elapsed_ms=%.1f",
            image_source,
            len(prompt),
            image_width,
            image_height,
            len(json.dumps(payload)),
            _elapsed_ms(payload_start),
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            attempt_start = time.perf_counter()
            try:
                logger.info(
                    "Calling OpenRouter VLM: attempt=%s/%s model=%s",
                    attempt,
                    self.max_retries,
                    self.model_id,
                )
                request_start = time.perf_counter()
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                logger.debug(
                    "OpenRouter HTTP completed: attempt=%s status_code=%s elapsed_ms=%.1f",
                    attempt,
                    response.status_code,
                    _elapsed_ms(request_start),
                )

                json_start = time.perf_counter()
                response_payload = response.json()
                logger.debug(
                    "Decoded OpenRouter JSON: attempt=%s elapsed_ms=%.1f",
                    attempt,
                    _elapsed_ms(json_start),
                )

                parse_start = time.perf_counter()
                parsed = self._parse_response_payload(
                    response_payload,
                    image_width=image_width,
                    image_height=image_height,
                )
                logger.debug(
                    "Validated VLM response: attempt=%s elapsed_ms=%.1f",
                    attempt,
                    _elapsed_ms(parse_start),
                )
                logger.info(
                    "VLM call succeeded: goal_side=%s goalkeeper=%s non_players=%s attempt_elapsed_ms=%.1f total_elapsed_ms=%.1f",
                    parsed.goal_side,
                    parsed.goalkeeper is not None,
                    len(parsed.non_players),
                    _elapsed_ms(attempt_start),
                    _elapsed_ms(analysis_start),
                )
                return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "VLM call failed: attempt=%s/%s elapsed_ms=%.1f error=%s",
                    attempt,
                    self.max_retries,
                    _elapsed_ms(attempt_start),
                    exc,
                )
                if attempt < self.max_retries:
                    logger.debug(
                        "Sleeping before retry: seconds=%s",
                        self.retry_delay_seconds,
                    )
                    time.sleep(self.retry_delay_seconds)

        raise RuntimeError(
            f"Failed to analyze image after {self.max_retries} attempts"
        ) from last_error

    def _build_request_payload(
        self, prompt: str, image_data_url: str
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        logger.debug(
            "Built request payload: keys=%s elapsed_ms=%.1f",
            sorted(payload.keys()),
            _elapsed_ms(start_time),
        )
        return payload

    def _parse_response_payload(
        self,
        payload: dict[str, Any],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> FootballImageAnalysis:
        start_time = time.perf_counter()
        raw_text = payload["choices"][0]["message"].get("content", "")
        cleaned_text = _strip_code_fences(raw_text)
        logger.debug(
            "Prepared response text for validation: raw_chars=%s cleaned_chars=%s elapsed_ms=%.1f",
            len(raw_text),
            len(cleaned_text),
            _elapsed_ms(start_time),
        )
        json_start = time.perf_counter()
        parsed_json = json.loads(cleaned_text)
        logger.debug(
            "Parsed response JSON text: elapsed_ms=%.1f", _elapsed_ms(json_start)
        )

        validate_start = time.perf_counter()
        result = _validate_analysis_dict(
            parsed_json, image_width=image_width, image_height=image_height
        )
        logger.debug(
            "Validated/coerced response model: elapsed_ms=%.1f total_parse_elapsed_ms=%.1f",
            _elapsed_ms(validate_start),
            _elapsed_ms(start_time),
        )
        return result


def analyze_football_image(
    image_path: str | Path,
    prompt: str = DEFAULT_FOOTBALL_ANALYSIS_PROMPT,
    api_key: str | None = None,
    model_id: str = DEFAULT_OPENROUTER_MODEL,
    timeout: int = 120,
    max_retries: int = 5,
    retry_delay_seconds: float = 2.0,
    reasoning_enabled: bool = False,
) -> FootballImageAnalysis:
    """Convenience wrapper for one-off VLM analysis calls."""
    analyzer = OpenRouterVLMAnalyzer(
        api_key=api_key,
        model_id=model_id,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        reasoning_enabled=reasoning_enabled,
    )
    return analyzer.analyze_image(image_path=image_path, prompt=prompt)


def analyze_football_frame(
    image: Any,
    prompt: str = DEFAULT_FOOTBALL_ANALYSIS_PROMPT,
    api_key: str | None = None,
    model_id: str = DEFAULT_OPENROUTER_MODEL,
    timeout: int = 120,
    max_retries: int = 5,
    retry_delay_seconds: float = 2.0,
    reasoning_enabled: bool = False,
) -> FootballImageAnalysis:
    """Convenience wrapper for one-off VLM analysis calls on arrays."""
    analyzer = OpenRouterVLMAnalyzer(
        api_key=api_key,
        model_id=model_id,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        reasoning_enabled=reasoning_enabled,
    )
    return analyzer.analyze_array(image=image, prompt=prompt)


def call_vlm_with_retry(
    image_path: str | Path,
    prompt: str = DEFAULT_FOOTBALL_ANALYSIS_PROMPT,
    api_key: str | None = None,
    model_id: str = DEFAULT_OPENROUTER_MODEL,
    max_retries: int = 5,
    timeout: int = 120,
    retry_delay_seconds: float = 2.0,
    reasoning_enabled: bool = True,
    return_dict: bool = True,
) -> FootballImageAnalysis | dict[str, Any]:
    """Compatibility wrapper around the requests + Pydantic analyzer."""
    result = analyze_football_image(
        image_path=image_path,
        prompt=prompt,
        api_key=api_key,
        model_id=model_id,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        reasoning_enabled=reasoning_enabled,
    )
    if return_dict:
        return result.model_dump()
    return result


def draw_vlm_boxes(
    image_path: str | Path,
    response_json: str | dict[str, Any] | FootballImageAnalysis,
    show: bool = True,
) -> Any:
    start_time = time.perf_counter()
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_height, image_width = image_rgb.shape[:2]

    if isinstance(response_json, FootballImageAnalysis):
        response = response_json
    else:
        raw_response = (
            json.loads(_strip_code_fences(response_json))
            if isinstance(response_json, str)
            else response_json
        )
        response = _validate_analysis_dict(
            raw_response, image_width=image_width, image_height=image_height
        )

    logger.debug(
        "Drawing boxes: image_path=%s goalkeeper=%s non_players=%s",
        image_path,
        response.goalkeeper is not None,
        len(response.non_players),
    )

    def to_xyxy(norm_bbox: list[float]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = norm_bbox
        return (
            int(x1 * image_width),
            int(y1 * image_height),
            int(x2 * image_width),
            int(y2 * image_height),
        )

    def draw_box(bbox: list[float], label: str, color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = to_xyxy(bbox)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_rgb,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    if response.goalkeeper is not None:
        label = f"GK | {response.goal_side} | {response.goalkeeper.confidence:.2f}"
        draw_box(response.goalkeeper.bbox, label, (255, 0, 0))

    for item in response.non_players:
        label = f"{item.type} | {item.confidence:.2f}"
        draw_box(item.bbox, label, (0, 255, 0))

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"Goal side: {response.goal_side}")
        plt.show()

    logger.debug(
        "Rendered debug boxes: image_path=%s show=%s elapsed_ms=%.1f",
        image_path,
        show,
        _elapsed_ms(start_time),
    )

    return image_rgb


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _get_image_dimensions(image_path: str | Path) -> tuple[int | None, int | None]:
    start_time = time.perf_counter()
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(
            "Could not read image dimensions for %s elapsed_ms=%.1f",
            image_path,
            _elapsed_ms(start_time),
        )
        return None, None
    height, width = image.shape[:2]
    logger.debug(
        "Read image dimensions from disk: image_path=%s width=%s height=%s elapsed_ms=%.1f",
        image_path,
        width,
        height,
        _elapsed_ms(start_time),
    )
    return width, height


def _validate_analysis_dict(
    data: dict[str, Any],
    image_width: int | None = None,
    image_height: int | None = None,
) -> FootballImageAnalysis:
    start_time = time.perf_counter()
    coerced_dict = _coerce_analysis_dict(
        data, image_width=image_width, image_height=image_height
    )

    try:
        result = FootballImageAnalysis.model_validate(coerced_dict)
        logger.debug(
            "Validated analysis dict successfully: elapsed_ms=%.1f",
            _elapsed_ms(start_time),
        )
        return result
    except ValidationError as exc:
        logger.warning(
            "Validation failed after bbox coercion, attempting lenient sanitation: elapsed_ms=%.1f error=%s",
            _elapsed_ms(start_time),
            exc,
        )
        sanitize_start = time.perf_counter()
        sanitized_dict = _sanitize_analysis_dict(coerced_dict)
        result = FootballImageAnalysis.model_validate(sanitized_dict)
        logger.debug(
            "Validated sanitized analysis dict successfully: sanitize_elapsed_ms=%.1f total_elapsed_ms=%.1f",
            _elapsed_ms(sanitize_start),
            _elapsed_ms(start_time),
        )
        return result


def _coerce_analysis_dict(
    data: dict[str, Any],
    image_width: int | None,
    image_height: int | None,
) -> dict[str, Any]:
    normalized = dict(data)

    goalkeeper = normalized.get("goalkeeper")
    if isinstance(goalkeeper, dict):
        goalkeeper = dict(goalkeeper)
        goalkeeper["bbox"] = _coerce_bbox_payload(
            goalkeeper, image_width=image_width, image_height=image_height
        )
        goalkeeper.pop("point", None)
        normalized["goalkeeper"] = goalkeeper

    non_players = []
    for item in normalized.get("non_players", []):
        if isinstance(item, dict):
            item = dict(item)
            item["bbox"] = _coerce_bbox_payload(
                item, image_width=image_width, image_height=image_height
            )
            item.pop("point", None)
        non_players.append(item)
    normalized["non_players"] = non_players

    return normalized


def _coerce_bbox_payload(
    payload: dict[str, Any],
    image_width: int | None,
    image_height: int | None,
) -> Any:
    bbox = payload.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return _coerce_bbox(bbox, image_width=image_width, image_height=image_height)

    point = payload.get("point")
    if isinstance(point, (list, tuple)) and len(point) == 2:
        return _coerce_point_to_bbox(
            point, image_width=image_width, image_height=image_height
        )

    return bbox if bbox is not None else point


def _coerce_bbox(
    bbox: Any,
    image_width: int | None,
    image_height: int | None,
) -> Any:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return bbox

    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return bbox

    if any(value > 1.0 for value in values) and image_width and image_height:
        normalized = [
            values[0] / image_width,
            values[1] / image_height,
            values[2] / image_width,
            values[3] / image_height,
        ]
        normalized = _clip_bbox(normalized)
        logger.warning(
            "Converted pixel bbox to normalized coordinates: raw=%s normalized=%s image_size=%sx%s",
            values,
            normalized,
            image_width,
            image_height,
        )
        return normalized

    clipped = _clip_bbox(values)
    if clipped != values:
        logger.warning(
            "Clipped out-of-range bbox values: raw=%s clipped=%s", values, clipped
        )
    return clipped


def _coerce_point_to_bbox(
    point: Any,
    image_width: int | None,
    image_height: int | None,
) -> Any:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return point

    try:
        values = [float(value) for value in point]
    except (TypeError, ValueError):
        return point

    if any(value > 1.0 for value in values) and image_width and image_height:
        normalized_point = [values[0] / image_width, values[1] / image_height]
        normalized_point = _clip_point(normalized_point)
        bbox = _point_to_bbox(normalized_point)
        logger.warning(
            "Converted pixel point to normalized bbox: raw=%s bbox=%s image_size=%sx%s",
            values,
            bbox,
            image_width,
            image_height,
        )
        return bbox

    clipped_point = _clip_point(values)
    bbox = _point_to_bbox(clipped_point)
    logger.warning("Converted point to bbox: raw=%s bbox=%s", values, bbox)
    return bbox


def _clip_bbox(values: list[float]) -> list[float]:
    x1, y1, x2, y2 = values
    clipped = [
        min(max(x1, 0.0), 1.0),
        min(max(y1, 0.0), 1.0),
        min(max(x2, 0.0), 1.0),
        min(max(y2, 0.0), 1.0),
    ]
    if clipped[2] < clipped[0]:
        clipped[0], clipped[2] = clipped[2], clipped[0]
    if clipped[3] < clipped[1]:
        clipped[1], clipped[3] = clipped[3], clipped[1]
    return clipped


def _clip_point(values: list[float]) -> list[float]:
    return [min(max(values[0], 0.0), 1.0), min(max(values[1], 0.0), 1.0)]


def _point_to_bbox(values: list[float], radius: float = 0.01) -> list[float]:
    x, y = values
    return _clip_bbox([x - radius, y - radius, x + radius, y + radius])


def _sanitize_analysis_dict(data: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        "goal_side": data.get("goal_side"),
        "goalkeeper": None,
        "non_players": [],
    }

    goalkeeper = data.get("goalkeeper")
    if goalkeeper is not None:
        try:
            sanitized["goalkeeper"] = GoalkeeperDetection.model_validate(
                goalkeeper
            ).model_dump()
        except ValidationError as exc:
            logger.warning("Dropping invalid goalkeeper payload: %s", exc)

    for item in data.get("non_players", []):
        try:
            sanitized["non_players"].append(
                NonPlayerDetection.model_validate(item).model_dump()
            )
        except ValidationError as exc:
            logger.warning("Dropping invalid non-player payload: %s", exc)

    return sanitized
