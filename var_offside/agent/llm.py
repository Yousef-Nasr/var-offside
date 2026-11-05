"""One vision call through OpenRouter, returning a validated answer or a clear error.

Any error here means the check has no answer. The caller must turn that into a
question for the user, and never into a cleared frame.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, TypeVar, Union

import cv2
import numpy as np
import requests
from pydantic import BaseModel, ValidationError

from var_offside.agent.config import AgentConfig
from var_offside.agent.schemas import VisionAnswer

logger = logging.getLogger(__name__)

MAX_IMAGE_SIDE = 1280
JPEG_QUALITY = 85
MAX_TOKENS = 1000
RETRY_DELAY_SECONDS = 2.0
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

AnswerT = TypeVar("AnswerT", bound=BaseModel)
Image = Union[str, Path, np.ndarray]


class LLMError(RuntimeError):
    """The check has no usable answer."""


class NotConfigured(LLMError):
    """The key or model is missing."""


class RejectedByService(LLMError):
    """OpenRouter refused the request, and retrying won't help."""


class ServiceUnavailable(LLMError):
    """The network or service failed, including after the retry."""


class InvalidAnswer(LLMError):
    """The reply wasn't valid JSON for this check, including after the retry."""


class LimitReached(LLMError):
    """This run hit its spending or call limit."""


@dataclass
class Usage:
    """What this client has spent so far. One client serves one run."""

    calls: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0


def encode_image(image: Image, max_side: int = MAX_IMAGE_SIDE) -> str:
    """Return a JPEG data URL, shrunk so the longest side fits ``max_side``."""
    try:
        frame = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
    except OSError:  # ultralytics patches cv2.imread to raise on a missing file
        frame = None
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError(f"Could not read image: {image if isinstance(image, (str, Path)) else 'array'}")
    longest = max(frame.shape[:2])
    if longest > max_side:
        scale = max_side / longest
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _parse_json(text: str) -> Any:
    """Parse a reply, allowing for a Markdown code fence around the JSON."""
    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.S)
    return json.loads(fenced.group(1) if fenced else text)


def _reply_text(payload: Dict[str, Any]) -> str:
    """Pull the message text from a response, which may be a string or a list of parts."""
    if payload.get("error"):
        raise ServiceUnavailable(f"OpenRouter returned an error: {payload['error']}")
    choices = payload.get("choices")
    if not choices:
        raise ServiceUnavailable("OpenRouter returned no choices")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content or not str(content).strip():
        raise InvalidAnswer("The model returned an empty reply")
    return str(content)


class OpenRouterClient:
    """Sends vision checks for one run and enforces that run's limits."""

    def __init__(
        self,
        config: AgentConfig,
        session: Optional[Any] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.usage = Usage()
        self._session = session or requests.Session()
        self._sleep = sleep

    def ask(
        self,
        task: str,
        system: str,
        prompt: str,
        schema: Type[AnswerT],
        images: Sequence[Image] = (),
    ) -> VisionAnswer[AnswerT]:
        """Ask one check. It retries once on a failure that a retry could fix."""
        if not self.config.vision_available:
            raise NotConfigured("Set OPENROUTER_API_KEY and OPENROUTER_MODEL_ID to use vision checks.")

        wrapper = VisionAnswer[schema]  # type: ignore[valid-type]
        body = self._request_body(task, system, prompt, wrapper, images)
        last_error: Optional[LLMError] = None
        for attempt in (1, 2):
            self._check_limits()
            try:
                return self._attempt(task, body, wrapper)
            except (ServiceUnavailable, InvalidAnswer) as exc:
                last_error = exc
                logger.warning("Vision check %s failed on attempt %s: %s", task, attempt, exc)
                if attempt == 1:
                    self._sleep(RETRY_DELAY_SECONDS)
        assert last_error is not None
        raise last_error

    def _check_limits(self) -> None:
        if self.usage.calls >= self.config.max_calls:
            raise LimitReached(f"This run reached its limit of {self.config.max_calls} model calls.")
        if self.usage.cost_usd >= self.config.max_cost_usd:
            raise LimitReached(f"This run reached its spending limit of ${self.config.max_cost_usd:.2f}.")

    def _request_body(
        self,
        task: str,
        system: str,
        prompt: str,
        wrapper: Type[BaseModel],
        images: Sequence[Image],
    ) -> Dict[str, Any]:
        schema_text = json.dumps(wrapper.model_json_schema(), separators=(",", ":"))
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": encode_image(image)}} for image in images
        ]
        return {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "usage": {"include": True},
            "messages": [
                {
                    "role": "system",
                    "content": f'{system}\n\nSet "task" to "{task}". '
                    f"Return only JSON that matches this schema:\n{schema_text}",
                },
                {"role": "user", "content": content},
            ],
        }

    def _attempt(self, task: str, body: Dict[str, Any], wrapper: Type[VisionAnswer]) -> VisionAnswer:
        self.usage.calls += 1
        try:
            response = self._session.post(
                f"{self.config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "VAR Offside",
                },
                json=body,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            # The exception type alone: its text can repeat request details.
            raise ServiceUnavailable(f"Could not reach OpenRouter ({type(exc).__name__})") from None

        status = response.status_code
        if status in RETRYABLE_STATUS:
            raise ServiceUnavailable(f"OpenRouter returned HTTP {status}")
        if status in (401, 403):
            raise RejectedByService("OpenRouter rejected the API key. Check OPENROUTER_API_KEY.")
        if status == 402:
            raise RejectedByService("The OpenRouter account is out of credit.")
        if status >= 400:
            raise RejectedByService(
                f"OpenRouter refused the request (HTTP {status}). Check that "
                f"{self.config.model} exists and accepts images and JSON output."
            )

        try:
            payload = response.json()
        except ValueError:
            raise ServiceUnavailable("OpenRouter returned a reply that isn't JSON") from None
        self._record_cost(payload)

        text = _reply_text(payload)
        try:
            answer = wrapper.model_validate(_parse_json(text))
        except (ValueError, ValidationError) as exc:
            raise InvalidAnswer(f"The reply didn't match the {task} schema ({type(exc).__name__})") from None
        if answer.task != task:
            raise InvalidAnswer(f'The reply was for task "{answer.task}", not "{task}"')
        return answer

    def _record_cost(self, payload: Dict[str, Any]) -> None:
        cost = (payload.get("usage") or {}).get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            self.usage.cost_usd += float(cost)
        else:
            self.usage.unpriced_calls += 1
