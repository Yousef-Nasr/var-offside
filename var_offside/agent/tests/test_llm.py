"""OpenRouter client tests. A fake session stands in for the network, so no key is needed."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytest
import requests

from var_offside.agent.config import AgentConfig, load_config
from var_offside.agent.llm import (
    InvalidAnswer,
    LimitReached,
    NotConfigured,
    OpenRouterClient,
    RejectedByService,
    ServiceUnavailable,
    encode_image,
)
from var_offside.agent.schemas import FrameCheck

KEY = "test-secret-key"
GOOD = {
    "task": "frame",
    "status": "answered",
    "certainty": "clear",
    "answer": {"camera_type": "main_side", "usable": "yes", "problems": []},
    "evidence": ["pitch lines are visible"],
}


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, bad_json: bool = False) -> None:
        self.status_code = status_code
        self._payload = payload
        self._bad_json = bad_json

    def json(self) -> Any:
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


def reply(content: Any, cost: Optional[float] = 0.001) -> FakeResponse:
    text = content if isinstance(content, (str, list)) else json.dumps(content)
    usage = {} if cost is None else {"cost": cost}
    return FakeResponse(payload={"choices": [{"message": {"content": text}}], "usage": usage})


class FakeSession:
    """Plays back scripted responses and records each request."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(session: FakeSession, **config: Any) -> OpenRouterClient:
    settings = {"api_key": KEY, "model": "test/vision", **config}
    return OpenRouterClient(AgentConfig(**settings), session=session, sleep=lambda _: None)


def ask(client: OpenRouterClient, images: Any = ()) -> Any:
    return client.ask("frame", "system text", "prompt text", FrameCheck, images=images)


# ---------------------------------------------------------------------------
# Successful answers
# ---------------------------------------------------------------------------


def test_valid_reply_returns_a_typed_answer():
    answer = ask(make_client(FakeSession(reply(GOOD))))
    assert answer.status == "answered"
    assert answer.answer.camera_type == "main_side"
    assert answer.evidence == ["pitch lines are visible"]


def test_request_has_the_model_image_and_schema_and_no_key_in_the_body():
    session = FakeSession(reply(GOOD))
    ask(make_client(session), images=[np.zeros((20, 30, 3), dtype=np.uint8)])

    call = session.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert call["timeout"] > 0
    body = call["json"]
    assert body["model"] == "test/vision"
    assert body["response_format"] == {"type": "json_object"}
    assert body["usage"] == {"include": True}
    system, user = body["messages"]
    assert '"camera_type"' in system["content"]
    assert 'Set "task" to "frame"' in system["content"]
    assert user["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert KEY not in json.dumps(body)


def test_reply_in_a_code_fence_is_accepted():
    answer = ask(make_client(FakeSession(reply(f"```json\n{json.dumps(GOOD)}\n```"))))
    assert answer.answer.usable == "yes"


def test_reply_as_a_list_of_text_parts_is_accepted():
    answer = ask(make_client(FakeSession(reply([{"type": "text", "text": json.dumps(GOOD)}]))))
    assert answer.answer.usable == "yes"


def test_cannot_tell_with_no_answer_is_valid():
    payload = {"task": "frame", "status": "cannot_tell", "certainty": "unclear", "answer": None}
    answer = ask(make_client(FakeSession(reply(payload))))
    assert answer.status == "cannot_tell"
    assert answer.answer is None


# ---------------------------------------------------------------------------
# Bad answers: retried once, then an error, and never a guess
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("this is not json", id="not-json"),
        pytest.param({**GOOD, "verdict": "offside"}, id="verdict-in-wrapper"),
        pytest.param({**GOOD, "answer": {**GOOD["answer"], "margin_m": 0.4}}, id="margin-in-answer"),
        pytest.param({**GOOD, "answer": None}, id="answered-with-no-answer"),
        pytest.param({**GOOD, "task": "keeper"}, id="wrong-task"),
        pytest.param({**GOOD, "certainty": 0.99}, id="numeric-certainty"),
        pytest.param({**GOOD, "answer": {**GOOD["answer"], "usable": "maybe"}}, id="unknown-choice"),
    ],
)
def test_invalid_reply_is_retried_once_then_fails(bad):
    session = FakeSession(reply(bad), reply(bad))
    with pytest.raises(InvalidAnswer):
        ask(make_client(session))
    assert len(session.calls) == 2


def test_invalid_reply_recovers_on_the_retry():
    session = FakeSession(reply("not json"), reply(GOOD))
    assert ask(make_client(session)).answer.usable == "yes"
    assert len(session.calls) == 2


def test_empty_reply_is_invalid():
    with pytest.raises(InvalidAnswer):
        ask(make_client(FakeSession(reply("  "), reply("  "))))


# ---------------------------------------------------------------------------
# Service failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(FakeResponse(429), id="rate-limit"),
        pytest.param(FakeResponse(503), id="server-error"),
        pytest.param(requests.Timeout("timed out"), id="timeout"),
        pytest.param(requests.ConnectionError("no route"), id="network"),
        pytest.param(FakeResponse(bad_json=True), id="not-json-body"),
        pytest.param(FakeResponse(payload={"error": {"message": "overloaded"}}), id="error-in-200"),
        pytest.param(FakeResponse(payload={"choices": []}), id="no-choices"),
    ],
)
def test_temporary_failure_gets_one_retry(failure):
    session = FakeSession(failure, reply(GOOD))
    assert ask(make_client(session)).answer.usable == "yes"
    assert len(session.calls) == 2


def test_temporary_failure_twice_raises():
    session = FakeSession(FakeResponse(429), FakeResponse(429))
    with pytest.raises(ServiceUnavailable, match="429"):
        ask(make_client(session))


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "API key"), (403, "API key"), (402, "credit"), (400, "test/vision"), (404, "test/vision")],
)
def test_refused_request_is_not_retried_and_says_why(status, message):
    session = FakeSession(FakeResponse(status))
    with pytest.raises(RejectedByService, match=message) as caught:
        ask(make_client(session))
    assert len(session.calls) == 1
    assert KEY not in str(caught.value)


def test_network_error_text_is_not_repeated_in_the_message():
    leak = requests.ConnectionError(f"failed with header Bearer {KEY}")
    with pytest.raises(ServiceUnavailable) as caught:
        ask(make_client(FakeSession(leak, leak)))
    assert KEY not in str(caught.value)


def test_missing_key_or_model_makes_no_request():
    for config in ({"api_key": None}, {"model": None}):
        session = FakeSession()
        with pytest.raises(NotConfigured):
            ask(make_client(session, **config))
        assert session.calls == []


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_spending_limit_stops_the_next_call():
    session = FakeSession(reply(GOOD, cost=0.30), reply(GOOD, cost=0.30), reply(GOOD))
    client = make_client(session, max_cost_usd=0.50)
    ask(client)
    ask(client)
    assert client.usage.cost_usd == pytest.approx(0.60)
    with pytest.raises(LimitReached, match="spending"):
        ask(client)
    assert len(session.calls) == 2


def test_call_limit_also_covers_replies_with_no_price():
    session = FakeSession(reply(GOOD, cost=None), reply(GOOD, cost=None), reply(GOOD, cost=None))
    client = make_client(session, max_calls=2)
    ask(client)
    ask(client)
    assert client.usage.unpriced_calls == 2
    with pytest.raises(LimitReached, match="2 model calls"):
        ask(client)
    assert len(session.calls) == 2


def test_a_retry_counts_toward_the_limit():
    session = FakeSession(FakeResponse(429), reply(GOOD))
    client = make_client(session, max_calls=1)
    with pytest.raises(LimitReached):
        ask(client)
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# Images and settings
# ---------------------------------------------------------------------------


def _decode(data_url: str) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def test_large_image_is_shrunk_and_keeps_its_shape():
    decoded = _decode(encode_image(np.zeros((1080, 1920, 3), dtype=np.uint8)))
    assert decoded.shape[:2] == (720, 1280)


def test_small_image_keeps_its_size():
    decoded = _decode(encode_image(np.zeros((100, 200, 3), dtype=np.uint8)))
    assert decoded.shape[:2] == (100, 200)


def test_unreadable_image_raises(tmp_path):
    with pytest.raises(ValueError, match="Could not read"):
        encode_image(tmp_path / "missing.jpg")


def test_config_reads_the_file_and_the_environment_wins(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        'OPENROUTER_API_KEY="from-file"\nOPENROUTER_MODEL_ID=file/model\nOPENROUTER_MAX_CALLS=7\n',
        encoding="utf-8",
    )
    config = load_config(env, environ={"OPENROUTER_MODEL_ID": "env/model"})
    assert config.api_key == "from-file"
    assert config.model == "env/model"
    assert config.max_calls == 7
    assert config.vision_available


def test_config_with_no_file_or_key_means_no_vision(tmp_path):
    config = load_config(tmp_path / "missing.env", environ={})
    assert not config.vision_available


def test_key_is_not_in_the_config_repr():
    assert KEY not in repr(AgentConfig(api_key=KEY, model="test/vision"))
