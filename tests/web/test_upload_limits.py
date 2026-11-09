"""Upload validation: size limit (413) and image/video type check (415)."""

from __future__ import annotations

import pytest

from var_offside.config import get_settings
from var_offside.web import server

from tests.web.test_server import (  # noqa: F401 - client and isolated_state are fixtures
    _png_bytes,
    client,
    isolated_state,
)


@pytest.fixture
def tiny_limit(monkeypatch):
    """A 1 MB upload limit, so the tests stay small."""
    monkeypatch.setattr(get_settings().server, "max_upload_mb", 1, raising=False)


def test_a_normal_image_is_still_accepted(client):
    response = client.post("/api/sessions", files={"file": ("frame.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["kind"] == "image"


def test_a_file_over_the_limit_is_refused_with_413(client, tiny_limit):
    big = b"\0" * (2 * 1024 * 1024)
    response = client.post("/api/sessions", files={"file": ("clip.mp4", big, "video/mp4")})
    assert response.status_code == 413
    assert "1 MB upload limit" in response.json()["detail"]
    assert "server.max_upload_mb" in response.json()["detail"]


def test_the_streamed_size_check_stops_a_body_without_a_content_length(tiny_limit):
    class FakeUpload:
        def __init__(self, size: int) -> None:
            self.left = size

        async def read(self, n: int) -> bytes:
            n = min(n, self.left)
            self.left -= n
            return b"\0" * n

    import asyncio

    with pytest.raises(server.HTTPException) as caught:
        asyncio.run(server._read_upload(FakeUpload(3 * 1024 * 1024)))
    assert caught.value.status_code == 413
    assert asyncio.run(server._read_upload(FakeUpload(1000))) == b"\0" * 1000


@pytest.mark.parametrize("name", ["notes.txt", "model.pt", "page.html", "archive.zip"])
def test_other_file_types_are_refused_with_415(client, name):
    response = client.post("/api/sessions", files={"file": (name, b"hello", "application/octet-stream")})
    assert response.status_code == 415
    assert "Upload an image" in response.json()["detail"]


def test_an_image_name_with_unreadable_content_is_refused_with_415(client):
    response = client.post("/api/sessions", files={"file": ("frame.png", b"not really a png", "image/png")})
    assert response.status_code == 415


def test_a_file_without_an_extension_is_judged_by_its_content_type(client):
    ok = client.post("/api/sessions", files={"file": ("frame", _png_bytes(), "image/png")})
    assert ok.status_code == 200
    refused = client.post("/api/sessions", files={"file": ("frame", b"x", "text/plain")})
    assert refused.status_code == 415


def test_the_limit_comes_from_the_server_setting():
    assert get_settings().server.max_upload_mb == 500
