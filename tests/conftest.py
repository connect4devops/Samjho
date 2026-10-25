"""Shared test doubles for the cloud Gemini API inference client.

No network and no API key are used. What is faked is only the transport: a
Gemini client whose ``generate_content`` returns a scripted reply, or raises a
scripted error. The real SDK's ``types`` objects are still used to build and
inspect the request, so prompt assembly, image attachment, sampler config and
error translation all run for real.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Iterable

import pytest

from app import ai_client, validation


class FakeAPIError(Exception):
    """Stands in for google.genai.errors.APIError.

    The client reads ``code`` and ``message`` off the exception; that is the
    whole surface it depends on.
    """

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.message = message
        self.code = code


class FakeResponse:
    """The subset of GenerateContentResponse the client reads."""

    def __init__(
        self,
        text: str = "",
        finish_reason: str | None = None,
        block_reason: str | None = None,
    ):
        self.text = text
        self.candidates = (
            [SimpleNamespace(finish_reason=finish_reason)] if finish_reason else []
        )
        self.prompt_feedback = SimpleNamespace(block_reason=block_reason)


class _FakeModels:
    def __init__(self, client: "FakeGenAI"):
        self._client = client

    def generate_content(self, model, contents, config):
        client = self._client
        client.calls.append({"model": model, "contents": contents, "config": config})
        if client.error is not None and client._fails_on(len(client.calls)):
            raise client.error
        reply = client.replies.pop(0) if client.replies else ""
        return client.reply_factory(reply)


class FakeGenAI:
    """Minimal stand-in for google.genai.Client."""

    def __init__(self, replies=None, error=None, reply_factory=None):
        self.replies = list(replies or [])
        self.error = error
        # Which 1-based call indices raise. A single int is that one call; an
        # iterable is those calls, e.g. every attempt on a saturated model.
        self.error_on: int | Iterable[int] | None = None
        self.reply_factory = reply_factory or FakeResponse
        self.calls: list[dict] = []
        self.models = _FakeModels(self)

    def _fails_on(self, index: int) -> bool:
        if self.error_on is None:
            return True
        if isinstance(self.error_on, int):
            return self.error_on == index
        return index in self.error_on


@pytest.fixture(autouse=True)
def no_real_sleeps(monkeypatch):
    """Record backoff sleeps instead of performing them."""
    slept: list[float] = []
    monkeypatch.setattr(ai_client, "_SLEEP", slept.append)
    return slept


@pytest.fixture(autouse=True)
def isolated_key_store(monkeypatch, tmp_path):
    """Point the credential store at a temporary directory.

    The Settings screen writes a real key to disk. Without this, a key saved
    on the developer's machine would leak into the suite and change what
    ``api_key()`` returns.
    """
    monkeypatch.setenv("SCRIBBLE2STRUCTURE_HOME", str(tmp_path / "store"))
    return tmp_path / "store"


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    """A key is present unless a test removes it."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMMA_MODEL", ai_client.DEFAULT_MODEL)
    monkeypatch.setenv("GEMMA_MODEL_FALLBACKS", "")


@pytest.fixture
def image():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (30, 20), "#ffffff").save(buf, format="PNG")
    return validation.validate_upload(buf.getvalue(), "image/png", "board.png")


@pytest.fixture
def client_env(monkeypatch):
    """Install a scripted Gemini client.

    Returns a factory: ``client_env(replies=[...], error=..., chain="a,b")``.
    """

    def install(replies=None, error=None, error_on=None, chain="", reply_factory=None):
        monkeypatch.setattr(ai_client, "_client", None, raising=False)
        monkeypatch.setattr(ai_client, "_client_error", None, raising=False)
        if chain:
            monkeypatch.setenv("GEMMA_MODEL_FALLBACKS", chain)

        client = FakeGenAI(replies or [], error, reply_factory)
        client.error_on = error_on
        monkeypatch.setattr(ai_client, "_get_client", lambda: client)
        return client

    return install


def part_texts(contents) -> list[str]:
    """Every text part in a request, in order."""
    out: list[str] = []
    for content in contents:
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                out.append(part.text)
    return out


def reply_json(payload: dict) -> str:
    return json.dumps(payload)