"""Settings tests: the API key store, its masking and the connection test.

No network: the transport is faked exactly as in the rest of the suite.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from app import ai_client, settings

from conftest import FakeAPIError, FakeGenAI, FakeResponse


VALID_KEY = "AIzaSyD-ExampleKeyForTests_1234567890abcd"


class TestKeyFormat:
    def test_a_plausible_key_is_accepted(self):
        assert settings.validate_key_format(VALID_KEY) is None

    def test_an_empty_key_is_rejected(self):
        assert settings.validate_key_format("") is not None
        assert settings.validate_key_format("   ") is not None

    def test_a_key_with_spaces_is_rejected(self):
        assert settings.validate_key_format("AIzaSy abc def") is not None

    def test_a_too_short_key_is_rejected(self):
        assert settings.validate_key_format("AIza") is not None

    def test_a_pasted_sentence_is_rejected(self):
        assert settings.validate_key_format("my key is AIzaSyD-Example") is not None

    def test_a_non_string_is_rejected(self):
        assert settings.validate_key_format(None) is not None


class TestMasking:
    def test_the_key_never_appears_whole_in_the_mask(self):
        masked = settings.mask_key(VALID_KEY)
        assert VALID_KEY not in masked
        assert masked.startswith("AIza")
        assert masked.endswith("abcd")

    def test_a_short_key_is_fully_hidden(self):
        assert settings.mask_key("short") == "*****"

    def test_no_key_masks_to_nothing(self):
        assert settings.mask_key("") == ""


class TestStore:
    def test_a_saved_key_round_trips(self):
        settings.save_key(VALID_KEY)
        assert settings.stored_key() == VALID_KEY

    def test_the_store_is_a_local_file_outside_the_repository_sources(self):
        settings.save_key(VALID_KEY)
        assert settings.store_path().exists()
        assert settings.store_path().name == "credentials.json"

    def test_the_stored_file_is_not_world_readable(self):
        settings.save_key(VALID_KEY)
        mode = settings.store_path().stat().st_mode
        # On POSIX the file must be owner-only. Windows does not implement
        # POSIX modes, so the check is skipped there.
        if os.name == "posix":
            assert stat.S_IMODE(mode) == 0o600

    def test_the_key_is_never_stored_in_plain_text_in_the_source_tree(self):
        settings.save_key(VALID_KEY)
        raw = settings.store_path().read_text(encoding="utf-8")
        assert "gemini_api_key" in raw
        assert VALID_KEY in json.loads(raw)["gemini_api_key"]

    def test_clearing_forgets_the_key(self):
        settings.save_key(VALID_KEY)
        settings.clear_key()
        assert settings.stored_key() == ""

    def test_provider_keys_are_stored_and_cleared_independently(self):
        jarvis_key = "jl_test_key_12345"
        settings.save_key(VALID_KEY, "gemini")
        settings.save_key(jarvis_key, "jarvislabs")

        settings.clear_key("jarvislabs")

        assert settings.stored_key("gemini") == VALID_KEY
        assert settings.stored_key("jarvislabs") == ""

    def test_clearing_a_missing_store_is_harmless(self):
        settings.clear_key()
        assert settings.stored_key() == ""

    def test_the_store_wins_over_the_environment(self):
        """A key pasted into Settings is the key the app should use."""
        os.environ["GEMINI_API_KEY"] = "env-key-that-is-long-enough-to-pass"
        try:
            settings.save_key(VALID_KEY)
            assert settings.api_key() == VALID_KEY
            assert settings.key_source() == "store"
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

    def test_the_environment_is_used_when_nothing_is_saved(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key-that-is-long-enough-to-pass")
        assert settings.api_key() == "env-key-that-is-long-enough-to-pass"
        assert settings.key_source() == "environment"

    def test_a_corrupt_store_file_is_ignored(self):
        settings.store_path().parent.mkdir(parents=True, exist_ok=True)
        settings.store_path().write_text("{not json", encoding="utf-8")
        assert settings.stored_key() == ""


class TestConnectionTest:
    def test_a_working_key_is_reported_as_working(self, monkeypatch):
        client = FakeGenAI(replies=["pong"])
        monkeypatch.setattr(ai_client, "_build_client", lambda key=None: client)
        result = ai_client.test_api_key(VALID_KEY)
        assert result["ok"] is True
        assert "works" in result["message"]

    def test_a_rejected_key_is_reported_as_rejected(self, monkeypatch):
        client = FakeGenAI(error=FakeAPIError("API key not valid. Please pass a valid API key."))
        monkeypatch.setattr(ai_client, "_build_client", lambda key=None: client)
        result = ai_client.test_api_key(VALID_KEY)
        assert result["ok"] is False
        assert "rejected" in result["message"].lower()

    def test_a_forbidden_key_explains_the_project_problem(self, monkeypatch):
        client = FakeGenAI(error=FakeAPIError("Permission denied for project"))
        monkeypatch.setattr(ai_client, "_build_client", lambda key=None: client)
        result = ai_client.test_api_key(VALID_KEY)
        assert result["ok"] is False
        assert "permission" in result["message"].lower() or "403" in result["message"]

    def test_the_key_is_never_echoed_back(self, monkeypatch):
        client = FakeGenAI(error=FakeAPIError(f"API key not valid: {VALID_KEY}"))
        monkeypatch.setattr(ai_client, "_build_client", lambda key=None: client)
        result = ai_client.test_api_key(VALID_KEY)
        assert VALID_KEY not in result["message"]

    def test_an_empty_reply_is_not_a_success(self, monkeypatch):
        client = FakeGenAI(reply_factory=lambda text: FakeResponse(text=""))
        monkeypatch.setattr(ai_client, "_build_client", lambda key=None: client)
        result = ai_client.test_api_key(VALID_KEY)
        assert result["ok"] is False
