"""Cloud inference-client tests for the Gemini API (Gemini or Gemma models).

No network and no API key are used: the transport is scripted (see conftest)
while request assembly, config, error translation, JSON repair, Mermaid
validation and telemetry all run for real.
"""

from __future__ import annotations

import io
import json
import pathlib

import pytest

from app import ai_client

from conftest import FakeAPIError, FakeResponse, part_texts, reply_json
import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROVIDER = ai_client.provider


def inline_parts(contents):
    return [p.inline_data for c in contents for p in (c.parts or []) if p.inline_data]


def text_parts(contents):
    return part_texts(contents)


class TestModelNaming:
    def test_default_model_is_a_known_model(self):
        assert ai_client.DEFAULT_MODEL in ai_client.KNOWN_MODELS

    def test_model_id_follows_env(self, monkeypatch):
        monkeypatch.setenv("GEMMA_MODEL", "gemma-3-12b-it")
        assert ai_client.model_id() == "gemma-3-12b-it"
        monkeypatch.delenv("GEMMA_MODEL")
        assert ai_client.model_id() == ai_client.DEFAULT_MODEL

    def test_model_id_accepts_the_google_standard_alias(self, monkeypatch):
        monkeypatch.delenv("GEMMA_MODEL")
        monkeypatch.setenv("GEMINI_MODEL", "gemma-3-4b-it")
        assert ai_client.model_id() == "gemma-3-4b-it"

    def test_both_families_are_configurable_through_one_variable(self, monkeypatch):
        """GEMMA_MODEL is historical naming; Gemini ids must work too."""
        for model in ("gemini-2.5-flash-lite", "gemini-2.5-flash", "gemma-3-27b-it"):
            monkeypatch.setenv("GEMMA_MODEL", model)
            assert ai_client.model_id() == model

    def test_model_chain_is_the_model_alone_by_default(self):
        assert ai_client.model_chain() == [ai_client.DEFAULT_MODEL]

    def test_model_chain_appends_fallbacks_without_duplicates(self, monkeypatch):
        monkeypatch.setenv("GEMMA_MODEL_FALLBACKS", "gemini-2.5-flash, gemma-3-4b-it")
        assert ai_client.model_chain() == [
            ai_client.DEFAULT_MODEL,
            "gemini-2.5-flash",
            "gemma-3-4b-it",
        ]

    def test_a_fallback_naming_the_primary_is_not_repeated(self, monkeypatch):
        monkeypatch.setenv("GEMMA_MODEL_FALLBACKS", ai_client.DEFAULT_MODEL)
        assert ai_client.model_chain() == [ai_client.DEFAULT_MODEL]

    def test_endpoint_is_the_gemini_api(self):
        assert ai_client.endpoint_url() == (
            "https://generativelanguage.googleapis.com/v1beta"
        )

    def test_endpoint_honours_overrides(self, monkeypatch):
        monkeypatch.setenv("GEMINI_BASE_URL", "https://proxy.example/")
        monkeypatch.setenv("GEMINI_API_VERSION", "v1alpha")
        assert ai_client.endpoint_url() == "https://proxy.example/v1alpha"

    def test_known_models_are_all_vision_capable(self):
        """The 1B Gemma 3 size is text-only and would fail on every upload."""
        assert "gemma-3-1b-it" not in ai_client.KNOWN_MODELS

    def test_known_models_span_both_families(self):
        assert any(m.startswith("gemini") for m in ai_client.KNOWN_MODELS)
        assert any(m.startswith("gemma") for m in ai_client.KNOWN_MODELS)

    def test_telemetry_reports_configured_model(self, image, client_env):
        client_env(replies=[reply_json({"title": "T", "notes": []})])
        _, telemetry = ai_client.analyze(image)
        assert telemetry["model"] == ai_client.DEFAULT_MODEL
        assert telemetry["model_used"] == ai_client.DEFAULT_MODEL
        assert telemetry["endpoint"] == ai_client.endpoint_url()
        assert telemetry["model_fallbacks"] == []

    def test_docs_and_env_example_match_default(self):
        """The shipped default must not drift from the documented one."""
        readme = io.open(ROOT / "readme.md", encoding="utf-8").read()
        env_example = io.open(ROOT / ".env.example", encoding="utf-8").read()
        assert (
            f"JARVISLABS_MODEL={ai_client.DEFAULT_JARVISLABS_MODEL}" in readme
        )
        assert (
            f"JARVISLABS_MODEL={ai_client.DEFAULT_JARVISLABS_MODEL}" in env_example
        )

    def test_env_example_documents_the_api_key(self):
        env_example = io.open(ROOT / ".env.example", encoding="utf-8").read()
        assert "JARVIS_API_TOKEN=" in env_example
        assert "GEMINI_API_KEY=" not in env_example
        assert "JARVISLABS_BASE_URL=" not in env_example
        assert "JARVISLABS_MODEL_FALLBACKS=" not in env_example

    def test_requirements_no_longer_pull_local_inference(self):
        reqs = io.open(ROOT / "requirements.txt", encoding="utf-8").read()
        assert "google-genai" in reqs
        for gone in ("torch", "transformers", "accelerate", "huggingface_hub"):
            assert gone not in reqs, f"{gone} is still a dependency"


class TestReadiness:
    def test_configuration_error_stays_inside_the_inference_handler(self):
        """app/main.py catches InferenceError; subclasses keep the 502 path."""
        assert issubclass(ai_client.ConfigurationError, ai_client.InferenceError)
        assert issubclass(ai_client.ModelUnavailable, ai_client.InferenceError)

    def test_key_present(self):
        assert ai_client.key_present() is True

    def test_not_ready_without_a_key(self, monkeypatch):
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert ai_client.key_present() is False

    def test_google_api_key_alias_is_accepted(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "abc")
        assert ai_client.api_key() == "abc"

    def test_missing_key_is_a_clean_error_not_a_crash(self, monkeypatch, image):
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(ai_client.InferenceError, match="No Gemini API key"):
            ai_client.analyze(image)

    def test_a_missing_key_is_not_memoized(self, monkeypatch, image):
        """Pasting a key into .env under a running server must just work.

        A first request made before the key existed must not poison the
        singleton, or the app stays dead until it is restarted. This drives the
        real _get_client; stubbing it out would prove nothing.
        """
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(ai_client, "_client", None, raising=False)
        monkeypatch.setattr(ai_client, "_client_error", None, raising=False)

        with pytest.raises(ai_client.ConfigurationError, match="No Gemini API key"):
            ai_client.analyze(image)
        assert ai_client._client_error is None

        # Building a client makes no network call, so this stays hermetic.
        monkeypatch.setenv("GEMINI_API_KEY", "added-later")
        assert ai_client._get_client() is not None

    def test_a_broken_client_build_is_memoized(self, monkeypatch, image):
        """An install that cannot work is remembered, not retried per request."""
        monkeypatch.setattr(ai_client, "_client", None, raising=False)
        monkeypatch.setattr(ai_client, "_client_error", None, raising=False)
        monkeypatch.setattr(
            ai_client,
            "_build_client",
            lambda: (_ for _ in ()).throw(ai_client.InferenceError("cannot build")),
        )
        for _ in range(2):
            with pytest.raises(ai_client.InferenceError, match="cannot build"):
                ai_client.analyze(image)
        assert ai_client._client_error is not None


class TestRequestAssembly:
    def test_image_is_sent_inline_with_its_real_mime(self, image, client_env):
        client = client_env(replies=[reply_json({"title": "T", "notes": []})])
        ai_client.analyze(image)

        sent = inline_parts(client.calls[0]["contents"])
        assert len(sent) == 1
        assert sent[0].mime_type == "image/png"
        assert sent[0].data == image.data

    def test_system_prompt_is_folded_into_the_user_turn(self, image, client_env):
        client = client_env(replies=[reply_json({"title": "ER", "notes": []})])
        ai_client.analyze(image, mode="er")

        texts = text_parts(client.calls[0]["contents"])
        assert len(texts) == 1
        # Both prompt blocks arrive in one user turn, because Gemma
        # rejects a system role.
        assert "Database ER Diagram" in texts[0]
        assert "Return ONLY a single valid JSON object" in texts[0]
        assert "---" in texts[0]

    def test_only_a_user_role_is_sent(self, image, client_env):
        client = client_env(replies=[reply_json({"title": "T", "notes": []})])
        ai_client.analyze(image)
        assert {c.role for c in client.calls[0]["contents"]} == {"user"}

    def test_json_mode_is_never_requested(self, image, client_env):
        """JSON mode is never requested; every Gemma id rejects it."""
        client = client_env(replies=[reply_json({"title": "T", "notes": []})])
        ai_client.analyze(image)

        config = client.calls[0]["config"]
        assert config.response_mime_type is None
        assert config.response_schema is None

    def test_system_instruction_is_never_sent(self, image, client_env):
        """Gemma was not trained with a system role and rejects the field."""
        client = client_env(replies=[reply_json({"title": "T", "notes": []})])
        ai_client.analyze(image)
        assert client.calls[0]["config"].system_instruction is None

    def test_sampler_knobs_are_omitted_by_default(self, image, client_env):
        """Omitted so one config works for both families: Gemma takes only the default sampler."""
        client = client_env(replies=[reply_json({"title": "T", "notes": []})])
        ai_client.analyze(image)

        config = client.calls[0]["config"]
        assert config.temperature is None
        assert config.top_p is None
        assert config.max_output_tokens == ai_client.MAX_OUTPUT_TOKENS

    def test_sampler_knobs_are_sent_when_explicitly_configured(
        self, image, client_env, monkeypatch
    ):
        monkeypatch.setenv("GEMMA_TEMPERATURE", "0.2")
        monkeypatch.setenv("GEMMA_TOP_P", "0.9")
        monkeypatch.setenv("GEMMA_MAX_OUTPUT_TOKENS", "512")
        client = client_env(replies=[reply_json({"title": "T", "notes": []})])
        ai_client.analyze(image)

        config = client.calls[0]["config"]
        assert config.temperature == 0.2
        assert config.top_p == 0.9
        assert config.max_output_tokens == 512

    def test_text_only_retry_sends_no_image(self, image, client_env):
        client = client_env(
            replies=[
                "not json",
                reply_json(
                    {"title": "T", "notes": [{"heading": "H", "content": ["C"]}]}
                ),
            ]
        )
        ai_client.analyze(image)

        retry = client.calls[1]["contents"]
        assert inline_parts(retry) == []
        assert "could not be parsed as JSON" in text_parts(retry)[0]


class TestErrorTranslation:
    def test_empty_response_is_rejected(self, image, client_env):
        client_env(replies=["   "])
        with pytest.raises(ai_client.InferenceError, match="empty response"):
            ai_client.analyze(image)

    def test_a_blocked_response_explains_the_safety_filter(self, image, client_env):
        client_env(reply_factory=lambda text: FakeResponse(text="", block_reason="SAFETY"))
        with pytest.raises(ai_client.InferenceError, match="safety filter"):
            ai_client.analyze(image)

    def test_an_early_stop_is_reported_as_such(self, image, client_env):
        client_env(reply_factory=lambda text: FakeResponse(text="", finish_reason="MAX_TOKENS"))
        with pytest.raises(ai_client.InferenceError, match="MAX_TOKENS"):
            ai_client.analyze(image)

    def test_rejected_key_is_a_clean_error(self, image, client_env):
        client_env(error=FakeAPIError("API key not valid. Please pass a valid API key.", 400))
        with pytest.raises(ai_client.InferenceError, match="rejected"):
            ai_client.analyze(image)

    def test_json_mode_rejection_is_surfaced_not_retried(self, image, client_env):
        """The failure mode the module docstring warns about."""
        client = client_env(
            error=FakeAPIError("JSON mode is not enabled for models/gemma-3-27b-it", 400),
            chain="gemma-3-12b-it",
        )
        with pytest.raises(ai_client.InferenceError, match="rejected this request"):
            ai_client.analyze(image)
        # A fatal request error must not burn quota walking the chain.
        assert len(client.calls) == 1

    def test_system_instruction_rejection_is_surfaced(self, image, client_env):
        client_env(
            error=FakeAPIError(
                "Developer instruction is not enabled for models/gemma-3-27b-it", 400
            )
        )
        with pytest.raises(ai_client.InferenceError, match="rejected this request"):
            ai_client.analyze(image)

    def test_saturation_retries_then_falls_back(
        self, image, client_env, no_real_sleeps
    ):
        primary = ai_client.DEFAULT_MODEL
        client = client_env(
            replies=[
                reply_json(
                    {"title": "Recovered", "notes": [{"heading": "H", "content": ["C"]}]}
                )
            ],
            error=FakeAPIError("Resource has been exhausted", 429),
            error_on=[1, 2, 3],
            chain="gemini-2.5-flash",
        )
        ext, telemetry = ai_client.analyze(image)

        assert ext.title == "Recovered"
        assert [c["model"] for c in client.calls] == [
            primary,
            primary,
            primary,
            "gemini-2.5-flash",
        ]
        assert telemetry["model_used"] == "gemini-2.5-flash"
        assert telemetry["model_fallbacks"] == [primary]
        assert len(no_real_sleeps) == 2

    def test_a_missing_model_is_skipped_without_retrying(self, image, client_env):
        primary = ai_client.DEFAULT_MODEL
        client = client_env(
            replies=[
                reply_json(
                    {"title": "Recovered", "notes": [{"heading": "H", "content": ["C"]}]}
                )
            ],
            error=FakeAPIError(f"models/{primary} is not found", 404),
            error_on=1,
            chain="gemini-2.5-flash",
        )
        _ext, telemetry = ai_client.analyze(image)
        assert telemetry["model_used"] == "gemini-2.5-flash"
        # A model the API does not serve must not burn the backoff budget.
        assert [c["model"] for c in client.calls] == [primary, "gemini-2.5-flash"]

    def test_an_exhausted_chain_raises_the_last_error(self, image, client_env):
        client = client_env(
            error=FakeAPIError("Service unavailable", 503), chain="gemini-2.5-flash"
        )
        with pytest.raises(ai_client.InferenceError, match="Service unavailable"):
            ai_client.analyze(image)
        assert {c["model"] for c in client.calls} == {
            ai_client.DEFAULT_MODEL,
            "gemini-2.5-flash",
        }

    def test_unparseable_output_degrades_gracefully(self, image, client_env):
        client_env(replies=["not json at all", "still not json"])
        ext, telemetry = ai_client.analyze(image)
        assert ext.notes == []
        assert any("could not parse" in w for w in ext.warnings)
        assert telemetry["raw_preview"]

    def test_a_retry_back_on_the_primary_clears_the_fallback_record(
        self, image, client_env
    ):
        """The model reported must be the one that produced the answer.

        The first call saturates onto the fallback; the content retry then
        succeeds on the primary. Stacking the skip lists would report the
        primary as both used and skipped.
        """
        primary = ai_client.DEFAULT_MODEL
        client = client_env(
            replies=[
                json.dumps(TestEmptyContentRetry.EMPTY),
                json.dumps(TestEmptyContentRetry.REAL),
            ],
            error=FakeAPIError("Resource has been exhausted", 429),
            error_on=[1, 2, 3],
            chain="gemini-2.5-flash",
        )
        ext, telemetry = ai_client.analyze(image, mode="flowchart")

        assert ext.title == "OS Process State Model"
        assert telemetry["model_used"] == primary
        assert telemetry["model_fallbacks"] == []
        assert [c["model"] for c in client.calls][:4] == [
            primary,
            primary,
            primary,
            "gemini-2.5-flash",
        ]

    def test_the_reported_model_is_the_one_that_served_the_reply(
        self, image, client_env
    ):
        """A first answer that falls back is reported as having done so."""
        client_env(
            replies=[
                reply_json(
                    {"title": "Recovered", "notes": [{"heading": "H", "content": ["C"]}]}
                )
            ],
            error=FakeAPIError("Resource has been exhausted", 429),
            error_on=[1, 2, 3],
            chain="gemini-2.5-flash",
        )
        _ext, telemetry = ai_client.analyze(image)
        assert telemetry["model_used"] == "gemini-2.5-flash"
        assert telemetry["model_fallbacks"] == [ai_client.DEFAULT_MODEL]


class TestEmptyContentRetry:
    """A reply that parses but carries nothing means the model never read
    the image; one image-aware retry is worth its latency."""

    EMPTY = {"title": "Untitled Extraction", "notes": [], "takeaways": [], "diagram": None}
    REAL = {
        "title": "OS Process State Model",
        "notes": [{"heading": "States", "content": ["New", "Ready"], "confidence": "high"}],
        "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    N[New] --> R[Ready]"},
    }

    def test_retry_with_image_recovers_content(self, image, client_env):
        client_env(replies=[json.dumps(self.EMPTY), json.dumps(self.REAL)])
        ext, telemetry = ai_client.analyze(image, mode="flowchart")

        assert telemetry["content_attempts"] == 1
        assert telemetry["recovered"] is True
        assert ext.title == "OS Process State Model"
        assert len(ext.notes) == 1
        assert ext.diagram is not None

    def test_retry_still_image_aware(self, image, client_env):
        client = client_env(replies=[json.dumps(self.EMPTY), json.dumps(self.REAL)])
        ai_client.analyze(image, mode="flowchart")

        assert len(client.calls) == 2
        # The image must be attached to the retry too, or it reads nothing.
        assert client.calls[1]["contents"][0].parts[0].inline_data is not None
        assert "no usable content" in text_parts(client.calls[1]["contents"])[0]

    def test_degrades_with_a_raw_preview_when_retry_fails(self, image, client_env):
        client_env(replies=[json.dumps(self.EMPTY), json.dumps(self.EMPTY)])
        ext, telemetry = ai_client.analyze(image)

        assert telemetry["degraded"] is True
        assert "recovered" not in telemetry
        assert telemetry["raw_preview"]
        assert ext.notes == []

    def test_no_retry_when_content_was_extracted(self, image, client_env):
        client = client_env(replies=[json.dumps(self.REAL)])
        ext, telemetry = ai_client.analyze(image)
        assert "content_attempts" not in telemetry
        assert len(client.calls) == 1
        assert ext.diagram is not None

    def test_retry_failure_is_not_fatal(self, image, client_env):
        client_env(
            replies=[json.dumps(self.EMPTY)],
            error=FakeAPIError("Resource has been exhausted", 429),
            error_on=2,
        )
        ext, telemetry = ai_client.analyze(image)
        assert telemetry["content_retry_error"]
        assert telemetry["degraded"] is True
        assert ext.notes == []


class TestCorrectiveRetries:
    def test_json_repair_on_malformed_initial_response(self, image, client_env):
        client = client_env(
            replies=[
                "Here is your output but it is not valid JSON {{",
                reply_json(
                    {"title": "Recovered", "notes": [{"heading": "H", "content": ["C"]}]}
                ),
            ]
        )
        ext, telemetry = ai_client.analyze(image)
        assert telemetry["attempts"] == 2
        assert ext.title == "Recovered"
        assert len(client.calls) == 2

    def test_diagram_repair_on_invalid_mermaid(self, image, client_env):
        bad = {
            "title": "Bad Diagram",
            "notes": [{"heading": "H", "content": ["C"]}],
            "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    A[Unclosed --> B"},
        }
        client = client_env(
            replies=[json.dumps(bad), "flowchart TD\n    A[Fixed bracket] --> B[Next]"]
        )
        ext, telemetry = ai_client.analyze(image)
        assert telemetry["diagram_attempts"] == 2
        assert ext.diagram is not None
        assert "Fixed bracket" in ext.diagram["mermaid"]
        assert len(client.calls) == 2

    def test_valid_diagram_needs_no_retry(self, image, client_env):
        good = {
            "title": "Good Diagram",
            "notes": [{"heading": "H", "content": ["C"]}],
            "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    A[Write] --> B[Read]"},
        }
        client = client_env(replies=[json.dumps(good)])
        ext, telemetry = ai_client.analyze(image)
        assert telemetry["diagram_attempts"] == 1
        assert len(client.calls) == 1
        assert ext.diagram["mermaid"].startswith("flowchart TD")

    def test_unrepairable_diagram_does_not_lose_notes(self, image, client_env):
        bad = {
            "title": "Broken Diagram",
            "notes": [{"heading": "H", "content": ["C"]}],
            "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    A[Unclosed --> B"},
        }
        client_env(replies=[json.dumps(bad), "flowchart TD\n    still ] broken"])
        ext, telemetry = ai_client.analyze(image)
        assert ext.diagram is None
        assert telemetry["diagram_attempts"] == 2
        assert telemetry["diagram_error"]
        assert len(ext.notes) == 1
        assert any("Mermaid validation" in w for w in ext.warnings)


class TestMissingDiagramRetry:
    """A mode whose deliverable is a diagram, answered without one."""

    NO_DIAGRAM = {
        "title": "Employee records",
        "detected_mode": "er",
        "notes": [{"heading": "Entities", "content": ["EMP", "COMPANY"]}],
        "coverage": [{"label": "EMP", "kind": "entity"}],
    }

    def test_a_missing_diagram_is_asked_for_again(self, image, client_env):
        diagram = (
            "erDiagram\n"
            "    EMP {\n"
            "        string id\n"
            "    }\n"
            "    COMPANY {\n"
            "        string id\n"
            "    }\n"
        )
        client_env(replies=[json.dumps(self.NO_DIAGRAM), diagram])

        ext, telemetry = ai_client.analyze(image, mode="er")

        assert ext.diagram is not None
        assert "EMP" in ext.diagram["mermaid"]
        assert telemetry["diagram_retry"] == "ok"

    def test_the_retry_is_only_made_when_a_diagram_is_owed(self, image, client_env):
        """Notes mode owes no diagram, so no request is wasted."""
        client_env(replies=[json.dumps({"title": "T", "notes": [{"heading": "H", "content": ["C"]}]})])
        _ext, telemetry = ai_client.analyze(image, mode="notes")
        assert "diagram_retry" not in telemetry

    def test_a_failed_retry_is_not_fatal(self, image, client_env):
        client_env(
            replies=[json.dumps(self.NO_DIAGRAM), "sorry, I could not do that"],
        )
        ext, telemetry = ai_client.analyze(image, mode="er")
        assert ext.diagram is None
        assert telemetry["diagram_retry_error"]
        assert len(ext.notes) == 1


class FakeJarvisClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers, json):
        self.calls.append({"url": url, "headers": headers, "json": json})
        result = self.responses.pop(0)
        return httpx.Response(
            result[0],
            json=result[1],
            request=httpx.Request("POST", url),
        )


class TestJarvisLabsProvider:
    def jarvis_env(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "jarvislabs")
        monkeypatch.setenv("JARVISLABS_API_KEY", "test-jarvis-key")
        monkeypatch.setenv(
            "JARVISLABS_BASE_URL",
            "https://serverlessn.jarvislabs.net/openai/test-deployment/v1",
        )
        monkeypatch.setenv("JARVISLABS_MODEL", "gemma-4-26b-a4b-it")

    def test_jarvislabs_is_the_runtime_default(self, monkeypatch):
        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.delenv("JARVISLABS_MODEL", raising=False)
        monkeypatch.setenv("JARVISLABS_API_KEY", "test-jarvis-key")
        monkeypatch.setenv("JARVISLABS_BASE_URL", "https://example.test/openai/id/v1")
        monkeypatch.setenv("JARVISLABS_MODEL_FALLBACKS", "other-model")
        assert ai_client.provider() == "jarvislabs"
        assert ai_client.model_id() == ai_client.DEFAULT_JARVISLABS_MODEL
        assert ai_client.endpoint_url() == ai_client.DEFAULT_JARVISLABS_BASE_URL
        assert ai_client.model_chain() == [ai_client.DEFAULT_JARVISLABS_MODEL]

    def test_provider_cannot_be_switched_through_environment(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        assert DEFAULT_PROVIDER() == "jarvislabs"

    def test_image_and_prompt_are_sent_as_openai_vision_content(
        self, image, monkeypatch
    ):
        self.jarvis_env(monkeypatch)
        response_body = reply_json(
            {
                "title": "Jarvis Test",
                "notes": [{"heading": "Diagram", "content": ["Alpha connects to Beta"]}],
            }
        )
        client = FakeJarvisClient(
            [(200, {"choices": [{"message": {"content": response_body}}]})]
        )
        monkeypatch.setattr(ai_client, "_get_client", lambda: client)

        extraction, telemetry = ai_client.analyze(image, mode="notes")

        sent = client.calls[0]
        assert sent["url"].endswith("/chat/completions")
        assert sent["headers"]["Authorization"] == "Bearer test-jarvis-key"
        assert sent["json"]["model"] == "gemma-4-26b-a4b-it"
        parts = sent["json"]["messages"][0]["content"]
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert parts[1]["type"] == "text"
        assert "Output" in parts[1]["text"]
        assert telemetry["provider_used"] == "jarvislabs"
        assert telemetry["model_used"] == "gemma-4-26b-a4b-it"
        assert extraction.title == "Jarvis Test"

    def test_transient_jarvislabs_failures_are_retried(
        self, image, monkeypatch, no_real_sleeps
    ):
        self.jarvis_env(monkeypatch)
        client = FakeJarvisClient(
            [
                (503, {"error": {"message": "No GPUs available"}}),
                (
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": reply_json(
                                        {
                                            "title": "T",
                                            "notes": [
                                                {
                                                    "heading": "Diagram",
                                                    "content": ["Alpha connects to Beta"],
                                                }
                                            ],
                                        }
                                    )
                                }
                            }
                        ]
                    },
                ),
            ]
        )
        monkeypatch.setattr(ai_client, "_get_client", lambda: client)

        extraction, _telemetry = ai_client.analyze(image, mode="notes")

        assert extraction.title == "T"
        assert len(client.calls) == 2
        assert no_real_sleeps == [1.0]

    def test_empty_jarvislabs_response_is_retried_once(
        self, image, monkeypatch, no_real_sleeps
    ):
        self.jarvis_env(monkeypatch)
        recovered = reply_json(
            {
                "title": "Recovered",
                "notes": [{"heading": "Diagram", "content": ["Alpha connects to Beta"]}],
            }
        )
        client = FakeJarvisClient(
            [
                (200, {"choices": [{"message": {"content": ""}}]}),
                (200, {"choices": [{"message": {"content": recovered}}]}),
            ]
        )
        monkeypatch.setattr(ai_client, "_get_client", lambda: client)

        extraction, _telemetry = ai_client.analyze(image, mode="notes")

        assert extraction.title == "Recovered"
        assert len(client.calls) == 2
        assert no_real_sleeps == [1.0]

    def test_bad_deployment_configuration_is_not_ready(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "jarvislabs")
        monkeypatch.setenv("JARVISLABS_API_KEY", "test-jarvis-key")
        monkeypatch.delenv("JARVISLABS_BASE_URL", raising=False)
        assert ai_client.key_present() is True
        assert ai_client.configured() is True

    def test_jarvis_api_token_variable_is_supported(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "jarvislabs")
        monkeypatch.delenv("JARVISLABS_API_KEY", raising=False)
        monkeypatch.setenv("JARVIS_API_TOKEN", "jarvis-api-token")
        assert ai_client.api_key() == "jarvis-api-token"

    def test_connection_test_uses_jarvislabs_and_the_submitted_key(self, monkeypatch):
        self.jarvis_env(monkeypatch)
        client = FakeJarvisClient(
            [(200, {"choices": [{"message": {"content": "pong"}}]})]
        )
        monkeypatch.setattr(ai_client, "_build_client", lambda key=None: client)

        result = ai_client.test_api_key("submitted-jarvis-key")

        assert result["ok"] is True
        assert client.calls[0]["json"]["max_tokens"] >= 32
        assert client.calls[0]["headers"]["Authorization"] == (
            "Bearer submitted-jarvis-key"
        )
