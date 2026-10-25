"""End-to-end pipeline test with a scripted cloud model.

Proves the whole path -- validate -> infer -> normalize -> JSON payload the UI
consumes -- without a network call or a real API key.
"""

from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient

from app import ai_client, main

from conftest import FakeAPIError, reply_json

client = TestClient(main.app)

REALISTIC_RESPONSE = reply_json(
    {
        "title": "Cache Invalidation Strategy",
        "detected_mode": "flowchart",
        "summary": "TTL combined with write-through cache invalidation to eliminate stale replica reads.",
        "notes": [
            {
                "heading": "Problem",
                "content": [
                    "Read replicas serve stale data after a write",
                    "Observed on the product listing page",
                ],
                "confidence": "high",
            },
            {
                "heading": "Options considered",
                "content": ["TTL only - simple but serves stale data", "Write-through invalidation"],
                "confidence": "medium",
            },
        ],
        "takeaways": [
            "TTL alone is not enough; pair it with invalidation on write",
            "The listing page is the only confirmed stale surface",
        ],
        "action_items": [
            {"task": "Add TTL to the cache layer", "owner": None, "deadline": None},
            {"task": "Prototype invalidation on write", "owner": None, "deadline": None},
        ],
        "uncertain_text": [
            {"text_or_location": "bottom-right annotation", "reason": "handwriting is smudged"}
        ],
        "diagram": {
            "type": "flowchart",
            "mermaid": "flowchart TD\n    W[Write] --> I[Invalidate cache]\n    I --> R[Replica catches up]",
        },
        "diagram_explanation": "A write triggers invalidation, then replicas catch up.",
        "suggestions": [
            "Benchmark replica replication delay under peak load"
        ],
    }
)


def _png_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (60, 40), "#336699").save(buf, format="PNG")
    return buf.getvalue()


class TestMetadataEndpoints:
    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["api_key"] == "present"
        assert data["model"] == ai_client.DEFAULT_MODEL
        assert data["endpoint"] == ai_client.endpoint_url()
        assert data["model_chain"] == [ai_client.DEFAULT_MODEL]

    def test_config_endpoint_reports_the_configured_model(self):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key_present"] is True
        assert data["model"] == ai_client.DEFAULT_MODEL
        assert "modes" in data
        assert "auto" in data["modes"]

    def test_health_is_503_and_actionable_without_a_key(self, monkeypatch):
        """The first thing a new user hits is a missing key, not a 500."""
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
            monkeypatch.delenv(name, raising=False)

        resp = client.get("/api/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "needs_key"
        assert data["api_key"] == "missing"
        # The model is still reported so the UI can show what it would use.
        assert data["model"] == ai_client.DEFAULT_MODEL

    def test_config_still_serves_metadata_without_a_key(self, monkeypatch):
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
            monkeypatch.delenv(name, raising=False)

        resp = client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json()["api_key_present"] is False

    def test_a_missing_key_is_502_not_an_unhandled_500(self, monkeypatch):
        """ConfigurationError must stay inside the InferenceError handler."""
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(ai_client, "_client", None, raising=False)
        monkeypatch.setattr(ai_client, "_client_error", None, raising=False)

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "auto"})
        assert resp.status_code == 502
        detail = resp.json()["detail"]
        assert "No Gemini API key" in detail
        assert "aistudio.google.com/apikey" in detail

    def test_samples_endpoint(self):
        resp = client.get("/api/samples")
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) >= 6
        modes = [s["mode"] for s in samples]
        for mode in ("notes", "flowchart", "architecture", "mindmap", "actions", "er"):
            assert mode in modes


class TestEndToEndPipeline:
    def test_analyze_end_to_end(self, client_env):
        genai = client_env(replies=[REALISTIC_RESPONSE])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "auto"})
        assert resp.status_code == 200
        payload = resp.json()

        assert payload["title"] == "Cache Invalidation Strategy"
        assert payload["detected_mode"] == "flowchart"
        assert payload["summary"].startswith("TTL combined")
        assert len(payload["notes"]) == 2
        assert len(payload["takeaways"]) == 2
        assert len(payload["action_items"]) == 2
        assert payload["diagram"]["type"] == "flowchart"
        assert "flowchart TD" in payload["diagram"]["mermaid"]
        assert len(payload["uncertain_text"]) == 1
        assert len(payload["suggestions"]) == 1
        assert payload["source"]["filename"] == "board.png"
        assert payload["telemetry"]["model"] == ai_client.DEFAULT_MODEL
        assert payload["telemetry"]["endpoint"] == ai_client.endpoint_url()
        assert payload["telemetry"]["elapsed_ms"] >= 0
        assert len(genai.calls) == 1

    def test_inference_error_becomes_502(self, client_env):
        client_env(error=FakeAPIError("Service unavailable", 503))

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "auto"})
        assert resp.status_code == 502
        assert "Service unavailable" in resp.json()["detail"]

    def test_bad_upload_is_rejected_before_inference(self, client_env):
        genai = client_env(replies=[REALISTIC_RESPONSE])

        files = {"image": ("board.png", b"not an image at all", "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "auto"})
        assert resp.status_code == 400
        assert genai.calls == []


class TestDiagramValidationIsServerSide:
    def test_invalid_mermaid_is_omitted_but_notes_survive(self, client_env):
        payload = {
            "title": "Bad Diagram",
            "notes": [{"heading": "H", "content": ["C"]}],
            "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    A[Unclosed --> B"},
        }
        client_env(replies=[json.dumps(payload), "flowchart TD\n    still ] broken"])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "flowchart"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["diagram"] is None
        assert len(data["notes"]) == 1
        assert data["telemetry"]["diagram_attempts"] == 2
        assert data["warnings"]


ER_RESPONSE = reply_json(
    {
        "title": "Employee records",
        "detected_mode": "er",
        "summary": "Employees work for companies and have a salary.",
        "notes": [
            {
                "heading": "Entities",
                "content": ["EMP holds id and name", "COMPANY holds id and name"],
                "confidence": "high",
            }
        ],
        "takeaways": ["An employee belongs to exactly one company."],
        "coverage": [
            {"label": "EMP", "kind": "entity", "detail": "id, name", "confidence": "high"},
            {"label": "COMPANY", "kind": "entity", "detail": "id, name", "confidence": "high"},
        ],
        "diagram": {
            "type": "erDiagram",
            # The attribute names have no data type: the exact shape that
            # used to break Mermaid in the browser.
            "mermaid": (
                "erDiagram\n"
                "    EMP {\n"
                "        id\n"
                "        name\n"
                "    }\n"
                "    COMPANY {\n"
                "        id\n"
                "        name\n"
                "    }\n"
                "    EMP ||--o{ COMPANY : \"works for\"\n"
            ),
        },
        "sql_schema": "CREATE TABLE emp (id INT PRIMARY KEY, name TEXT);",
        "deviations": [
            {
                "change": "Read the unclear cardinality on the EMP arrow as one-to-many",
                "why": "The drawing shows one company box and one employee box",
                "basis": "inferred",
            }
        ],
        "guide": {
            "overview": "An employee record system with two tables.",
            "steps": [{"title": "Start with EMP", "detail": "It is the box on the left."}],
            "concepts": [{"term": "PK", "meaning": "Primary key"}],
            "next_steps": ["Add a payroll table"],
        },
        "suggestions": ["Consider a salary table"],
    }
)


class TestErDiagramIsRepairedServerSide:
    def test_an_er_diagram_without_types_is_repaired(self, client_env):
        client_env(replies=[ER_RESPONSE])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "er"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["diagram"] is not None
        assert "string id" in data["diagram"]["mermaid"]
        assert "string name" in data["diagram"]["mermaid"]

    def test_the_repair_is_reported_as_a_deviation(self, client_env):
        client_env(replies=[ER_RESPONSE])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "er"})
        data = resp.json()

        changes = [d["change"] for d in data["deviations"]]
        assert any("data type" in change for change in changes)
        assert all(d["why"] for d in data["deviations"])

    def test_the_model_deviation_is_kept(self, client_env):
        client_env(replies=[ER_RESPONSE])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "er"})
        data = resp.json()

        changes = [d["change"] for d in data["deviations"]]
        assert any("cardinality" in change for change in changes)

    def test_the_coverage_inventory_is_kept(self, client_env):
        client_env(replies=[ER_RESPONSE])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "er"})
        data = resp.json()

        assert [c["label"] for c in data["coverage"]] == ["EMP", "COMPANY"]

    def test_the_guide_is_kept(self, client_env):
        client_env(replies=[ER_RESPONSE])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "er"})
        data = resp.json()

        assert data["guide"]["overview"].startswith("An employee")
        assert data["guide"]["steps"][0]["title"] == "Start with EMP"
        assert data["guide"]["concepts"][0]["term"] == "PK"

    def test_the_extraction_is_graded(self, client_env):
        client_env(replies=[ER_RESPONSE])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "er"})
        data = resp.json()

        assert 0 <= data["assessment"]["score"] <= 100
        assert data["assessment"]["criteria"]
        assert data["assessment"]["rationale"]


class TestMissingElementsAreCompleted:
    def test_a_diagram_missing_an_element_is_completed(self, client_env):
        payload = json.loads(ER_RESPONSE)
        payload["diagram"]["mermaid"] = (
            "erDiagram\n"
            "    EMP {\n"
            "        string id\n"
            "    }\n"
        )
        completed = (
            "erDiagram\n"
            "    EMP {\n"
            "        string id\n"
            "    }\n"
            "    COMPANY {\n"
            "        string id\n"
            "    }\n"
            "    EMP ||--o{ COMPANY : \"works for\"\n"
        )
        client_env(replies=[json.dumps(payload), completed])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "er"})
        data = resp.json()

        assert "COMPANY" in data["diagram"]["mermaid"]
        assert data["telemetry"].get("diagram_gap_fill") is True
        assert any("COMPANY" in d["change"] for d in data["deviations"])

    def test_a_completion_that_drops_elements_is_rejected(self, client_env):
        """A rewrite that loses what the user drew is worse than the gap."""
        payload = json.loads(ER_RESPONSE)
        payload["diagram"]["mermaid"] = (
            "erDiagram\n"
            "    EMP {\n"
            "        string id\n"
            "    }\n"
        )
        # The completion fills the gap but throws EMP away.
        worse = (
            "erDiagram\n"
            "    COMPANY {\n"
            "        string id\n"
            "    }\n"
        )
        client_env(replies=[json.dumps(payload), worse])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "er"})
        data = resp.json()

        assert "EMP" in data["diagram"]["mermaid"]
        assert data["telemetry"].get("diagram_gap_fill") is None
        assert data["telemetry"].get("diagram_gap_error")


class TestSettingsEndpoints:
    def test_settings_report_the_key_without_revealing_it(self, client_env):
        from app import settings

        settings.save_key("AIzaSyD-ExampleKeyForTests_1234567890abcd")
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key_present"] is True
        assert data["api_key_source"] == "store"
        assert "AIzaSyD-ExampleKeyForTests_1234567890abcd" not in resp.text

    def test_a_malformed_key_is_rejected_without_a_request(self):
        resp = client.post("/api/settings/api-key", json={"api_key": "nope"})
        assert resp.status_code == 400
        assert resp.json()["ok"] is False

    def test_a_key_that_fails_the_test_is_not_saved(self, client_env, monkeypatch):
        from app import ai_client, settings

        monkeypatch.setattr(
            ai_client,
            "test_api_key",
            lambda key, model=None: {
                "ok": False,
                "message": "The Gemini API rejected that key.",
                "model": "gemini-2.5-flash-lite",
                "elapsed_ms": 1,
            },
        )
        resp = client.post(
            "/api/settings/api-key",
            json={"api_key": "AIzaSyD-ExampleKeyForTests_1234567890abcd"},
        )
        assert resp.status_code == 502
        assert settings.stored_key() == ""

    def test_a_working_key_is_tested_and_saved(self, client_env, monkeypatch):
        from app import ai_client, settings

        monkeypatch.setattr(
            ai_client,
            "test_api_key",
            lambda key, model=None: {
                "ok": True,
                "message": "The key works.",
                "model": "gemini-2.5-flash-lite",
                "elapsed_ms": 1,
            },
        )
        resp = client.post(
            "/api/settings/api-key",
            json={"api_key": "AIzaSyD-ExampleKeyForTests_1234567890abcd"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["api_key_present"] is True
        assert "AIzaSyD-ExampleKeyForTests_1234567890abcd" not in resp.text
        assert settings.stored_key() == "AIzaSyD-ExampleKeyForTests_1234567890abcd"

    def test_the_key_can_be_removed(self, client_env):
        from app import settings

        settings.save_key("AIzaSyD-ExampleKeyForTests_1234567890abcd")
        resp = client.delete("/api/settings/api-key")
        assert resp.status_code == 200
        # The saved key is gone. The suite also sets a key in the
        # environment, so that one takes over from here.
        assert settings.stored_key() == ""
        assert resp.json()["api_key_source"] == "environment"

    def test_the_connection_test_does_not_save_the_key(self, client_env, monkeypatch):
        from app import ai_client, settings

        monkeypatch.setattr(
            ai_client,
            "test_api_key",
            lambda key, model=None: {
                "ok": True,
                "message": "The key works.",
                "model": "gemini-2.5-flash-lite",
                "elapsed_ms": 1,
            },
        )
        resp = client.post(
            "/api/settings/api-key/test",
            json={"api_key": "AIzaSyD-ExampleKeyForTests_1234567890abcd"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert settings.stored_key() == ""

    def test_a_broken_key_never_replaces_a_working_one(self, client_env, monkeypatch):
        """A failed test must leave the saved key alone."""
        from app import ai_client, settings

        settings.save_key("AIzaSyD-GoodKeyForTests_1234567890abcd")
        monkeypatch.setattr(
            ai_client,
            "test_api_key",
            lambda key, model=None: {
                "ok": False,
                "message": "The Gemini API rejected that key.",
                "model": "gemini-2.5-flash-lite",
                "elapsed_ms": 1,
            },
        )
        resp = client.post(
            "/api/settings/api-key",
            json={"api_key": "AIzaSyD-BrokenKeyForTests_1234567890abcd"},
        )
        assert resp.status_code == 502
        assert settings.stored_key() == "AIzaSyD-GoodKeyForTests_1234567890abcd"

    def test_a_key_from_another_site_is_refused(self, client_env):
        """The key endpoint must not accept a cross-origin post."""
        resp = client.post(
            "/api/settings/api-key",
            json={"api_key": "AIzaSyD-ExampleKeyForTests_1234567890abcd"},
            headers={"Origin": "https://evil.example"},
        )
        assert resp.status_code == 403
        assert resp.json()["ok"] is False

    def test_a_same_origin_post_is_accepted(self, client_env, monkeypatch):
        from app import ai_client

        monkeypatch.setattr(
            ai_client,
            "test_api_key",
            lambda key, model=None: {
                "ok": True,
                "message": "The key works.",
                "model": "gemini-2.5-flash-lite",
                "elapsed_ms": 1,
            },
        )
        resp = client.post(
            "/api/settings/api-key",
            json={"api_key": "AIzaSyD-ExampleKeyForTests_1234567890abcd"},
            headers={"Origin": "http://testserver"},
        )
        assert resp.status_code == 200

    def test_a_saved_key_is_used_for_the_next_analysis(self, client_env):
        """The key saved in Settings is the key the app then talks to."""
        from app import settings

        settings.save_key("AIzaSyD-ExampleKeyForTests_1234567890abcd")
        assert ai_client.api_key() == "AIzaSyD-ExampleKeyForTests_1234567890abcd"
