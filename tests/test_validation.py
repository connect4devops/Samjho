"""Validation tests. No network, no API key required."""

import io

import pytest

from app import validation


def _png_bytes(width=40, height=20):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), "#123456").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (30, 30), "#abcdef").save(buf, format="JPEG")
    return buf.getvalue()


def _webp_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (30, 30), "#abcdef").save(buf, format="WEBP")
    return buf.getvalue()


class TestUploadValidation:
    def test_valid_png_accepted(self):
        img = validation.validate_upload(_png_bytes(), "image/png", "board.png")
        assert img.mime == "image/png"
        assert img.extension == "png"
        assert img.width == 40 and img.height == 20

    def test_valid_jpeg_accepted(self):
        img = validation.validate_upload(_jpeg_bytes(), "image/jpeg", "note.jpg")
        assert img.mime == "image/jpeg"

    def test_valid_webp_accepted(self):
        img = validation.validate_upload(_webp_bytes(), "image/webp", "note.webp")
        assert img.mime == "image/webp"

    def test_unsupported_extension_rejected(self):
        with pytest.raises(validation.ValidationError, match="not a supported file type"):
            validation.validate_upload(_png_bytes(), "image/png", "notes.txt")

    def test_missing_extension_rejected(self):
        with pytest.raises(validation.ValidationError):
            validation.validate_upload(_png_bytes(), "image/png", "noextension")

    def test_empty_file_rejected(self):
        with pytest.raises(validation.ValidationError, match="empty"):
            validation.validate_upload(b"", "image/png", "a.png")

    def test_oversized_file_rejected(self, monkeypatch):
        monkeypatch.setenv("MAX_UPLOAD_MB", "10")
        big = b"\x89PNG\r\n\x1a\n" + b"0" * (10 * 1024 * 1024 + 10)
        with pytest.raises(validation.ValidationError, match="limit is"):
            validation.validate_upload(big, "image/png", "big.png")

    def test_non_image_bytes_rejected(self):
        with pytest.raises(validation.ValidationError, match="does not appear to be a real image"):
            validation.validate_upload(b"just some text here", "image/png", "fake.png")

    def test_mime_mismatch_rejected(self):
        with pytest.raises(validation.ValidationError, match="declared"):
            validation.validate_upload(_jpeg_bytes(), "image/png", "mismatch.jpg")

    def test_extension_mismatch_rejected(self):
        with pytest.raises(validation.ValidationError, match="Rename it to match"):
            validation.validate_upload(_jpeg_bytes(), "image/jpeg", "mismatch.png")


class TestNormalizeExtraction:
    def test_none_payload_returns_safe_extraction(self):
        ext = validation.normalize_extraction(None, requested_mode="flowchart")
        assert ext.title == "Extraction unavailable"
        assert ext.mode == "flowchart"
        assert len(ext.warnings) > 0

    def test_complete_payload_normalized(self):
        payload = {
            "title": "Architecture Overview",
            "detected_mode": "architecture",
            "summary": "This document describes the high-level cloud architecture.",
            "notes": [
                {
                    "heading": "API Tier",
                    "content": ["Handles TLS termination", "Applies rate limiting"],
                    "confidence": "high",
                }
            ],
            "takeaways": ["Microservices communicate via gRPC internally"],
            "action_items": [
                {"task": "Configure Redis cluster", "owner": "Marcus", "deadline": "Friday"},
                {"task": "Review PR", "owner": "null", "deadline": "unknown"},
            ],
            "uncertain_text": [
                {"text_or_location": "margin note", "reason": "smudged ink"}
            ],
            "sql_schema": "CREATE TABLE users (id INT PRIMARY KEY);",
            "suggestions": ["Consider adding circuit breakers"],
        }
        ext = validation.normalize_extraction(payload, requested_mode="auto")
        assert ext.title == "Architecture Overview"
        assert ext.detected_mode == "architecture"
        assert ext.summary == "This document describes the high-level cloud architecture."
        assert len(ext.notes) == 1
        assert ext.notes[0]["confidence"] == "high"
        assert len(ext.takeaways) == 1
        assert len(ext.action_items) == 2
        # Verify null strings collapse to None
        assert ext.action_items[0]["owner"] == "Marcus"
        assert ext.action_items[1]["owner"] is None
        assert ext.action_items[1]["deadline"] is None
        assert ext.sql_schema.startswith("CREATE TABLE")
        assert len(ext.suggestions) == 1
        assert len(ext.uncertain_text) == 1

    def test_coverage_labels_are_preserved_when_notes_are_missing(self):
        payload = {
            "title": "Untitled Extraction",
            "notes": [],
            "takeaways": [],
            "coverage": [
                {"label": "NP-hard", "kind": "node"},
                {"label": "NP-complete", "kind": "node", "detail": "inside NP-hard"},
                {"label": "P", "kind": "node"},
            ],
            "diagram": None,
        }

        ext = validation.normalize_extraction(payload, requested_mode="notes")

        assert ext.title == "Extracted Labels"
        assert ext.notes[0]["heading"] == "Visible labels"
        assert ext.notes[0]["content"] == [
            "NP-hard",
            "NP-complete: inside NP-hard",
            "P",
        ]
        assert not any("No readable structured text" in warning for warning in ext.warnings)

    def test_diagram_only_result_is_not_reported_as_unreadable(self):
        payload = {
            "title": "Complexity classes",
            "notes": [],
            "takeaways": [],
            "diagram": {
                "type": "flowchart",
                "mermaid": "flowchart TD\n    A[NP] --> B[P]",
            },
        }

        ext = validation.normalize_extraction(
            payload,
            requested_mode="notes",
            diagram_validated={
                "type": "flowchart",
                "mermaid": "flowchart TD\n    A[NP] --> B[P]",
            },
        )

        assert ext.diagram is not None
        assert not any("No readable structured text" in warning for warning in ext.warnings)


class TestSchemaEchoRejection:
    """A model that repeats the prompt's template has not read the image."""

    ECHOED = {
        "title": "Short descriptive title of the content",
        "detected_mode": "notes|flowchart|mindmap|architecture|actions|concept|er",
        "summary": "1-3 sentence summary of the source content",
        "notes": [
            {
                "heading": "Section heading",
                "content": ["Statement or bullet point 1", "Statement or bullet point 2"],
                "confidence": "high",
            }
        ],
        "takeaways": ["Key takeaway, concept, or confirmed decision"],
        "action_items": [{"task": "Explicit task statement", "owner": None, "deadline": None}],
        "uncertain_text": [
            {
                "text_or_location": "Fragment or region that is illegible or ambiguous",
                "reason": "Why it is uncertain (e.g. smudged, illegible handwriting, cut off)",
            }
        ],
        "diagram": {
            "type": "flowchart|mindmap|erDiagram|sequenceDiagram",
            "mermaid": "Raw Mermaid syntax (no ``` code fences)",
        },
        "diagram_explanation": "What relationships or flow the diagram shows",
        "suggestions": [
            "AI suggestion or recommended next step (clearly distinguished from source content)"
        ],
    }

    def test_echoed_fields_are_dropped(self):
        ext = validation.normalize_extraction(dict(self.ECHOED), requested_mode="auto")
        assert ext.title == "Untitled Extraction"
        assert ext.summary == ""
        assert ext.notes == []
        assert ext.takeaways == []
        assert ext.action_items == []
        assert ext.uncertain_text == []
        assert ext.suggestions == []
        assert ext.diagram is None
        assert ext.detected_mode is None

    def test_echoed_payload_is_reported_as_a_failed_read(self):
        ext = validation.normalize_extraction(dict(self.ECHOED), requested_mode="auto")
        assert any("JSON template" in w for w in ext.warnings)

    def test_real_content_alongside_an_echo_is_kept(self):
        payload = dict(self.ECHOED)
        payload["title"] = "OS Process State Model"
        payload["summary"] = "Five-state lifecycle for a scheduled process."
        payload["takeaways"] = ["A waiting process returns to the ready queue."]
        ext = validation.normalize_extraction(payload, requested_mode="flowchart")
        assert ext.title == "OS Process State Model"
        assert ext.summary.startswith("Five-state")
        assert ext.takeaways == ["A waiting process returns to the ready queue."]
        assert not any("JSON template" in w for w in ext.warnings)

    def test_detected_mode_must_be_a_real_mode(self):
        for bogus in ("notes|flowchart", "chart", "FLOWCHART!", "", "auto"):
            payload = {"title": "T", "detected_mode": bogus, "notes": [{"heading": "H", "content": ["C"]}]}
            ext = validation.normalize_extraction(payload, requested_mode="auto")
            assert ext.detected_mode is None, bogus

    def test_detected_mode_accepts_a_real_mode(self):
        payload = {"title": "T", "detected_mode": "Mindmap", "notes": []}
        ext = validation.normalize_extraction(payload, requested_mode="auto")
        assert ext.detected_mode == "mindmap"

    def test_requested_mode_is_the_fallback(self):
        payload = {"title": "T", "notes": []}
        ext = validation.normalize_extraction(payload, requested_mode="er")
        assert ext.detected_mode == "er"

    def test_is_schema_echo_helper(self):
        assert validation.is_schema_echo("1-3 sentence summary of the source content")
        assert validation.is_schema_echo("  Key takeaway, concept, or confirmed decision  ")
        assert validation.is_schema_echo("high|medium|low")
        assert not validation.is_schema_echo("New process enters the ready queue.")
        assert not validation.is_schema_echo("")

    def test_extract_json_object_from_fenced_and_prose(self):
        text = """Here is the result:
        ```json
        {
          "title": "Clean Result",
          "notes": [{"heading": "H", "content": ["Line 1"]}]
        }
        ```
        Hope this helps!"""
        obj = validation.extract_json_object(text)
        assert obj is not None
        assert obj["title"] == "Clean Result"
