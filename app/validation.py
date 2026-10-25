"""Input validation and model-response normalization.

Two jobs, kept separate because they fail for unrelated reasons:

* ``validate_upload`` runs before inference and decides whether we are willing
  to spend a request. Handles file size, extensions, content sniffing, integrity,
  and EXIF orientation transposition.
* ``normalize_extraction`` runs after inference and ensures the model's
  JSON is validated and safe to render.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

# Magic-byte prefixes. A declared Content-Type from a browser is a hint, not
# evidence, so we look at the actual bytes before trusting the extension.
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]

_WEBP_PREFIX = b"RIFF"
_WEBP_MARK = b"WEBP"

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def _get_max_bytes() -> int:
    try:
        return int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    except (ValueError, TypeError):
        return 10 * 1024 * 1024

MAX_BYTES = _get_max_bytes()
MAX_PIXELS = 40_000_000  # ~40 MP; guards decompression-bomb style inputs

_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class ValidationError(ValueError):
    """Raised when an upload is unusable. Message is safe to show a user."""


@dataclass
class ValidatedImage:
    data: bytes
    mime: str
    filename: str
    width: int | None = None
    height: int | None = None
    extension: str = ""


def sniff_mime(head: bytes) -> str | None:
    """Return the MIME type implied by the file's leading bytes, or None."""
    for sig, mime in _SIGNATURES:
        if head.startswith(sig):
            return mime
    if head.startswith(_WEBP_PREFIX) and _WEBP_MARK in head[:16]:
        return "image/webp"
    return None


def validate_upload(
    data: bytes, declared_mime: str | None, filename: str
) -> ValidatedImage:
    """Validate an upload on both declared type and real content.

    Applies EXIF orientation correction and dimension checks.
    Raises ValidationError with a user-facing message on any failure.
    """
    name = (filename or "").strip()
    if not name:
        raise ValidationError("No filename provided.")

    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"'{name}' is not a supported file type. Use PNG, JPG, JPEG, or WEBP."
        )

    if not data:
        raise ValidationError("The uploaded file is empty.")

    max_b = _get_max_bytes()
    if len(data) > max_b:
        raise ValidationError(
            f"Image is {len(data) // (1024 * 1024)} MB; the limit is "
            f"{max_b // (1024 * 1024)} MB."
        )

    actual = sniff_mime(data[:32])
    if actual is None:
        raise ValidationError(
            "That file does not appear to be a real image. The bytes do not match "
            "PNG, JPEG, GIF, or WEBP."
        )
    if actual == "image/gif":
        raise ValidationError("GIF is not supported. Please upload a PNG, JPG, or WEBP.")

    # The declared type and the extension must agree with the real bytes.
    if declared_mime and declared_mime.startswith("image/") and declared_mime != actual:
        raise ValidationError(
            f"File content is {actual} but the upload declared {declared_mime}. "
            "Please re-export the image and try again."
        )
    if _MIME_BY_EXT[ext] != actual:
        raise ValidationError(
            f"File is named '.{ext}' but its content is {actual}. Rename it to match "
            "or re-export it."
        )

    width = height = None
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(data)) as im:
            im.verify()

        with Image.open(io.BytesIO(data)) as im:
            # Correct EXIF orientation where applicable
            transposed = ImageOps.exif_transpose(im)
            if transposed is not None and transposed is not im:
                im = transposed
                buf = io.BytesIO()
                fmt = "PNG" if actual == "image/png" else ("WEBP" if actual == "image/webp" else "JPEG")
                im.save(buf, format=fmt)
                data = buf.getvalue()

            width, height = im.size

        if width * height > MAX_PIXELS:
            raise ValidationError(
                f"Image is {width}x{height} pixels, which exceeds the "
                f"{MAX_PIXELS // 1_000_000} MP limit."
            )
    except ValidationError:
        raise
    except Exception:
        # Truncated or otherwise corrupt files fail decode. Content sniffing
        # already passed, so this is a genuine corruption signal.
        raise ValidationError("That image file is corrupt or incomplete.")

    return ValidatedImage(
        data=data, mime=actual, filename=name, width=width, height=height, extension=ext
    )


class _BytesReader:
    """Minimal file-like wrapper so Pillow can open from memory."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            out = self._data[self._pos :]
            self._pos = len(self._data)
            return out
        out = self._data[self._pos : self._pos + n]
        self._pos += len(out)
        return out

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        else:
            self._pos = len(self._data) + pos
        return self._pos

    def tell(self) -> int:
        return self._pos


# --------------------------------------------------------------------------
# Model response normalization
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

VALID_CONFIDENCE = {"high", "medium", "low"}

VALID_MODES = {
    "auto",
    "notes",
    "flowchart",
    "mindmap",
    "architecture",
    "actions",
    "concept",
    "er",
}


def _schema_echoes() -> frozenset[str]:
    """Placeholder strings quoted inside the JSON schema we send the model.

    Small models frequently copy these example values straight back instead of
    reading the image, so anything matching one is discarded as a non-answer.
    Derived from the prompt so it can never drift out of sync with it.
    """
    from . import prompts

    return frozenset(
        match.strip().lower()
        for match in re.findall(r'"([^"\\]{4,})"', prompts._JSON_SCHEMA)
    )


_SCHEMA_ECHOES = _schema_echoes()


_ENUM_TOKENS = VALID_MODES | VALID_CONFIDENCE | {
    "flowchart",
    "mindmap",
    "erDiagram",
    "sequenceDiagram",
}


def is_schema_echo(value: Any) -> bool:
    """True when a string is the prompt's placeholder rather than real content."""
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    if not text:
        return False
    if text in _SCHEMA_ECHOES:
        return True
    # Enum fields get echoed back as the raw option list, e.g.
    # "notes|flowchart|mindmap|architecture|actions|concept|er".
    parts = [p.strip() for p in text.split("|")]
    return len(parts) > 1 and all(p in _ENUM_TOKENS for p in parts)


@dataclass
class Extraction:
    """A validated, renderable extraction. Every field is safe to display."""

    title: str
    mode: str = "auto"
    detected_mode: str | None = None
    summary: str = ""
    notes: list[dict[str, Any]] = field(default_factory=list)
    takeaways: list[str] = field(default_factory=list)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    uncertain_text: list[dict[str, str]] = field(default_factory=list)
    diagram: dict[str, str] | None = None
    diagram_explanation: str = ""
    sql_schema: str | None = None
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Everything the model found in the image, one entry per drawn element.
    coverage: list[dict[str, str]] = field(default_factory=list)
    # What the app or the model changed in the original drawing, and why.
    deviations: list[dict[str, str]] = field(default_factory=list)
    # The teaching layer: a plain-language walkthrough of the idea.
    guide: dict[str, Any] = field(default_factory=dict)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model response.

    Handles bare JSON, fenced JSON, and JSON surrounded by preamble prose.
    Returns None if no parseable object is found.
    """
    if not text:
        return None

    candidates: list[str] = []
    fenced = _FENCE.findall(text)
    candidates.extend(fenced)

    stripped = text.strip()
    candidates.append(stripped)

    # Scan for the first balanced {...} block, respecting strings and escapes.
    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start : i + 1])
                    start = -1
                    break
        if start == -1:
            break
        start = stripped.find("{", start + 1)

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            if not is_schema_echo(item):
                out.append(item.strip())
        elif isinstance(item, dict):
            for key in ("text", "content", "item", "takeaway", "suggestion"):
                if isinstance(item.get(key), str) and item[key].strip():
                    if not is_schema_echo(item[key]):
                        out.append(item[key].strip())
                    break
    return out


def _as_notes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str):
            if item.strip() and not is_schema_echo(item):
                out.append({"heading": "Notes", "content": [item.strip()], "confidence": "medium"})
            continue
        if not isinstance(item, dict):
            continue
        content = _as_str_list(item.get("content")) or _as_str_list(item.get("lines"))
        if not content:
            continue
        raw_heading = item.get("heading") or item.get("title") or "Notes"
        heading = "Notes" if is_schema_echo(raw_heading) else str(raw_heading).strip()[:200]
        if not heading:
            heading = "Notes"
        conf = str(item.get("confidence", "medium")).strip().lower()
        if conf not in VALID_CONFIDENCE:
            conf = "medium"
        out.append(
            {
                "heading": heading,
                "content": content,
                "confidence": conf,
            }
        )
    return out


def _as_action_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                out.append({"task": item.strip(), "owner": None, "deadline": None})
            continue
        if not isinstance(item, dict):
            continue
        task = item.get("task") or item.get("action") or item.get("text")
        if not isinstance(task, str) or not task.strip():
            continue
        if is_schema_echo(task):
            continue
        owner = item.get("owner")
        deadline = item.get("deadline")
        out.append(
            {
                "task": task.strip(),
                # Never let a model invent an owner or a date. Empty strings and
                # the literal "null"/"none" all collapse to None.
                "owner": (
                    owner.strip()
                    if isinstance(owner, str)
                    and owner.strip().lower() not in {"null", "none", "n/a", "unknown", "", "unassigned"}
                    else None
                ),
                "deadline": (
                    deadline.strip()
                    if isinstance(deadline, str)
                    and deadline.strip().lower() not in {"null", "none", "n/a", "unknown", "", "tbd"}
                    else None
                ),
            }
        )
    return out


def _as_uncertain(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip() and not is_schema_echo(item):
            out.append({"text_or_location": item.strip(), "reason": "Flagged as uncertain"})
            continue
        if not isinstance(item, dict):
            continue
        loc = item.get("text_or_location") or item.get("location") or item.get("text")
        reason = item.get("reason") or item.get("why") or "Illegible or ambiguous"
        if isinstance(loc, str) and loc.strip() and not is_schema_echo(loc):
            out.append(
                {
                    "text_or_location": loc.strip(),
                    "reason": "Illegible or ambiguous"
                    if is_schema_echo(reason)
                    else str(reason).strip(),
                }
            )
    return out


def _as_coverage(value: Any) -> list[dict[str, str]]:
    """The inventory of elements the model found in the image.

    Each entry is one drawn thing: a box, a table, a node, a heading. The
    app checks the diagram against this list, so a label that was echoed back
    from the template has to be dropped here rather than counted as coverage.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            label = item.strip()
            if not label or is_schema_echo(label):
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"label": label, "kind": "", "detail": "", "confidence": ""})
            continue
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("name") or item.get("text")
        if not isinstance(label, str) or not label.strip() or is_schema_echo(label):
            continue
        label = label.strip()
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)

        kind = item.get("kind") or item.get("type") or ""
        kind = "" if not isinstance(kind, str) or is_schema_echo(kind) else kind.strip()
        detail = item.get("detail") or item.get("description") or ""
        detail = "" if not isinstance(detail, str) or is_schema_echo(detail) else detail.strip()
        confidence = item.get("confidence") or ""
        confidence = (
            ""
            if not isinstance(confidence, str) or is_schema_echo(confidence)
            else confidence.strip().lower()
        )
        out.append(
            {
                "label": label,
                "kind": kind,
                "detail": detail,
                "confidence": confidence,
            }
        )
    return out


def _as_deviations(value: Any) -> list[dict[str, str]]:
    """Changes made to the original drawing, with the reason for each.

    This is the record of "what did the app do to my idea, and why". A
    deviation without a reason is not a deviation, it is an overwrite, so an
    entry without one is dropped.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if not text or is_schema_echo(text):
                continue
            out.append({"change": text, "why": "", "basis": ""})
            continue
        if not isinstance(item, dict):
            continue
        change = item.get("change") or item.get("what") or item.get("changed")
        if not isinstance(change, str) or not change.strip() or is_schema_echo(change):
            continue
        why = item.get("why") or item.get("reason") or item.get("because")
        why = "" if not isinstance(why, str) or is_schema_echo(why) else why.strip()
        basis = item.get("basis") or item.get("kind") or ""
        basis = "" if not isinstance(basis, str) or is_schema_echo(basis) else basis.strip()
        out.append({"change": change.strip(), "why": why, "basis": basis})
    return out


def _as_guide(value: Any) -> dict[str, Any]:
    """The teaching layer: a plain-language walkthrough of the idea."""
    if not isinstance(value, dict):
        return {}

    def _text(raw: Any) -> str:
        if not isinstance(raw, str) or not raw.strip() or is_schema_echo(raw):
            return ""
        return raw.strip()

    overview = _text(value.get("overview"))

    steps: list[dict[str, str]] = []
    raw_steps = value.get("steps")
    if isinstance(raw_steps, list):
        for item in raw_steps:
            if isinstance(item, str):
                title, detail = "", item.strip()
                if detail and not is_schema_echo(detail):
                    steps.append({"title": "", "detail": detail})
                continue
            if not isinstance(item, dict):
                continue
            title = _text(item.get("title") or item.get("name"))
            detail = _text(item.get("detail") or item.get("text") or item.get("content"))
            if not title and not detail:
                continue
            steps.append({"title": title, "detail": detail})

    concepts: list[dict[str, str]] = []
    raw_concepts = value.get("concepts")
    if isinstance(raw_concepts, list):
        for item in raw_concepts:
            if isinstance(item, str):
                meaning = item.strip()
                if meaning and not is_schema_echo(meaning):
                    concepts.append({"term": "", "meaning": meaning})
                continue
            if not isinstance(item, dict):
                continue
            term = _text(item.get("term") or item.get("name"))
            meaning = _text(item.get("meaning") or item.get("definition") or item.get("detail"))
            if not term and not meaning:
                continue
            concepts.append({"term": term, "meaning": meaning})

    next_steps = _as_str_list(value.get("next_steps"))

    if not (overview or steps or concepts or next_steps):
        return {}
    return {
        "overview": overview,
        "steps": steps,
        "concepts": concepts,
        "next_steps": next_steps,
    }


def _string_values(value: Any, depth: int = 0) -> list[str]:
    """Every string anywhere in the payload, used to spot a schema echo."""
    if depth > 6:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_string_values(item, depth + 1))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_string_values(item, depth + 1))
        return out
    return []


def normalize_extraction(
    payload: dict[str, Any] | None,
    *,
    requested_mode: str = "auto",
    diagram_validated: dict | None = None,
    extra_deviations: list[dict[str, str]] | None = None,
) -> Extraction:
    """Coerce arbitrary model JSON into a renderable Extraction.

    Never raises. Missing or wrong-typed fields degrade to empty rather than
    crashing, because notes and text must survive even when diagram fails.
    Placeholder values copied back out of the prompt are dropped, never shown.
    """
    warnings: list[str] = []
    if payload is None:
        return Extraction(
            title="Extraction unavailable",
            mode=requested_mode,
            warnings=["The model did not return a parseable JSON object."],
        )

    raw_title = payload.get("title")
    if not isinstance(raw_title, str) or not raw_title.strip() or is_schema_echo(raw_title):
        title = "Untitled Extraction"
    else:
        title = raw_title.strip()[:160]

    raw_detected = payload.get("detected_mode")
    detected_mode: str | None = None
    if isinstance(raw_detected, str):
        candidate = raw_detected.strip().lower()
        # Only a real mode name counts; "notes|flowchart|..." is an echoed enum.
        if candidate in VALID_MODES and candidate != "auto":
            detected_mode = candidate
    if detected_mode is None and requested_mode != "auto":
        detected_mode = requested_mode

    raw_summary = payload.get("summary")
    summary = (
        str(raw_summary).strip()
        if isinstance(raw_summary, str) and raw_summary.strip() and not is_schema_echo(raw_summary)
        else ""
    )

    notes = _as_notes(payload.get("notes"))
    takeaways = _as_str_list(payload.get("takeaways"))
    actions = _as_action_items(payload.get("action_items"))
    uncertain = _as_uncertain(payload.get("uncertain_text"))
    suggestions = _as_str_list(payload.get("suggestions"))
    coverage = _as_coverage(payload.get("coverage"))
    deviations = _as_deviations(payload.get("deviations"))
    guide = _as_guide(payload.get("guide"))

    raw_sql = payload.get("sql_schema")
    sql_schema = (
        str(raw_sql).strip()
        if isinstance(raw_sql, str) and raw_sql.strip() and not is_schema_echo(raw_sql)
        else None
    )

    raw_diagram = payload.get("diagram")
    diagram: dict[str, str] | None = None
    explanation = payload.get("diagram_explanation")
    explanation = (
        str(explanation).strip()
        if isinstance(explanation, str)
        and explanation.strip()
        and not is_schema_echo(explanation)
        else ""
    )

    if isinstance(raw_diagram, dict):
        validated = diagram_validated if diagram_validated is not None else None
        if validated and validated.get("mermaid"):
            diagram = {
                "type": validated.get("type", "flowchart"),
                "mermaid": validated["mermaid"],
            }
            if not explanation:
                explanation = validated.get("explanation", "")
        else:
            warnings.append(
                "A diagram was generated but did not pass Mermaid validation, so it "
                "was omitted. The structured notes and takeaways are unaffected."
            )
    elif isinstance(raw_diagram, str) and raw_diagram.strip():
        validated = diagram_validated if diagram_validated is not None else None
        if validated and validated.get("mermaid"):
            diagram = {"type": validated.get("type", "flowchart"), "mermaid": validated["mermaid"]}
        else:
            warnings.append(
                "Diagram source failed validation and was omitted; notes are unaffected."
            )

    if not notes and coverage:
        coverage_lines = [
            f"{item['label']}: {item['detail']}" if item["detail"] else item["label"]
            for item in coverage
        ]
        notes = [
            {
                "heading": "Visible labels",
                "content": coverage_lines,
                "confidence": "medium",
            }
        ]
        if title == "Untitled Extraction":
            title = "Extracted Labels"

    if not notes and not takeaways and not actions and not diagram and not coverage:
        warnings.append(
            "The model did not return readable content from this image. This does not "
            "necessarily mean the image is unclear. Try Auto Detect or another mode, or "
            "inspect the model response under “Why is this empty?”."
        )

    free_text = _string_values(payload)
    echoed = [v for v in free_text if is_schema_echo(v)]
    # "confidence": "high" and similar enum values are never real content.
    real_content = [
        v
        for v in free_text
        if v.strip()
        and not is_schema_echo(v)
        and v.strip().lower() not in VALID_CONFIDENCE
        and not v.lstrip().startswith(
            ("{", "flowchart", "graph", "mindmap", "erDiagram", "sequenceDiagram")
        )
    ]

    if echoed and not real_content and diagram is None:
        # Every free-text field came back as placeholder: the model described
        # the schema instead of reading the image. Say so rather than showing
        # empty panels as if the extraction were merely sparse.
        warnings.append(
            "The model returned the requested JSON template instead of reading the image, "
            "so nothing could be reconstructed. Try Auto Detect or another mode, and "
            "check that the JarvisLabs deployment is serving the configured vision model."
        )

    # Changes the app made to the drawing itself (a repaired ER attribute, a
    # renamed entity) are deviations too, and belong in the same record as
    # the model's own: what changed, and why.
    for extra in extra_deviations or []:
        change = (extra.get("change") or "").strip()
        if not change or any(d["change"] == change for d in deviations):
            continue
        deviations.append(
            {
                "change": change,
                "why": (extra.get("why") or "").strip(),
                "basis": (extra.get("basis") or "normalization").strip(),
            }
        )

    return Extraction(
        title=title,
        mode=requested_mode,
        detected_mode=detected_mode,
        summary=summary,
        notes=notes,
        takeaways=takeaways,
        action_items=actions,
        uncertain_text=uncertain,
        diagram=diagram,
        diagram_explanation=explanation,
        sql_schema=sql_schema,
        suggestions=suggestions,
        warnings=warnings,
        coverage=coverage,
        deviations=deviations,
        guide=guide,
    )
