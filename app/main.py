"""Scribble2Structure -- from messy thinking to structured knowledge.

FastAPI app: validation, cloud Gemma inference via the Gemini API,
transformation mode routing, sample gallery endpoint, structured knowledge
responses, objective grading of the reconstruction, and the Settings screen
where the API key is entered and tested.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from urllib.parse import urlsplit

from . import ai_client, grading, prompts, settings, validation

app = FastAPI(
    title="Scribble2Structure",
    description="Turn handwritten notes, whiteboards, sketches, and diagrams into structured knowledge.",
)


def _analyze_timeout_s() -> int:
    try:
        return max(10, int(os.getenv("MAX_ANALYZE_SECONDS", "300")))
    except (TypeError, ValueError):
        return 300

SAMPLES = [
    {
        "id": "01_ml_notes",
        "title": "Machine Learning Notes",
        "description": "Handwritten gradient descent formulas, learning rate, and L2 regularization.",
        "mode": "notes",
        "filename": "01_ml_notes.png",
        "url": "/samples/01_ml_notes.png",
    },
    {
        "id": "02_os_process",
        "title": "OS Process State Model",
        "description": "Operating system process transitions: New, Ready, Running, Waiting, Terminated.",
        "mode": "flowchart",
        "filename": "02_os_process.png",
        "url": "/samples/02_os_process.png",
    },
    {
        "id": "03_software_architecture",
        "title": "Software Architecture Sketch",
        "description": "Cloud e-commerce microservices, API Gateway, DB replica, Redis, and Kafka.",
        "mode": "architecture",
        "filename": "03_software_architecture.png",
        "url": "/samples/03_software_architecture.png",
    },
    {
        "id": "04_student_productivity",
        "title": "Productivity Brainstorm",
        "description": "Mind map of study habits, task systems, health goals, and tech tools.",
        "mode": "mindmap",
        "filename": "04_student_productivity.png",
        "url": "/samples/04_student_productivity.png",
    },
    {
        "id": "05_meeting_notes",
        "title": "Meeting Sync & Action Items",
        "description": "Sprint planning notes with decisions, confirmed assignees, deadlines, and questions.",
        "mode": "actions",
        "filename": "05_meeting_notes.png",
        "url": "/samples/05_meeting_notes.png",
    },
    {
        "id": "06_database_er",
        "title": "Database ER Diagram",
        "description": "Relational entities (Customer, Order, Item, Product) with PK, FK, and cardinalities.",
        "mode": "er",
        "filename": "06_database_er.png",
        "url": "/samples/06_database_er.png",
    },
]


@app.get("/api/health")
def health() -> JSONResponse:
    ready = ai_client.key_present()
    return JSONResponse(
        {
            "status": "ok" if ready else "needs_key",
            "api_key": "present" if ready else "missing",
            "model": ai_client.model_id(),
            "model_chain": ai_client.model_chain(),
            "endpoint": ai_client.endpoint_url(),
            "modes": prompts.MODES,
            "supported_types": sorted(validation.ALLOWED_EXTENSIONS),
            "max_mb": validation.MAX_BYTES // (1024 * 1024),
        },
        status_code=200 if ready else 503,
    )


@app.get("/api/config")
def config() -> dict:
    """Model, capability, and transformation mode info for the UI."""
    return {
        "model": ai_client.model_id(),
        "api_key_present": ai_client.key_present(),
        "endpoint": ai_client.endpoint_url(),
        "max_mb": validation.MAX_BYTES // (1024 * 1024),
        "supported": sorted(validation.ALLOWED_EXTENSIONS),
        "modes": prompts.MODES,
    }


# ---------------------------------------------------------------------------
# Settings: the API key, entered and tested in the app itself
# ---------------------------------------------------------------------------

# Connection tests are cheap but not free, and they are the one endpoint
# that takes a secret from the browser. Both facts are handled here rather
# than in the client: at most a few tests a minute, and a key that is never
# echoed back.
_MAX_TESTS_PER_MINUTE = 6

_test_times: list[float] = []
_test_lock = threading.Lock()


class ApiKeyRequest(BaseModel):
    """The body of a key submission. The key is write-only."""

    api_key: str = Field(default="", max_length=400)
    model: str | None = Field(default=None, max_length=120)


def _same_origin(request: Request) -> bool:
    """True when the request came from this app's own page.

    The key endpoint is the one place a secret is accepted, so a page on
    another site must not be able to post to it. A request with no Origin
    header is a same-origin fetch or a command-line client, both fine.
    """
    origin = request.headers.get("origin")
    if not origin:
        return True
    host = request.headers.get("host", "")
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return bool(parsed.netloc) and parsed.netloc == host


def _reject_cross_origin(request: Request) -> JSONResponse | None:
    """A 403 response when the request did not come from this app."""
    if _same_origin(request):
        return None
    return JSONResponse(
        {
            "ok": False,
            "message": "This request came from another site, so it was refused.",
        },
        status_code=403,
    )


def _rate_limited() -> bool:
    """True when too many connection tests were made in the last minute."""
    now = time.monotonic()
    with _test_lock:
        _test_times[:] = [stamp for stamp in _test_times if now - stamp < 60.0]
        if len(_test_times) >= _MAX_TESTS_PER_MINUTE:
            return True
        _test_times.append(now)
    return False


def _settings_status() -> dict:
    """What the Settings screen needs to show. Never the key itself."""
    key = ai_client.api_key()
    return {
        "api_key_present": bool(key),
        "api_key_masked": settings.mask_key(key),
        "api_key_source": settings.key_source(),
        "model": ai_client.model_id(),
        "model_chain": ai_client.model_chain(),
        "endpoint": ai_client.endpoint_url(),
        "known_models": list(ai_client.KNOWN_MODELS),
        "store": settings.display_path(),
    }


@app.get("/api/settings")
def get_settings() -> dict:
    """The current settings, with the key masked."""
    return _settings_status()


@app.post("/api/settings/api-key/test")
def test_api_key(payload: ApiKeyRequest, request: Request) -> JSONResponse:
    """Check a key without saving it.

    The key is validated for shape first, so a pasted sentence costs no
    request, then used for a single minimal call. It is never stored and
    never returned.
    """
    refused = _reject_cross_origin(request)
    if refused is not None:
        return refused

    if _rate_limited():
        return JSONResponse(
            {
                "ok": False,
                "message": "Too many connection tests in a row. Wait a minute and try again.",
                "model": ai_client.model_id(),
            },
            status_code=429,
        )

    problem = settings.validate_key_format(payload.api_key)
    if problem:
        return JSONResponse(
            {
                "ok": False,
                "message": problem,
                "model": ai_client.model_id(),
            },
            status_code=400,
        )

    result = ai_client.test_api_key(payload.api_key.strip(), payload.model)
    return JSONResponse(result, status_code=200 if result.get("ok") else 502)


@app.post("/api/settings/api-key")
def save_api_key(payload: ApiKeyRequest, request: Request) -> JSONResponse:
    """Test a key and, if it works, save it for this machine.

    The key is stored in a local file that only this user can read and is
    never sent to the browser again -- the Settings screen only ever shows
    a masked form of it.
    """
    refused = _reject_cross_origin(request)
    if refused is not None:
        return refused

    if _rate_limited():
        return JSONResponse(
            {
                "ok": False,
                "message": "Too many attempts in a row. Wait a minute and try again.",
            },
            status_code=429,
        )

    problem = settings.validate_key_format(payload.api_key)
    if problem:
        return JSONResponse({"ok": False, "message": problem}, status_code=400)

    key = payload.api_key.strip()
    result = ai_client.test_api_key(key, payload.model)
    if not result.get("ok"):
        return JSONResponse(result, status_code=502)

    settings.save_key(key)
    # The cached client still holds the old key; drop it so the next request
    # uses the one that was just saved.
    ai_client.reset_client()
    status = _settings_status()
    status["ok"] = True
    status["message"] = result.get("message", "The key works and has been saved.")
    return JSONResponse(status)


@app.delete("/api/settings/api-key")
def delete_api_key(request: Request) -> JSONResponse:
    """Forget the saved key."""
    refused = _reject_cross_origin(request)
    if refused is not None:
        return refused

    settings.clear_key()
    ai_client.reset_client()
    status = _settings_status()
    status["ok"] = True
    return JSONResponse(status)


@app.get("/api/samples")
def get_samples() -> list[dict]:
    """Return available sample inputs for the sample gallery."""
    return SAMPLES


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    mode: str = Form("auto"),
) -> JSONResponse:
    raw = await image.read()

    # Pre-inference validation (format, size, dimensions, EXIF)
    try:
        checked = validation.validate_upload(raw, image.content_type, image.filename or "")
    except validation.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Inference is synchronous and can take minutes. Running it inline would
    # block the event loop, so the whole server stops answering -- including
    # /api/health -- for the duration. Hand it to a worker thread.
    budget = _analyze_timeout_s()
    try:
        extraction, telemetry = await asyncio.wait_for(
            run_in_threadpool(ai_client.analyze, checked, mode=mode),
            timeout=budget,
        )
    except asyncio.TimeoutError as exc:
        # The worker thread cannot be cancelled mid-request; it finishes on its
        # own. The client stops waiting, which is the part that matters.
        raise HTTPException(
            status_code=504,
            detail=(
                f"Analysis exceeded {budget}s and was abandoned. The upstream model was "
                "too slow or unreachable. Try a smaller model via GEMMA_MODEL, set "
                "GEMMA_MODEL_FALLBACKS so a saturated model is skipped sooner, or raise "
                "MAX_ANALYZE_SECONDS."
            ),
        ) from exc
    except ai_client.InferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    diagram_source = ""
    diagram_type = ""
    if extraction.diagram:
        diagram_source = extraction.diagram.get("mermaid", "")
        diagram_type = extraction.diagram.get("type", "")

    assessment = grading.assess(
        extraction,
        coverage=extraction.coverage,
        diagram_source=diagram_source,
        diagram_type=diagram_type,
        mode=extraction.detected_mode or extraction.mode or "auto",
    )

    return JSONResponse(
        {
            "title": extraction.title,
            "mode": extraction.mode,
            "detected_mode": extraction.detected_mode,
            "summary": extraction.summary,
            "notes": extraction.notes,
            "takeaways": extraction.takeaways,
            "action_items": extraction.action_items,
            "uncertain_text": extraction.uncertain_text,
            "diagram": extraction.diagram,
            "diagram_explanation": extraction.diagram_explanation,
            "sql_schema": extraction.sql_schema,
            "suggestions": extraction.suggestions,
            "warnings": extraction.warnings,
            "coverage": extraction.coverage,
            "deviations": extraction.deviations,
            "guide": extraction.guide,
            "assessment": assessment,
            "source": {
                "filename": checked.filename,
                "width": checked.width,
                "height": checked.height,
                "bytes": len(checked.data),
            },
            "telemetry": telemetry,
            "system_prompt": prompts.build_system_prompt(extraction.mode),
        }
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


# Mount static assets and sample images
os.makedirs("samples", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")
