"""Cloud inference client for Scribble2Structure -- Gemini API.

Handles:
  1. Build one Gemini API client from the API key (module-level singleton).
  2. Build a multimodal user turn from the uploaded image + mode prompt.
  3. Call generate_content() on the configured model and read the reply.
  4. Walk a model chain, moving to the next model when one is saturated.
  5. Parse the JSON object out of the reply (with at most 1 corrective retry).
  6. Validate the Mermaid diagram (with at most 1 corrective retry).
  7. Normalize everything into an Extraction.

The configured model defaults to ``gemini-2.5-flash-lite``. It may also be a
Gemma id (Gemma 3 4B/12B/27B, Gemma 4), which share the endpoint and accept
images, so the request is built to the *safest common denominator* of both
families rather than to the capabilities of whichever one is configured:

  * No JSON mode. ``response_mime_type``/``response_schema`` are rejected by
    every Gemma id ("JSON mode is not enabled for models/<id>"). The JSON
    schema therefore lives in the prompt (app/prompts.py) and the reply is
    parsed afterwards. Gemini could honour a response schema, but relying on it
    would make the app silently model-dependent, so the prompt-driven contract
    is kept for both.
  * No system role. Gemma was not trained with one and rejects
    ``system_instruction``, so the system prompt is folded into the single user
    turn. This is accepted by Gemini too.
  * No sampler knobs unless configured. Gemma is served with the default
    sampler, so ``temperature`` and ``top_p`` are omitted unless the operator
    sets them, rather than being sent with a value Gemini would honour and
    Gemma would reject.

That conservatism costs a little of what Gemini could offer, and buys one
codebase that keeps working if ``GEMMA_MODEL`` is pointed at a Gemma id.

Images are sent inline as base64 in the request; nothing is written to disk.
"""

from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Any

from dotenv import load_dotenv

from . import diagram as diagram_mod
from . import prompts, settings, validation

load_dotenv(override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Vision-capable ids served by the Gemini API, Gemini first. The env var is
# named GEMMA_MODEL for continuity, but any of these may be set there.
#
# Gemma 3 4B/12B/27B and Gemma 4 accept image input. The Gemma 3 1B size is
# text-only and is deliberately absent.
KNOWN_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-4-26b-a4b-it",
)

# Gemini API public endpoint and version.
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_API_VERSION = "v1beta"

# Enough room for a full Mermaid diagram plus the notes payload. Comfortably
# inside both ceilings: Gemma 3 caps generation at 8192 tokens and
# gemini-2.5-flash-lite at 65536.
MAX_OUTPUT_TOKENS = 8500

# How much of the model's own reply to keep for diagnosis when a run yields
# nothing. Enough to see what went wrong, short enough to return in JSON.
RAW_PREVIEW_CHARS = 1200

# Wall-clock ceiling for a single HTTP call to the Gemini API. The SDK's own
# default leaves a stalled connection waiting indefinitely, which presents to
# the user as a request that never returns. Always bounded.
def _call_timeout_ms() -> int:
    """Per-request HTTP timeout in milliseconds. Never zero."""
    return max(5_000, _int_env("GEMINI_TIMEOUT_MS", 120_000))

# Backoff for a saturated or erroring endpoint, before moving to the next model
# in the chain. Replaced by a no-op in tests so the suite never really sleeps.
_SLEEP = time.sleep
RETRY_DELAYS = (1.0, 3.0)

# Errors that mean the request itself is wrong. Walking the model chain cannot
# fix these, and retrying only burns quota, so they fail immediately.
_FATAL_MESSAGE_HINTS = (
    "json mode is not enabled",
    "developer instruction is not enabled",
    "function calling is not enabled",
    "api key not valid",
    "permission denied",
)


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _int_env(name: str, fallback: int) -> int:
    try:
        return int(_env(name))
    except (ValueError, TypeError):
        return fallback


def _float_env(name: str) -> float | None:
    raw = _env(name)
    if not raw:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def api_key() -> str:
    """The Gemini API key.

    The key saved in the Settings screen wins; otherwise the documented
    environment variable is used, with the two aliases accepted because that
    is what the Google tooling and most .env templates write.
    """
    stored = settings.stored_key()
    if stored:
        return stored
    return _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY") or _env("GEMINI_KEY")


def reset_client() -> None:
    """Drop the cached client so the next call rebuilds it.

    Called after the key is changed in Settings: a client built with the old
    key would keep sending the old key, and the new one would look broken.
    """
    global _client, _client_error
    with _client_lock:
        _client = None
        _client_error = None


def model_id() -> str:
    """The configured model, falling back to the shipped default."""
    return _env("GEMMA_MODEL") or _env("GEMINI_MODEL") or DEFAULT_MODEL


def model_chain() -> list[str]:
    """The configured model followed by any opt-in fallbacks, deduplicated.

    GEMMA_MODEL_FALLBACKS is a comma-separated list of ids tried in order when
    the primary is saturated, rate-limited, or missing. It is empty by default
    so a failure is reported rather than silently answered by a smaller model.
    """
    chain: list[str] = []
    for candidate in [model_id(), *_env("GEMMA_MODEL_FALLBACKS").split(",")]:
        name = candidate.strip()
        if name and name not in chain:
            chain.append(name)
    return chain


def endpoint_url() -> str:
    """The API base URL this client talks to, for /api/health and /api/config."""
    base = _env("GEMINI_BASE_URL") or DEFAULT_BASE_URL
    version = _env("GEMINI_API_VERSION") or DEFAULT_API_VERSION
    return f"{base.rstrip('/')}/{version.strip('/')}"


def key_present() -> bool:
    """True when inference can run: a Gemini API key is set."""
    return bool(api_key())


def max_output_tokens() -> int:
    return _int_env("GEMMA_MAX_OUTPUT_TOKENS", MAX_OUTPUT_TOKENS)


# ---------------------------------------------------------------------------
# Public error types
# ---------------------------------------------------------------------------


class InferenceError(RuntimeError):
    """A user-facing inference failure."""


class ModelUnavailable(InferenceError):
    """A model-level failure that may not affect the next model in the chain.

    ``retryable`` distinguishes the two cases that look alike from here:
    a saturated endpoint is worth another try after a backoff, while a model
    the API does not serve will fail identically forever.

    ``max_attempts`` caps how many times this specific failure is retried
    against the same model. A timeout that has already consumed its full HTTP
    budget should be tried once more, not three times: three 120s timeouts are
    six minutes of apparent hang before the chain moves on.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        max_attempts: int | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.max_attempts = max_attempts


class ConfigurationError(InferenceError):
    """A setup problem the operator can fix without restarting the process.

    Kept distinct from other failures because it must never be memoized: a key
    pasted into .env under a running --reload server should start working
    immediately, not be poisoned by the first request that lacked it.
    """


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Any = None
_client_lock = Lock()
_client_error: str | None = None


def _build_client(key: str | None = None):
    """Construct a Gemini API client bound to the configured key and endpoint."""
    key = key if key is not None else api_key()
    if not key:
        raise ConfigurationError(
            "No Gemini API key is set. Open Settings in the app and paste your key, "
            "or add GEMINI_API_KEY to your .env file "
            "(create one at https://aistudio.google.com/apikey)."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise InferenceError(
            "The google-genai package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        options: dict[str, Any] = {}
        base = _env("GEMINI_BASE_URL")
        version = _env("GEMINI_API_VERSION")
        if base:
            options["base_url"] = base
        if version:
            options["api_version"] = version
        timeout_ms = _call_timeout_ms()
        options["timeout"] = timeout_ms
        return genai.Client(
            api_key=key,
            http_options=types.HttpOptions(**options) if options else None,
        )
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(
            f"Could not initialize the Gemini API client: {str(exc)[:300]}"
        ) from exc


def _get_client():
    """Return the shared client, creating it on first call (thread-safe).

    A broken SDK install or an unusable endpoint is remembered for the process
    lifetime: neither fixes itself between requests, and re-attempting on every
    upload would add latency to every failure. A missing key is NOT remembered
    (see ConfigurationError).
    """
    global _client, _client_error

    if _client_error is not None:
        raise InferenceError(f"Gemini API client failed to initialize: {_client_error}")
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client
        if _client_error is not None:
            raise InferenceError(
                f"Gemini API client failed to initialize: {_client_error}"
            )
        try:
            _client = _build_client()
        except ConfigurationError:
            # Recoverable in place: never poison the singleton over a missing key.
            raise
        except InferenceError as exc:
            _client_error = str(exc)[:400]
            raise
        logger.info("Gemini API client ready. Endpoint: %s", endpoint_url())
        return _client


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status carried by an SDK error, if it has one."""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _message_of(exc: BaseException) -> str:
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message
    return str(exc)


def _looks_like_timeout(exc: BaseException) -> bool:
    """True for a stalled/expired request rather than a server-side error."""
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "deadline" in name:
        return True
    lowered = _message_of(exc).lower()
    return "timed out" in lowered or "timeout" in lowered


def _classify(exc: BaseException) -> InferenceError:
    """Translate an SDK error into a user-facing error.

    Model-level failures (saturation, a missing model, a 5xx) become
    ModelUnavailable so the chain moves on; request-level failures stay fatal
    because no other model would accept the request either.
    """
    status = _status_of(exc)
    message = _message_of(exc)
    lowered = message.lower()

    if any(hint in lowered for hint in _FATAL_MESSAGE_HINTS):
        return InferenceError(
            "The Gemini API rejected this request: " f"{message[:300]}"
        )
    if status in (401, 403):
        return InferenceError(
            "The Gemini API key was rejected (HTTP {0}). Check GEMINI_API_KEY in your "
            ".env file -- see https://aistudio.google.com/apikey".format(status)
        )
    if status == 404:
        # The endpoint does not serve this id. Retrying it cannot help.
        return ModelUnavailable(
            f"Model not available on the Gemini API: {message[:200]}", retryable=False
        )
    if _looks_like_timeout(exc):
        # Already consumed a full HTTP budget. One more try, then move on.
        return ModelUnavailable(
            f"The Gemini API did not respond within {_call_timeout_ms() // 1000}s.",
            retryable=True,
            max_attempts=2,
        )
    # Saturation and 5xx: the same model may well succeed shortly.
    return ModelUnavailable(message[:300], retryable=True)


def _call_once(model: str, contents: list[Any], config: Any) -> str:
    """One generate_content() call, returning the reply text."""
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
    except InferenceError:
        raise
    except Exception as exc:
        raise _classify(exc) from exc
    return _response_text(response, model)


def _response_text(response: Any, model: str) -> str:
    """Pull the reply text out of a response, explaining an empty one.

    An empty reply is almost always a safety filter or a finish reason rather
    than a bug, so say which instead of failing with a decoding error.
    """
    text = ""
    try:
        text = response.text or ""
    except (AttributeError, ValueError):
        text = ""

    if text.strip():
        return text.strip()

    reason = "the model returned an empty response"
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = getattr(candidates[0], "finish_reason", None)
            if finish:
                reason = f"the model stopped early ({finish})"
        feedback = getattr(response, "prompt_feedback", None)
        block = getattr(feedback, "block_reason", None)
        if block:
            reason = f"the request was blocked by a safety filter ({block})"
    except Exception:
        pass

    raise InferenceError(f"{reason} while using '{model}'.")


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------


def _types():
    """The SDK's schema module, or a user-facing error when it is missing."""
    try:
        from google.genai import types
    except ImportError as exc:
        raise InferenceError(
            "The google-genai package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return types


def _build_config():
    """Generation config restricted to what every supported model accepts.

    Sampler knobs are omitted unless explicitly configured, and JSON mode and
    system instructions are never requested. See the module docstring: the
    default model is Gemini, but a Gemma id must keep working from the same
    codebase, so the config stays at their intersection.
    """
    types = _types()

    kwargs: dict[str, Any] = {"max_output_tokens": max_output_tokens()}

    temperature = _float_env("GEMMA_TEMPERATURE")
    if temperature is not None:
        kwargs["temperature"] = temperature
    top_p = _float_env("GEMMA_TOP_P")
    if top_p is not None:
        kwargs["top_p"] = top_p

    # response_mime_type / response_schema are intentionally NOT set: every
    # Gemma id rejects JSON mode on this endpoint. See the module docstring.
    return types.GenerateContentConfig(**kwargs)


def _fuse(system_prompt: str, user_prompt: str) -> str:
    """Join the system prompt and the request into one user turn.

    Gemma rejects a system role outright, so the instruction block and the
    per-request text are concatenated with a rule between them. Gemini accepts
    the same shape, which keeps one request format for both families.
    """
    return f"{system_prompt.strip()}\n\n---\n\n{user_prompt.strip()}"


def _build_image_contents(image: validation.ValidatedImage, system_prompt: str, user_prompt: str):
    """A single user turn holding the image bytes and the fused prompt."""
    types = _types()

    part_image = types.Part.from_bytes(data=image.data, mime_type=image.mime)
    part_text = types.Part.from_text(text=_fuse(system_prompt, user_prompt))
    return [types.Content(role="user", parts=[part_image, part_text])]


def _build_text_contents(prompt: str):
    """A single user turn holding only text, for corrective retries."""
    types = _types()

    return [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------


def _call_chain(contents: list[Any], config: Any) -> tuple[str, str, list[str]]:
    """Run one generation, walking the model chain on model-level failures.

    Returns (text, model_used, fallbacks_skipped).
    """
    chain = model_chain()
    last: InferenceError | None = None

    for index, model in enumerate(chain):
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                return _call_once(model, contents, config), model, chain[:index]
            except ModelUnavailable as exc:
                last = exc
                # A saturated endpoint is worth a short backoff. A model the
                # API does not serve is not: hand off at once. A failure may
                # also cap its own attempts (a timeout has already spent its
                # budget, so retrying it three times just multiplies the wait).
                ceiling = (
                    exc.max_attempts
                    if exc.max_attempts is not None
                    else len(RETRY_DELAYS) + 1
                )
                if not exc.retryable or attempt + 1 >= ceiling:
                    break
                _SLEEP(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
            except InferenceError:
                raise

    raise last or InferenceError("No model in the chain is available.")


def _generate(
    image: validation.ValidatedImage, system_prompt: str, user_prompt: str
) -> tuple[str, str, list[str]]:
    """Run one multimodal generation round. Returns (text, model, fallbacks)."""
    return _call_chain(_build_image_contents(image, system_prompt, user_prompt), _build_config())


def _generate_text_only(prompt: str) -> tuple[str, str, list[str]]:
    """Text-only generation for corrective retries (no image)."""
    return _call_chain(_build_text_contents(prompt), _build_config())


# ---------------------------------------------------------------------------
# Diagram repair
# ---------------------------------------------------------------------------


def _repair_diagram(
    raw_source: str, error: str, expected_type: str = "flowchart"
) -> tuple[str | None, str | None]:
    try:
        repair_content = diagram_mod.build_repair_prompt(raw_source, error, expected_type=expected_type)
        fixed, _model, _fallbacks = _generate_text_only(repair_content)
    except InferenceError as exc:
        return None, f"Diagram repair could not run: {exc}"

    try:
        validated = diagram_mod.validate_mermaid(fixed)
        return validated["mermaid"], None
    except diagram_mod.DiagramError as exc:
        return None, f"Diagram repair failed: {exc}"


# Kinds of source element that belong in a diagram. A coverage entry with a
# different kind (a heading, a note) is not something the diagram is missing.
_DIAGRAM_KINDS = {"entity", "node", "box", "table", "component", "service"}


def _gap_candidates(coverage: list[dict[str, str]]) -> list[dict[str, str]]:
    """The coverage entries that should have appeared in the diagram."""
    out: list[dict[str, str]] = []
    for item in coverage:
        label = (item.get("label") or "").strip()
        if not label:
            continue
        kind = (item.get("kind") or "").strip().lower()
        if kind:
            if kind not in _DIAGRAM_KINDS:
                continue
        elif " " in label:
            # No kind and a multi-word phrase: a heading, not a box name.
            continue
        out.append(item)
    return out


def _change_records(changes: list[str]) -> list[dict[str, str]]:
    """Turn the app's change log into deviation records.

    The log is written as "what changed: why", so the two halves become the
    two fields the user reads.
    """
    records: list[dict[str, str]] = []
    for text in changes:
        change, _, why = text.partition(": ")
        records.append(
            {
                "change": change.strip(),
                "why": why.strip() or text.strip(),
                "basis": "normalization",
            }
        )
    return records


def _complete_diagram(
    diagram: dict, missing: list[dict[str, str]], expected_type: str
) -> tuple[dict | None, str | None]:
    """Ask the model to add the elements the first diagram left out.

    A diagram that quietly drops a box is the most common way a
    reconstruction stops being faithful, so the gap is worth one extra
    round trip.
    """
    listed = "\n".join(
        f"- {item.get('label', '')}"
        + (f" ({item.get('detail', '')})" if item.get("detail") else "")
        for item in missing
    )
    prompt = (
        "Your Mermaid diagram is missing elements that are in the source image:\n"
        f"{listed}\n\n"
        "Return the COMPLETE corrected diagram with those elements added. Keep every "
        "node, label, attribute and relationship you already drew exactly as it is; "
        "only add what is missing. Use the names above. Return only the diagram, with "
        "no prose and no code fence.\n\n"
        f"The diagram you produced:\n{diagram.get('mermaid', '')}"
    )

    try:
        fixed, _model, _fallbacks = _generate_text_only(prompt)
    except InferenceError as exc:
        return None, f"Diagram completion could not run: {exc}"

    try:
        validated = diagram_mod.validate_mermaid(fixed)
    except diagram_mod.DiagramError as exc:
        return None, f"Diagram completion failed: {exc}"

    completed = {
        "type": validated.get("type", expected_type),
        "mermaid": validated["mermaid"],
    }
    if not completed["mermaid"].strip():
        return None, "Diagram completion produced an empty diagram."

    # The completed diagram must be a superset of the first one. A rewrite
    # that quietly drops something the user drew is worse than the gap it
    # was asked to fill.
    original_labels = set(diagram_mod.diagram_labels(diagram.get("mermaid", "")))
    completed_labels = set(diagram_mod.diagram_labels(completed["mermaid"]))
    dropped = original_labels - completed_labels
    if dropped:
        return None, (
            "Diagram completion dropped "
            + ", ".join(sorted(dropped)[:5])
            + ", so the original diagram was kept."
        )

    return completed, None


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


def test_api_key(key: str, model: str | None = None) -> dict:
    """Check that a key works, without saving it.

    Makes one minimal request against the configured model: enough to prove
    the key is accepted and the model is reachable, cheap enough to be free
    in practice. The key itself is never logged or returned.

    Returns {"ok", "message", "model", "elapsed_ms"}.
    """
    target = model or model_id()
    started = time.perf_counter()
    try:
        client = _build_client(key)
    except InferenceError as exc:
        return {
            "ok": False,
            "message": str(exc),
            "model": target,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    try:
        types = _types()
    except InferenceError as exc:
        return {
            "ok": False,
            "message": str(exc),
            "model": target,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    config = types.GenerateContentConfig(max_output_tokens=8)
    try:
        response = client.models.generate_content(
            model=target,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text="ping")])],
            config=config,
        )
        _response_text(response, target)
    except Exception as exc:
        message = _message_of(exc)
        lowered = message.lower()
        if "api key not valid" in lowered or "invalid api key" in lowered:
            message = "The Gemini API rejected that key. Copy it again from AI Studio."
        elif "permission" in lowered:
            message = (
                "That key was rejected (HTTP 403). It may belong to a project that "
                "does not have the Gemini API enabled."
            )
        elif "not found" in lowered:
            message = (
                f"The key was accepted, but the model '{target}' is not available on "
                "this endpoint. Pick another model in Settings."
            )
        return {
            "ok": False,
            "message": message[:300],
            "model": target,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    return {
        "ok": True,
        "message": f"The key works. '{target}' answered the test request.",
        "model": target,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _extraction_is_empty(extraction: validation.Extraction) -> bool:
    """True when nothing usable came back, so a retry could still help."""
    return not (
        extraction.diagram
        or extraction.notes
        or extraction.takeaways
        or extraction.action_items
        or extraction.sql_schema
    )


# Modes whose whole point is a diagram. If one of these comes back without
# a diagram, the answer is incomplete and one focused retry is worth it.
_DIAGRAM_MODES = {"flowchart", "mindmap", "architecture", "concept", "er"}


def _expects_diagram(mode: str, detected: str | None) -> bool:
    """True when this answer is supposed to carry a diagram."""
    if detected:
        return detected in _DIAGRAM_MODES
    return mode in _DIAGRAM_MODES


def _retry_diagram_only(image, mode: str) -> tuple[dict | None, str | None]:
    """Ask for the diagram alone, in one focused turn.

    A mode like ER or flowchart that comes back with good notes but no
    diagram is a partial answer. Rather than re-running the whole
    extraction, the model is asked for the one thing that is missing.
    """
    prompt = (
        "Your previous answer did not include the diagram, and the diagram is the "
        "main deliverable for this mode. Look at the image again and return ONLY "
        "the Mermaid diagram for it, with no prose, no explanation and no code "
        "fence. Follow the syntax rules for this mode exactly, include every "
        "element that is in the image, and use the names as they are written there."
    )
    try:
        text, _model, _fallbacks = _generate(image, prompts.build_system_prompt(mode), prompt)
    except InferenceError as exc:
        return None, f"Diagram retry could not run: {exc}"

    try:
        validated = diagram_mod.validate_mermaid(text)
    except diagram_mod.DiagramError as exc:
        return None, f"Diagram retry failed: {exc}"
    return validated, None


def _extract_diagram(
    payload: dict, telemetry: dict
) -> tuple[dict | None, list[dict[str, str]]]:
    """Validate the payload's diagram, with one corrective retry on failure.

    Returns (diagram, changes). The changes are the edits the app made to
    the model's source -- a repaired ER attribute, a renamed entity -- and
    they are reported to the user rather than applied silently.
    """
    raw_diagram = payload.get("diagram")
    raw_source, explanation, raw_type = "", "", "flowchart"
    if isinstance(raw_diagram, dict):
        raw_source = raw_diagram.get("mermaid") or ""
        explanation = raw_diagram.get("explanation") or ""
        raw_type = raw_diagram.get("type") or "flowchart"
    elif isinstance(raw_diagram, str):
        raw_source = raw_diagram

    if not raw_source:
        return None, []

    applied: list[str] = []
    telemetry["diagram_attempts"] = 1
    try:
        validated = diagram_mod.validate_mermaid(raw_source, explanation, changes=applied)
        telemetry["diagram_changes"] = list(applied)
        return validated, _change_records(applied)
    except diagram_mod.DiagramError as exc:
        fixed, repair_error = _repair_diagram(raw_source, str(exc), expected_type=raw_type)
        telemetry["diagram_attempts"] = 2
        if fixed:
            return (
                {"type": raw_type, "mermaid": fixed, "explanation": explanation},
                _change_records(applied),
            )
        telemetry["diagram_repair_error"] = repair_error
        telemetry["diagram_error"] = str(exc)
        return None, _change_records(applied)


def _record_model(telemetry: dict, model_used: str, fallbacks: list[str]) -> None:
    """Fold a completed call's model choice into the response telemetry.

    Replaces rather than accumulates: the extraction on screen came from the
    last successful call, so its model is the one reported. Stacking the skip
    lists of every attempt would describe a chain walk that never happened --
    a retry that lands back on the primary must not leave the primary listed
    as skipped.
    """
    telemetry["model_used"] = model_used
    telemetry["model_fallbacks"] = list(fallbacks)


def analyze(
    image: validation.ValidatedImage, mode: str = "auto"
) -> tuple[validation.Extraction, dict]:
    """Run a full extraction using cloud inference via the Gemini API.

    Returns (extraction, telemetry).
    """
    chain = model_chain()
    telemetry: dict = {
        "model": chain[0],
        "model_chain": chain,
        "model_used": chain[0],
        "model_fallbacks": [],
        "mode": mode,
        "endpoint": endpoint_url(),
        "attempts": 0,
        "diagram_attempts": 0,
        "elapsed_ms": 0,
    }
    started = time.perf_counter()

    sys_prompt = prompts.build_system_prompt(mode)
    usr_prompt = prompts.build_user_prompt(mode)

    try:
        text, model_used, fallbacks = _generate(image, sys_prompt, usr_prompt)
    except InferenceError:
        telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        raise
    telemetry["attempts"] = 1
    _record_model(telemetry, model_used, fallbacks)

    payload = validation.extract_json_object(text)
    if payload is None:
        telemetry["attempts"] = 2
        repair = (
            "Your previous reply could not be parsed as JSON. Reply with ONLY the valid JSON "
            "object matching the schema — no markdown fences, no notes, no commentary. "
            f"Here is what you previously returned:\n\n{text[:2000]}"
        )
        try:
            text, model_used, fallbacks = _generate_text_only(repair)
            _record_model(telemetry, model_used, fallbacks)
            payload = validation.extract_json_object(text)
        except InferenceError:
            payload = None

    if payload is None:
        telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        telemetry["raw_preview"] = text[:RAW_PREVIEW_CHARS]
        extraction = validation.normalize_extraction(None, requested_mode=mode)
        extraction.warnings.append(
            "The model replied in a format this app could not parse. Please try again."
        )
        return extraction, telemetry

    validated_diagram, diagram_changes = _extract_diagram(payload, telemetry)
    extraction = validation.normalize_extraction(
        payload,
        requested_mode=mode,
        diagram_validated=validated_diagram,
        extra_deviations=diagram_changes,
    )

    # The diagram must show what the drawing contains. When the model's own
    # inventory lists an element the diagram does not have, one extra round
    # trip asks for the complete version rather than shipping a diagram with
    # a hole in it.
    if extraction.diagram:
        candidates = _gap_candidates(extraction.coverage)
        labels = [item.get("label", "") for item in candidates]
        missing_labels = diagram_mod.missing_labels(extraction.diagram["mermaid"], labels)
        if missing_labels:
            missing_items = [
                item for item in candidates if item.get("label") in missing_labels
            ]
            telemetry["diagram_gaps"] = missing_labels
            completed, gap_error = _complete_diagram(
                extraction.diagram, missing_items, extraction.diagram.get("type", "flowchart")
            )
            if completed:
                extraction.diagram = completed
                telemetry["diagram_gap_fill"] = True
                extraction.deviations.append(
                    {
                        "change": (
                            "Added "
                            + ", ".join(missing_labels[:6])
                            + (" and others" if len(missing_labels) > 6 else "")
                            + " to the diagram"
                        ),
                        "why": (
                            "These were in your drawing but missing from the first "
                            "diagram, so the diagram now shows the whole idea."
                        ),
                        "basis": "source",
                    }
                )
            elif gap_error:
                telemetry["diagram_gap_error"] = gap_error

    # A mode whose deliverable is a diagram, answered without one, is a
    # partial answer. One focused turn asks for the missing diagram.
    if (
        extraction.diagram is None
        and not _extraction_is_empty(extraction)
        and _expects_diagram(mode, extraction.detected_mode)
    ):
        telemetry["diagram_retry"] = 1
        diagram_only, diagram_retry_error = _retry_diagram_only(image, mode)
        if diagram_only:
            extraction.diagram = {
                "type": diagram_only.get("type", "flowchart"),
                "mermaid": diagram_only["mermaid"],
            }
            telemetry["diagram_retry"] = "ok"
        elif diagram_retry_error:
            telemetry["diagram_retry_error"] = diagram_retry_error

    # A reply that parses but carries nothing usable means the model did not
    # read the image. Retry once WITH the image, under an explicit instruction to
    # transcribe rather than describe.
    if _extraction_is_empty(extraction):
        telemetry["content_attempts"] = 1
        strict = (
            "Your previous answer contained no usable content from this image. "
            "Look at the image again and transcribe only what is actually written or drawn in it. "
            "List each box, arrow and label you can see. If a section is illegible, say so in "
            "\"uncertain_text\" instead of leaving everything empty. Do not describe the task and "
            "do not repeat any part of the JSON template."
        )
        try:
            retry_text, model_used, fallbacks = _generate(
                image, sys_prompt, f"{usr_prompt}\n\n{strict}"
            )
            telemetry["attempts"] = 2
            _record_model(telemetry, model_used, fallbacks)
            retry_payload = validation.extract_json_object(retry_text)
            if retry_payload is not None:
                retry_diagram, retry_changes = _extract_diagram(retry_payload, telemetry)
                retry_extraction = validation.normalize_extraction(
                    retry_payload,
                    requested_mode=mode,
                    diagram_validated=retry_diagram,
                    extra_deviations=retry_changes,
                )
                if not _extraction_is_empty(retry_extraction):
                    extraction = retry_extraction
                    telemetry["recovered"] = True
        except InferenceError as exc:
            telemetry["content_retry_error"] = str(exc)[:200]

    if _extraction_is_empty(extraction):
        # Keep the model's own words so a failure is diagnosable instead of blank.
        telemetry["raw_preview"] = text[:RAW_PREVIEW_CHARS]
        telemetry["degraded"] = True

    telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return extraction, telemetry