"""Where inference API keys live, and how they are handled.

The active provider's key can come from these sources, in order:

1. the store written by the Settings screen (``.samjho/``),
2. the provider-specific environment variable in ``.env``.

The store wins because it is the most recent, deliberate action: if someone
pastes a key into Settings, that is the key the app should use, not one left
over in ``.env`` from an earlier setup.

Nothing here ever returns the key itself to a caller. The only form that
leaves this module is :func:`mask_key`, which keeps the first four and the
last four characters so a person can recognise the right key without the
key being readable in a screenshot, a log line or a browser devtools tab.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

# Google AI Studio keys are 39 characters and start with AIza. Other
# deployments (proxies, Vertex, emulator) use other shapes, so the format
# check is a sanity check, not a whitelist.
_KEY_RE = re.compile(r"^[A-Za-z0-9_\-\.]{1,500}$")

_GEMINI_MIN_LENGTH = 20
_GEMINI_MAX_LENGTH = 200
_MAX_LENGTH = 500

_PROVIDER_KEYS = {
    "jarvislabs": (
        "jarvislabs_api_key",
        ("JARVISLABS_API_KEY", "JARVIS_API_TOKEN"),
    ),
    "gemini": (
        "gemini_api_key",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"),
    ),
}

_HOME_ENV = "SAMJHO_HOME"
_STORE_DIRNAME = ".samjho"
_STORE_FILENAME = "credentials.json"


def store_dir() -> Path:
    """The directory holding the local credential store."""
    override = (os.getenv(_HOME_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / _STORE_DIRNAME


def store_path() -> Path:
    """The file holding the locally saved API key."""
    return store_dir() / _STORE_FILENAME


def display_path() -> str:
    """A short description of where the key is stored.

    Shown in the Settings screen so the user knows where the file is, but
    relative to the project so the server's directory layout is not handed
    out to the browser.
    """
    path = store_path()
    root = Path(__file__).resolve().parent.parent
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def mask_key(key: str) -> str:
    """A recognisable, non-usable form of the key."""
    if not key:
        return ""
    text = key.strip()
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * max(4, len(text) - 8)}{text[-4:]}"


def validate_key_format(key: str, provider: str = "gemini") -> str | None:
    """Check a key's shape. Returns an error message, or None when it is fine.

    This is a cheap sanity check performed before a network call, so an
    obviously wrong value (empty, whitespace, a pasted sentence) is rejected
    without spending a request.
    """
    if not isinstance(key, str):
        return "The API key must be text."
    text = key.strip()
    if not text:
        return f"Enter your {provider} API key."
    if any(char.isspace() for char in text):
        return "The API key contains a space or a line break. Paste the key only."
    minimum = _GEMINI_MIN_LENGTH if provider == "gemini" else 1
    maximum = _GEMINI_MAX_LENGTH if provider == "gemini" else _MAX_LENGTH
    if len(text) < minimum:
        return (
            f"That key is only {len(text)} characters long. "
            f"At least {minimum} characters are required."
        )
    if len(text) > maximum:
        return "That API key is too long."
    if not _KEY_RE.match(text):
        return (
            "That API key contains unsupported characters. Paste the key only."
        )
    return None


def _read_store() -> dict:
    path = store_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def stored_key(provider: str = "gemini") -> str:
    """The key saved through the Settings screen, if any."""
    store_key, _ = _PROVIDER_KEYS.get(provider, _PROVIDER_KEYS["gemini"])
    value = _read_store().get(store_key)
    return value.strip() if isinstance(value, str) else ""


def save_key(key: str, provider: str = "gemini") -> None:
    """Persist the key locally, readable only by this user."""
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    store_key, _ = _PROVIDER_KEYS.get(provider, _PROVIDER_KEYS["gemini"])
    data = _read_store()
    data[store_key] = key.strip()
    payload = json.dumps(data, indent=2)
    # Create with restrictive permissions where the OS supports them.
    path.write_text(payload, encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        path.parent.chmod(stat.S_IRWXU)
    except (OSError, NotImplementedError):
        # Windows and some network filesystems do not implement POSIX modes.
        pass


def clear_key(provider: str = "gemini") -> None:
    """Forget the saved key."""
    path = store_path()
    try:
        data = _read_store()
        store_key, _ = _PROVIDER_KEYS.get(provider, _PROVIDER_KEYS["gemini"])
        data.pop(store_key, None)
        if data:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        pass


def api_key(provider: str = "gemini") -> str:
    """The key the app should use, from any of the supported sources."""
    stored = stored_key(provider)
    if stored:
        return stored
    _, env_names = _PROVIDER_KEYS.get(provider, _PROVIDER_KEYS["gemini"])
    for name in env_names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def key_source(provider: str = "gemini") -> str:
    """Where the active key came from: ``store``, ``environment`` or ``none``."""
    if stored_key(provider):
        return "store"
    _, env_names = _PROVIDER_KEYS.get(provider, _PROVIDER_KEYS["gemini"])
    return (
        "environment"
        if any((os.getenv(name) or "").strip() for name in env_names)
        else "none"
    )
