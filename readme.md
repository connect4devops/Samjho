# Scribble2Structure

**From messy visual thinking to structured, editable knowledge.**

Scribble2Structure turns handwritten notes, whiteboard photographs, brainstorming
sketches, technical system drawings and meeting notes into structured artifacts with
rendered Mermaid diagrams, verifiable action plans and multi-format exports.

Inference runs in the cloud through the **Gemini API** (Google AI Studio), defaulting to
**`gemini-2.5-flash-lite`** — Google's cheapest, fastest multimodal model, recommended for
high-volume data extraction. There is no model to download, no GPU, and no VRAM to budget.
A single API key plus a `GEMMA_MODEL` value in `.env` are the whole setup — or paste the key
into the app's **Settings** screen and it is tested and saved for you. Your uploaded
images are sent to Google's servers to be analysed, so this is **not** an offline tool.

---

## Settings: your key, inside the app

The **Settings** button in the header opens a panel where the API key can be entered,
tested and saved without touching a file or restarting anything.

- **Test connection** sends one minimal request with the key you pasted. A key that is
  wrong, expired or belongs to a project without the Gemini API enabled comes back as a
  plain explanation, not a stack trace. The key is checked for shape first, so a pasted
  sentence costs no request.
- **Save key** tests it first and only stores it if the test passed, so a broken key can
  never be saved by accident.
- **Remove key** deletes it.

How the key is handled:

- It is **write-only**. The API never returns it — the Settings screen shows a masked form
  (`AIza…abcd`) and nothing else, and the browser never receives the key after saving.
- It is stored **on this machine only**, in `.scribble2structure/credentials.json`, created
  with owner-only permissions and gitignored.
- A key saved in Settings **takes precedence** over one in `.env`, because it is the more
  recent deliberate action.
- A key is **only saved if it works**. The connection test runs first, so a broken key can
  neither be saved nor replace a working one that is already there.
- The key endpoints **refuse cross-origin requests**, so a page on another site cannot post
  a key to a locally running app.
- Connection tests are **rate limited** (a handful per minute), since this is the one
  endpoint that takes a secret from the browser.
- If the page is not served over **HTTPS**, the panel says so, because the key travels to
  the server in the clear.

Once a key is saved, the app uses it for the next request without a restart: the cached
API client is rebuilt so it cannot keep sending the old key.

---

## Choosing a model

`GEMMA_MODEL` accepts any vision-capable id the Gemini API serves — Gemini or Gemma. The
name is historical; both families work from the same codebase, and picking one is a
quality/latency/cost tradeoff, nothing more.

| `GEMMA_MODEL` | Family | Image input | Notes |
| --- | --- | --- | --- |
| `gemini-2.5-flash-lite` *(default)* | Gemini | yes | Cheapest and fastest; the API's recommendation for high-volume extraction. Thinking is off by default on this size. |
| `gemini-2.5-flash` | Gemini | yes | Slower and pricier, stronger reasoning on dense images and multi-box diagrams. |
| `gemma-3-27b-it` | Gemma | yes | Strongest open-weight extraction. |
| `gemma-3-12b-it` | Gemma | yes | Lighter Gemma. |
| `gemma-3-4b-it` | Gemma | yes | Lightest vision size. |
| `gemma-4-26b-a4b-it` | Gemma | yes | Gemma 4 mixture-of-experts, if enabled on your key. |

`gemma-3-1b-it` is deliberately absent: the 1B size is text-only and rejects every image.

Output limit per model: 65,536 tokens for Gemini 2.5, 8,192 for Gemma 3. The app requests
4,096 (`GEMMA_MAX_OUTPUT_TOKENS`), which is comfortably inside both and enough for a full
Mermaid diagram plus the notes payload.

### Optional: a fallback chain

Set `GEMMA_MODEL_FALLBACKS` to a comma-separated list of ids to try in order when the
primary is saturated, rate-limited, or not served on your key:

```ini
GEMMA_MODEL_FALLBACKS=gemini-2.5-flash,gemma-3-27b-it
```

It is empty by default, so a failure is reported rather than silently answered by a
different model. Whichever model actually served the reply is reported as
`telemetry.model_used`, with `telemetry.model_fallbacks` listing what was skipped.

---

## What the API does and does not let us assume

The two families share one endpoint but not one feature set. Because a Gemma id must keep
working from the same `GEMMA_MODEL` setting, `app/ai_client.py` builds requests to the
**intersection** of what both accept rather than to the capabilities of the configured one.

| Feature | Gemini | Gemma on this endpoint | What the app does |
| --- | --- | --- | --- |
| JSON mode / structured output | yes | **no** — `JSON mode is not enabled for models/gemma-3-27b-it` | The JSON schema ships in the prompt; the reply is parsed, with one corrective retry. |
| System role | yes | **no** — `Developer instruction is not enabled for models/…` | The system prompt is folded into the single user turn. |
| Sampler knobs | yes | default sampler only | `temperature` / `top_p` omitted unless explicitly set in `.env`. |
| Thinking | on/off configurable | n/a | Not configured. Flash-Lite does not think by default, so no budget is stolen from the reply. |

That conservatism leaves some of Gemini's structured output unused. It buys one codebase
that behaves identically whichever id you configure, instead of branching on the model name
and silently changing extraction quality when someone edits `.env`.

### A known `gemini-2.5-flash-lite` quirk

Flash-Lite has a documented bug where it wraps its JSON in a markdown fence —
a leading ` ```json ` and a trailing ` ``` ` — where `gemini-2.5-flash` returns bare
JSON. The prompt-driven contract already tolerates this:
`validation.extract_json_object()` strips fences before parsing, so fenced replies parse
normally. This is handled rather than worked around, and is the main reason the fallback
chain lists `gemini-2.5-flash` as a sensible next hop.

---

## Features

- **Cloud inference, no local weights** — Gemini API; nothing to download, nothing to fit
  in VRAM, selected entirely from `.env` or from the Settings screen.
- **Model-flexible** — Gemini or Gemma vision ids, with an optional fallback chain, from
  one codebase and one request format.
- **Reconstruction as the deliverable** — diagrams, notes and action plans are rebuilt as
  editable artifacts rather than described in prose.
- **ER diagrams that actually draw** — Mermaid's ER grammar is strict in ways models
  break constantly (every attribute needs a data type, entity names cannot contain
  spaces). The server repairs those deterministically and validates the result before the
  browser ever sees it, so the diagram renders instead of failing with a parse error.
- **Objective grading** — every extraction is scored against the drawing it came from on
  five criteria (source coverage, diagram completeness, structure captured, relationship
  integrity, faithfulness). The score is computed from the artifacts the app holds, so the
  model cannot talk itself up.
- **A guide, not just a diagram** — a plain-language walkthrough of the idea: what to
  notice, what the terms mean, where to go next.
- **Nothing is changed behind your back** — every edit the app makes to your drawing, and
  every change the model made, is listed in **Needs review** with what changed and why.
- **8 transformation modes**:
  1. **Auto Detect** — picks the best mode from the image's layout and content.
  2. **Handwritten Notes → Structured Notes** — headings, bullets, concepts, formulas.
  3. **Diagram → Flowchart** — nodes, labels, arrows and decision paths as Mermaid.
  4. **Brainstorming → Mind Map** — themes, categories and radiating ideas as Mermaid.
  5. **Technical Diagram → Architecture** — tiers, services, protocols, storage, data flow.
  6. **Meeting Notes → Action Plan** — decisions, tasks, owners and deadlines, with no invented facts.
  7. **Scientific Concept Map** — equations, symbols, units and relationships preserved.
  8. **Database ER Diagram** — entities, attributes, PK/FK, cardinalities, plus SQL `CREATE TABLE`.
- **Rendered Mermaid with a live editor** — `flowchart`, `graph`, `mindmap`,
  `sequenceDiagram` and `erDiagram`; edit the source and re-render in place; export `.mmd`,
  `.svg` or `.png`. Labels are native SVG text so exported images stay faithful.
- **Side-by-side comparison** — the uploaded image next to the reconstructed diagram.
- **Editable output with state management** — live editing of titles, summaries, notes,
  action items and diagrams, with reset and unsaved-change protection.
- **Share image** — one click renders the extraction as a designed PNG card (title, notes,
  takeaways, action plan, SQL) drawn on a canvas, so formulas and identifiers are exact
  rather than regenerated by an image model.
- **Multi-format exports** — PNG card, Markdown, plain text, JSON, Mermaid and SVG.
- **Sample gallery** — six generated inputs, one per diagram-oriented mode, for zero-effort evaluation.

---

## Implementation

### Pipeline

```
upload → validate → prompt → generate → repair → normalize → render
         │                       │         │          │
     magic bytes,           image +     JSON retry,   schema-echo
     size, EXIF,            mode prompt Mermaid retry rejection
     pixel ceiling
```

1. **Validation** (`app/validation.py`) — the file's leading bytes are checked against real
   image signatures, not the browser-supplied content type; extension, declared MIME and
   actual format must agree. Size, megapixel ceiling and EXIF orientation are handled before
   any inference cost is incurred.
2. **Prompting** (`app/prompts.py`) — a mode-specific system prompt plus the full JSON schema.
   Instructions state that the image is data, never instructions.
3. **Inference** (`app/ai_client.py`) — one thread-safe Gemini API client is built from the
   key and reused. The image goes inline as base64 with its real MIME type; the system
   prompt and request are fused into one user turn, because Gemma rejects a system role.
   JSON mode is never requested. Saturated or missing models move along the fallback chain,
   each with a short backoff; a request the API rejects outright fails immediately instead
   of burning quota on retries that cannot succeed.
4. **Corrective retries** — one retry when the reply is not parseable JSON, and one retry when
   the generated Mermaid fails validation. Both are text-only follow-ups.
5. **Normalization** (`app/validation.py`) — arbitrary model JSON is coerced into a
   renderable `Extraction`. Never raises: missing fields degrade to empty, so notes survive
   even when the diagram does not. Values that merely echo the prompt's own template are
   discarded rather than displayed as extracted content.
6. **Diagram validation** (`app/diagram.py`) — fences and prose are stripped, script-like
   markup is rejected, then supported type, size, bracket/quote balance and node-label
   bracketing are checked before Mermaid ever reaches the browser. ER diagrams get their
   own pass: the source is rewritten into the strict subset Mermaid can parse (a data type
   in front of every attribute, underscores in names that contain spaces, a quoted
   relationship label) and then parsed and checked entity by entity.
7. **Grading** (`app/grading.py`) — the extraction is scored against the drawing on five
   criteria. Every criterion is computed from what the app holds: the diagram that passed
   validation, the source inventory the model reported, the notes that were extracted and
   the changes that had to be made. A model that is confidently wrong cannot raise its own
   score.
8. **Rendering** (`static/index.html`) — Mermaid.js draws the diagram; every model string is
   HTML-escaped before insertion, and each render uses a unique id so concurrent draws cannot
   collide.

### The ER diagram problem, and what is done about it

Mermaid's ER grammar is much stricter than its flowchart grammar, and the server-side
checks used to be flowchart-shaped, so an invalid ER diagram passed validation and died in
the browser:

```
Parse error on line 9: ... emp_id } SALARY { ------^ Expecting 'ATTRIBUTE_WORD', got 'BLOCK_STOP'
```

Mermaid reads the first word of an attribute as its **data type** and demands a second one.
A bare `emp_id` is therefore a hard parse error, not a style problem. Entity identifiers
have the same trap: they are `ALPHANUM`, so a name with a space breaks the line.

The fix has three layers:

1. **The prompt** states the ER rules and shows a complete, valid example.
2. **The server** rewrites the source into strict syntax before validating it —
   `name` becomes `string name`, `Employee Record` becomes
   `Employee_Record["Employee Record"]`, an unquoted multi-word label gets quotes.
3. **The repair prompt** is ER-specific. Sending a flowchart rule ("wrap every label in
   brackets") to an ER diagram is how a repair round reproduces the same broken diagram.

Every rewrite is recorded as a **change** and shown in *Needs review* with the reason, so
the repair is visible rather than silent.

### Fail-safe by design

A hostile, broken or simply weak model response degrades instead of breaking the product:
notes and takeaways survive a diagram failure, a rejected diagram falls back to showing its
source in a code block, and a rejected key or saturated endpoint is translated into an
actionable message rather than a stack trace.

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 2. Install

```bash
git clone https://github.com/rohan1252030019-netizen/scribble2structure.git
cd scribble2structure

python -m venv .venv
# Windows:      .\.venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

```ini
# Gemini API key - https://aistudio.google.com/apikey
GEMINI_API_KEY=your_actual_key_here

# Pick the model that matches the quality/latency tradeoff (default shown)
GEMMA_MODEL=gemini-2.5-flash-lite

# Optional: try other models in order when the primary is unavailable
# GEMMA_MODEL_FALLBACKS=gemini-2.5-flash,gemma-3-27b-it

# Optional image upload limit in MB (default: 10)
MAX_UPLOAD_MB=10
```

### 4. Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload
```

Open **http://127.0.0.1:8090**. If the key is missing, the UI says so on load instead of
failing on the first upload.

---

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Readiness, the active model and the endpoint. `200` when a key is set, `503` when not. |
| `GET` | `/api/config` | Model id, upload limit, supported formats, available modes. |
| `GET` | `/api/samples` | Sample gallery entries. |
| `POST` | `/api/analyze` | `multipart/form-data` with `image` and `mode`. `400` on a bad upload, `502` on an inference failure. |
| `GET` | `/api/settings` | The current settings. The key is masked; it is never returned. |
| `POST` | `/api/settings/api-key/test` | `{"api_key": "..."}`. Tests the key without saving it. `400` on a malformed key, `502` if the API rejects it. |
| `POST` | `/api/settings/api-key` | `{"api_key": "..."}`. Tests the key, then saves it if it works. |
| `DELETE` | `/api/settings/api-key` | Forgets the saved key. |

The key endpoints refuse requests that do not come from the app's own page (`403`).

### The analyze response

`POST /api/analyze` returns the extraction plus the parts that make it reviewable:

| Field | What it holds |
| --- | --- |
| `notes`, `takeaways`, `action_items`, `sql_schema` | The structured reading of the image. |
| `diagram` | The validated Mermaid source. |
| `coverage` | The inventory of elements found in the image. The app checks the diagram against it and asks for a complete version when something is missing. |
| `deviations` | Every change made to the original drawing — what changed, why, and whether it came from the source, an inference or a normalization. |
| `guide` | The plain-language walkthrough: overview, steps, concepts, next steps. |
| `assessment` | The grade: score, confidence band, per-criterion scores with evidence, and the gaps worth a second look. |
| `telemetry` | Model used, fallbacks skipped, attempts, elapsed time, repairs made. |

---

## Samples

Six generated inputs live in `samples/`, covering notes, flowcharts, architecture, mind maps,
meeting notes and ER diagrams. Regenerate them with:

```bash
python scripts/create_samples.py
```

---

## Tests

```bash
pytest
```

The suite runs with no network call and no API key. Only the transport is faked
(`tests/conftest.py`); request assembly, the real SDK's `types` objects, config, error
translation, corrective retries, Mermaid validation, normalization and HTTP routing all
execute for real.

- `tests/test_ai_client.py` — model selection and the fallback chain, request assembly, the
  API capability constraints above, error translation, retries.
- `tests/test_diagram.py` — Mermaid sanitization and validation, including the ER repairs.
- `tests/test_grading.py` — the grading criteria.
- `tests/test_settings.py` — key masking, format checks, the local store, connection tests.
- `tests/test_validation.py` — upload integrity and response normalization, including template-echo rejection.
- `tests/test_frontend.py` — frontend contracts: render targets exist, ids are unique, no hardcoded model, no stale local-inference copy.
- `tests/test_pipeline.py` — end-to-end `/api/analyze` and the Settings endpoints.

---

## Project structure

```
app/
  ai_client.py    Gemini API client: request assembly, fallback chain, retries
  validation.py   Upload integrity, EXIF transpose, response normalization
  diagram.py      Mermaid sanitization, ER rewriting, validation, repair prompts
  grading.py      Objective grading of a reconstruction against its source
  prompts.py      Mode-specific system prompts and the JSON schema
  settings.py     The API key store: masking, format checks, local storage
  main.py         FastAPI routes
static/index.html Frontend (Mermaid rendering, editors, guide, grade, exports)
samples/          Sample input images
scripts/          Sample generator
tests/            Test suite and shared test doubles
```

---

## Security & reliability

- **Secrets stay server-side.** `GEMINI_API_KEY` is read only by the inference layer and is
  never sent to the browser or committed; `.env` is gitignored. A key saved from the
  Settings screen lives in an owner-readable, gitignored file and is never returned by the
  API — only a masked form is.
- **Uploads are inspected, not trusted.** Magic-byte sniffing, format agreement, size and
  megapixel ceilings, and EXIF orientation correction.
- **Model output is treated as data.** Generated SQL, Mermaid and prose are never executed.
  Script-like markup in a diagram is rejected, and all rendered text is HTML-escaped.
- **Images are sent to a third party.** This is the deliberate cost of cloud inference: the
  uploaded image reaches Google's servers. Do not use it for material you cannot share.
- **No mocked responses in the product.** The real model runs in every mode; only the test
  suite substitutes fakes.