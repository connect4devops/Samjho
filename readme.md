# Samjho

**From messy visual thinking to structured, editable knowledge.**

Samjho turns handwritten notes, whiteboard photographs, brainstorming sketches, technical
system drawings and meeting notes into structured artifacts with rendered Mermaid
diagrams, verifiable action plans and multi-format exports.

By default, cloud inference runs through **JarvisLabs Serverless** using the
OpenAI-compatible API and the vision-capable **Gemma 4 26B A4B Instruct** model. The GPU runs
in the JarvisLabs deployment; Samjho sends image and prompt data to the fixed endpoint
and receives a structured response. This is not local or offline inference.

---

## Settings: your key, inside the app

The **Settings** button in the header opens a panel where the API key can be entered,
tested and saved without touching a file or restarting anything.

- **Test connection** sends a minimal request to the configured JarvisLabs deployment. A
  rejected key, missing deployment, or unavailable endpoint returns an actionable message.
- **Save key** tests it first and only stores it if the test passed, so a broken key can
  never be saved by accident.
- **Remove key** deletes it.

How the key is handled:

- It is **write-only**. The API never returns it — the Settings screen shows only a masked
  form, and the browser never receives the key after saving.
- It is stored **on this machine only**, in `.samjho/credentials.json`, created
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

## Inference provider and model

Samjho uses JarvisLabs Serverless exclusively. The endpoint is fixed to the project
deployment, and `JARVISLABS_MODEL` must match its model identifier (the default is
`gemma-4-26b-a4b-it`). There are no provider or model fallbacks: model and deployment
errors are returned to the user rather than silently switching models. Responses report
`provider_used` and `model_used` in telemetry.

JarvisLabs Serverless exposes an OpenAI-compatible `chat/completions` route. Samjho sends
the image as an inline `data:` image URL and the prompt in the same user message; JSON is
requested in the prompt and parsed/validated locally, avoiding provider-specific JSON
mode. Temporary HTTP 408, 429, 500, 502, 503 and 504 errors and an empty model response
receive bounded same-model retries. Authentication and invalid-request errors fail
clearly without retrying.

---

## Features

- **Cloud GPU inference** — JarvisLabs Serverless runs the deployed Gemma vision model;
  no model weights or GPU drivers are needed on the Samjho host.
- **JarvisLabs-only inference** — one fixed endpoint and configured model, with no
  provider or model fallback.
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
3. **Inference** (`app/ai_client.py`) — one authenticated JarvisLabs HTTP client is reused.
   The image is sent inline as a base64 data URL with its real MIME type, together with the
   prompt. The response follows the OpenAI chat-completion shape and is normalized to text.
   Transient service failures receive bounded retries; authentication and invalid-request
   failures surface immediately.
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
- A JarvisLabs account, a running Serverless deployment, and an API key

### 2. Install

```bash
git clone https://github.com/connect4devops/Samjho.git
cd Samjho

python -m venv .venv
# Windows:      .\.venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Create a JarvisLabs GPU deployment

1. Follow the [JarvisLabs Serverless guide](https://docs.jarvislabs.ai/serverless/) and
   open its [deployment dashboard](https://jarvislabs.ai/dashboard/serverless); choose
   **New deployment**.
2. Select a serving framework supported by your model (vLLM or SGLang) and choose the
   **Gemma 4 26B A4B Instruct** model from the model selector, or use the model from
   [your JarvisLabs model page](https://jarvislabs.ai/dashboard/models/gemma-4-31b-it).
3. Select a GPU type and count that satisfy the dashboard's memory requirements for this
   model. If the model does not fit on one GPU, select a supported multi-GPU configuration.
   If the model source requires Hugging Face access, provide an authorized token as the
   deployment's `HF_TOKEN` environment variable in JarvisLabs (not in Samjho's app config).
4. For lower idle cost, set **Min workers** to `0`; set it to `1` if you prefer a warm
   worker and faster first responses. Scale-to-zero stops worker GPU charges while idle,
   but persistent model storage continues to be billed.
5. Click **Create**, wait until deployment status is **Running** and its worker is healthy.
   Copy the OpenAI base URL shown for the deployment; its documented shape is
   `https://<region-base-url>/openai/<deployment_id>/v1`.
6. Create an API key under [JarvisLabs API settings](https://jarvislabs.ai/settings/api-keys).

GPU selection happens in the JarvisLabs deployment, not in Samjho's API request. The
chat-completion response does not report GPU utilization; check the deployment's Workers
and Usage dashboard for worker health and cost. The app's Settings **Test connection**
checks text reachability; analyze one sample image to confirm the deployed serving
framework/model accepts vision content.

### 4. Configure Samjho

```bash
cp .env.example .env      # Windows PowerShell: Copy-Item .env.example .env
```

Keep the three required deployment values separate in `.env`:

```ini
JARVIS_API_TOKEN=paste_your_jarvislabs_api_token_here
JARVISLABS_MODEL=gemma-4-26b-a4b-it
JARVISLABS_TIMEOUT_SECONDS=120
JARVISLABS_MAX_OUTPUT_TOKENS=4096

# Optional image upload limit in MB (default: 10)
MAX_UPLOAD_MB=10
```

The endpoint is fixed by Samjho; `JARVIS_API_TOKEN` is the private credential and
`JARVISLABS_MODEL` must match the deployment's model identifier. `JARVISLABS_API_KEY` is
also accepted as an alternative token variable. You can also enter the JarvisLabs API key
in **Settings**; it is tested against the fixed endpoint and saved locally. Keep `.env`
private and never commit API keys.

### 5. Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload
```

Open **http://127.0.0.1:8090**. The health endpoint reports missing credentials before
an upload fails.

---

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Readiness, provider, model, endpoint and key configuration state. `503` when not ready. |
| `GET` | `/api/config` | Provider/model id, upload limit, supported formats and modes. |
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
| `telemetry` | Provider/model used, retry attempts, elapsed time and repairs made. |

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

The suite runs with no network call or live API key. Provider transports are faked
(`tests/conftest.py`); request assembly, config, error translation, corrective retries,
Mermaid validation, normalization and HTTP routing all execute for real.

- `tests/test_ai_client.py` — JarvisLabs image-request assembly, error translation and
  bounded same-endpoint retries.
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
  ai_client.py    JarvisLabs transport, request assembly and bounded retries
  validation.py   Upload integrity, EXIF transpose, response normalization
  diagram.py      Mermaid sanitization, ER rewriting, validation, repair prompts
  grading.py      Objective grading of a reconstruction against its source
  prompts.py      Mode-specific system prompts and the JSON schema
  settings.py     JarvisLabs API key store: masking, format checks, local storage
  main.py         FastAPI routes
static/index.html Frontend (Mermaid rendering, editors, guide, grade, exports)
samples/          Sample input images
scripts/          Sample generator
tests/            Test suite and shared test doubles
```

---

## Security & reliability

- **Secrets stay server-side.** `JARVISLABS_API_KEY` / `JARVIS_API_TOKEN` are read only by the inference layer and
  is never sent to the browser or committed; `.env` is gitignored. A key saved from the
  Settings screen lives in an owner-readable, gitignored file and is never returned by the
  API — only a masked form is.
- **Uploads are inspected, not trusted.** Magic-byte sniffing, format agreement, size and
  megapixel ceilings, and EXIF orientation correction.
- **Model output is treated as data.** Generated SQL, Mermaid and prose are never executed.
  Script-like markup in a diagram is rejected, and all rendered text is HTML-escaped.
- **Images are sent to a third party.** With the default provider, uploaded images reach
  JarvisLabs and its configured model-serving workers. Do not use it for material you
  cannot share.
- **No mocked responses in the product.** The real model runs in every mode; only the test
  suite substitutes fakes.

---

## Project positioning

Samjho is an **AI-powered productivity and knowledge-management tool**, specifically
**document intelligence**: it digitizes handwriting, notes, whiteboards and diagrams,
then turns them into structured, editable information. For technical audiences, the
clearest category is **Document AI**; for a pitch or competition entry, position it as an
**AI productivity and knowledge-capture tool**.

- **Business model/category:** A B2B/B2C SaaS developer-and-productivity tool built on
  applied generative AI and multimodal document understanding.
- **Primary use:** Handwriting and whiteboard OCR, diagram digitization, structured
  extraction, validation, repair and grading. It sits alongside AI-assisted import and
  note-taking workflows such as Miro/Lucidchart import, Microsoft Lens and Notion AI.
- **Horizontal industry scope:** Education (study notes and concept maps), software and
  product teams (architecture and ER diagrams), and business operations (meeting notes and
  action plans).
- **Likely customers:** Students and educators; software/product teams; consultants and
  project managers facilitating workshops and whiteboard sessions.
- **Technology layer:** JarvisLabs-hosted Gemma 4 vision model by default, with validation,
  diagram repair, normalization and objective grading layered around generation.
- **Planned direction:** Cited idea validation expands Samjho toward AI research and
  decision-support tooling; pushing action items into GitHub issues expands it toward
  workflow automation and DevOps productivity.