# Task 2 implementation report

## Status

Implemented the bounded OpenAI/local controller, FastAPI application, guided and advanced
upload flows, durable report artifacts, responsive browser interface, and `satquery-serve`
CLI. Task 2 consumes the reviewed Task 1 contracts from commits `6bf5678` and `c247566`.
No Task 1 implementation file or package `__init__.py` was changed.

## Interfaces consumed

- `load_bundle(path, destination) -> InputBundle`; the destination is new and temporary.
- `InputBundle.summary()` for the only image-input facts sent to the language controller.
- `CoverageRuntime.available_capabilities`, `analyze(bundle)`, and
  `analyze_cached(features, selected_index=..., capability=...)`.
- `SceneResult` for deterministic evidence and optional local preview persistence.
- `execute_task(task, scenes, class_name=None, threshold=0.5)` with the fixed
  `coverage`, `presence`, `describe`, `locate`, and `change` allowlist.

The guided upload route generates the same schema-version-1 ZIP consumed by
`load_bundle`. A standard-order stack maps the documented optical or SAR channels to
distinct one-based TIFF band indexes. Separate files must carry every exact `_B01` … `_B12`
or `_VV`/`_VH` suffix. The server never infers missing bands, sensor type, acquisition date,
or SAR radiometry.

## Controller

- Added explicit `local` and `openai` providers. There is no automatic provider fallback or
  fabricated OpenAI success.
- The default language model is `gpt-4.1-mini-2025-04-14`, configurable through
  `SATQUERY_OPENAI_MODEL`. The OpenAI SDK reads `OPENAI_API_KEY` from its own environment;
  no API-key argument or browser form exists.
- The Responses API receives a question and safe input summary for routing, then compact
  deterministic measurements for optional wording. Pixel arrays, previews, dense evidence
  grids, filesystem paths, and raw imagery do not leave the server.
- Every function definition is a flat strict Responses tool with
  `additionalProperties: false` and every declared property required. One routing call and
  at most one wording call are allowed; both use `store=False`.
- Function-call arguments and compact outputs remain in the returned observable trace.
  Unknown tools, malformed JSON, extra/missing parameters, invalid thresholds, absent call
  identifiers, and multiple tool calls abstain rather than execute.
- Object-count questions are rejected before either provider is called because the coverage
  taxonomy cannot count buildings, roads, vehicles, ships, trees, or similar objects.
- The deterministic measurement answer always remains separate and visible. Optional wording
  is limited to 2,000 characters and is omitted if it introduces a numeric value absent from
  the computed result. Provider failures become bounded actionable errors without including
  the upstream exception or credential.

## Web application

- Added dependency-injected `create_app(...)`, `GET /api/status`, `POST /api/analyze`,
  `POST /api/analyze-files`, `POST /api/demo`, and opaque report, GeoJSON, and preview routes.
- Advanced ZIPs and guided multipart files are copied in bounded chunks into request-scoped
  temporary directories. The 64 MiB upload, multipart part/field/file, member-count, filename,
  and 1,000-character question limits apply before model work. Temporary content is removed on
  success and failure.
- Heavy inference and cached-demo prediction are serialized with one application lock. OpenAI
  configuration is checked before heavy inference when that provider is selected.
- Result IDs are random 32-character UUID hex values. Browser routes select only fixed artifact
  names under that ID. Invalid IDs, missing artifacts, artifact symlinks, result-directory
  symlinks, and configured path symlinks are rejected.
- The application has no CORS middleware, rejects mismatched `Origin` and `Sec-Fetch-Site:
  cross-site` API requests, restricts host headers to localhost/test hosts, and sends CSP,
  nosniff, and no-referrer headers.
- Reports and RFC 7946 GeoJSON are durable under the configured output directory. Request paths
  and output-directory paths are absent from reports.
- The browser interface supports ordinary band files, explicitly acknowledged standard-order
  stacks, optional second observations, advanced reusable ZIPs, and configured cached features.
  It exposes model/provider/demo status, including `demo_fit_all`, question examples,
  progress/errors, deterministic and optional language answers, preview, area table, clickable
  15 × 15 evidence grid, 19-class legend, limitations, trace, and downloads.
- All untrusted result content is assigned with `textContent` or created as text nodes. The UI
  contains no HTML-string rendering or generated-code execution.

## Packaging and CLI

Added an `assistant` optional dependency group containing FastAPI, httpx, OpenAI, Pillow,
python-multipart, and Uvicorn. A clean base installation receives a direct instruction to install
`satquery-preprocessing[assistant]` when the web dependencies are missing.

The `satquery-serve` entry point supports:

- `--croma-checkpoint`
- `--optical-head`, `--sar-head`, `--joint-head`
- `--demo-features`, `--demo-index`, `--demo-capability`, `--demo-preview`
- `--demo-fit-all`
- `--output-dir`, `--device`, `--host`, and `--port`

It binds `127.0.0.1` by default, checks regular non-symlink model/demo paths, validates the pinned
checkpoint and configured head compatibility before starting, and requires at least one usable
head.

## Test-first evidence

The controller tests were written before the Task 1 package or controller existed. The first run
failed at the intended missing-feature boundary:

```text
/tmp/satquery-nonlinear-env/bin/python -m pytest -q tests/test_assistant_controller.py
11 failed (ModuleNotFoundError: No module named 'satquery.assistant')
```

The web tests were then written before `web.py` existed:

```text
/tmp/satquery-nonlinear-env/bin/python -m pytest -q tests/test_assistant_web.py
9 failed (ModuleNotFoundError: No module named 'satquery.assistant.web')
```

The tests cover local routing, count abstention, unrelated questions, strict OpenAI schemas,
bounded call counts, compact function outputs, unknown tools, malformed arguments, provider
failure sanitization, numeric-grounding rejection, question/provider bounds, environment model
selection, status, capability/class/demo metadata, temporary cleanup, durable downloads,
cross-origin/request-size rejection, result path traversal and symlink protection, missing
OpenAI configuration, CLI arguments, cached demos, advanced ZIPs, and guided TIFF assembly.

The guided route integration tests create and submit real Rasterio GeoTIFFs: one 12-band optical
stack and two named SAR files. Both assembled archives pass the real Task 1 loader, proving the
web manifest matches the production input boundary.

## Verification

Focused Task 2 tests:

```text
/tmp/satquery-nonlinear-env/bin/python -m pytest -q tests/test_assistant_controller.py
16 passed

/tmp/satquery-nonlinear-env/bin/python -m pytest -q tests/test_assistant_web.py
11 passed
```

Integrated Task 1 and Task 2 tests:

```text
/tmp/satquery-nonlinear-env/bin/python -m pytest -q \
  tests/test_assistant_inputs.py tests/test_assistant_tools.py \
  tests/test_assistant_controller.py tests/test_assistant_web.py
57 passed
```

Full repository suite:

```text
/tmp/satquery-nonlinear-env/bin/python -m pytest -q
174 passed, 5 skipped
```

The full run reports existing/upstream Starlette and Rasterio deprecation warnings; it has no
test failures.

Static checks:

```text
/tmp/satquery-nonlinear-env/bin/python -m ruff check src tests
All checks passed!

/tmp/satquery-nonlinear-env/bin/python -m ruff format --check \
  src/satquery/assistant tests/test_assistant_inputs.py tests/test_assistant_tools.py \
  tests/test_assistant_controller.py tests/test_assistant_web.py
10 files already formatted

node --check src/satquery/assistant/static/app.js
exit 0

PYTHONPATH=src /tmp/satquery-nonlinear-env/bin/python -m satquery.assistant.web --help
exit 0; all documented CLI flags displayed

git diff --check
exit 0
```

## Browser verification

Served the dependency-injected app on `127.0.0.1:8765` with fake status dependencies, then
inspected it in a live Chromium page at 1200 × 900 and 390 × 844. Desktop and mobile layouts were
readable without horizontal overflow. Status showed optical/SAR/joint capabilities, explicit
OpenAI unavailability, and `demo_fit_all: yes`. Guided and advanced panels switched correctly;
choosing SAR revealed and enabled the radiometry selector. The browser console contained zero
messages.

## Remaining integration fact

No live OpenAI call was made in this task because the current environment key returns HTTP 401.
Fake-client tests cover both Responses calls and error sanitization. The app reports the
authentication failure when OpenAI is explicitly selected and continues to offer the local
provider; it never labels that path as OpenAI success.
