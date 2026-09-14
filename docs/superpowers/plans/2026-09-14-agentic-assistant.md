# SatQuery agentic application implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development to execute and review each bounded task. Keep work modular and reuse the existing pipeline.

**Goal:** Build the approved interactive assistant around the existing coverage
pipeline, with OpenAI orchestration and reusable adaptation/evaluation workflows.

**Architecture:** Label-free raster input boundary → modality-specific CROMA and
coverage runtime → spatial tools → bounded OpenAI controller → FastAPI browser
application. Image-text adaptation and evaluation remain independent modules.

**Tech Stack:** Python, PyTorch, Rasterio, FastAPI, OpenAI Responses API, plain
HTML/CSS/JavaScript. Existing runtime: /tmp/satquery-nonlinear-env/bin/python.

**Spec:** docs/superpowers/specs/2026-09-14-agentic-assistant.md

## Global constraints

- Work in /tmp/satquery-assistant-20260914, branch codex/agentic-assistant. Do not
  push, merge, modify baseline artifacts or run bulk downloads/training.
- No new agent dispatches from workers. Write tests first for meaningful behavior.
- Current head expects 768-dimensional CROMA features and 19 class fractions.
  Verify checkpoint contract and feature key; never substitute modalities.
- User API key exists in environment; never print it or save it in source,
  reports, client code or logs. OpenAI is the selected provider.
- Inference uses imagery only. Do not load reference maps, text answers or labels
  into the coverage inference path.
- Real datasets/checkpoints live under the original checkout's ignored data/
  or EDU Drive. Fixtures are synthetic; do not claim benchmark results from them.
- Preserve train/validation/test splits and explicit unsupported capabilities.

## Task 1: Label-free inputs and specialist runtime

Create src/satquery/assistant/ package with inputs.py, runtime.py, tools.py and
tests/test_assistant_inputs.py, tests/test_assistant_tools.py. Do not edit API/UI,
pyproject, documentation outside your report, or adaptation files.

Implement a bounded ZIP ingestion API `load_bundle(path, destination)` returning
an InputBundle object. ZIP has request.json schema_version=1, observations list;
each has id, modality (optical or SAR), sensor (Sentinel-2 or Sentinel-1), acquired
(ISO date), bands mapping names to {file, band:1}, and SAR radiometry='db'.
Paths must be relative, regular files within extraction root. Reject traversal,
symlinks, duplicates, excessive member/file/total sizes and >4 observations;
initial accepted configurations are 1 optical, 1 SAR, 1 optical+1 SAR, or 2
same-modality dated observations. Use explicit config validation, no guessing.
Keep uploads <=64 MiB and expanded total <=128 MiB. Accept TIFFs only.

Reuse DEFAULT_PROFILE and normalize_patch. Optical resampling onto B02's
120x120 10m grid follows existing preprocessing. SAR is 120x120 10m. Require
metre projected north-up CRS, same extent/CRS across bands and pair, native S2
resolutions or already aligned 10m bands, finite unmasked data. Reject mismatched
footprints, date reversals/equal temporal dates, single-band SAR, RGB and unknown
sensors explicitly. Distinguish date-pair from cross-modal pair. Return per-scene
normalized tensors plus metadata/geotransform and PNG-ready RGB or SAR preview.
InputBundle exposes kind ('single','cross_modal','temporal'), observations,
summary() with no server paths or raw pixels, and common grid metadata.

Runtime `CoverageRuntime(checkpoint, heads: dict, device='cpu')` uses the existing
CROMA vendor directly in optical/SAR/both mode and load_head; the caller provides
mapping keys optical/SAR/joint to compatible head files. Preserve checkpoint hash,
normalization/channel/source contract. Add `analyze(bundle)` returning one scene
result for single/cross-modal or two for temporal: arrays of shape (15,15,19),
preview, grid, provenance. Lazy model loading, torch.inference_mode, CPU supported.
If matching head isn't installed return explicit CapabilityError. No random head
or zero-filled missing sensor fallback. Define a cached-feature demo path using
FeatureBatches + selected index and supplied matching head, avoiding CROMA reruns.

Tools implement `execute_task(task, scenes, class_name=None, threshold=0.5)` with
fixed allowlist: coverage, presence, describe, locate, change. Return JSON-safe
dict with deterministic answer, measurements, evidence grid/GeoJSON, limitations,
trace (selected tool/model/parameters). Resolve class schema names plus forest,
water, farmland, built-up aliases without guessing unsupported objects. Sum
appropriate class fractions for aliases (forest = three forest classes; water =
inland+marine water; explicit mapping for farmland). Area is fractions*6400 m²
per cell. Presence and location are threshold-based estimates, not confidence;
do not report absence merely because a softmax area is small. Mark thresholds
heuristic/unvalidated. Describe top land-cover proportions. Locate uses coarse
cell footprints with correct affine bounds, no exact object boxes. Change needs
two aligned dated scenes of the same modality and model contract, yields per-cell
class deltas and before/after scene areas; label provisional/unvalidated. Include
abstention for unsupported questions/classes. Expose available capabilities from
installed heads. Do not return raw tensors in JSON.

Test real GeoTIFF fixtures with distinctive bands/channel normalization, grid
alignment failures, malicious ZIPs, date checks, reference-free loading, spatial
area/alias pooling, localization coordinates and temporal deltas. Fake runtime
models for unit tests are fine; clearly distinguish them from real integration.
Run changed tests + existing relevant tests, write report with commands/results,
and commit task files only.

## Task 2: OpenAI controller and browser application

Create assistant/controller.py, assistant/web.py, assistant/static/*,
tests/test_assistant_controller.py, tests/test_assistant_web.py; modify pyproject
for app dependencies/entrypoint. Consume Task 1 interfaces after reading its
report. OpenAI Responses API strict function calling, fixed tool registry,
bounded calls/parameters, store=False. Never execute generated code. Default
model gpt-4.1-mini-2025-04-14 configurable by SATQUERY_OPENAI_MODEL. Read key via
SDK environment only. No implicit remote imagery uploads: send question, input
summary and compact tool measurements only. Responses function-call inputs and
outputs stay in observable trace; omit internal reasoning and server paths.

Support deterministic explicit local mode as fallback; report chosen provider,
never silently fake OpenAI success. Unsupported objects, benchmark sensors,
missing dates/heads lead to specific actionable results. Natural-language output
must be grounded in computed evidence and display deterministic measurement
answer alongside optional LLM wording. Bounded error handling with no key leak.

FastAPI `create_app(...)` dependency injection for tests, /api/status,
/api/analyze multipart ZIP+question+provider, /api/demo for configured cached
feature demo, per-result report JSON and GeoJSON/download routes. Request files
are temporary; results durable under configured output directory. Use opaque
IDs; never accept arbitrary server paths from browser. Bind localhost by default,
serialize heavy inference, validate configured model paths before use. Do not
serve arbitrary filesystem paths. Keep CORS closed and protect local API against
cross-origin use. Body/query limits, clean temporary files on errors.

The primary GUI also accepts ordinary TIFF files with observation modality/date
controls and builds the strict internal ZIP/manifest. Support separate band files
with unambiguous band-name suffixes and one stacked TIFF in a user-acknowledged
standard channel order; do not guess missing bands or SAR radiometry. A bounded
/api/analyze-files endpoint can assemble this contract. Test both layouts and
multipart aggregate/member limits. Keep ZIP upload as an advanced option.

Responsive GUI shows input options, ZIP instructions/example schema, question
examples, clear capability and demo-model status, progress/error states, source
preview and clickable class grid/legend, area table, deterministic answer and LLM
wording, trace, limitations and downloadable report/evidence. Render untrusted
text with textContent. Works in local mode without key; selected OpenAI mode
explains missing API config. Provide CLI `satquery-serve` with model/head/features
and artifact-directory options. No secret forms in UI.

Tests exercise bounded tool calls, invalid arguments/unknown tools, unsupported
questions, OpenAI failures, valid upload+question results, limits/path protections,
local mode and report downloads. Use fake client for unit tests. Root will run a
live OpenAI and real model smoke after integration. Commit task files only.

## Task 3: Reusable image-text adaptation and task evaluation

Create assistant/text_adaptation.py, assistant/task_evaluation.py,
scripts/colab_adapt_text.py, tests/test_text_adaptation.py,
tests/test_task_evaluation.py. Preserve existing coverage/annotation code.

Build training pairs from verified match_annotations output and FeatureBatches,
using mean-pooled spatial features (record feature key), exact patch_id+s1_name
matching, report hashes and source/image splits. Only type=captioning and
training_eligible rows may train; validation rows select; test/bench/conflicts
must never enter fitting or training caption retrieval bank. Do not turn boxes
into per-cell percentage targets. Deduplicate same-image captions and quarantine
known invalid annotation IDs via configurable exclusions recorded in manifest.

Use OpenAI text-embedding-3-small 512-dimensional frozen text targets (explicitly
configured, resumable content/model/dim hash cache, no key persisted). Cache only
requested eligible splits. Pair preparation accepts an injectable embedder for
offline tests and records synthetic/caller-supplied provenance; CLI uses real
OpenAI embedding. Add small PyTorch 768→256→512 projection (GELU) with normalized
outputs, train-only feature normalization, cosine target loss averaged per-image,
seeded training and validation early stopping. Save best checkpoint and source
hashes exclusively; preserve prior outputs. Real training may need Drive access;
do not label test-fixture training as a production-adapted model.

Expose loading and image/query similarity or top-k training-caption retrieval;
scores are similarities not calibrated confidence. This is an experimental
scene-level image-text adapter; no open-ended caption/VQA quality claim. Test
image features are never associated with their own held-out answers at inference.
CLI prepare/train/evaluate accepts separate paths and respects split quarantine.
Evaluation reports held-out image-text retrieval ranks, candidate-set details,
counts and unsupported/absent data; never promote a checkpoint based on test.
Colab script calls reusable package functions and writes under supplied EDU P,
without embedding notebook-only implementation or rerunning image extraction.

Task evaluation consumes explicit JSONL reference/prediction records with IDs,
dataset, split, task, answer/abstention and optional class fractions/change labels.
Strict ID/split matching; duplicate/leakage fail. Report normalized exact-match
VQA (clearly provisional, not official scorer), abstention/coverage and per-task
counts; numerical area MAE uses explicit units; no invented joint score. Add
benchmark compatibility manifest for VRSBench/RSVQA/CDVQA/SAC expected modality
and current unsupported gaps. Do not pretend benchmark adapters/models are
available. Tests verify split exclusions, ordering, cache resume, held-out answer
exclusion, training updates/best checkpoint, evaluation and provenance guards.
Commit task files only and write full report.

## Task 4: Integration, documentation and acceptance

Integrate real original three-sample demo artifacts (explicit demo status) and
train optical/SAR demo heads from existing cached features via existing trainer
without recomputing CROMA. Run real label-free optical/SAR/paired smoke, live
OpenAI routing, UI screenshot/interaction checks and full existing/new suite.
Create reusable bundle-builder CLI for existing raw sample inputs if necessary,
using existing metadata mapping. Update README, docs/architecture.md and new
docs/assistant.md plus hackathon requirement-to-status table. Explain exact
config/launch, modular training and remaining true benchmark/model gaps. Persist
all code in the original project checkout and Git branch; prepare durable EDU
code snapshot (upload only if available). Do not merge main. Record evidence in
reports/assistant-integration.json without secrets/local sensitive paths. Final
response distinguishes running capabilities from pending real model adaptation,
benchmark validation and sensor compatibility. Commit verified integration.
