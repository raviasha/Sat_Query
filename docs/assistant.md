# SatQuery browser assistant

**Status: 14 September 2026.** The browser application is a working, bounded prototype around
the existing CROMA land-cover pipeline. This guide separates verified behavior from demo artifacts
and capabilities that still need model training or benchmark evaluation.

## What runs today

The application accepts validated Sentinel-2 optical imagery, Sentinel-1 SAR imagery, a
co-registered optical–SAR pair, or two aligned and dated observations of the same modality. It
normalizes the image bands, runs the official pinned CROMA Base checkpoint, applies the matching
optical, SAR, or joint coverage head, and routes one question to an allowlisted deterministic tool.
It returns:

- a deterministic answer and structured measurements;
- the input preview and a clickable 15 × 15 evidence grid when the selected tool has spatial
  evidence;
- model/head hashes, selected tool, permitted parameters, provider role, and function-call trace;
- limitations and explicit abstentions; and
- downloadable JSON, GeoJSON, and preview PNG artifacts.

Each grid cell is an 80 m spatial anchor for the current 120 × 120 at 10 m input profile. CROMA
attention can include neighboring cells and wider scene context, so a token is not an isolated
8 × 8-pixel classifier. Fractions are estimated class area, not confidence. Presence/location
thresholds and image-text similarity values are uncalibrated.

## Install and launch

Use Python 3.11 or later and install the assistant extra:

```bash
uv sync --locked --no-editable \
  --extra dev --extra downloads --extra features --extra assistant
```

Start the local service with the pinned CROMA checkpoint and the heads you intend to expose:

```bash
uv run --no-sync satquery-serve \
  --croma-checkpoint artifacts/CROMA_base.pt \
  --optical-head artifacts/optical/head.pt \
  --sar-head artifacts/sar/head.pt \
  --joint-head artifacts/joint/head.pt \
  --output-dir data/assistant-results \
  --device cpu
```

Open `http://127.0.0.1:8000`. CPU, CUDA, and MPS are selectable, subject to availability. The
default host is `127.0.0.1`; the application is designed as a local service and rejects
cross-origin API calls. Configure only compatible heads. Missing modalities appear as unavailable
in `/api/status` and produce an actionable capability error rather than a random or zero-filled
fallback.

To display a cached-feature demonstration without rerunning CROMA, add:

```bash
  --demo-features artifacts/features \
  --demo-capability joint \
  --demo-index 0 \
  --demo-fit-all
```

`--demo-fit-all` is a visible disclosure. The integration samples were fit on all three supplied
demo areas and provide no held-out or production-performance evidence. Model weights, datasets,
and the 742 MiB CROMA checkpoint are intentionally outside Git.

## Input contracts

### Guided TIFF upload

The default UI builds the strict internal request from ordinary TIFF files. For each observation,
choose the modality and acquisition date. A second observation may be a co-registered other sensor
or a later observation of the same modality.

Accepted layouts are:

- twelve separate Sentinel-2 files ending in `_B01.tif`, `_B02.tif`, `_B03.tif`, `_B04.tif`,
  `_B05.tif`, `_B06.tif`, `_B07.tif`, `_B08.tif`, `_B8A.tif`, `_B09.tif`, `_B11.tif`, and
  `_B12.tif`;
- two separate Sentinel-1 files ending in `_VV.tif` and `_VH.tif`, with `dB` radiometry selected;
- one 12-band optical stack after explicitly acknowledging the standard order above; or
- one two-band SAR stack in `VV, VH` order after explicitly selecting `dB` radiometry.

The server never infers a missing band or guesses stack order/radiometry. Guided upload already
assembles the same bounded manifest consumed by the strict loader, so a separate bundle-builder CLI
would duplicate this implementation and was not added.

### Advanced request ZIP

The advanced route accepts a ZIP with `request.json` at its root and explicitly referenced TIFFs.
A TIFF may provide one or several bands; `band` is a positive one-based TIFF band index.

```json
{
  "schema_version": 1,
  "observations": [
    {
      "id": "scene-a",
      "modality": "SAR",
      "sensor": "Sentinel-1",
      "acquired": "2026-02-03",
      "radiometry": "db",
      "bands": {
        "VV": {"file": "scene.tif", "band": 1},
        "VH": {"file": "scene.tif", "band": 2}
      }
    }
  ]
}
```

The loader accepts only the four current configurations: one optical observation, one SAR
observation, one optical plus one SAR observation, or two ascending dated observations of the same
modality. It validates ZIP boundaries, TIFF driver, band counts, masks/finite values, projected
metre CRS, north-up geometry, exact 1.2 km footprint, 10 m target grid, expected native Sentinel-2
resolutions, and pair alignment. Expanded requests and individual members are bounded. Uploads are
temporary and removed after processing.

Reference maps, target fractions, annotations, and text answers are neither required nor loaded at
inference. RGB images, arbitrary multispectral layouts, unknown sensors, missing Sentinel channels,
Cartosat-2S, and RISAT are rejected until a documented sensor profile and compatible model exist.

## Question routing and tool semantics

The question router exposes five tools:

| Tool | Current output | Scope and limitation |
|---|---|---|
| `coverage` | Estimated scene area/fraction for one supported class or explicit alias | Fractions are area estimates, not confidence. |
| `presence` | Cells above a supplied area-fraction threshold | A cell below threshold does not establish absence; threshold is heuristic. |
| `describe` | Top estimated class proportions | Deterministic coverage summary, not a learned caption generator. |
| `locate` | WGS84 GeoJSON polygons for cells above threshold | Coarse 80 m anchors, not exact object boxes or segmentation. |
| `change` | Before/after/net class area and per-cell deltas | Requires aligned, ordered dates; provisional coverage delta, not a benchmarked change model. |

The fixed class schema has nineteen CLC-derived groups. `forest`, `water`, `farmland`, and
`built-up` are explicit aliases that pool documented classes. Other requested objects produce an
abstention instead of a guessed mapping.

Local mode uses deterministic rules to select a tool. With OpenAI routing, set the key only in the
server environment:

```bash
export OPENAI_API_KEY='your-key'
export SATQUERY_OPENAI_MODEL='gpt-4.1-mini-2025-04-14'  # optional override
```

The browser has no key field. The OpenAI Responses request contains the question, a compact safe
input summary, and strict tool schemas. Raw imagery, dense feature/coverage grids, filesystem paths,
secrets, and computed tool output are not sent. OpenAI may select exactly one tool and bounded
arguments in one call; all measurements and answer text are produced locally. Model-authored prose
is discarded so it cannot invent a measurement. Provider authentication/rate-limit/timeout errors
are sanitized and visible. Selecting OpenAI never silently falls back to local routing; the user may
choose local mode explicitly.

## Outputs and evidence

Successful runs create an opaque result ID beneath the configured output directory. The browser
uses server-generated download URLs only:

- `report.json` contains the input summary, deterministic result, trace, limitations, model hashes,
  and download links;
- `evidence.geojson` contains WGS84 feature polygons for spatial results, or an empty feature
  collection when the tool has no polygons; and
- `preview.png` is the input-derived optical or SAR preview when available.

Reports are JSON-safe and omit normalized tensors, raw pixels, temporary/server paths, and secrets.
The observable trace records the selected task/tool, fixed model/provider role, accepted parameters,
and bounded function-call result. It intentionally excludes internal model reasoning.

## BigEarthNet.txt image-text adaptation

The adaptation module is separate from the coverage head. It reads the verified annotation match
and saved `FeatureBatches`, mean-pools a selected 225 × 768 CROMA spatial feature, embeds eligible
captions with OpenAI `text-embedding-3-small` at 512 dimensions, and trains a
768→256→512 GELU projection with cosine loss. It does not modify CROMA.

Only `captioning` records marked training-eligible may train. Image-level partition quarantine is
preserved; train-only statistics normalize features; validation captions select the checkpoint;
test/benchmark/conflict captions never enter fitting or the training retrieval bank. Prepared
records, splits, eligibility, and source hashes are bound in the artifact and revalidated on load.
Embedding cache keys include content, model, and dimension, and no API key is persisted.

The Colab-friendly driver consumes existing features and writes new artifacts under a supplied
persistent root; it never reruns image preparation or CROMA extraction:

```bash
PYTHONPATH=src python scripts/colab_adapt_text.py --edu-p /persistent/SatQuery prepare \
  --annotations /persistent/SatQuery/annotations/matched \
  --features /persistent/SatQuery/features --name text-pairs

PYTHONPATH=src python scripts/colab_adapt_text.py --edu-p /persistent/SatQuery train \
  --prepared /persistent/SatQuery/text-adaptation/text-pairs --name text-adapter

PYTHONPATH=src python scripts/colab_adapt_text.py --edu-p /persistent/SatQuery evaluate \
  --prepared /persistent/SatQuery/text-adaptation/text-pairs \
  --checkpoint /persistent/SatQuery/text-adaptation/text-adapter/best.pt \
  --split test --name retrieval-test.json
```

Evaluation reports held-out retrieval rank over the declared candidate set. Test never promotes a
checkpoint. Similarity is not calibrated confidence. The current repository contains the workflow
and offline tests, but no successfully trained real BigEarthNet.txt adapter; the configured OpenAI
key used during integration returned HTTP 401.

## Task evaluation

`satquery.assistant.task_evaluation` joins explicit JSONL references/predictions by
`dataset + split + task + id`, rejects duplicates and split mismatches, and scores only fixed
`vqa`, `area`, and `change` records. It reports task counts, abstention/coverage, provisional
normalized exact match, class-fraction error, unit-specific numerical area MAE, and exact change
label-set match. It computes no invented joint score.

```bash
PYTHONPATH=src python -m satquery.assistant.task_evaluation \
  --references data/task-references.jsonl \
  --predictions data/task-predictions.jsonl \
  --output data/task-evaluation.json \
  --compatibility-output data/benchmark-compatibility.json
```

The generated compatibility manifest explicitly reports VRSBench, RSVQA, CDVQA, and the hidden
ISRO/SAC evaluation configuration as unsupported because their adapters/models/official scorers or
sensor profiles are not integrated.

## SIH26167 requirement status

| Requirement | Status on 14 September 2026 | Evidence or remaining work |
|---|---|---|
| Input upload and compatibility checking | **Implemented for the current Sentinel profile** | Guided TIFF and strict ZIP loaders; RGB benchmark files and Cartosat/RISAT profiles remain open. |
| Remote-sensing-adapted visual or vision-language component | **Visual specialist implemented; text adaptation workflow only** | Official pretrained CROMA plus BigEarthNet-derived coverage heads run. A successfully trained BigEarthNet.txt image-text adapter remains pending. |
| Single-image VQA | **Prototype coverage-question subset** | Deterministic land-cover coverage/presence questions work. A validated open-ended VQA model and official RSVQA/VRSBench evaluation remain pending. |
| Additional single-image captioning or grounding | **Prototype description/location tools** | Coverage summary and coarse class-cell GeoJSON work. A validated caption generator or text-conditioned grounding model remains pending. |
| Bi-temporal change description or change VQA | **Provisional prototype** | Aligned dated inputs yield coverage deltas. A benchmarked temporal change model and CDVQA adapter/scorer remain pending. |
| Co-registered optical–SAR pair analysis | **Implemented prototype** | Both tensors enter CROMA's joint encoder; optional optical/SAR/joint summaries are descriptive and do not prove fusion improves accuracy. |
| Agentic query routing and specialist orchestration | **Implemented bounded prototype** | Local rules or one OpenAI tool-routing call select an allowlisted specialist; execution trace is returned. |
| Visual evidence, confidence, trace, reports | **Evidence/trace/reports implemented; confidence pending** | Preview, grid/GeoJSON, model hashes and downloads work. Fractions/thresholds/similarities are not calibrated confidence. |
| Public benchmark evaluation | **Framework only** | Official VRSBench/RSVQA/CDVQA adapters and scorers, prescribed splits, and benchmark runs remain pending. |
| ISRO/SAC evaluation and target sensors | **Pending** | Direct Cartosat-2S/RISAT profiles, hidden SAC evaluation, and calibration remain pending. |

## Verification boundary

The three-sample optical, SAR, and joint heads used for integration are explicitly
`demo_fit_all`, training-data-only artifacts. They verify end-to-end wiring and input contracts;
they do not replace the recorded held-out 1,000-area coverage experiment and must not be quoted as
accuracy. The integration acceptance report is in
[`reports/assistant/integration.json`](../reports/assistant/integration.json).

The current open gaps are: a successfully trained BigEarthNet.txt image-text adapter; official
VRSBench/RSVQA/CDVQA evaluation adapters and scorers; a validated open-ended VQA/caption generator;
a benchmarked temporal change model; direct Cartosat-2S/RISAT sensor profiles; SAC evaluation; and
calibrated confidence for each supported task.
