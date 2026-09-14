# SatQuery AI

**A modular satellite-image learning pipeline, with a vision-language assistant under design.**

SatQuery currently downloads selected BigEarthNet v2 optical/SAR image pairs, prepares tensors, extracts frozen CROMA features, generates aligned land-cover percentage targets, trains a coverage model, evaluates it on held-out areas, and matches BigEarthNet.txt annotations to the images and features.

The intended application is an agentic remote-sensing assistant for SIH26167: upload imagery, ask questions, describe scenes, locate regions, and compare dates. **This version has no conversational UI, web API, trained Q&A/captioning/grounding model, or temporal-change model.** The working product is a Python package and reproducible Colab workflows.

Read the [architecture and module guide](docs/architecture.md) for code navigation, interfaces and extension points.

## Current capabilities

Status: **10 September 2026**. The recorded 1,000-area experiment ran on 9 September.

| Functionality | Status | Implementation |
|---|---|---|
| Selectively download paired imagery and reference maps | Implemented; 1,000 areas downloaded | `download_bigearthnet.py`, `remote_lmdb.py` |
| Validate bands/grids/pairing and create normalized tensors | Implemented and run | `preprocessing.py`, `prepare_croma.py` |
| Extract optical, SAR and joint CROMA features | Implemented and run | `features.py`, `extract_croma.py` |
| Align 19 class fractions with every spatial token | Implemented; exhaustive geometry/count audit completed | `targets.py`, `prepare_targets.py` |
| Train a separate coverage head and select its epoch on validation | Implemented and run | `prediction.py`, `validation_training.py` |
| Test on held-out areas and report per-class errors/support | Implemented and run | `evaluation.py`, `colab_stage6_7.py` |
| Export predictions with matching historical reliability | Implemented and run | `predict_cover.py` |
| Match official text records to images and saved features | Implemented and run | `match_annotations.py`, `colab_match_annotations.py` |
| Parse boxes, pool region features and learn image–text alignment | Proposed; not implemented | No module yet |
| Q&A, captions, grounding, RAG, change analysis, controller and GUI | Pending | No modules yet |

Modules live in [src/satquery](src/satquery) or [scripts](scripts). Current inference produces **class fractions per spatial block**, not textual answers or pixel-level segmentation.

## Verified experiment and limitations

The subset preserves official assignments: **600 train / 200 validation / 200 test areas**, approximately 1.2 × 1.2 km each, with 100 areas from each of ten countries. This country-balanced subset is an initial experiment, not the full official benchmark distribution.

| Measure | Recorded result |
|---|---:|
| Paired areas / TIFFs | 1,000 / 15,000 |
| TIFFs per area | 12 optical + 2 SAR + 1 reference map |
| Spatial features | 225 per area, 768 values each |
| Eligible train / validation / test blocks | 134,255 / 44,848 / 44,723 |
| Test dominant-class accuracy | 53.87% |
| Mean coverage error over test blocks and 19 classes | 6.16 percentage points |
| Constant training-mean baseline error | 9.21 percentage points |
| Whole-area bootstrap 95% interval for aggregate coverage error | 5.88–6.40 percentage points |
| Selected epoch | 5; 15 epochs run with early stopping |

The independent spatial audit checked **14.4 million original reference pixels**, **225,000 block boundaries**, and **4.275 million class counts**. See the [complete results and class table](reports/pipeline-1000/results.md), [test report](reports/pipeline-1000/evaluation/test-report.json), and [per-class CSV](reports/pipeline-1000/per-class-metrics.csv).

- Dominant accuracy compares the largest predicted and reference fractions. It excludes 173 tied-reference blocks, leaving 44,550 categorical comparisons.
- **6.16 percentage-point error is not 93.84% accuracy.** Errors where a class actually occurs are often much larger than averages including absent classes.
- Coastal wetlands have no eligible positive test support; some other classes have very little support.
- Predicted fractions are coverage estimates, not presence probabilities. Historical class errors are not calibrated confidence for a new prediction.
- Exact alignment with CLC maps does not establish independent ground-truth accuracy at 10 m or 80 m.
- Future model/threshold choices should use validation data. Reserve fresh held-out data for a later final comparison after this examined baseline.

## Run with Colab and Google Drive

Run these notebooks in order. They embed a checksum-verified package wheel, so cloning inside Colab is unnecessary.

1. [Download subset](notebooks/SatQuery_BigEarthNet_Drive_Download.ipynb) — [open in Colab](https://colab.research.google.com/github/raviasha/Sat_Query/blob/main/notebooks/SatQuery_BigEarthNet_Drive_Download.ipynb). CPU is sufficient; mount your own Google Drive when prompted.
2. [Coverage pipeline](notebooks/SatQuery_Pipeline_1000.ipynb) — [open in Colab](https://colab.research.google.com/github/raviasha/Sat_Query/blob/main/notebooks/SatQuery_Pipeline_1000.ipynb). Run cells individually and inspect receipts. The recorded run used a free T4 GPU; CPU extraction is supported but slower.
3. [Text matching](notebooks/SatQuery_BigEarthNet_Text_Matching.ipynb) — [open in Colab](https://colab.research.google.com/github/raviasha/Sat_Query/blob/main/notebooks/SatQuery_BigEarthNet_Text_Matching.ipynb). CPU is sufficient; requires downloaded metadata and completed features.

Default Drive layout:

```text
MyDrive/SatQuery/
├── bigearthnet-v2-1000/          ZIP shards, combined metadata, source cache, receipts
├── annotation-source-cache/     Pinned official text table
└── pipeline-1000/
    ├── prepared/               Image tensors and reference maps
    ├── features/               Frozen CROMA outputs
    ├── targets/                Reference class fractions
    ├── croma-checkpoint/        Pretrained weights
    ├── model/                  Trained head and training history
    ├── evaluation/             Test predictions, metrics and reliability
    ├── predictions/            Inference exports
    ├── annotations/            Matched text and feature links
    └── stage-*-report.json      Completion/provenance receipts
```

The original download plus source cache used approximately **0.558 GB**. This excludes downstream tensors/features, the roughly **778 MB** CROMA checkpoint and the **466.8 MB** text table. Matched text outputs occupy about **14.5 MB**. Allow additional storage for intermediates.

Completed downloads and stages are checked and reused. Most core exporters require a **new output directory**; choose new experiment directories when changing data/configuration. Rerunning a completed notebook does not automatically retrain the model.

**Portability boundary:** package APIs support other subset sizes, but current Colab scripts assert 1,000 areas and a 600/200/200 split and use fixed paths. Adapt the scripts or call package APIs for a different experiment. They are not a general workflow scheduler.

## Local installation

Requires Python **3.11+** and [uv](https://docs.astral.sh/uv/). Public data/checkpoint downloads require network access; local processing does not after inputs are available.

```bash
git clone https://github.com/raviasha/Sat_Query.git
cd Sat_Query
uv sync --locked --no-editable --extra dev --extra downloads --extra features
```

Distribution name: `satquery-preprocessing`; import name: `satquery`. There is no server to start. In a separate environment, `pip install '.[dev,downloads,features]'` is also supported, but does not reproduce the uv lock exactly. Non-editable installation avoids an editable-import problem observed on the development Mac.

### Download and extract

```bash
uv run --no-sync satquery-download-bigearthnet \
  --destination data/subset-1000 --work data/download-work \
  --train 600 --validation 200 --test 200
```

Destination and temporary work must be separate directories, neither nested inside the other. The downloader uses selected HTTP byte ranges instead of fetching the entire 155 GB LMDB. See [sources and resume rules](docs/download-bigearthnet.md).

Extract ZIP shards into a raw-data root. Each ZIP includes only its own metadata: when combining shards, copy the download directory's combined `metadata.parquet` into the raw root **after extraction**. Preserve the nested archive hierarchy under `BigEarthNet-S2/`, `BigEarthNet-S1/` and `Reference_Maps/`.

### Prepare, extract features and build targets

Assuming extracted imagery and selected metadata under `data/raw`:

```bash
uv run --no-sync satquery-prepare-croma \
  --input data/raw --output data/prepared --batch-size 32

uv run --no-sync satquery-extract-croma \
  --input data/prepared --output data/features --device cpu --batch-size 4

uv run --no-sync satquery-prepare-targets \
  --input data/prepared --features data/features --output data/targets
```

CROMA also supports `--device cuda` or `--device mps` on compatible hardware. The first run downloads the pinned checkpoint unless `--checkpoint` supplies an existing verified copy. Use each command's `--help` for filtering, limits and metadata options.

### Train and evaluate

Two routes exist:

- **Validation-selected experiment:** `evaluation.load_splits` → `validation_training.fit_with_validation` → `prediction.save_head` → held-out `evaluation.coverage_metrics`. The pipeline notebook implements this route with full provenance and reliability reports. A standalone CLI for this complete workflow is pending.
- **Simple training CLI:** the following trains on official training areas and reports fit metrics. It does not perform validation epoch selection or held-out evaluation.

```bash
uv run --no-sync satquery-train-head \
  --features data/features --targets data/targets --output data/model
```

The explicit `--demo-fit-all` option exists for the historical three-area smoke demonstration. It trains across supplied splits and must not be used to claim held-out performance. The original three demo areas are excluded from the 1,000-area subset.

### Predict

With a compatible saved coverage head:

```bash
uv run --no-sync satquery-predict-cover \
  --features data/features --checkpoint data/model/head.pt \
  --output data/predictions
```

Optionally append `--reliability data/evaluation/reliability.json` when a matching report has been generated through evaluation. The exporter checks checkpoint hash, feature contract and class order.

For new images, repeat preprocessing and feature extraction with the same profile. **The current raw loader requires reference maps**; an inference-only image loader without labels is pending. The prediction module itself consumes features and requires no targets.

## Data and model contracts

| Artifact | Shape for batch size N |
|---|---|
| Optical input | float32 `[N,12,120,120]` |
| SAR input | float32 `[N,2,120,120]` |
| Original reference map | int64 `[N,120,120]` |
| Each optical/SAR/joint spatial output | `[N,225,768]` |
| Each scene-level feature output | `[N,768]` |
| Class-fraction targets/predictions | `[N,225,19]` |

- Optical order: **B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12**. SAR: **VV, VH**. See [channel evidence and qualifications](docs/channel-order.md).
- B02 defines the 120 × 120, 10 m grid. Coarse optical values are cast to float32 before bilinear resampling. Masks, finite values, CRS, orientation and bounds are validated. SAR gets no additional log transform.
- Normalization is per image/channel: mean ± 2 sample standard deviations, clipping, uint8 quantization, then float32 /255. Constant channels map to 127/255. It is independent of batch composition, not a claim of exact original benchmark preprocessing.
- Tokens form a **15 × 15 grid**, each anchored to 8 × 8 pixels or **80 × 80 m** here. Index: `row * 15 + column`. CROMA attention includes wider context; token features are not exclusively local measurements.
- Targets use the official 19-class mapping. Fractions are counts /64; unknown/excluded labels retain their share rather than being redistributed. Current training/evaluation uses fully labeled blocks only.
- The coverage model is a separate `Linear(768,19)` head with softmax fractions. CROMA stays frozen. Validation training fits feature scaling on training data only and folds it into the saved head.
- Optical/SAR-only features are saved, but the recorded baseline uses joint features. A controlled fusion-benefit comparison remains pending.

Class names/codes are defined in [targets.py](src/satquery/targets.py). Manifests preserve sample identities, spatial contracts, source revisions and hashes; consumers must not pair artifacts by incidental file order.

## Text annotations

The matcher trains no model. The notebook downloads the official table once and selects annotations by both Sentinel IDs.

```bash
uv run --no-sync python -m satquery.match_annotations \
  --metadata data/raw/metadata.parquet \
  --source /path/to/BigEarthNet.txt.parquet \
  --output data/annotations
```

The source file must already exist; its download is implemented in `scripts/colab_download_annotations.py` and the notebook. The core matcher writes `annotations.parquet`, `image-annotations.jsonl` and `report.json`. **Feature links are added by `colab_match_annotations.py`, not by the matcher CLI.**

| Annotation type | Records across all matched partitions |
|---|---:|
| Binary Q&A | 7,529 |
| Multiple-choice Q&A | 6,818 |
| Scene captions | 955 |
| Text-referenced bounding boxes | 2,477 |
| Point-guided bounding boxes | 2,674 |
| **Total** | **20,453** |

955 images matched; 45 have no records in the pinned source. Matched image partitions are **569 train / 197 validation / 188 test / 1 benchmark-held-out**. There are **12,090 training records**; the table above is not training-only.

Original strings, including boxes, remain unchanged. Boxes are **not yet parsed or mapped to tokens**. Scene captions cannot be copied onto every 80 m block. Point-guided instructions may not name a class. Missing annotations are not negative labels.

Any benchmark annotation holds out its whole image; other image/text split disagreements are quarantined. See the [matching receipt](reports/pipeline-1000/stage-8-annotations-report.json).

## Pending work

These are design directions and known gaps, **not implemented features or a finalized implementation plan**.

| Work area | Pending outcome |
|---|---|
| Annotation semantics and geometry | Validate coordinates, box sizes/shapes/overlaps, class phrases and scope; retain provenance for derived training records. |
| Region learning | Compare pooled and spatial representations; learn appearance/alignment from verified named regions; handle mixed boxes and multiple valid matches. |
| Presence and area Q&A | Establish coverage-based baselines, class synonyms and validation-selected thresholds; compare with learned answers. Nonzero coverage alone is not presence. |
| Counting and spatial relationships | Learn connected regions, adjacency and relative position; coverage totals alone are insufficient. |
| Multiple-choice Q&A | Preserve options and underlying task meaning; test option reordering and question-only shortcuts. |
| Grounding | Train full-image + text → box and full-image + point → box. Evaluate predicted regions and absent-target requests without supplying reference boxes at inference. |
| Captioning / language adaptation | Train a remote-sensing-adapted language component with spatial features and realistic specialist evidence; check factuality. |
| Metadata questions | Route date/location/climate questions to permitted metadata or tools; avoid filename and benchmark-answer leakage. |
| Optional RAG | Compare train-only scene/region retrieval with adapted models; learn alignment for direct image–text comparison. Similar examples do not establish current-image facts. |
| Coverage quality and uncertainty | Expand class/geographic support, compare models on validation, calibrate task-specific reliability and reserve a final held-out evaluation. |
| Single-sensor / fusion | Compare optical-only, SAR-only and joint variants; demonstrate whether SAR helps; support image loading without reference labels. |
| Temporal analysis | Obtain genuine aligned dated pairs, train change description/VQA and evaluate on prescribed temporal data. Optical/SAR pairing is not temporal change. |
| Sensor portability | Add appropriate Cartosat-2S/RISAT and approved RGB benchmark profiles/adaptation; do not fabricate Sentinel bands. |
| Agentic backend | Implement query interpretation, compatibility checks, permitted tool registry, routing, evidence combination and observable execution summaries. |
| User application | Add API, upload/preview, text questions, region selection, overlays and downloadable reports. |
| Experiment tooling | Replace fixed Colab paths/counts with configuration; add validation/evaluation CLIs, scalable streaming and model artifact distribution. |
| Delivery | Resolve VRSBench/RSVQA/CDVQA protocols, run compliant evaluations, add CI/deployment and decide a project license. |

The proposed architecture retains the coverage specialist and adds spatial image–language learning. Preserve full feature grids alongside optional pooled vectors. See [proposed extension boundaries](docs/architecture.md#proposed-extensions-not-implemented).

## Development and verification

After installing all extras:

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
uv build --wheel
```

Tests cover preprocessing, export contracts, taxonomy/alignment, features, training, metrics, prediction provenance, selective downloads and text matching. Some integration tests require the original local three-area fixture and skip without it. Unit tests do not repeat the 1,000-area cloud experiment; its historical receipts are included separately.

Publication verification on 10 September 2026: **93 tests passed**, Ruff passed, and the wheel built successfully. The test run emitted 44 Rasterio/Affine pending-deprecation warnings. All three notebooks were rebuilt; their code cells parse, outputs are empty, and embedded package source matches the current repository. The cloud experiment was not rerun for this documentation release.

Rebuild notebooks after package or stage-script changes:

```bash
uv build --wheel
uv run --no-sync python scripts/build_download_notebook.py
uv run --no-sync python scripts/build_annotation_notebook.py
```

The annotation builder also rebuilds the main pipeline notebook. Builders embed the current package wheel and checksum, not imagery, trained weights or credentials. Colab uses its runtime/dependency installation rather than the local uv lock.

Git contains source, tests, notebooks, documentation, the lockfile and small reports. Datasets, prepared tensors, checkpoints, predictions, caches and virtual environments are excluded. **Cloning does not retrieve the recorded Drive artifacts or trained head.** Recreate them with the workflows or supply compatible artifacts.

## Documentation and provenance

- [Architecture and codebase guide](docs/architecture.md)
- [Selective downloading](docs/download-bigearthnet.md)
- [Channel-order evidence](docs/channel-order.md)
- [CROMA feature API](docs/croma-features.md)
- [Reference-map targets](docs/patch-targets.md)
- [Coverage head / simple training](docs/prediction-layer.md)
- [Historical 1,000-area results](reports/pipeline-1000/results.md)
- [Historical design notes](docs/superpowers) — current status is summarized here and in the architecture guide.

Sources: [BigEarthNet](https://bigearth.net/), [BigEarthNet.txt](https://txt.bigearth.net/), [official CROMA code](https://github.com/antofuller/CROMA) and [weights](https://huggingface.co/antofuller/CROMA). Downloads use pinned [unofficial native-array LMDB](https://huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB) and [metadata/reference-map](https://huggingface.co/datasets/torchgeo/bigearthnet) mirrors. TIFFs are reconstructed from native arrays; headers need not match originals. All 14 band arrays were checked against originals for one demo area, not every mirrored record.

Vendored CROMA retains its [MIT license](src/satquery/_vendor/CROMA_LICENSE) and [provenance](src/satquery/_vendor/provenance.json). Upstream data/model licenses apply separately. The project's original code does not yet have a specified repository license.

## Nonlinear head comparison

[Experiment guide](docs/nonlinear-comparison.md) · [Colab notebook](notebooks/SatQuery_Nonlinear_Comparison.ipynb). Reuses the existing CROMA features and coverage targets to compare linear and MLP heads across three seeds.
