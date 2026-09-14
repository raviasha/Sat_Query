# SatQuery application milestone

User-approved direction: preserve the existing coverage baseline and build the
query-driven application required by SIH26167. OpenAI is the selected language
provider; its key stays on the server. This milestone implements the application,
model integration, text-adaptation training path, and task evaluation path. A
trained/validated benchmark VLM and validated temporal model cannot be claimed
merely because their workflows exist.

## Design

Keep the existing download, tensor, feature, target and prediction modules intact.
Add a label-free inference boundary for documented Sentinel-2 and Sentinel-1
inputs, a model registry, deterministic spatial tools, an OpenAI Responses tool
controller, and a local FastAPI application with a browser interface. The GUI
must allow uploads and questions, show actual evidence and an observable trace,
and download JSON and visual evidence. No generated scripts are executed.

Inputs use a ZIP containing request.json plus explicitly mapped TIFF bands.
Each observation declares sensor, modality, acquisition date and band-to-file
mapping (optionally a band index into a multi-band TIFF). One optical, one SAR,
co-registered optical/SAR, or two dated observations of the same modality are
accepted. Validate CRS, footprint, metre resolution, channel order, radiometry,
finite valid pixels and temporal ordering. Do not infer missing spectral bands
from RGB or silently turn Cartosat/RISAT into Sentinel inputs. Unsupported
benchmarks/sensors get a specific compatibility result and remain explicit gaps.

CROMA supports single encoders, but each modality requires a head trained on the
corresponding feature key. Reuse frozen features to train these heads. Existing
three-sample demo weights may be used for a clearly marked integration demo;
they are never presented as the 5,000-area selected model or held-out evidence.

Tools answer land-cover presence, area, coarse location and description; paired
analysis uses joint features. Temporal analysis compares spatially aligned
predictions from genuinely dated inputs, with estimated change clearly separated
from validated change detection. Every answer includes model hashes, units,
thresholds and limitations. Area fractions are not confidence. An OpenAI answer
cannot invent a measurement that is absent from tool outputs.

Text adaptation is a separate reusable image-text projection experiment: pair
verified CROMA scene features with publisher caption embeddings, train only on
eligible training images, select with validation images, and evaluate retrieval
without using held-out captions as an inference answer bank. This is a modest
adapted image-text component, not a claim of open-ended benchmark VQA success.
Retain annotation provenance, known annotation errors, and split quarantine.

Task evaluation records task-level answers, errors and abstentions independently
of coverage MAE. Keep public benchmark ingestion and sensor/model compatibility
visible; do not invent official metric weights or report synthetic tests as
benchmark results.

## Acceptance

- Working browser upload/question/evidence/report flow, using the existing real
  CROMA checkpoint and compatible coverage head for a local sample.
- OpenAI tool routing exercised with a live API call when a key is available;
  fail visibly on API problems, with an explicit local mode available.
- Single/paired/temporal contracts and spatial calculations tested with small
  deterministic fixtures, including invalid/unaligned uploads and unsupported
  questions. No reference maps are required at inference.
- Reusable text-adaptation preparation/training/evaluation commands and provenance
  tests. Real training status is separate from implemented code status.
- Existing pipeline tests pass; README and architecture explain capabilities,
  installation and the remaining hackathon gaps.
