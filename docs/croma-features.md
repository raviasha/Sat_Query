# Extract CROMA features from prepared tensors

> Status note (10 September 2026): this guide retains the original three-area API/demo examples. The 1,000-area pipeline, validation-selected training, held-out evaluation and annotation matching have since completed. See the [current README](../README.md), [architecture](architecture.md) and [recorded results](../reports/pipeline-1000/results.md). Counts and limitations in the historical demo sections refer to those earlier runs.

This is the second stage of the pipeline. It reads the outputs of preprocessing and runs **pretrained CROMA-Base**. It does not read TIFFs, resample bands, normalize inputs again, or train any model.

## Install and run

```bash
uv sync --no-editable --locked --extra dev --extra features
uv run --no-editable --extra features satquery-extract-croma \
  --input data/croma/three-samples \
  --output data/features/three-samples \
  --device cpu --batch-size 2
```

The first run downloads the official 778 MB Base checkpoint to `data/models/croma/`. Later runs reuse it after verifying its SHA-256. You can supply an existing official checkpoint with `--checkpoint /path/to/CROMA_base.pt`. `--cache-dir` changes the default download directory. The checkpoint is ignored by Git.

`--device mps` uses Apple Silicon acceleration; `--device cuda` uses an available NVIDIA GPU. CPU is the portable default. Choose a smaller `--batch-size` if device memory is limited. Output directories must be new and outside the input directory.

For a larger collection, point `--input` at the top-level output of the preprocessing command. The feature exporter consumes its batch index, verifies input tensor/metadata hashes and sample order, and preserves separate output batches. Inference mini-batches are bounded by `--batch-size`; one full prepared tensor batch and its resulting features remain in host memory at a time. No dataset size or sample IDs are hard-coded.

## Output tensors

For the present three areas, `features.pt` contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `optical_encodings` | `(3,225,768)` | Optical spatial features |
| `SAR_encodings` | `(3,225,768)` | SAR spatial features |
| `joint_encodings` | `(3,225,768)` | Fused SAR–optical spatial features |
| `optical_GAP` | `(3,768)` | Optical mean pooling followed by the pretrained feedforward head |
| `SAR_GAP` | `(3,768)` | SAR mean pooling followed by the pretrained feedforward head |
| `joint_GAP` | `(3,768)` | Mean-pooled joint spatial features |

All saved feature tensors are CPU float32 values. Feature values are not clipped or L2-normalized; they are the official model outputs.

`batch.json` retains the ordered patch/SAR IDs, source georeferencing and the spatial-grid convention. Token index is `row * 15 + column`. Reshape `joint_encodings` to `(N,15,15,768)` when a spatial grid is useful. Each token is anchored to an 8×8 input region, about 80×80 m for these data, but attention adds context from elsewhere in the image. Do not interpret a token as measurements exclusively within its 80 m cell.

`manifest.json` records the model and checkpoint hashes, upstream revision, device, batch size, source manifest hash and feature-file hashes. `validation.json` records the completed shape, finiteness, pairing and save/reload checks. Original input tensors and label maps are unchanged.

## Reusable Python interface

```python
from pathlib import Path
import torch
from satquery.features import CromaFeatureExtractor, extract_features

extractor = CromaFeatureExtractor.from_pretrained(device="cpu")
inputs = torch.load("data/croma/three-samples/croma_inputs.pt", weights_only=True)
features = extractor(inputs["optical_images"], inputs["SAR_images"], batch_size=2)
joint_grid = features["joint_encodings"].reshape(-1, 15, 15, 768)

# Alternatively export a complete preprocessing output with provenance:
# extract_features(Path("/path/to/prepared"), Path("/path/to/new-features"), extractor)
```

The direct tensor interface expects already-normalized, paired float32 inputs in [0,1], with shapes `(N,12,120,120)` and `(N,2,120,120)`. It cannot infer band names from tensor values; callers must honor the documented profile. The manifest-based exporter checks the declared order and normalization profile as well.

## Model source and limits of validation

- Official inference code: [antofuller/CROMA at 59505a6](https://github.com/antofuller/CROMA/blob/59505a6bcadbf36ba20767270154bf9f3067c5e7/use_croma.py).
- Official Base weights: [Hugging Face at 0dd28e3](https://huggingface.co/antofuller/CROMA/blob/0dd28e3d633bd6715856ae9890e8c49360040598/CROMA_base.pt).
- Expected checkpoint SHA-256: `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63` (publisher's LFS object digest).

The MIT-licensed code is vendored in `src/satquery/_vendor/`, with its license and provenance. The model architecture and forward calculations are unchanged. The only loader adaptation reads the checkpoint once, explicitly on CPU using `weights_only=True`, before device placement.

Running the model successfully verifies that the prepared inputs execute through the actual checkpoint and produce finite features. It does not establish downstream classification accuracy or independently prove spectral channel semantics. The [channel-order evidence](channel-order.md) remains source-dataset verification with inferred applicability to CROMA. The three areas are still a small adjacent test-split sample.

## Verified local run

The three prepared areas were processed on 2026-09-09 using the verified official Base checkpoint on CPU, with batch size 2. Results are saved under `data/features/three-samples/`. All six output tensors have the shapes above, contain finite float32 values, and differ across the three areas. Sample linkage, file hashes, save/reload equality and joint mean pooling were checked. The automated suite passes all 33 tests.
