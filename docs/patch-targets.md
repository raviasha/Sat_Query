# Match CROMA features to land-cover percentages

> Status note (10 September 2026): this guide retains the original three-area API/demo examples. The 1,000-area pipeline, validation-selected training, held-out evaluation and annotation matching have since completed. See the [current README](../README.md), [architecture](architecture.md) and [recorded results](../reports/pipeline-1000/results.md). Counts and limitations in the historical demo sections refer to those earlier runs.

This third module reads the prepared integer reference maps and the feature export. It assigns a 19-class coverage vector to every CROMA spatial token, without running CROMA again. The taxonomy follows Table 1 of the [official BigEarthNet v2 description](https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf).

## Target definition

Each 120×120 reference map is divided into 15×15 non-overlapping blocks of 8×8 pixels. For the prepared 10 m grid, each block is 80×80 m and contains 64 equal-area reference-map pixels. Token index is `row * 15 + column`, exactly matching the CROMA feature export.

For class `k`, `class_fractions[..., k] = class_counts[..., k] / 64`. Multiply by 100 for percentages. For example, 32 broad-leaved forest pixels, 16 coniferous forest pixels and 16 unlabeled pixels give 50%, 25%, and 25% unlabeled. Known classes are **not** renormalized to hide the missing coverage. Fractions are exact multiples of 1/64, so percentage resolution is 1.5625 percentage points.

The 19 classes have a fixed order in the output manifest, including classes absent from this small sample. Several original CLC codes merge into one class. Other CLC codes are excluded by the official 19-class mapping and count as unlabeled, as do the official unlabeled code 999 and preserved raster nodata code 0. Unexpected codes cause an error.

These are percentages of the supplied reference-map labels. Rasterizing the CLC labels onto a 10 m grid does not establish independent ground-truth accuracy at 10 m or 80 m. The module does not infer class proportions from the scene-level label list.

## Run

```bash
uv sync --no-editable --locked --extra dev --extra features
uv run --no-editable satquery-prepare-targets \
  --input data/croma/three-samples \
  --features data/features/three-samples \
  --output data/targets/three-samples
```

If the project environment has filesystem/import problems, use a separate environment for both commands by setting `UV_PROJECT_ENVIRONMENT` to a local directory outside the project, for example `export UV_PROJECT_ENVIRONMENT="$HOME/.cache/satquery/venv"`.

Additional prepared batches use the same command. The output must be new and outside both input directories. The exporter checks the feature export's source-manifest hash, batch order, sample IDs and metadata, feature-file and reference-map hashes, token grid, and the projected 10 m ground grid before publishing targets. It preserves source batches and uses only one batch at a time.

## Outputs

Each output batch contains `targets.pt` and `batch.json`. The top-level `manifest.json` records class order, CLC mapping, denominator and source/output hashes; `validation.json` records completed checks and coverage counts.

| Tensor key | Shape for three areas | Meaning |
| --- | --- | --- |
| `class_counts` | `(3,225,19)` | Integer count of each class's pixels |
| `class_fractions` | `(3,225,19)` | Float32 coverage on a 0–1 scale |
| `unlabeled_counts` | `(3,225)` | Pixels without a retained 19-class label |
| `unlabeled_fraction` | `(3,225)` | Unlabeled share of the full block |
| `fully_labeled_mask` | `(3,225)` | True when all 64 pixels have retained labels |
| `has_labels_mask` | `(3,225)` | True when at least one retained label is present |
| `bounds` | `(3,225,4)` | Float64 `[left,bottom,right,top]` in each sample's projected CRS |

`class_fractions.sum(-1) + unlabeled_fraction` equals 1 for every token. Fully unlabeled blocks have zero class fractions and both masks false. No tokens are silently dropped. Patch IDs, SAR IDs, dataset split, georeferencing and feature-file linkage are retained in `batch.json`.

## Use in the next ML step

```python
import torch

features = torch.load("data/features/three-samples/features.pt", weights_only=True)
targets = torch.load("data/targets/three-samples/targets.pt", weights_only=True)

X = features["joint_encodings"].reshape(-1, 768)  # all 675 spatial feature vectors
y = targets["class_fractions"].reshape(-1, 19)   # same 675 sample/token positions
mask = targets["fully_labeled_mask"].reshape(-1)
X_complete, y_complete = X[mask], y[mask]
```

This creates fractional coverage targets for a later model, rather than choosing a single majority label. The fully labeled mask gives an explicit subset whose 19 fractions sum to one. If using partially labeled blocks, define a missing-label-aware objective; do not treat their omitted coverage as a known land-cover class. Preserve geographic/sample grouping when splitting later datasets. The present three adjacent areas retain their official **test** split and serve as a pipeline check.

For tensor-only callers, `satquery.targets.patch_label_targets(reference_maps)` accepts integer `(N,120,120)` maps and returns the six coverage/count/mask tensors. `satquery.targets.export_targets(prepared, features, output)` adds feature alignment, ground bounds and provenance.

## Verified three-area result

The exported targets are `(3,225,19)`, paired with the existing `(3,225,768)` spatial features. All 675 tokens have full 19-class label coverage; 159 contain more than one class. Every target count was independently checked against its original 8×8 slice of the prepared reference-map tensor.

For example, sample 0, token 13 (row 0, column 13) has 28 arable-land pixels and 36 broad-leaved-forest pixels: **43.75% arable land and 56.25% broad-leaved forest**. All other class percentages for that token are zero.

Verification: all 70 repository tests and Ruff checks pass; the package wheel builds successfully. Independent code review verified the taxonomy, denominator and spatial alignment. All 45 original TIFFs also matched their preprocessing provenance hashes. This run used the separate local environment described above; initial file-access stalls/errors cleared after fully reading the affected files, without changing their contents.
