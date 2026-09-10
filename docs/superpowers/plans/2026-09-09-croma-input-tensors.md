# CROMA Input Tensors Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Convert our three paired BigEarthNet v2 areas into validated optical and SAR input tensors, with reference maps kept separately.

**Architecture:** A reusable loader resolves metadata into band files and reads aligned raw tensors. A normalization function transforms each patch independently; a CLI exports batched tensors and provenance.

**Tech Stack:** Python, Rasterio, NumPy, pandas/PyArrow, PyTorch, pytest. Use an isolated environment with mutually compatible versions and record resolved versions; do not alter global packages.

**Spec:** `docs/superpowers/specs/2026-09-09-croma-input-tensors-design.md` (accepted for implementation on 2026-09-09).

## Implementation outcome

Implemented on 2026-09-09. The suite has 23 passing tests; formatting and lint checks pass. The real three-sample export is in `data/croma/three-samples/`. A separate lazy Dataset and CLI support additional metadata selections and bounded tensor batches.

Refinements made during implementation:
- Exact source-dataset band order is verified at a pinned revision. Applying it to CROMA is explicitly recorded as an inference from its pretraining source, not checkpoint-specific proof; see `docs/channel-order.md`. The binary verification flag was replaced with this evidence level. Custom unverified orders are allowed only for raw export.
- Cast optical rasters to float32 before bilinear resampling: integer-source Rasterio reads otherwise round interpolated fractions. A uint16-ramp regression verifies 9.75 rather than 10.0.
- Detect identical fractional pixels directly before standard-deviation reduction; float32 roundoff otherwise misses constant channels. Three fractional-constant regressions verify the 127/255 fallback.
- Preserve reference-map unlabeled code 0 even when declared nodata. Invalid imagery pixels remain errors.
- Export multiple batches when the selected image count exceeds `--batch-size`; source metadata remains in memory.

## Global constraints

- Preserve original TIFFs and metadata; write generated artifacts to a separate output directory.
- Process on CPU using float32 imagery; preserve floating-point SAR measurements when reading.
- Pair samples using metadata identifiers, never directory iteration order.
- Keep reference-map class codes separate from the CROMA imagery inputs. Read maps as integers without interpolation or normalization.
- Do not claim checkpoint compatibility from tensor dimensions alone: verify the channel-order profile first.
- This plan contains no model-weight download, inference, training, or full-dataset download.

## Files and interfaces

Create `src/satquery/preprocessing.py` for sample discovery, raster loading and normalization, `src/satquery/prepare_croma.py` for export, and `src/satquery/__init__.py`. Add `pyproject.toml`, `.gitignore`, `README.md`, and `tests/test_preprocessing.py` as part of the associated tasks. Ignore downloaded/generated data and environments in Git.

Use `PatchRecord(patch_id: str, s1_name: str, split: str, country: str)` and `ChannelProfile(optical: tuple[str, ...], sar: tuple[str, ...], evidence: str, verified: bool)` data classes in `preprocessing.py`.

Interfaces:

```python
read_records(root: Path) -> list[PatchRecord]
load_raw_patch(root: Path, record: PatchRecord,
               profile: ChannelProfile) -> dict
normalize_patch(image: torch.Tensor) -> tuple[torch.Tensor, dict]
export_inputs(root: Path, output: Path, profile: ChannelProfile) -> dict
```

`load_raw_patch` returns `optical_images`, `SAR_images`, `reference_map`, and JSON-serializable `provenance`. `normalize_patch` accepts one `(C,H,W)` raw tensor and returns normalized imagery plus statistics. `export_inputs` returns the saved manifest. Batch assembly uses `torch.stack` separately for each modality, in metadata order.

### Task 1: Establish the channel profile and metadata pairing

Files: `pyproject.toml`, `.gitignore`, `src/satquery/__init__.py`, `src/satquery/preprocessing.py`, `tests/test_preprocessing.py`, `README.md`.

- [x] Inspect the official benchmark collection and author documentation for exact optical and SAR channel sequences. Record an immutable source revision and the evidence in `ChannelProfile`. If evidence is unavailable, support only a labeled raw export and report the unresolved compatibility requirement.
- [x] Set up an isolated environment and package with the libraries listed above, then record the resolved versions.
- [x] Add failing tests for metadata-based pairing, deterministic row order, duplicate metadata IDs, and exact band membership. Create tiny temporary TIFF fixtures whose band values differ, so an alphabetical B8A misordering or VV/VH swap is observable.
- [x] Implement `PatchRecord`, `ChannelProfile`, `read_records`, and explicit band-path resolution. Reject missing, duplicated, or unexpected imagery bands. Error messages must include patch ID and band name.
- [x] Run `python -m pytest tests/test_preprocessing.py -k 'pairing or order or missing or duplicate' -v`; require all selected tests to pass.

Example fixture assertion (the fixture writes constants 1 through 12 in its declared optical sequence):

```python
patch = load_raw_patch(root, record, profile)
assert patch['optical_images'][:, 0, 0].tolist() == list(range(1, 13))
assert patch['SAR_images'][:, 0, 0].tolist() == [-12.25, -18.75]
```

### Task 2: Load and align raw imagery and label maps

Files: `src/satquery/preprocessing.py`, `tests/test_preprocessing.py`.

- [x] Add failing cases for shifted transforms despite identical shapes, differing CRS, unreadable/masked/nonfinite imagery, and fractional SAR values. Assert that an unchanged native 10 m band is numerically identical after loading.
- [x] Implement `load_raw_patch`. Read B02's CRS, transform and bounds as the target; require 120 × 120 pixels at 10 m. Check each original grid against the same footprint and expected native resolution. Read coarse optical rasters to the target shape with Rasterio bilinear resampling and float32 output. Read SAR directly as float32 and reference maps as integers. Do not silently reproject mismatched footprints.
- [x] Use a synthetic coarse constant raster to verify upsampling preserves its value; use a spatial ramp to verify orientation and interior interpolation. Compare the implementation against an independently calculated expected interior value rather than repeating its resampling call in the assertion.
- [x] Run `python -m pytest tests/test_preprocessing.py -k 'grid or resample or sar or invalid or map' -v`; require all selected tests to pass.
- [x] Run the loader on all three real metadata rows and check `(12,120,120)`, `(2,120,120)`, and `(120,120)` outputs, including matching footprints and unchanged map class-code sets.

### Task 3: Normalize patches reproducibly

Files: `src/satquery/preprocessing.py`, `tests/test_preprocessing.py`.

- [x] Add tests for a known small array, constant channels, clipping, nonfinite input rejection, and independence from other samples.
- [x] Implement the spec's patchwise mean ± 2 sample-standard-deviations transformation. Quantize to uint8 by truncation after clipping, then convert to float32 / 255. Handle constant channels explicitly and record statistics and constant-channel flags.
- [x] Compare nonconstant test inputs against the official README calculation with N=1. This establishes equivalence for the chosen patchwise policy, not for the README's arbitrary larger-batch statistics.
- [x] Run `python -m pytest tests/test_preprocessing.py -k normalize -v`; require all selected tests to pass.

Concrete edge-case and numeric assertions:

```python
x = torch.tensor([[[0., 1.], [2., 3.]]])
y, stats = normalize_patch(x)
sigma = (5 / 3) ** 0.5  # sample standard deviation of 0,1,2,3
expected = (((x - (1.5 - 2*sigma)) / (4*sigma)).clamp(0, 1)
            * 255).to(torch.uint8).float() / 255
torch.testing.assert_close(y, expected)
constant, stats = normalize_patch(torch.full((2, 120, 120), -7.))
torch.testing.assert_close(constant, torch.full_like(constant, 127/255))
```

### Task 4: Export and validate the three-sample batch

Files: `src/satquery/prepare_croma.py`, `tests/test_preprocessing.py`, `README.md`.

- [x] Add an integration test that calls `export_inputs` on the existing small-sample directory and a fresh temporary output directory. Assert batch IDs retain metadata order and saved/reloaded tensors are identical. Add an error case for an existing output directory so earlier artifacts cannot be silently overwritten.
- [x] Implement a CLI using `--input` and `--output`. Reuse the functions above, assemble batches with `torch.stack`, and write all artifacts specified in the design. Use temporary outputs and rename only after checks pass. Include hashes, channel evidence, normalization statistics and source georeferencing in the manifest.
- [x] If the profile is unverified, require an explicit `--raw-only` mode and omit `croma_inputs.pt`; the default command must explain why compatibility is unresolved instead of silently exporting model-ready inputs.
- [x] Document a single command and a short example loading `croma_inputs.pt` and passing its dictionary keys to CROMA. Explain that model invocation is a subsequent phase.
- [x] Run the complete suite once after implementation: `python -m pytest -q`.
- [x] Run the proposed command below on the real samples and inspect `validation.json`, output shapes, range, dtype, pairing, and reload equality.

```bash
python -m satquery.prepare_croma \
  --input data/bigearthnet-v2/small-sample \
  --output data/croma/three-samples
```

Final checks for the exported batch:

```python
batch = torch.load(output / 'croma_inputs.pt', weights_only=True)
assert batch['optical_images'].shape == (3, 12, 120, 120)
assert batch['SAR_images'].shape == (3, 2, 120, 120)
for tensor in batch.values():
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert 0 <= tensor.min() <= tensor.max() <= 1
```

## Review and handoff

The plan covers pairing, source validation, channel order, alignment, float SAR preservation, normalization, constant/invalid data, batching, separate labels, artifact provenance, and meaningful synthetic and real-data checks. Execution is complete for input preparation. A successful export establishes input-preparation checks; model behavior remains untested until an actual CROMA inference run.
