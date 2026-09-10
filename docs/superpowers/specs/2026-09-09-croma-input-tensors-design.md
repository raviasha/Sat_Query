# Proposed design: BigEarthNet TIFFs to CROMA tensors

Status: accepted and implemented on 2026-09-09. This document records the original design; resolved channel-order evidence and implementation refinements are documented in `docs/channel-order.md` and the implementation plan.

## Purpose and approach

Build a small reusable Python loader with a command-line exporter. Start with the three downloaded BigEarthNet v2 areas and allow the same functions to process additional metadata rows later.

A standalone notebook would be easy to explore but harder to reuse and validate. A full training DataModule adds unnecessary scope now. A reusable loader plus a small export command provides inspectable outputs and a path to a future PyTorch Dataset.

Input directory: `data/bigearthnet-v2/small-sample/`. Its `metadata.parquet` identifies three optical patches and the matching SAR patches using `patch_id` and `s1_name`. There are 12 optical bands, two SAR bands, and one reference map per area. The existing verification report records successful reads and matching bounds/CRS for all 45 TIFFs.

## Data contract

- Preserve original TIFFs and metadata; write generated artifacts to a separate output directory.
- Process on CPU using float32 imagery; preserve floating-point SAR measurements when reading.
- Pair samples using metadata identifiers, never directory iteration order.
- Use the optical B02 raster as the target 120 × 120, 10 m grid.
- Require matching CRS and ground bounds, valid north-up transforms, and the expected native grids; reject unexpected alignment rather than silently repairing it.
- Read optical bands on the target grid using bilinear resampling. Keep native 10 m pixels unchanged. Bilinear interpolation is a project choice, not a claim of exact reproduction of the authors' benchmark preprocessing.
- Require the SAR bands to match the target grid; no new logarithmic conversion or integer cast is applied without source evidence about the existing SAR units.
- Keep reference-map class codes separate from the CROMA imagery inputs. Read maps as integers without interpolation or normalization.
- Reject missing/duplicate bands, invalid pixels, nonfinite values, empty selections, or duplicate metadata IDs with an error naming the patch and file.

Per area: optical `(12, 120, 120)`, SAR `(2, 120, 120)`. For the three areas: optical `(3, 12, 120, 120)`, SAR `(3, 2, 120, 120)`. The first dimension follows the metadata row order for both modalities.

## Channel order: evidence and implementation resolution

The official model interface confirms channel counts, but the README and `use_croma.py` inspected on 2026-09-09 do not explicitly name the channel sequence. Shape checks alone cannot establish correct spectral ordering.

The implementation inspected author-provided materials and traced the channel sequence to the pinned SSL4EO-S12 source loader. See `docs/channel-order.md` for direct evidence and the explicitly inferred CROMA connection. The conventional candidate optical order is B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12; SAR candidates must explicitly distinguish VV/VH from VH/VV. Do not infer either sequence from filename sorting.

The resolved source-backed profile is used for normalized export, with `source_dataset_verified_croma_order_inferred` recorded in its manifest. This is a documented refinement of the original conservative raw-only fallback. Custom unverified profiles remain restricted to raw export, and checkpoint-specific inference validation remains a separate phase.

## Normalization proposal

Use the authors' demonstrated mean ± two-standard-deviations clipping and 8-bit quantization path, followed by float32 conversion and division by 255. Apply it independently to each patch and channel, then batch the results. Use the same sample-standard-deviation convention as the reference PyTorch call.

This matches applying the README function separately with N=1; the README example otherwise aggregates statistics across its supplied batch. Per-patch processing is an explicit project choice that makes a patch independent of its batch companions. It is not a claim of exact reproduction of the original training distribution.

For a zero standard deviation, define the normalized value as 0.5 before quantization (127/255 afterward), record the constant channel, and avoid division by zero. Reject invalid pixels instead of estimating statistics from them. Save the per-patch/channel statistics and profile name `croma_readme_patch_8bit_v1` in the manifest.

## Outputs and interfaces

- `raw_inputs.pt`: a dictionary with `optical_images` and `SAR_images` containing aligned, unnormalized CPU tensors.
- `croma_inputs.pt`: the same dictionary keys, containing normalized CPU float32 tensors in [0, 1], once the channel profile is verified.
- `reference_maps.pt`: integer tensor `(3, 120, 120)` in unchanged CLC codes.
- `manifest.json`: ordered patch IDs and SAR IDs, split/country, channel sequences and their evidence, source file hashes, target CRS/transform, normalization statistics, source revision, package versions, output shapes/dtypes, and artifact hashes.
- `validation.json`: checks and results, including whether checkpoint compatibility has been established.

The model interface is `model(SAR_images=batch['SAR_images'], optical_images=batch['optical_images'])`. CROMA creates its own 8 × 8 patches internally; this loader does not patchify images.

## Acceptance criteria and scope

All three areas produce correctly paired, ordered tensors with the specified shapes. Original 10 m imagery is preserved in raw outputs. Coarser imagery matches the target geospatial grid. Normalization is finite, reproducible, and independent of batching. Maps retain their original class codes. Saved artifacts reload identically. A manifest connects every tensor position to its source files.

Use small synthetic tests for channel ordering, interpolation, alignment errors, missing bands, SAR fractional-value preservation, invalid pixels, and normalization, plus one integration test over the actual three samples.

This phase prepares inputs. Downloading model weights, running CROMA, producing embeddings, reducing labels to 8 × 8 targets, and training a classifier are later phases. All three current samples belong to the official test split and must not be presented as a representative training set.

## Sources inspected

- Official preprocessing example: https://github.com/antofuller/CROMA/blob/main/README.md
- Official model interface: https://github.com/antofuller/CROMA/blob/main/use_croma.py
- Author benchmark collection to inspect for channel provenance: https://huggingface.co/datasets/antofuller/CROMA_benchmarks
- Local raster evidence: `data/bigearthnet-v2/small-sample/verification.json`
