# CROMA feature extraction implementation plan

User authorized a separate module to run the three prepared areas through CROMA. This extends the existing prepared-tensor interface; preprocessing remains independent.

- [x] Pin the official CROMA inference code and Base checkpoint; retain its license and source attribution. Load trusted tensor weights with `weights_only=True` and CPU mapping, then move the model to the selected device.
- [x] Provide `CromaFeatureExtractor` for direct tensor calls, setting eval mode and inference mode. Validate input shapes, float32 dtype, range, equal batch length and finite outputs. Preserve all six official outputs, with spatial encodings `(N,225,768)` and pooled encodings `(N,768)` for Base.
- [x] Provide a separate CLI consuming preprocessing manifests, validating declared channel order/normalization and source file hashes, and saving one feature file per prepared batch. Slice inference into bounded mini-batches, with CPU as portable default and explicit MPS/CUDA options.
- [x] Carry sample IDs, original ground bounds/transform and row-major 15×15 token-grid interpretation into output metadata. Record checkpoint/source hashes, device, versions and the inherited channel-order inference caveat. Do not present the contextual transformer features as independent 80 m observations.
- [x] Cover invalid inputs, small neural-encoder evaluation, order preservation, saved/reloaded output and exporter protection with tests. Run the real pretrained Base checkpoint on the existing three samples and inspect all feature shapes, finiteness and hashes.
- [x] Document the command and Python interface. Perform independent code review and fix material findings before delivery.

Output: `features.pt` (all six official feature keys), `batch.json` (sample linkage, token-grid details), `manifest.json` and `validation.json`. Multiple prepared batches retain separate output folders. The original input artifacts and labels are unchanged.

## Completed verification

On 2026-09-09, ran the official SHA-256-verified CROMA-Base checkpoint on all three prepared areas on CPU, with inference batch size 2. Saved `data/features/three-samples/features.pt`: three spatial tensors `(3,225,768)` and three pooled tensors `(3,768)`. Confirmed finite float32 outputs, distinct area features, unchanged sample linkage, source/output hashes and joint mean-pooling consistency. The feature file is 6,251,425 bytes.

All 33 tests and Ruff checks pass. Independent review findings about checkpoint-validation claims and source-manifest provenance were fixed with regression tests. CLI import was verified using the documented non-editable installation after macOS repeatedly marked the editable package path file hidden.
