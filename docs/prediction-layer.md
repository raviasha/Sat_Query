# Predict land-cover fractions from CROMA features

> Status note (10 September 2026): this guide retains the original three-area API/demo examples. The 1,000-area pipeline, validation-selected training, held-out evaluation and annotation matching have since completed. See the [current README](../README.md), [architecture](architecture.md) and [recorded results](../reports/pipeline-1000/results.md). Counts and limitations in the historical demo sections refer to those earlier runs.

The prediction head is a separate trainable `Linear(768, 19)` layer. CROMA's exported features stay fixed. The head produces logits during training and softmax fractions during prediction; these are estimates of class coverage, not confidence that the whole patch belongs to one class.

Training uses the existing 19-class fractional targets with unweighted soft-target cross-entropy. Only fully labeled tokens participate, so each target sums to one. Class order is identical to the target export. By default, training uses only samples marked `train`. The explicit `--demo-fit-all` option includes every fully labeled token, including validation/test samples, for the user-requested same-data demonstration. Original split labels remain unchanged. The checkpoint and predictions identify demo training, and reported metrics describe training-data fit only.

Separate modules own the neural head, verified artifact loading, training, and prediction export. Training streams one prepared feature/target batch at a time and uses configurable mini-batches within it. Checkpoints retain class order, feature selection, CROMA checkpoint and preprocessing profile, training input hashes, seed and hyperparameters. The inference command checks these input contracts, preserves sample/token order and rejects overwriting existing outputs.

Synthetic learning tests verify that the layer can learn fractional mixtures. They are not evidence of land-cover accuracy. Geographic train/validation data is needed before measuring real prediction quality.


## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `prediction.py` | Neural head, fractional-target loss, checkpoint save/load |
| `prediction_data.py` | Feature/target hashes, class/feature contracts, sample alignment and split selection |
| `train_prediction.py` | Reproducible fitting, fit metrics and training CLI |
| `predict_cover.py` | Checkpoint inference, ordered prediction export and inference CLI |

The default input is `joint_encodings`. Training also accepts `--feature-key optical_encodings` or `--feature-key SAR_encodings`; the selection is saved with the head. Pooled whole-area features are not used for patch prediction.

## Train the current three-area demo

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/satquery/venv"
uv sync --no-editable --locked --extra dev --extra features
uv run --no-editable satquery-train-head \
  --features data/features/three-samples \
  --targets data/targets/three-samples \
  --output data/models/coverage-head/three-samples-demo \
  --demo-fit-all --epochs 200 --learning-rate 0.01 --batch-size 128 --seed 17
```

Outputs are `head.pt` (14,611 learned parameters and input/class metadata) and `training.json` (loss history, fit metrics, hyperparameters and checkpoint hash). Output directories must be new. This is CPU training; CROMA is not loaded or fine-tuned.

For a later training dataset, use its feature and target exports and omit `--demo-fit-all`. Only `train` samples and fully labeled tokens will then be fitted. This initial module reports fit metrics on those same training tokens; it does not yet implement a separate validation/evaluation command.

## Predict

```bash
uv run --no-editable satquery-predict-cover \
  --features data/features/three-samples \
  --checkpoint data/models/coverage-head/three-samples-demo/head.pt \
  --output data/predictions/three-samples-demo
```

Each batch contains `predictions.pt` with:

- `class_fractions`: `(N,225,19)`, float32, sum to one. Multiply by 100 for coverage percentages.
- `dominant_class`: `(N,225)`, integer index of the largest predicted fraction. Ties select the lowest class index.
- `dominant_fraction`: `(N,225)`, the largest predicted coverage fraction; this is not calibrated confidence.

`batch.json` retains sample IDs, split labels, georeferencing and token order. `manifest.json` records checkpoint/source hashes, feature contract, class order and whether the input feature manifest equals the training feature manifest. That equality is provenance information, not a general test for geographic overlap. `validation.json` records input-contract, finite-output, fraction-sum and save/reload checks. Source batches remain separate, and mini-batch inference is controlled by `--batch-size`.

## Python interface

```python
import torch
from satquery.prediction import load_head

head, metadata = load_head("data/models/coverage-head/three-samples-demo/head.pt")
features = torch.load("data/features/three-samples/features.pt", weights_only=True)
fractions = head.predict(features[metadata["feature_key"]])  # (3,225,19)
percentages = fractions * 100
```

The direct interface checks tensor shape, dtype and finiteness; callers must supply features compatible with the checkpoint metadata. Use the CLI/exporter for automatic feature-contract and source-hash checking.

Coverage MAE is reported in percentage points averaged across all tokens and all 19 classes, including absent classes. Dominant-class agreement compares the largest target and predicted fractions (lowest index wins ties). Soft-target cross-entropy includes the target distribution's own entropy, so a perfect fit on mixed labels does not have zero loss.

## Completed three-area demo

Trained on all 675 fully labeled tokens from the three existing test areas with the explicitly requested demo option, using 200 epochs, Adam learning rate 0.01, batch size 128 and seed 17. The saved head has 14,611 parameters. Training-data fit: dominant-class agreement **655/675 (97.04%)**, mean coverage error **0.284 percentage points** averaged over all 19 classes, and soft-target cross-entropy **0.1340**. These are same-data fit measurements, not held-out accuracy.

The saved checkpoint was reloaded by the separate inference command and produced `(3,225,19)` fraction predictions for all 675 tokens. Class order, sample linkage, fraction sums, checkpoint/source/output hashes and reloaded-head predictions were checked. All **78 repository tests** and Ruff checks pass; the wheel builds. Independent review found no substantive issues.
