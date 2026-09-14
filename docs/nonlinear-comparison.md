# Linear versus nonlinear coverage heads

This experiment reuses the existing frozen CROMA features and aligned 19-class percentage targets. It does not download imagery, prepare tensors, extract features, or rebuild labels.

The MLP is 768 → 256 → 128 → 19, with GELU and 0.1 dropout after each hidden layer. Softmax converts logits to coverage fractions. The default linear head and existing version-1 checkpoints continue to work.

## Modules

- `prediction.py`: both head architectures and compatible checkpoint save/load.
- `validation_training.py`: shared train-only normalization, AdamW, soft-target cross entropy, and validation early stopping. Normalization is folded into the first affine layer for both heads, so inference still consumes raw CROMA features. Dropout RNG is seeded for the entire fit.
- `evaluation.py`: existing verified split loader, batched prediction, per-class errors/F1 and whole-area bootstrap intervals.
- `compare_heads.py`: reusable experiment orchestration, resume, model selection and reports.
- `predict_cover.py`: unchanged inference export, compatible with either architecture and its corresponding reliability report.

## Run

```sh
python -m satquery.compare_heads \
  --pipeline '/content/drive/MyDrive/SatQuery (1)/pipeline-5000' \
  --device cuda --seeds 17 29 43 --source-revision <git-commit>
```

Default output: `pipeline-5000/experiments/linear-vs-mlp-v1/` in the same EDU-backed folder as the source features and targets. `SatQuery (1)` is the personal account shortcut to that folder. The original `pipeline-5000/model/` and `evaluation/` remain the historical linear baseline.

Each architecture gets `seed-17/`, `seed-29/` and `seed-43/` folders containing `head.pt`, `training.json`, `test-report.json`, `test-predictions.pt`, `per-class-metrics.csv`, and `reliability.json`. Test predictions store plain lists and tensors and support `torch.load(..., weights_only=True)`.

`config.json` records hashes, software version, model configuration and seed list; `splits.json` records area identities. Completed seed folders are verified and reused after a disconnect. An interrupted unfinished seed is trained again; the input extraction stages are never repeated. Changing inputs or configuration requires a new output folder. `comparison.json` records aggregate metrics, all run results and output checksums.

Both architectures use 60 maximum epochs, patience 10, learning rate 0.001, batch size 1024, and AdamW weight decay 0.01. Each run's epoch is selected by validation cross entropy. The architecture is selected by mean validation cross entropy across all three seeds, then its checkpoint by the lowest validation cross entropy. This decision is saved to `selection.json` before any test evaluation. Test data must not influence subsequent tuning of this experiment. Deployment is not automatically changed.

Report mean and sample standard deviation across seeds for coverage MAE and dominant-class accuracy, plus per-class MAE, present-class MAE, F1 and area support. The standard deviation measures training variation, not uncertainty across landscapes. Test areas have already been used for earlier comparisons, so this is not a new untouched benchmark. Coverage percentages and historical class errors are not calibrated confidence scores.

## Code preservation

The Colab launcher saves a ZIP of the exact Git revision under `SatQuery/code/` on the EDU-backed Drive. This contains reusable modules, tests, documentation and the launch notebook. The revision is also recorded in the experiment configuration; Colab's local extraction of that ZIP is temporary.

## Completed 5,000-area experiment — 14 September 2026

All six fits completed on a Colab T4 using source revision `79dc1eebd5e53f3b11edcce75d677f7e548af48f`. Inputs were reused: 4,600 training areas (1,029,204 eligible cells), 200 validation areas (44,848 cells), and 200 test areas (44,723 cells). No imagery extraction, tensor preparation, CROMA inference or label aggregation was repeated.

| Metric, mean ± sample SD over seeds 17, 29, 43 | Linear | MLP |
| --- | ---: | ---: |
| Validation soft-target cross entropy | 1.36337 ± 0.00377 | 1.26091 ± 0.01159 |
| Test dominant-class accuracy | 56.84% ± 0.20 pp | 59.85% ± 0.49 pp |
| Test coverage MAE | 5.710 ± 0.067 pp | 5.177 ± 0.099 pp |

The MLP gained 3.00 percentage points of dominant-class accuracy and reduced mean coverage MAE by 0.533 pp (9.33% relative). Linear seed 17 exactly reproduced the earlier 5,000-area baseline metrics: 57.0685% accuracy and 5.63167 pp MAE.

Validation selected **MLP seed 29, epoch 1**, stored at `mlp/seed-29/head.pt`; SHA-256 `7261904cf89430c2c8f3a60c7d1db61471381555af8158592614626cd9e4dda9`. This particular checkpoint achieved **59.7082%** test dominant accuracy and **5.23148 pp** coverage MAE. Seed 43 had better test metrics but was not selected because the decision was fixed using validation. Dominant accuracy excludes 173 tied ground-truth cells, leaving 44,550 cells; coverage MAE includes all 44,723 eligible test cells.

Across seeds, per-class overall MAE decreased for 18 of 19 classes. Examples: broad-leaved forest 15.79→13.81 pp, coniferous forest 9.50→7.94 pp, arable land 14.84→12.97 pp. Agro-forestry increased slightly from 1.84→1.88 pp. Low overall errors for rare classes can be dominated by absence: coastal wetlands have no test presence, and several other classes occur in fewer than 20 test areas. Use the saved present-class errors, precision/recall/F1 and support alongside overall MAE.

MLP validation selected epochs 1, 1 and 2; later epochs overfit. The next controlled experiment should focus on validation-only regularization/capacity tuning and better rare-class training coverage. A fresh geographic holdout is needed for a stronger final assessment after these repeated comparisons. The MLP is the recommended candidate for inference; this experiment preserves the original deployed baseline.

Verification checked all 39 indexed artifact hashes, input manifest hashes, prediction dimensions and normalization, 200 distinct test area IDs per run, and the unchanged original baseline checksum. The code test suite passed **104 tests, with 5 optional integration tests skipped**. Full metrics, predictions and reliability files are under the EDU-backed `SatQuery/pipeline-5000/experiments/linear-vs-mlp-v1/`; `summary-for-review.json` adds a compact per-class comparison. Source snapshots, launch notebook and architecture documentation are also preserved in the EDU SatQuery folder.
