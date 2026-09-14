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
