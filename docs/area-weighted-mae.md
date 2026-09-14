# Area-weighted MAE

Our headline progress score is `area_weighted_mae_pp`; lower is better. It uses actual class coverage as a weight within each fully labelled 80 × 80 metre block, then gives each evaluated block equal weight:

`100 × mean_over_blocks(sum_over_classes(truth × abs(prediction − truth)))`

Inputs are fractions between 0 and 1, summing to 1 across classes. Output is in percentage points. There is no division by the number of classes or by two. A block with truth `[.02, .80, .18]` and prediction `[.10, .72, .18]` scores `6.56` points. Absent classes have zero direct weight. Major land cover matters more than small features; retain per-class reports for diagnosis. This score is neither ordinary MAE nor coverage allocation error, and is not prediction confidence.

## Reuse

The separate module `src/satquery/area_weighted_mae.py` owns the calculation, input validation, and saved-prediction scoring. The existing `evaluation.coverage_metrics` calls it automatically, adding `area_weighted_mae_pp` to new evaluation reports alongside the original metrics. Training loss and checkpoint selection are unchanged.

```python
from satquery.area_weighted_mae import area_weighted_mae
score_pp = area_weighted_mae(predicted_fractions, true_fractions)
```

For any new head, score its saved predictions without loading CROMA or training again:

```sh
python -m satquery.area_weighted_mae \
  --predictions /path/to/test-predictions.pt \
  --output /path/to/area-weighted-mae.json
```

The safe PyTorch file must contain `predicted_fractions` and `true_fractions` as aligned CPU N × C tensors, plus `area_ids` as a list of strings, one per row. Both matrices must be finite, nonnegative, at most one, with row sums within 1e-5 of one. Do not include unlabelled blocks or independently reorder either matrix. The old NumPy-pickled legacy baseline file is not loaded; its reproduced linear seed-17 export uses the supported format.

For the six saved linear/MLP runs:

```sh
python -m satquery.area_weighted_mae \
  --comparison '/content/drive/MyDrive/SatQuery (1)/pipeline-5000/experiments/linear-vs-mlp-v1' \
  --output '/content/drive/MyDrive/SatQuery (1)/pipeline-5000/experiments/linear-vs-mlp-v1/area-weighted-mae.json'
```

This verifies prediction and model hashes, confirms identical ordered targets and area IDs across runs, and reports scores per seed plus means and sample standard deviations per architecture. It reports the previously selected checkpoint's score without changing selection. The independent output records the metric definition and source hashes; original model and evaluation files are preserved.

For meaningful progress comparisons, keep the test blocks, label schema and masking unchanged. Equal block weighting differs from giving each 1.2 km scene equal weight when their eligible block counts differ. The same test set has already informed earlier comparisons; reserve fresh geographic data for a final assessment. Use validation scores for model tuning.

## Recorded results: 14 September 2026

Recomputed on all 44,723 eligible test blocks from the same 200 areas, without retraining. Metric source: `b0e62acb0da52e54c7326ab358d4939411210a3f`. The full JSON report is `pipeline-5000/experiments/linear-vs-mlp-v1/area-weighted-mae.json` on the EDU-backed Drive.

| Architecture | Seed 17 | Seed 29 | Seed 43 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| Linear | 50.3669 | 51.4107 | 51.3888 | **51.0555 ± 0.5964** |
| MLP | 46.8608 | 46.7711 | 45.2263 | **46.2861 ± 0.9189** |

Units are percentage points; lower is better. Across-seed mean error fell by 4.7694 points (9.34% relative). The previously validation-selected MLP seed 29 scores 46.7711 points. Selection remains unchanged. Do not compare these absolute values with the old ordinary MAE of approximately 5.18: the new measure emphasizes actual major land cover and is a different weighted average. It also is not a classification accuracy or a percentage of pixels misclassified.

All six prediction/checkpoint hash pairs and identical ordered test targets were checked by the reusable runner. The code suite passed 116 tests with 5 optional integration tests skipped.
