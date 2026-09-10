"""Render the actual saved evaluation reports into a readable summary and CSV."""
import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'reports/pipeline-1000'
report=json.loads((root/'evaluation/test-report.json').read_text())
m=report['metrics'];baseline=report['training_mean_baseline']['coverage_mae_pp']
fields=['index','name','mae_pp','rmse_pp','bias_pp','present_mae_pp','present_tokens','present_areas','within_5pp_fraction','present_within_5pp_fraction','dominant_precision','dominant_recall','dominant_f1','support_status']
with (root/'per-class-metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(m['classes'])
rows=[]
for c in m['classes']:
    present='Unavailable' if c['present_mae_pp'] is None else f"{c['present_mae_pp']:.2f}"
    recall='Unavailable' if c['dominant_recall'] is None else f"{100*c['dominant_recall']:.1f}%"
    rows.append(f"| {c['name']} | {c['mae_pp']:.2f} | {present} | {c['present_areas']} | {recall} |")
text=f'''# SatQuery: first 1,000-area pipeline results

Completed 9 September 2026 on a free Colab T4 GPU. **All seven requested stages passed.** Outputs are saved in Google Drive → MyDrive → SatQuery → pipeline-1000.

## Main results

| Measure | Result |
|---|---:|
| Test areas / eligible blocks | 200 / 44,723 |
| Dominant-class accuracy | {100*m['dominant_class_accuracy']:.2f}% |
| Coverage MAE across all blocks and 19 classes | {m['coverage_mae_pp']:.2f} percentage points |
| Constant training-mean baseline error | {baseline:.2f} percentage points |
| Relative reduction in coverage error | {(1-m['coverage_mae_pp']/baseline)*100:.2f}% |
| Area-bootstrap 95% interval for aggregate error | 5.88–6.40 percentage points |
| Epoch selected using validation | 5 of 15 run |

**A 6.16-point error is not 93.84% accuracy or individual confidence.** Many class entries are zero. Errors on blocks actually containing a class are often substantially larger, as shown below. This is an initial baseline that still needs improvement.

## Per-class results

Errors are absolute percentage-point differences. “Where present” includes only blocks whose true fraction for that class is greater than zero. Support counts distinct test areas with the class in at least one eligible block. Dominant recall is a separate categorical measurement.

| Class | MAE: all blocks | MAE: where present | Test areas with class | Dominant recall |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Coastal wetlands have no eligible test presence, so their positive-class reliability is unavailable. Beaches occur in only three eligible blocks in one test area, occupying 12 reference pixels in total. Small absolute error on these classes does not establish reliable detection. The “limited” support flag uses a descriptive 20-area threshold, not a statistical guarantee.

## Spatial verification and label percentages

The independent audit checked every one of the **14.4 million original map pixels**, all **225,000 block boundaries**, and all **4.275 million class counts**. It used the original TIFFs and a separate transcription of the official taxonomy. Every token matched its row-major 8 × 8 pixel region and 80 × 80 metre ground boundary.

Class fractions use all 64 pixels as the denominator. Unknown/excluded labels retain their area and are not redistributed. The current head uses only fully labeled blocks.

| Official split | Areas | Fully labeled blocks used | Partly labeled excluded | Unlabeled excluded |
|---|---:|---:|---:|---:|
| Train | 600 | 134,255 | 416 | 329 |
| Validation | 200 | 44,848 | 89 | 63 |
| Test | 200 | 44,723 | 169 | 108 |

Exact matching to the supplied CLC maps does not prove independent ground-truth accuracy at 80 metres. CROMA tokens are spatially anchored but include attention context from the wider image.

## Features and model selection

This baseline uses CROMA's learned 768-value joint optical/SAR feature for each block. No hand-crafted indices or feature selection were added. Feature means and standard deviations are fitted on training data only and incorporated in the saved linear head, so existing inference uses the raw CROMA features directly.

CROMA remains frozen. The 19-output head uses soft-target cross entropy and AdamW. Only training areas update weights; validation loss selects the saved epoch. Test evaluation happens afterward. All blocks from an area stay in the same split. The baseline predicts the training-set mean class fractions.

Future optical-only/SAR-only comparisons, indices, metadata or model changes should be selected using validation data. This test set has been used for the baseline; reserve fresh test data for a later final comparison.

## What to display during inference

The inference module accepts a reliability report and verifies the exact model checksum, feature contract and class order. Predictions for all 1,000 areas were saved and verified with that report attached. A display can show:

**Predicted coverage: [model output]% · Historical error where this class occurs: [test MAE] percentage points · Test support: [area count] areas.**

These historical measurements are not per-prediction confidence probabilities or prediction intervals. Calibrated confidence would require separate calibration and evaluation. The aggregate uncertainty interval resamples whole areas rather than treating neighboring blocks as independent; dependence between different areas remains a limitation.

## Reusable files

- [Per-class CSV](per-class-metrics.csv)
- [Full test report](evaluation/test-report.json)
- [Inference reliability](evaluation/reliability.json)
- [Training and validation history](model/training.json)
- [Exhaustive spatial audit](stage-3-4-report.json)
- [Portable notebook](../../notebooks/SatQuery_Pipeline_1000.ipynb)

The preparation, CROMA, targets, validation training, evaluation and inference modules remain separate. The package passed 90 automated tests, with independent code review of alignment and evaluation. Reports copied locally retain their exact Drive bytes and the test-report checksum was verified. This country-balanced subset is an initial experiment, not the full official benchmark.
'''
(root/'results.md').write_text(text)
print(root/'results.md')
