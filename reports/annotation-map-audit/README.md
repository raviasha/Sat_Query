# Annotation/reference-map sample check

Image: `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_39`. Split: test. Checked 22 publisher records for one image. This is an illustrative sample, not a dataset-wide evaluation.

Publisher methodology: https://arxiv.org/html/2603.29630v1 (section 3.1). Land-cover attributes come from reference maps; country, season and climate have additional sources.

| Class | Map area (m²) | Caption area (m²) | Named class within its box |
|---|---:|---:|---:|
| Broad-leaved forest | 585,700 | 586,000 | 43.27% |
| Inland waters | 485,700 | 486,000 | 53.54% |
| Arable land | 368,600 | 369,000 | 61.16% |

All three caption areas agree after rounding to the nearest 1,000 m². All three boxes agree with the corresponding class extent when minima are rounded down and maxima up to two decimal places. This coordinate convention is inferred from this sample. Box coverage uses fractional pixel overlap with the published coordinates.

Additional checks: water presence=yes; agro-forestry presence=no; arable land between 60–70%=no (map: 25.60%).

These annotations do not supply independent land-cover truth beyond the maps. They can still train language/grounding tasks on training images and evaluate held-out tasks. Do not convert boxes to 100% class targets or provide ground-truth answers as inputs when measuring inference quality.

Existing export report records SatQuery/pipeline-1000/annotations, with 20,453 records across 955 images. EDU Drive search has not yet located that export.

The full official text source was subsequently downloaded and SHA-256 verified in the EDU Drive at `SatQuery/pipeline-5000/annotations/source/72d865f2146f0a85b720f7f3ca1cdbaeafc3d316/BigEarthNet.txt.parquet` (466,819,745 bytes; 9,553,962 rows).
