# Expanded annotation/reference-map audit

Six additional test images were selected before inspecting their annotation answers: one image from each of the first six alphabetically sorted countries in the 5,000-image metadata, sampling with random seed 37 after sorting by patch ID. The 124 matching publisher records and the six original reference maps were saved in the EDU Drive. This is a small, deliberately geographically varied sample, not an estimate of dataset-wide annotation accuracy.

| Check | Consistent with reference maps |
|---|---:|
| Class-presence Q&A (binary and selected MCQ answers) | 18 / 18 |
| Area Q&A (binary and selected MCQ answer ranges) | 18 / 18 |
| Named-class boxes matching connected-region extents | 13 / 13 |
| Caption total-class area claims, allowing nearest-1,000 m² rounding | 15 / 17 |
| Explicit caption connected-region counts | 5 / 5 |

## Two caption discrepancies

| Annotation ID | Image / class | Caption area | Map area | Map coverage |
|---|---|---:|---:|---:|
| 17046 | Austria; land principally occupied by agriculture, with significant areas of natural vegetation | 72,000 m² | 38,800 m² | 2.6944% |
| 2970546 | Lithuania; coniferous forest | 72,000 m² | 61,400 m² | 4.2639% |

Both classes occupy less than 5% of their image. The stated 72,000 m² equals exactly 5% of a 1,200 × 1,200 m image. This suggests that an upper threshold for a marginal class may have become an approximate area in generated wording; that explanation is an inference, not a verified reconstruction of the publisher's generation process. Neither discrepancy is explained by rounding to the nearest 1,000 m². Flag these two caption area claims before using them as quantitative evaluation answers; retain the original records unchanged.

One apparent third mismatch was a rounding convention, not an annotation error: 1,194,500 m² is equally close to 1,194,000 and 1,195,000 m². The audit accepts either nearest value (absolute error at most 500 m²), rather than requiring Python's ties-to-even result.

## Spatial checks

The 13 named-class boxes exactly match a connected region of their named class after normalizing pixel-edge bounds by 120 and rounding lower coordinates downward and upper coordinates upward to two decimals. This coordinate convention is inferred from the observed agreement. The extent matches hold for both 4- and 8-neighbor connectivity in these examples; this does not establish the convention globally or independently validate every smallest/largest qualifier.

Named-class coverage inside the published boxes ranges from **42.338% to 81.845%**, measured using fractional overlap at boundary pixels. None of these boxes warrants a 100%-class target for every enclosed CROMA cell.

The five explicit caption region counts agree with 4-neighbor components: Finland inland water 3; Ireland pastures 2 and arable land 2; Kosovo transitional woodland/shrub 6; Lithuania mixed forest 2. The Kosovo map has only 5 components with 8-neighbor connectivity, so diagonal adjacency matters when reproducing this answer.

## Selected image IDs

| Country | Patch ID |
|---|---|
| Austria | S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_51_44 |
| Belgium | S2B_MSIL2A_20170829T105019_N9999_R051_T31UER_67_37 |
| Finland | S2A_MSIL2A_20170701T093031_N9999_R136_T35VPK_47_36 |
| Ireland | S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_44_49 |
| Kosovo | S2A_MSIL2A_20171208T093351_N9999_R136_T34TEN_24_45 |
| Lithuania | S2A_MSIL2A_20180413T095031_N9999_R079_T34UEG_46_31 |

## Reproduction and interpretation

The full evidence is saved under `SatQuery/pipeline-5000/annotation-map-audit/expanded-six/` in the EDU Drive:

- `source-records.json`: 124 unmodified matching publisher records.
- `reference-map-samples.npz` and `map-counts.json`: six map arrays, class counts, original TIFF hashes and pixel areas.
- `sample_audit.py`: the executed selection and arithmetic comparison recipe, including manually transcribed source claims and their annotation IDs.
- `comparison.json` and `summary.json`: detailed numerical comparisons and summary, read back after saving.

Source: pinned BigEarthNet.txt revision `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`, SHA-256 `d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554`. Original map pixels were read from `bigearthnet-v2-5000/Reference_Maps-selected.zip`, with the existing 19-class taxonomy used to combine CLC codes. MCQ checks validate the selected answer; they do not exhaustively prove all distractors are false. Point boxes, all adjacency/count/relative-position Q&A, all caption wording, and climate metadata were not exhaustively validated.

The publisher explicitly derives land-cover spatial attributes from reference maps, while country, season and climate have additional sources and caption wording is generated. See [BigEarthNet.txt methodology, section 3.1](https://arxiv.org/html/2603.29630v1). The sampled agreement supports that method; it cannot prove that every record is map-derived or correct. The two caption discrepancies show why map-derived text is not automatically reliable quantitative ground truth.

Continue training the coverage model with reference-map percentages. Use checked annotations on held-out images for question-answering evaluation, keeping raw annotations separate from verified or corrected evaluation targets. These texts do not supply independent new land-cover truth beyond the maps.
