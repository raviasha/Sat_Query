# Selective BigEarthNet v2 downloads

Open the [download notebook in Colab](https://colab.research.google.com/github/raviasha/Sat_Query/blob/main/notebooks/SatQuery_BigEarthNet_Drive_Download.ipynb) (or upload `notebooks/SatQuery_BigEarthNet_Drive_Download.ipynb`), choose a standard CPU runtime, and run all cells. The notebook includes the project wheel and asks Google to mount your Drive. It saves to `MyDrive/SatQuery/bigearthnet-v2-1000`.

The default is **1,000 paired areas: train 600, validation 200, test 200**, retaining official splits. A seeded country-balanced selection excludes the original three demo test areas and keeps unique optical/SAR pairs. Each area has 12 native optical band TIFFs, two native SAR TIFFs, and one original reference-map TIFF. The expected stored size, including the 327 MB reference-map archive, is approximately 0.5–1 GB; the notebook reports the actual total.

## Resuming

Run the same notebook with the same destination and settings after a disconnect. Downloads of metadata/reference-map cache files resume at the last written byte and are verified against pinned publisher SHA256 values. Completed 100-area ZIP batches are verified against their receipts and skipped. An interrupted batch is fetched again. Changing the selection requires a new destination. Do not run two instances against the same destination simultaneously.

The combined selected metadata is atomically regenerated on resume, including after an interrupted metadata write. ZIP files are written locally, copied to Drive with a temporary name, hash-checked, renamed and given a completion receipt. Colab only needs temporary space for selected reference maps and one batch of imagery. Original optical/SAR archives and the full LMDB are never downloaded.

## Sources and limits

- Original metadata and reference maps: [TorchGeo mirror](https://huggingface.co/datasets/torchgeo/bigearthnet/tree/3cf3a5910a5302d449fdb8e570e5b78de24fe07f/V2), pinned revision `3cf3a5910a5302d449fdb8e570e5b78de24fe07f`.
- Native optical/SAR arrays: [unofficial LMDB pre-conversion](https://huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB/tree/118d1b6285c080ba8e4078414e1b8a243b18c9bd), pinned revision `118d1b6285c080ba8e4078414e1b8a243b18c9bd`.

The downloader reads the LMDB tree and selected values with strict HTTP byte ranges. It validates the stored safetensors band sets, native dimensions, dtypes and finite values. TIFFs are **reconstructed** using the original reference map's CRS and bounds. The TIFF header bytes need not match the original image archives. All 14 native arrays and the resulting aligned imagery tensors for the existing `_61_39` demo area were checked for exact equality against originals; this sample comparison does not certify every record in the unofficial mirror. Source revisions, record hashes and reconstructed-file hashes are stored in `download-provenance.json` inside every ZIP.

This is an initial country-balanced experiment, not the official complete benchmark distribution. Keep area-level official splits when training on the 225 spatial CROMA features per area. Snow/cloud flags and labels remain in the selected metadata. Imagery masking or other failures detected by the separate preprocessing module are not silently bypassed.

## Using downloaded batches

Every ZIP contains the existing raw input hierarchy and metadata for that batch. Extract one batch and point `satquery-prepare-croma` at it, then use the separate feature, target and prediction modules. If extracting all batches into one directory, copy the top-level combined `metadata.parquet` into that directory **after** extraction; each ZIP contains only its own batch metadata.

The CLI is also reusable outside Colab:

```bash
uv sync --no-editable --extra downloads
uv run --no-editable --extra downloads satquery-download-bigearthnet \
  --destination /path/to/persistent/subset-1000 \
  --work /path/to/temporary/work \
  --train 600 --validation 200 --test 200
```

The persistent destination and temporary work directory must be separate, with neither nested inside the other. Use `--workers 2` on a limited connection; the default is four.

## Maintenance and validation

After changing package code, rebuild the portable notebook:

```bash
uv build --wheel
python scripts/build_download_notebook.py
```

The builder embeds only the built project wheel and checks its SHA256 on installation. No satellite files or credentials are embedded. Tests cover native LMDB leaf/overflow records, strict range responses, partial cache resume, checksum rejection, country-balanced split selection, safe selective map extraction, native TIFF reconstruction and complete-batch resume with interrupted metadata. Run tests with both `dev` and `downloads` extras installed.

## Completed Drive run — 9 September 2026

The saved Colab notebook completed the 1,000-area run in approximately 15 minutes and its final integrity cell passed. It verified all ten ZIP hashes plus the combined metadata and selection hashes. Output: `MyDrive/SatQuery/bigearthnet-v2-1000`, 15,000 TIFFs in ten ZIP batches; 600 training, 200 validation and 200 test areas. The selected metadata contains all 19 labels and 100 areas from each of ten countries. Total stored size reported by Colab, including the source cache, was **0.558 GB**. Downstream tensor preparation, CROMA extraction, target auditing, validation-selected training, held-out evaluation and text matching have also completed. See the [results](../reports/pipeline-1000/results.md) and [current architecture](architecture.md).
