# Code, Colab and Drive

GitHub is the versioned source of the pipeline. The local checkout is where changes are edited and tested. Colab is a temporary execution environment; Drive holds the durable inputs and outputs.

| Location | Contents | Reuse |
| --- | --- | --- |
| `src/satquery/` | Tensor preparation, CROMA extraction, targets, linear/MLP heads, evaluation, annotation matching | Import the same modules for new experiments |
| `scripts/` | Stage orchestration, notebook builders and the fixed annotation audit | Configure inputs; preserve prior experiment directories |
| `notebooks/` | Clean templates with a checksum-verified package snapshot | Rebuild after source changes; run one stage at a time |
| GitHub | Committed source, templates, tests, docs and small reports | Pin a commit for a recorded run |
| EDU Drive `SatQuery/` | Downloaded archives, selected imagery, tensors, features, targets, models, evaluations and text source | Reuse validated artifacts rather than extract again |
| EDU Drive code snapshots | Versioned ZIP of source/templates/docs plus a commit identifier; historical executed notebook copies | Backup and provenance; edit the repository for the next version |
| Colab `/content/` | Installed package and temporary runtime files | Disposable when the session disconnects |

## EDU storage and account shortcuts

The recorded 5,000-image run uses `SatQuery/pipeline-5000` and `SatQuery/bigearthnet-v2-5000` owned by the EDU account. From the personal Colab account these are accessible through shortcuts beneath `MyDrive/SatQuery (1)`. A shortcut's parent folder can belong to the personal account even when the target belongs to EDU. Verify the target's owner in Drive before saving a code backup alongside the data.

The full text source is already stored at:

```text
SatQuery/pipeline-5000/annotations/
  source/72d865f2146f0a85b720f7f3ca1cdbaeafc3d316/
    BigEarthNet.txt.parquet
    download-verification.json
  matched/                         # created when matching is run
```

It contains 9,553,962 records (466,819,745 bytes). The existing resumable downloader checks size and SHA-256, reuses a valid file, and writes a verification receipt. The source and matched-output folders must be separate. Downloading the source does **not** mean all 5,000 images have already been matched.

## Run annotation steps

Open `notebooks/SatQuery_BigEarthNet_Text_Matching.ipynb`, install its included package, mount Drive, then edit its configuration cell. Set `P`, `IMAGE_ROOT`, `TEXT_SOURCE` and `OUTPUT`. There is no fixed image-count assertion in the matching stage. It verifies metadata, both Sentinel IDs, splits, feature manifest/batch-index hashes and feature offsets. Existing matching outputs are reused only after their recorded hashes agree. This linkage does not reload every large feature tensor; tensor integrity belongs to feature-stage validation.

The old 1,000-image layout remains supported by setting `TEXT_SOURCE` to its original `annotation-source-cache` file and `OUTPUT` to its existing `pipeline-1000/annotations` directory. No automatic relocation is performed.

The optional final cells reproduce the six-image reference-map audit. Alternatively, from an installed checkout:

```bash
python scripts/audit_annotation_samples.py \
  --pipeline /path/to/pipeline-5000 \
  --images /path/to/bigearthnet-v2-5000 \
  --output /path/to/pipeline-5000/annotation-map-audit/recheck-six
```

Install SciPy for this diagnostic. `--source` can point to a shared annotation source cache. The audit requires a new output directory and the original 5,000-image metadata. Its class/claim interpretations are manually transcribed for the six recorded images; changing the selection requires reviewing those interpretations. It is not a general natural-language parser. See [the findings and limitations](../reports/annotation-map-audit/expanded-six.md).

The port comes from the exported live notebook. `reports/annotation-map-audit/executed-six.py` preserves the exact historical recipe, SHA-256 `ae5c6c2c80969a0f11f7c9dae876f97fc5f60bdac8f0cd2a738b4e623f5a5329`, matching the saved Drive receipt. Use the configurable script to rerun; the historical copy retains the original paths and is evidence, not a default workflow.

## Keep future changes synchronized

1. Edit reusable modules or stage scripts in the repository. If a fix is tried in Colab, bring it back into the relevant script before treating it as the maintained version.
2. Run relevant tests and rebuild affected notebooks. The annotation builder also refreshes the 1,000-image notebook's embedded package.
3. Commit and push. Record the commit alongside the run's parameters and artifact hashes.
4. Save a code snapshot and executed notebook to EDU Drive. Keep historical notebooks as records; do not use **Run all** on a notebook containing old recovery/debug cells.

Preparing templates or saving a backup does not retrain models or regenerate tensors. Existing frozen CROMA features can be reused by both the linear and nonlinear heads.
