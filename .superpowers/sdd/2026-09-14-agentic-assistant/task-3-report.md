# Task 3 report: reusable image-text adaptation and task evaluation

## Status

Implemented the requested experimental scene-level image-text adaptation and
explicit-record task evaluation workflows. This work did not call the live
OpenAI API, run a production adaptation, download data, extract CROMA features,
or alter the existing coverage/annotation implementation.

The supplied environment status says its real API test returned HTTP 401. The
implementation therefore uses an injected deterministic embedder in tests and
keeps the real OpenAI SDK call confined to the CLI `prepare` path. Authentication
is left to the SDK environment and no key is accepted by, logged by, or persisted
from this code.

The plan's `assistant/...` path shorthand was resolved to
`src/satquery/assistant/...` so the reusable modules are included by the
project's existing Hatch `src/satquery` package target. No Task 1 package files
were added or modified.

## Files

- `src/satquery/assistant/text_adaptation.py`
  - Reads a verified `match_annotations` artifact and integrity-checked
    `FeatureBatches`.
  - Builds deterministic caption/scene pairs from exact `patch_id` + `s1_name`
    matches and mean-pooled 15x15 spatial features.
  - Implements resumable frozen OpenAI embedding targets, the PyTorch projection,
    validation-selected training, checkpoint loading, cosine similarity,
    train-caption retrieval, and held-out retrieval reporting.
  - Provides `prepare`, `train`, and `evaluate` CLI subcommands.
- `src/satquery/assistant/task_evaluation.py`
  - Strictly matches JSONL reference/prediction records.
  - Reports provisional normalized exact-match VQA, abstention/coverage,
    per-task counts, unit-specific numeric area MAE, scene-fraction MAE, and
    exact change-label-set match. It does not produce a joint score.
  - Exposes the candid VRSBench/RSVQA/CDVQA/SAC compatibility manifest and a
    JSONL evaluation CLI.
- `scripts/colab_adapt_text.py`
  - Calls reusable package functions over existing annotation/features/pair
    artifacts and constrains every output beneath the supplied `--edu-p`.
  - It contains no notebook-only trainer and never invokes image/CROMA extraction.
- `tests/test_text_adaptation.py`, `tests/test_task_evaluation.py`
  - Cover filtering, ordering, cache resume, provenance, source mutation,
    fitting/checkpoint behavior, held-out exclusion, retrieval reports, Colab
    routing, task metrics, abstentions, duplicate/leakage failures, units, and
    compatibility gaps.

## Split and provenance controls

Preparation only accepts `captioning` records whose annotation split, image
split, and conservative `use_partition` agree. Train additionally requires
`training_eligible=true`; validation/test require it to be false. `bench`,
`bench_holdout`, and `split_conflict_holdout` records are quarantined before any
embedding request. Boxes, binary questions, MCQs, and other non-caption rows are
never transformed into training targets.

Default excluded annotation IDs are `17046` and `2970546`. Callers can add
exclusions, and the complete set is recorded in both the pair and cache
contracts. Duplicate caption text for one image is collapsed deterministically;
the first annotation in source-image/annotation-ID order is retained.

The pair manifest records:

- selected feature key and `mean_over_15x15_spatial_tokens` pooling;
- source annotation and source image split counts, plus selected caption/image
  counts per requested split;
- annotation Parquet, annotation report, publisher source, selected metadata,
  and feature manifest SHA-256 values;
- feature checkpoint/channel/normalization contract;
- excluded IDs and quarantine counts;
- embedding model/dimension and explicit synthetic/caller/OpenAI provenance;
- pair tensor and embedding-cache hashes.

The matcher report's `files["annotations.parquet"]` hash is checked before the
Parquet is read. Annotation files, the feature manifest, and every referenced
feature batch file are checked again after embedding and immediately before
publishing the pair manifest. A resumed
embedding cache must match source hashes, splits, exclusions, feature key,
embedding model/dimension, and provenance. Its content key is the SHA-256 of the
canonical text/model/dimension tuple. Only selected eligible captions are sent
to or appended to the cache.

## Training and inference contract

`TextProjection` is `768 -> 256 -> 512` with GELU and normalized output. Feature
mean/std are fitted on unique train images only. Cosine target loss first averages
captions belonging to one image and then averages images. Training is seeded;
validation loss alone selects the retained best state and controls early
stopping. Test rows are not used for optimization or selection.

The output directory must be new. It contains only `best.pt` and `training.json`;
an existing result is never overwritten. The checkpoint includes source hashes,
the train-only normalizer, model/feature contracts, and a caption retrieval bank
containing train captions only. Test/validation captions and their embeddings are
not serialized into the checkpoint. A retrieval evaluation verifies the exact
prepared-manifest hash used by training and is labeled
`report_only_never_checkpoint_promotion`.

Inference can compare a scene feature with a caller-supplied 512-dimensional
query embedding or return top-k captions from the train-only bank. Returned
scores are labeled `cosine_similarity_not_calibrated_confidence`. The module is
explicitly described as an experimental scene-level adapter and makes no
open-ended captioning or VQA quality claim.

## CLI usage

Run from the repository checkout with the package source and repository root on
`PYTHONPATH`:

```bash
PYTHONPATH=src:. python -m satquery.assistant.text_adaptation prepare \
  --annotations /path/to/matched-annotations \
  --features /path/to/features \
  --output /path/to/text-pairs \
  --feature-key joint_encodings \
  --splits train,validation,test

PYTHONPATH=src:. python -m satquery.assistant.text_adaptation train \
  --prepared /path/to/text-pairs \
  --output /path/to/text-adapter \
  --epochs 100 --patience 10 --seed 17

PYTHONPATH=src:. python -m satquery.assistant.text_adaptation evaluate \
  --prepared /path/to/text-pairs \
  --checkpoint /path/to/text-adapter/best.pt \
  --split test \
  --output /path/to/retrieval-test.json

PYTHONPATH=src:. python -m satquery.assistant.task_evaluation \
  --references /path/to/references.jsonl \
  --predictions /path/to/predictions.jsonl \
  --output /path/to/task-evaluation.json \
  --compatibility-output /path/to/benchmark-compatibility.json
```

The real embedding call is made only by `prepare` through
`client.embeddings.create(input=..., model="text-embedding-3-small",
dimensions=512)`. The SDK obtains its API key from the environment.

For Colab/Drive, outputs are derived beneath `EDU_P/text-adaptation` while inputs
remain explicit and may live elsewhere:

```bash
PYTHONPATH=src:. python scripts/colab_adapt_text.py \
  --edu-p "$EDU_P" prepare \
  --annotations "$EDU_P/annotations" \
  --features "$EDU_P/features" \
  --name text-pairs \
  --splits train,validation,test

PYTHONPATH=src:. python scripts/colab_adapt_text.py \
  --edu-p "$EDU_P" train \
  --prepared "$EDU_P/text-adaptation/text-pairs" \
  --name text-adapter

PYTHONPATH=src:. python scripts/colab_adapt_text.py \
  --edu-p "$EDU_P" evaluate \
  --prepared "$EDU_P/text-adaptation/text-pairs" \
  --checkpoint "$EDU_P/text-adaptation/text-adapter/best.pt" \
  --name retrieval-test.json --split test
```

## TDD and verification evidence

The implementation followed red-green cycles:

1. The initial 11 behavior tests failed with `ModuleNotFoundError` before either
   module existed.
2. Pair preparation, training, retrieval, and task evaluation were implemented;
   the focused suite reached 11 passing tests.
3. The missing Colab driver test failed because the script did not exist, then
   passed after adding the reusable driver.
4. New image-split and optional-abstention tests failed on the missing guards,
   then passed after the fixes.
5. New cache-provenance and prepared-manifest provenance tests failed before
   their checks were implemented, then passed.
6. Annotation-output tampering plus concurrent annotation/feature source-mutation
   tests failed before the matcher-receipt and end-of-operation checks were
   added, then passed.

Focused verification command:

```bash
/tmp/satquery-nonlinear-env/bin/python -m pytest \
  tests/test_text_adaptation.py tests/test_task_evaluation.py -q
```

Latest focused result before final verification: `18 passed`.

Lint command:

```bash
/tmp/satquery-nonlinear-env/bin/python -m ruff check \
  src/satquery/assistant/text_adaptation.py src/satquery/assistant/task_evaluation.py \
  scripts/colab_adapt_text.py tests/test_text_adaptation.py \
  tests/test_task_evaluation.py
```

Complete repository verification command:

```bash
/tmp/satquery-nonlinear-env/bin/python -m pytest -q
```

Final fresh result: `135 passed, 5 skipped` in 9.55 seconds. The 44 warnings are
existing Rasterio affine-multiplication pending-deprecation warnings from the
download tests.

CLI/import smoke verification:

```bash
PYTHONPATH=src:. python -m satquery.assistant.text_adaptation --help
PYTHONPATH=src:. python -m satquery.assistant.task_evaluation --help
PYTHONPATH=src:. python scripts/colab_adapt_text.py --help
python -m compileall -q src/satquery/assistant scripts/colab_adapt_text.py
```

All four commands exited successfully. `git diff --check` also exited cleanly.

Wheel packaging verification:

```bash
uv build --wheel --out-dir /tmp/satquery-task3-wheel-20260914
```

The build succeeded, and archive inspection found both
`satquery/assistant/text_adaptation.py` and
`satquery/assistant/task_evaluation.py` in the wheel.

## Remaining gaps

- No real OpenAI embeddings were obtained in this task because the supplied
  environment reports HTTP 401. The SDK boundary is exercised offline through
  injection, but a successful authenticated prepare run remains pending.
- All training tests use synthetic caller-supplied embeddings/features. No file
  produced by these tests is a production-adapted model or benchmark result.
- VRSBench, RSVQA, CDVQA, and SAC dataset adapters, official scorers, benchmark
  models, and CDVQA temporal image encoder are not implemented. The compatibility
  manifest reports each as unsupported.
- VQA exact match is intentionally provisional and is not an official dataset
  scorer. Retrieval evaluation measures only the explicit candidate set supplied
  in the prepared artifact.

## Review round 1 fixes

The prepared artifact now stores a canonical SHA-256 digest of `feature_key`,
`requested_splits`, `image_records`, and `caption_records` inside `pairs.pt` and
in the manifest. Loading recomputes and compares both digests before using any
record. It then independently validates unique image identities/source indices,
allowed requested splits, caption-to-image ownership, matching optical/SAR IDs,
matching source feature indices, annotation-ID uniqueness, and the exact rule
that only train captions have `training_eligible=true` and `use_partition=train`.

Embedding provenance validation now walks nested objects/lists, rejects
credential-like keys at every depth, requires a nonblank `kind`, accepts only
finite JSON-safe values, and enforces depth/item/string/serialized-size bounds.
Task evaluation accepts only the fixed `area`, `change`, and `vqa` task names and
reports that enum. The optional `assistant` dependency extra now declares
`openai>=1.68`; `OpenAIEmbedder` gives an actionable
`satquery-preprocessing[assistant]` installation error when absent. `uv.lock` was
refreshed without installing OpenAI into the base preprocessing dependency set.

Regression-first red command:

```bash
/tmp/satquery-nonlinear-env/bin/python -m pytest \
  tests/test_text_adaptation.py::test_prepared_manifest_cannot_relabel_heldout_records_as_training \
  tests/test_text_adaptation.py::test_embedding_provenance_is_recursive_nonempty_and_bounded \
  tests/test_text_adaptation.py::test_missing_openai_extra_has_actionable_error \
  tests/test_task_evaluation.py::test_rejects_unknown_task_instead_of_silently_leaving_it_unscored -q
```

Red result before fixes: `6 failed` (held-out relabel accepted; all three invalid
provenance cases accepted; raw `ModuleNotFoundError`; unknown task accepted).

The independent semantic-relationship validator was mutation-checked by
temporarily removing its load-time call and running:

```bash
/tmp/satquery-nonlinear-env/bin/python -m pytest \
  tests/test_text_adaptation.py::test_prepared_loader_rejects_internally_bound_invalid_split_relationship -q
```

Red result: `1 failed` because an internally digested test caption relabeled as
train was accepted. Restoring the validator produced `1 passed`.

Focused green checks during the fixes:

- Semantic digest plus held-out relationship guards: `3 passed`, then the two
  dedicated relabel regressions: `2 passed`.
- Recursive/bounded provenance and cache provenance: `5 passed`.
- Fixed task enum plus existing evaluation behavior: `2 passed`.
- Missing OpenAI extra error: `1 passed`.

No live OpenAI request or real adaptation training was performed in this review
round.

Final review-round verification:

```bash
/tmp/satquery-nonlinear-env/bin/python -m ruff check \
  src/satquery/assistant/text_adaptation.py \
  src/satquery/assistant/task_evaluation.py scripts/colab_adapt_text.py \
  tests/test_text_adaptation.py tests/test_task_evaluation.py
/tmp/satquery-nonlinear-env/bin/python -m pytest \
  tests/test_text_adaptation.py tests/test_task_evaluation.py -q
/tmp/satquery-nonlinear-env/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/satquery-task3-round1-wheel-20260914
```

Results: Ruff clean; `25 passed` focused; `142 passed, 5 skipped` repository-wide
in 8.69 seconds. The wheel built successfully, imported the packaged adaptation
module after extraction, and its metadata contains both `Provides-Extra:
assistant` and `Requires-Dist: openai>=1.68; extra == 'assistant'`.
