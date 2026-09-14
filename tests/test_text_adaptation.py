import builtins
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def _row(**values):
    return values


def _refresh_annotation_hash(root):
    report_path = root / "report.json"
    report = json.loads(report_path.read_text())
    report["files"] = {"annotations.parquet": _sha256(root / "annotations.parquet")}
    _write_json(report_path, report)


def _semantic_digest(manifest):
    payload = {
        "feature_key": manifest["feature_key"],
        "requested_splits": manifest["requested_splits"],
        "image_records": manifest["image_records"],
        "caption_records": manifest["caption_records"],
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _feature_artifact(tmp_path):
    root = tmp_path / "features"
    root.mkdir()
    samples = [
        {"patch_id": f"p{i}", "s1_name": f"s{i}", "split": split}
        for i, split in enumerate(["train", "validation", "test", "train", "train"], start=1)
    ]
    values = torch.zeros(5, 225, 768)
    for i in range(5):
        values[i, :, i] = float(i + 1)
    torch.save(
        {
            "joint_encodings": values,
            "optical_encodings": values + 10,
            "SAR_encodings": values + 20,
        },
        root / "features.pt",
    )
    _write_json(
        root / "batch.json",
        {
            "sample_count": 5,
            "samples": samples,
            "spatial_grid": {
                "height": 15,
                "width": 15,
                "token_order": "row-major",
                "input_patch_pixels": [8, 8],
            },
        },
    )
    manifest = {
        "format_version": 1,
        "sample_count": 5,
        "feature_dimension": 768,
        "model": {"checkpoint_sha256": "fixture-checkpoint"},
        "channel_profile": {"optical": ["fixture"], "sar": ["fixture"]},
        "normalization_profile": "fixture-normalization",
        "batches": [
            {
                "directory": ".",
                "start_index": 0,
                "sample_count": 5,
                "sha256": {
                    "features.pt": _sha256(root / "features.pt"),
                    "batch.json": _sha256(root / "batch.json"),
                },
            }
        ],
    }
    _write_json(root / "manifest.json", manifest)
    return root


def _annotation_artifact(tmp_path):
    root = tmp_path / "annotations"
    root.mkdir(parents=True)
    rows = [
        # Same-image duplicate captions must collapse deterministically.
        _row(
            ID=2,
            patch_id="p1",
            s1_name="s1",
            split="train",
            type="captioning",
            output="A forest scene",
            input="Caption",
            image_split="train",
            use_partition="train",
            training_eligible=True,
        ),
        _row(
            ID=1,
            patch_id="p1",
            s1_name="s1",
            split="train",
            type="captioning",
            output="A forest scene",
            input="Caption",
            image_split="train",
            use_partition="train",
            training_eligible=True,
        ),
        _row(
            ID=17046,
            patch_id="p1",
            s1_name="s1",
            split="train",
            type="captioning",
            output="Known faulty caption",
            input="Caption",
            image_split="train",
            use_partition="train",
            training_eligible=True,
        ),
        _row(
            ID=3,
            patch_id="p1",
            s1_name="s1",
            split="train",
            type="binary",
            output="yes",
            input="Forest?",
            image_split="train",
            use_partition="train",
            training_eligible=True,
        ),
        _row(
            ID=4,
            patch_id="p2",
            s1_name="s2",
            split="validation",
            type="captioning",
            output="Validation water",
            input="Caption",
            image_split="validation",
            use_partition="validation",
            training_eligible=False,
        ),
        _row(
            ID=5,
            patch_id="p3",
            s1_name="s3",
            split="test",
            type="captioning",
            output="Held out secret answer",
            input="Caption",
            image_split="test",
            use_partition="test",
            training_eligible=False,
        ),
        _row(
            ID=6,
            patch_id="p4",
            s1_name="s4",
            split="bench",
            type="captioning",
            output="Benchmark answer",
            input="Caption",
            image_split="train",
            use_partition="bench_holdout",
            training_eligible=False,
        ),
        _row(
            ID=7,
            patch_id="p5",
            s1_name="s5",
            split="train",
            type="captioning",
            output="Conflict answer",
            input="Caption",
            image_split="train",
            use_partition="split_conflict_holdout",
            training_eligible=False,
        ),
    ]
    pd.DataFrame(rows).to_parquet(root / "annotations.parquet", index=False)
    _write_json(
        root / "report.json",
        {
            "format_version": 1,
            "passed": True,
            "source_revision": "fixture-revision",
            "source_sha256": "a" * 64,
            "selected_metadata_sha256": "b" * 64,
            "files": {"annotations.parquet": _sha256(root / "annotations.parquet")},
        },
    )
    return root


class RecordingEmbedder:
    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def __call__(self, texts, *, model, dimensions):
        self.calls.extend(texts)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("interrupted")
        rows = []
        for text in texts:
            value = torch.zeros(dimensions)
            value[int(hashlib.sha256(text.encode()).hexdigest()[:4], 16) % dimensions] = 1
            rows.append(value.tolist())
        return rows


def test_prepare_quarantines_splits_deduplicates_and_records_provenance(tmp_path):
    from satquery.assistant.text_adaptation import load_prepared_pairs, prepare_text_pairs

    features = _feature_artifact(tmp_path)
    annotations = _annotation_artifact(tmp_path)
    output = tmp_path / "pairs"
    embedder = RecordingEmbedder()
    manifest = prepare_text_pairs(
        annotations,
        features,
        output,
        requested_splits=("train", "validation", "test"),
        embedder=embedder,
        embedding_provenance={"kind": "synthetic", "generator": "unit-test"},
    )

    assert embedder.calls == ["A forest scene", "Validation water", "Held out secret answer"]
    assert manifest["feature_key"] == "joint_encodings"
    assert manifest["embedding"] == {"model": "text-embedding-3-small", "dimensions": 512}
    assert manifest["excluded_annotation_ids"] == [17046, 2970546]
    assert manifest["counts_by_split"] == {"train": 1, "validation": 1, "test": 1}
    assert manifest["source_annotation_counts_by_split"] == {
        "bench": 1,
        "test": 1,
        "train": 5,
        "validation": 1,
    }
    assert manifest["source_image_counts_by_split"] == {
        "test": 1,
        "train": 3,
        "validation": 1,
    }
    assert manifest["quarantine_counts"]["bench_or_conflict"] == 2
    assert manifest["annotation_source"]["source_revision"] == "fixture-revision"
    assert len(manifest["source_hashes"]["annotations_parquet_sha256"]) == 64
    assert len(manifest["source_hashes"]["feature_manifest_sha256"]) == 64
    assert len(manifest["embedding_cache"]["manifest_sha256"]) == 64
    assert len(manifest["embedding_cache"]["entries_sha256"]) == 64

    pairs = load_prepared_pairs(output)
    assert pairs.image_features.shape == (3, 768)
    assert pairs.caption_embeddings.shape == (3, 512)
    assert [record["annotation_id"] for record in pairs.caption_records] == [1, 4, 5]
    assert [record["split"] for record in pairs.caption_records] == ["train", "validation", "test"]
    assert [record["training_eligible"] for record in pairs.caption_records] == [
        True,
        False,
        False,
    ]
    assert pairs.image_records[0]["patch_id"] == "p1"
    assert pairs.image_features[0, 0].item() == 1.0  # mean pooled from 225 tokens


def test_embedding_cache_resumes_by_content_model_and_dimension(tmp_path):
    from satquery.assistant.text_adaptation import prepare_text_pairs

    features = _feature_artifact(tmp_path)
    annotations = _annotation_artifact(tmp_path)
    output = tmp_path / "pairs"
    interrupted = RecordingEmbedder(fail_after=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_text_pairs(
            annotations,
            features,
            output,
            requested_splits=("train", "validation"),
            embedder=interrupted,
            embedding_provenance={"kind": "synthetic"},
            embedding_batch_size=1,
        )
    assert interrupted.calls == ["A forest scene", "Validation water"]

    resumed = RecordingEmbedder()
    manifest = prepare_text_pairs(
        annotations,
        features,
        output,
        requested_splits=("train", "validation"),
        embedder=resumed,
        embedding_provenance={"kind": "synthetic"},
        embedding_batch_size=1,
    )
    assert resumed.calls == ["Validation water"]
    assert manifest["embedding_cache"]["hits"] == 1
    assert manifest["embedding_cache"]["misses"] == 1


def test_embedding_cache_rejects_provenance_change_on_resume(tmp_path):
    from satquery.assistant.text_adaptation import prepare_text_pairs

    output = tmp_path / "pairs"
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_text_pairs(
            _annotation_artifact(tmp_path),
            _feature_artifact(tmp_path),
            output,
            requested_splits=("train", "validation"),
            embedder=RecordingEmbedder(fail_after=1),
            embedding_provenance={"kind": "synthetic", "generator": "first"},
            embedding_batch_size=1,
        )
    with pytest.raises(ValueError, match="provenance"):
        prepare_text_pairs(
            tmp_path / "annotations",
            tmp_path / "features",
            output,
            requested_splits=("train", "validation"),
            embedder=RecordingEmbedder(),
            embedding_provenance={"kind": "caller_supplied", "generator": "second"},
            embedding_batch_size=1,
        )


@pytest.mark.parametrize(
    "provenance",
    [
        {"kind": "synthetic", "details": {"api_key": "must-not-persist"}},
        {"kind": "  "},
        {"kind": "synthetic", "details": "x" * 5000},
    ],
)
def test_embedding_provenance_is_recursive_nonempty_and_bounded(tmp_path, provenance):
    from satquery.assistant.text_adaptation import prepare_text_pairs

    with pytest.raises((TypeError, ValueError), match="provenance|credentials|bounded|kind"):
        prepare_text_pairs(
            _annotation_artifact(tmp_path),
            _feature_artifact(tmp_path),
            tmp_path / "bad-provenance",
            requested_splits=("train",),
            embedder=RecordingEmbedder(),
            embedding_provenance=provenance,
        )


def test_prepare_rejects_identity_mismatch_and_unverified_source(tmp_path):
    from satquery.assistant.text_adaptation import prepare_text_pairs

    features = _feature_artifact(tmp_path)
    annotations = _annotation_artifact(tmp_path)
    frame = pd.read_parquet(annotations / "annotations.parquet")
    frame.loc[frame.ID == 1, "s1_name"] = "wrong"
    frame.to_parquet(annotations / "annotations.parquet", index=False)
    _refresh_annotation_hash(annotations)
    with pytest.raises(ValueError, match="SAR|identity"):
        prepare_text_pairs(
            annotations,
            features,
            tmp_path / "bad",
            requested_splits=("train",),
            embedder=RecordingEmbedder(),
            embedding_provenance={"kind": "synthetic"},
        )
    report = json.loads((annotations / "report.json").read_text())
    report["passed"] = False
    _write_json(annotations / "report.json", report)
    with pytest.raises(ValueError, match="verified"):
        prepare_text_pairs(
            annotations,
            features,
            tmp_path / "bad2",
            requested_splits=("train",),
            embedder=RecordingEmbedder(),
            embedding_provenance={"kind": "synthetic"},
        )


def test_prepare_rejects_image_split_mismatch_and_secret_provenance(tmp_path):
    from satquery.assistant.text_adaptation import prepare_text_pairs

    features = _feature_artifact(tmp_path)
    annotations = _annotation_artifact(tmp_path)
    frame = pd.read_parquet(annotations / "annotations.parquet")
    frame.loc[frame.patch_id == "p1", "image_split"] = "validation"
    frame.loc[frame.patch_id == "p1", "split"] = "validation"
    frame.loc[frame.patch_id == "p1", "use_partition"] = "validation"
    frame.loc[frame.patch_id == "p1", "training_eligible"] = False
    frame.to_parquet(annotations / "annotations.parquet", index=False)
    _refresh_annotation_hash(annotations)
    with pytest.raises(ValueError, match="image split"):
        prepare_text_pairs(
            annotations,
            features,
            tmp_path / "bad-split",
            requested_splits=("validation",),
            embedder=RecordingEmbedder(),
            embedding_provenance={"kind": "synthetic"},
        )
    with pytest.raises(ValueError, match="credentials"):
        prepare_text_pairs(
            _annotation_artifact(tmp_path / "fresh"),
            features,
            tmp_path / "bad-provenance",
            requested_splits=("train",),
            embedder=RecordingEmbedder(),
            embedding_provenance={"kind": "synthetic", "api_key": "must-not-persist"},
        )


def test_prepare_rejects_annotation_output_hash_tampering(tmp_path):
    from satquery.assistant.text_adaptation import prepare_text_pairs

    annotations = _annotation_artifact(tmp_path)
    frame = pd.read_parquet(annotations / "annotations.parquet")
    frame.loc[frame.ID == 1, "output"] = "tampered caption"
    frame.to_parquet(annotations / "annotations.parquet", index=False)
    with pytest.raises(ValueError, match="hash"):
        prepare_text_pairs(
            annotations,
            _feature_artifact(tmp_path),
            tmp_path / "tampered",
            requested_splits=("train",),
            embedder=RecordingEmbedder(),
            embedding_provenance={"kind": "synthetic"},
        )


def test_prepare_detects_annotation_source_change_during_embedding(tmp_path):
    from satquery.assistant.text_adaptation import prepare_text_pairs

    annotations = _annotation_artifact(tmp_path)
    delegate = RecordingEmbedder()

    def mutating_embedder(texts, *, model, dimensions):
        with (annotations / "report.json").open("a") as stream:
            stream.write(" ")
        return delegate(texts, model=model, dimensions=dimensions)

    with pytest.raises(ValueError, match="changed during"):
        prepare_text_pairs(
            annotations,
            _feature_artifact(tmp_path),
            tmp_path / "changed",
            requested_splits=("train",),
            embedder=mutating_embedder,
            embedding_provenance={"kind": "synthetic"},
        )


def test_prepare_detects_feature_batch_change_during_embedding(tmp_path):
    from satquery.assistant.text_adaptation import prepare_text_pairs

    annotations = _annotation_artifact(tmp_path)
    features = _feature_artifact(tmp_path)
    delegate = RecordingEmbedder()

    def mutating_embedder(texts, *, model, dimensions):
        with (features / "features.pt").open("ab") as stream:
            stream.write(b"tamper")
        return delegate(texts, model=model, dimensions=dimensions)

    with pytest.raises(ValueError, match="Feature source changed during"):
        prepare_text_pairs(
            annotations,
            features,
            tmp_path / "changed-features",
            requested_splits=("train",),
            embedder=mutating_embedder,
            embedding_provenance={"kind": "synthetic"},
        )


def test_training_updates_saves_best_and_keeps_heldout_answers_out_of_checkpoint(tmp_path):
    from satquery.assistant.text_adaptation import (
        TextProjection,
        load_text_adapter,
        prepare_text_pairs,
        top_k_training_captions,
        train_text_adapter,
    )

    pairs_dir = tmp_path / "pairs"
    prepare_text_pairs(
        _annotation_artifact(tmp_path),
        _feature_artifact(tmp_path),
        pairs_dir,
        requested_splits=("train", "validation", "test"),
        embedder=RecordingEmbedder(),
        embedding_provenance={"kind": "synthetic", "generator": "unit-test"},
    )
    torch.manual_seed(19)
    initial = TextProjection().state_dict()["input.weight"].clone()
    result = train_text_adapter(
        pairs_dir,
        tmp_path / "model",
        epochs=12,
        learning_rate=0.02,
        patience=4,
        seed=19,
    )
    assert result["selection_split"] == "validation"
    assert result["best_epoch"] <= len(result["history"])
    model, metadata = load_text_adapter(tmp_path / "model" / "best.pt")
    assert not torch.equal(model.state_dict()["input.weight"], initial)
    assert metadata["feature_normalization"]["fit_split"] == "train"
    assert (
        metadata["source_hashes"]
        == json.loads((pairs_dir / "manifest.json").read_text())["source_hashes"]
    )
    assert {r["split"] for r in metadata["training_caption_bank"]} == {"train"}
    assert b"Held out secret answer" not in (tmp_path / "model" / "best.pt").read_bytes()
    hits = top_k_training_captions(
        model,
        metadata,
        torch.ones(768),
        k=5,
    )
    assert [hit["caption"] for hit in hits] == ["A forest scene"]
    assert all(hit["score_kind"] == "cosine_similarity_not_calibrated_confidence" for hit in hits)
    with pytest.raises(FileExistsError):
        train_text_adapter(pairs_dir, tmp_path / "model", epochs=1)


def test_prepared_manifest_cannot_relabel_heldout_records_as_training(tmp_path):
    from satquery.assistant.text_adaptation import load_prepared_pairs, prepare_text_pairs

    pairs_dir = tmp_path / "pairs"
    prepare_text_pairs(
        _annotation_artifact(tmp_path),
        _feature_artifact(tmp_path),
        pairs_dir,
        requested_splits=("train", "validation", "test"),
        embedder=RecordingEmbedder(),
        embedding_provenance={"kind": "synthetic"},
    )
    manifest_path = pairs_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    test_image = next(record for record in manifest["image_records"] if record["patch_id"] == "p3")
    test_image["split"] = "train"
    test_caption = next(
        record for record in manifest["caption_records"] if record["patch_id"] == "p3"
    )
    test_caption.update(split="train", use_partition="train", training_eligible=True)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="semantic|split|eligibility"):
        load_prepared_pairs(pairs_dir)


def test_prepared_loader_rejects_internally_bound_invalid_split_relationship(tmp_path):
    from satquery.assistant.text_adaptation import load_prepared_pairs, prepare_text_pairs

    pairs_dir = tmp_path / "pairs"
    prepare_text_pairs(
        _annotation_artifact(tmp_path),
        _feature_artifact(tmp_path),
        pairs_dir,
        requested_splits=("train", "validation", "test"),
        embedder=RecordingEmbedder(),
        embedding_provenance={"kind": "synthetic"},
    )
    manifest_path = pairs_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    test_caption = next(
        record for record in manifest["caption_records"] if record["patch_id"] == "p3"
    )
    test_caption.update(split="train", use_partition="train", training_eligible=True)
    semantic_digest = _semantic_digest(manifest)
    manifest["semantic_records_sha256"] = semantic_digest
    pair_path = pairs_dir / "pairs.pt"
    tensors = torch.load(pair_path, map_location="cpu", weights_only=True)
    tensors["semantic_records_sha256"] = semantic_digest
    torch.save(tensors, pair_path)
    manifest["pairs_sha256"] = _sha256(pair_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="split|eligibility"):
        load_prepared_pairs(pairs_dir)


def test_missing_openai_extra_has_actionable_error(monkeypatch):
    from satquery.assistant.text_adaptation import OpenAIEmbedder

    original_import = builtins.__import__

    def missing_openai(name, *args, **kwargs):
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_openai)
    with pytest.raises(RuntimeError, match=r"\[assistant\]"):
        OpenAIEmbedder()


def test_heldout_retrieval_evaluation_reports_candidate_set_without_promoting(tmp_path):
    from satquery.assistant.text_adaptation import (
        evaluate_retrieval,
        prepare_text_pairs,
        train_text_adapter,
    )

    pairs_dir = tmp_path / "pairs"
    prepare_text_pairs(
        _annotation_artifact(tmp_path),
        _feature_artifact(tmp_path),
        pairs_dir,
        requested_splits=("train", "validation", "test"),
        embedder=RecordingEmbedder(),
        embedding_provenance={"kind": "synthetic"},
    )
    model_dir = tmp_path / "model"
    train_text_adapter(pairs_dir, model_dir, epochs=3, patience=2, seed=3)
    report = evaluate_retrieval(model_dir / "best.pt", pairs_dir, split="test")
    assert report["split"] == "test"
    assert report["candidate_set"]["image_count"] == 1
    assert report["counts"] == {"images": 1, "captions": 1}
    assert report["selection_use"] == "report_only_never_checkpoint_promotion"
    assert report["limitations"]["scores"] == "similarities, not calibrated confidence"
    assert report["ranks"][0]["patch_id"] == "p3"

    manifest_path = pairs_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["limitations"].append("tampered after training")
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="prepared manifest"):
        evaluate_retrieval(model_dir / "best.pt", pairs_dir, split="test")


def test_colab_driver_trains_from_prepared_pairs_under_supplied_edu_root(tmp_path):
    from satquery.assistant.text_adaptation import prepare_text_pairs

    pairs_dir = tmp_path / "separate-input" / "pairs"
    prepare_text_pairs(
        _annotation_artifact(tmp_path),
        _feature_artifact(tmp_path),
        pairs_dir,
        requested_splits=("train", "validation"),
        embedder=RecordingEmbedder(),
        embedding_provenance={"kind": "synthetic"},
    )
    edu_root = tmp_path / "EDU"
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository), str(repository / "src"), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "colab_adapt_text.py"),
            "--edu-p",
            str(edu_root),
            "train",
            "--prepared",
            str(pairs_dir),
            "--name",
            "fixture-adapter",
            "--epochs",
            "1",
            "--patience",
            "1",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (edu_root / "text-adaptation" / "fixture-adapter" / "best.pt").is_file()
    assert "fixture_or_caller_data_training_not_a_production_adaptation_claim" in result.stdout
