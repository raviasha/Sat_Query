import json

import pandas as pd
import pytest


def test_matches_both_ids_preserves_all_rows_and_holds_out_benchmark(tmp_path):
    from satquery.match_annotations import match_annotations

    images = pd.DataFrame(
        {
            "patch_id": ["p1", "p2", "p3", "p4"],
            "s1_name": ["s1", "s2", "s3", "s4"],
            "split": ["train", "test", "train", "validation"],
            "country": ["A"] * 4,
        }
    )
    selection = tmp_path / "metadata.parquet"
    images.to_parquet(selection, index=False)
    rows = []
    for i, (p, s, split, kind) in enumerate(
        [
            ("p1", "s1", "train", "captioning"),
            ("p1", "s1", "train", "binary"),
            ("p2", "s2", "test", "mcq"),
            ("p3", "s3", "bench", "binary"),
            ("p3", "s3", "train", "captioning"),
            ("other", "other", "train", "binary"),
        ]
    ):
        rows.append(
            {
                "ID": i,
                "patch_id": p,
                "s1_name": s,
                "split": split,
                "type": kind,
                "category": "presence",
                "input": "Does it show forest?",
                "output": "yes",
                "country": "A",
            }
        )
    source = tmp_path / "source.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    result = match_annotations(
        selection, source, tmp_path / "matched", source_revision="test-revision", batch_size=2
    )
    assert result["matched_image_count"] == 3 and result["missing_image_count"] == 1
    assert result["annotation_count"] == 5 and result["training_eligible_annotation_count"] == 2
    matched = pd.read_parquet(tmp_path / "matched" / "annotations.parquet")
    assert matched.ID.tolist() == [0, 1, 2, 3, 4]
    assert matched.loc[matched.patch_id == "p3", "training_eligible"].tolist() == [False, False]
    assert set(matched.loc[matched.patch_id == "p3", "use_partition"]) == {"bench_holdout"}
    assert result["missing_patch_ids"] == ["p4"]
    scenes = [
        json.loads(s)
        for s in (tmp_path / "matched" / "image-annotations.jsonl").read_text().splitlines()
    ]
    assert (
        len(scenes) == 4 and len(scenes[0]["annotations"]) == 2 and scenes[3]["annotations"] == []
    )
    assert scenes[0]["image_split"] == "train"
    with pytest.raises(FileExistsError):
        match_annotations(selection, source, tmp_path / "matched", source_revision="test-revision")


def test_rejects_wrong_sar_pair_and_duplicate_annotation_ids(tmp_path):
    from satquery.match_annotations import match_annotations

    image = tmp_path / "metadata.parquet"
    pd.DataFrame([{"patch_id": "p1", "s1_name": "s1", "split": "train"}]).to_parquet(image, index=False)
    row = {
        "ID": 1,
        "patch_id": "p1",
        "s1_name": "wrong",
        "split": "train",
        "type": "binary",
        "input": "Forest?",
        "output": "yes",
        "category": "presence",
    }
    source = tmp_path / "source.parquet"
    pd.DataFrame([row]).to_parquet(source, index=False)
    with pytest.raises(ValueError, match="SAR"):
        match_annotations(image, source, tmp_path / "bad", source_revision="test")
    assert not (tmp_path / "bad").exists()
    row["s1_name"] = "s1"
    pd.DataFrame([row, row]).to_parquet(source, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        match_annotations(
            image, source, tmp_path / "duplicate", source_revision="test", batch_size=1
        )


def test_split_mismatch_is_retained_but_never_training_eligible(tmp_path):
    from satquery.match_annotations import match_annotations

    image = tmp_path / "metadata.parquet"
    source = tmp_path / "source.parquet"
    pd.DataFrame([{"patch_id": "p", "s1_name": "s", "split": "train"}]).to_parquet(image, index=False)
    pd.DataFrame(
        [
            {
                "ID": 1,
                "patch_id": "p",
                "s1_name": "s",
                "split": "test",
                "type": "binary",
                "input": "Forest?",
                "output": "no",
            }
        ]
    ).to_parquet(source, index=False)
    result = match_annotations(image, source, tmp_path / "matched", source_revision="test")
    assert result["training_eligible_annotation_count"] == 0
    assert result["split_conflict_patch_ids"] == ["p"]
