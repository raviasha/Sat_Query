import json

import pytest


def _jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_evaluates_explicit_records_without_joint_score(tmp_path):
    from satquery.assistant.task_evaluation import evaluate_task_records

    references = [
        {
            "id": "q1",
            "dataset": "RSVQA",
            "split": "test",
            "task": "vqa",
            "answer": "A Forest!",
            "abstention": False,
        },
        {
            "id": "q2",
            "dataset": "RSVQA",
            "split": "test",
            "task": "vqa",
            "answer": "water",
            "abstention": False,
        },
        {
            "id": "a1",
            "dataset": "SAC",
            "split": "test",
            "task": "area",
            "answer": 12.0,
            "unit": "hectare",
            "abstention": False,
            "class_fractions": {"forest": 0.25, "water": 0.75},
        },
        {
            "id": "c1",
            "dataset": "CDVQA",
            "split": "test",
            "task": "change",
            "answer": "changed",
            "abstention": False,
            "change_labels": ["new construction"],
        },
    ]
    predictions = [
        {
            "id": "q1",
            "dataset": "RSVQA",
            "split": "test",
            "task": "vqa",
            "answer": "a forest",
            "abstention": False,
        },
        {
            "id": "q2",
            "dataset": "RSVQA",
            "split": "test",
            "task": "vqa",
            "answer": None,
            "abstention": True,
        },
        {
            "id": "a1",
            "dataset": "SAC",
            "split": "test",
            "task": "area",
            "answer": 10.0,
            "unit": "hectare",
            "abstention": False,
            "class_fractions": {"forest": 0.35, "water": 0.65},
        },
        {
            "id": "c1",
            "dataset": "CDVQA",
            "split": "test",
            "task": "change",
            "answer": "changed",
            "abstention": False,
            "change_labels": ["new construction"],
        },
    ]
    reference_path, prediction_path = tmp_path / "references.jsonl", tmp_path / "predictions.jsonl"
    _jsonl(reference_path, references)
    _jsonl(prediction_path, predictions)
    report = evaluate_task_records(reference_path, prediction_path)

    assert report["matching"] == {"reference_count": 4, "prediction_count": 4, "matched_count": 4}
    assert report["coverage"] == pytest.approx(0.75)
    assert report["abstention_count"] == 1
    assert report["vqa"]["metric"] == "provisional_normalized_exact_match_not_official_scorer"
    assert report["vqa"]["answered_exact_match"] == 1.0
    assert report["vqa"]["coverage"] == 0.5
    assert report["numerical_area_mae"] == {"hectare": {"count": 1, "mae": 2.0}}
    assert report["class_fraction_mae"] == {
        "count": 2,
        "unit": "fraction_of_scene",
        "mae": pytest.approx(0.1),
        "percentage_point_mae": pytest.approx(10.0),
    }
    assert report["change_labels"]["exact_set_match"] == 1.0
    assert report["per_task_counts"] == {"area": 1, "change": 1, "vqa": 2}
    assert "joint_score" not in report


@pytest.mark.parametrize("mutation", ["duplicate", "split", "missing"])
def test_strict_identity_split_and_completeness_guards(tmp_path, mutation):
    from satquery.assistant.task_evaluation import evaluate_task_records

    reference = {
        "id": "q1",
        "dataset": "RSVQA",
        "split": "test",
        "task": "vqa",
        "answer": "forest",
        "abstention": False,
    }
    prediction = dict(reference)
    refs, preds = [reference], [prediction]
    if mutation == "duplicate":
        preds.append(dict(prediction))
    elif mutation == "split":
        preds[0]["split"] = "validation"
    else:
        preds.clear()
    rp, pp = tmp_path / "r.jsonl", tmp_path / "p.jsonl"
    _jsonl(rp, refs)
    _jsonl(pp, preds)
    with pytest.raises(ValueError, match="duplicate|matching|missing|split"):
        evaluate_task_records(rp, pp)


def test_rejects_cross_split_id_leakage_and_implicit_area_units(tmp_path):
    from satquery.assistant.task_evaluation import evaluate_task_records

    refs = [
        {
            "id": "same",
            "dataset": "SAC",
            "split": "train",
            "task": "area",
            "answer": 1.0,
            "abstention": False,
        },
        {
            "id": "same",
            "dataset": "SAC",
            "split": "test",
            "task": "area",
            "answer": 2.0,
            "abstention": False,
        },
    ]
    rp, pp = tmp_path / "r.jsonl", tmp_path / "p.jsonl"
    _jsonl(rp, refs)
    _jsonl(pp, refs)
    with pytest.raises(ValueError, match="leakage|duplicate"):
        evaluate_task_records(rp, pp)

    one = [
        {
            "id": "a",
            "dataset": "SAC",
            "split": "test",
            "task": "area",
            "answer": 2.0,
            "abstention": False,
        }
    ]
    _jsonl(rp, one)
    _jsonl(pp, one)
    with pytest.raises(ValueError, match="unit"):
        evaluate_task_records(rp, pp)


def test_rejects_unknown_task_instead_of_silently_leaving_it_unscored(tmp_path):
    from satquery.assistant.task_evaluation import evaluate_task_records

    record = {
        "id": "q1",
        "dataset": "RSVQA",
        "split": "test",
        "task": "vqaa",
        "answer": "forest",
        "abstention": False,
    }
    rp, pp = tmp_path / "r.jsonl", tmp_path / "p.jsonl"
    _jsonl(rp, [record])
    _jsonl(pp, [record])
    with pytest.raises(ValueError, match="Unsupported task"):
        evaluate_task_records(rp, pp)


def test_benchmark_compatibility_manifest_is_candid_about_gaps():
    from satquery.assistant.task_evaluation import benchmark_compatibility_manifest

    manifest = benchmark_compatibility_manifest()
    assert set(manifest["benchmarks"]) == {"VRSBench", "RSVQA", "CDVQA", "SAC"}
    assert (
        manifest["benchmarks"]["CDVQA"]["expected_modality"]
        == "paired temporal imagery plus question"
    )
    assert (
        manifest["benchmarks"]["SAC"]["expected_modality"]
        == "pre-georeferenced co-registered Cartosat-2S optical and RISAT SAR evaluation inputs"
    )
    assert "hidden evaluation" in manifest["benchmarks"]["SAC"]["current_gap"].lower()
    assert all(item["status"] == "unsupported" for item in manifest["benchmarks"].values())
    assert "adapter" in manifest["benchmarks"]["VRSBench"]["current_gap"].lower()


def test_abstention_does_not_require_optional_prediction_payload(tmp_path):
    from satquery.assistant.task_evaluation import evaluate_task_records

    reference = {
        "id": "a1",
        "dataset": "SAC",
        "split": "test",
        "task": "area",
        "answer": 4.0,
        "unit": "hectare",
        "abstention": False,
        "class_fractions": {"forest": 1.0},
    }
    prediction = {
        "id": "a1",
        "dataset": "SAC",
        "split": "test",
        "task": "area",
        "answer": None,
        "abstention": True,
    }
    rp, pp = tmp_path / "r.jsonl", tmp_path / "p.jsonl"
    _jsonl(rp, [reference])
    _jsonl(pp, [prediction])
    report = evaluate_task_records(rp, pp)
    assert report["coverage"] == 0
    assert report["numerical_area_mae"] == {}
    assert report["class_fraction_mae"]["count"] == 0
