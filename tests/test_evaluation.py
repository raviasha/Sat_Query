import numpy as np
import pytest
import torch


def test_per_class_errors_support_and_area_bootstrap():
    from satquery.evaluation import coverage_metrics

    truth = np.zeros((4, 19))
    truth[:, 0] = [1, 0.5, 0, 0]
    truth[:, 1] = 1 - truth[:, 0]
    pred = truth.copy()
    pred[:, 0] = [0.8, 0.6, 0.1, 0]
    pred[:, 1] = 1 - pred[:, 0]
    report = coverage_metrics(
        pred, truth, np.array(["a", "a", "b", "b"]), bootstrap_repeats=100, seed=3
    )
    a = report["classes"][0]
    assert a["mae_pp"] == pytest.approx(10)
    assert a["rmse_pp"] == pytest.approx(np.sqrt(0.015) * 100)
    assert a["bias_pp"] == pytest.approx(0)
    assert a["present_mae_pp"] == pytest.approx(15)
    assert a["present_tokens"] == 2 and a["present_areas"] == 1
    assert a["within_5pp_fraction"] == 0.25
    assert a["mae_pp_area_bootstrap_95ci"] == pytest.approx([5, 15])
    assert report["classes"][2]["present_mae_pp"] is None
    assert report["classes"][2]["dominant_recall"] is None
    assert report["dominant_evaluated_tokens"] == 3  # exclude true tied dominant classes
    assert report["dominant_class_accuracy"] == 1
    assert report["classes"][2]["support_status"] == "no_test_presence"
    for bad in (pred * 2, np.full_like(pred, np.nan)):
        with pytest.raises(ValueError):
            coverage_metrics(bad, truth, np.array(["a", "a", "b", "b"]))


def test_verified_reader_preserves_area_and_split_alignment(artifacts):
    from satquery.evaluation import load_splits
    from satquery.prediction_data import TrainingPairs

    features, targets = artifacts
    splits, provenance = load_splits(features, targets)
    assert set(splits) == {"train", "validation", "test"}
    assert len(splits["train"].x) == 224
    assert len(splits["test"].x) == 225
    assert splits["train"].patch_ids == ("p0",)
    assert splits["validation"].patch_ids == ("p1",)
    assert set(splits["test"].area_ids) == {"p2"}
    assert provenance["feature_key"] == "joint_encodings"
    assert sum(len(x) for x, y in TrainingPairs(features, targets)) == 224


def test_validation_fit_uses_train_only_and_folds_normalization():
    from satquery.evaluation import TokenSplit, predict_rows
    from satquery.validation_training import fit_with_validation

    torch.manual_seed(4)
    x = torch.ones(32, 768)  # Nonzero constant columns must not amplify folded weights.
    x[:, 0] = torch.arange(32) / 10 + 20
    y = torch.zeros(32, 19)
    y[:, 0] = torch.linspace(0.1, 0.9, 32)
    y[:, 1] = 1 - y[:, 0]
    train = TokenSplit("train", x, y, np.array(["a"] * 32), ("a",))
    val = TokenSplit("validation", x.clone(), y.clone(), np.array(["b"] * 32), ("b",))
    model, report = fit_with_validation(
        train, val, max_epochs=20, patience=20, learning_rate=0.1, batch_size=32
    )
    assert report["selected_epoch"] >= 1
    assert report["history"][-1]["validation_loss"] < report["history"][0]["validation_loss"]
    assert torch.isfinite(predict_rows(model, x)).all()
    assert report["folded_normalization_max_abs_error"] < 1e-4
    test = TokenSplit("test", x, y, np.array(["c"] * 32), ("c",))
    with pytest.raises(ValueError, match="split"):
        fit_with_validation(train, test, max_epochs=1)
    overlap = TokenSplit("validation", x, y, np.array(["a"] * 32), ("a",))
    with pytest.raises(ValueError, match="overlap"):
        fit_with_validation(train, overlap, max_epochs=1)


def test_inference_attaches_only_matching_test_reliability(artifacts, tmp_path):
    import json

    from satquery.predict_cover import export_predictions
    from satquery.prediction import CoverageHead, save_head
    from satquery.prediction_data import FeatureBatches, class_schema
    from satquery.preprocessing import sha256

    features, _ = artifacts
    reader = FeatureBatches(features)
    head = tmp_path / "head.pt"
    save_head(
        head, CoverageHead(), {"feature_contract": reader.contract, "classes": class_schema()}
    )
    report = {
        "format_version": 1,
        "checkpoint_sha256": sha256(head),
        "feature_contract": reader.contract,
        "classes": class_schema(),
        "evaluation_scope": "held_out_test_areas",
        "prediction_confidence_available": False,
        "class_metrics": [{"index": i, "mae_pp": 10.0} for i in range(19)],
    }
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(report))
    out = tmp_path / "predictions"
    manifest = export_predictions(features, head, out, reliability=path)
    assert manifest["historical_class_reliability"]["sha256"] == sha256(out / "reliability.json")
    assert (
        json.loads((out / "reliability.json").read_text())["prediction_confidence_available"]
        is False
    )
    report["checkpoint_sha256"] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="reliability"):
        export_predictions(features, head, tmp_path / "bad", reliability=path)
