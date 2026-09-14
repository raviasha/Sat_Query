import numpy as np
import pytest
import torch

from satquery.evaluation import TokenSplit, predict_rows
from satquery.prediction import CoverageHead, load_head, save_head
from satquery.validation_training import fit_with_validation


def xor_splits():
    x = torch.full((64, 768), 3.0)
    x[:, :2] = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]).repeat(16, 1)
    y = torch.zeros(64, 19)
    same = x[:, 0] == x[:, 1]
    y[:, 0] = torch.where(same, 0.8, 0.2)
    y[:, 1] = 1 - y[:, 0]
    return (
        TokenSplit("train", x, y, np.array(["a"] * 64), ("a",)),
        TokenSplit("validation", x.clone(), y.clone(), np.array(["b"] * 64), ("b",)),
    )


def test_mlp_learns_nonlinear_mixtures_and_round_trips(tmp_path):
    train, val = xor_splits()
    opts = {
        "architecture": "mlp",
        "max_epochs": 100,
        "patience": 30,
        "learning_rate": 0.01,
        "batch_size": 64,
        "seed": 7,
    }
    model, report = fit_with_validation(train, val, **opts)
    pred = predict_rows(model, val.x)
    assert (pred[:, :2] - val.y[:, :2]).abs().max() < 0.08
    assert report["folded_normalization_max_abs_error"] < 1e-4
    repeat, repeated = fit_with_validation(train, val, **opts)
    torch.testing.assert_close(pred, predict_rows(repeat, val.x), rtol=0, atol=0)
    assert repeated["selected_epoch"] == report["selected_epoch"]
    save_head(tmp_path / "mlp.pt", model, report)
    restored, metadata = load_head(tmp_path / "mlp.pt")
    torch.testing.assert_close(pred, predict_rows(restored, val.x), rtol=0, atol=0)
    assert metadata["architecture"] == "mlp"
    with pytest.raises(FileExistsError):
        save_head(tmp_path / "mlp.pt", model, report)


def test_old_linear_checkpoint_and_bad_architecture(tmp_path):
    model = CoverageHead()
    x = torch.randn(3, 768)
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format_version": 1,
            "architecture": "linear_768_19",
            "metadata": {},
            "state_dict": model.state_dict(),
        },
        path,
    )
    restored, _ = load_head(path)
    torch.testing.assert_close(restored.predict(x), model.predict(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match="architecture"):
        CoverageHead(architecture="unsupported")


def test_mlp_uses_existing_inference_export(artifacts, tmp_path):
    from satquery.predict_cover import export_predictions
    from satquery.prediction_data import FeatureBatches, class_schema

    features, _ = artifacts
    reader = FeatureBatches(features)
    model = CoverageHead(architecture="mlp")
    head = tmp_path / "mlp.pt"
    save_head(head, model, {"feature_contract": reader.contract, "classes": class_schema()})
    out = tmp_path / "predictions"
    export_predictions(features, head, out)
    result = torch.load(out / "predictions.pt", weights_only=True)
    assert result["class_fractions"].shape == (3, 225, 19)
    torch.testing.assert_close(result["class_fractions"].sum(-1), torch.ones(3, 225))


def test_mlp_rejects_test_selection():
    train, val = xor_splits()
    val.split = "test"
    with pytest.raises(ValueError, match="split"):
        fit_with_validation(train, val, architecture="mlp", max_epochs=1)


def test_comparison_reuses_completed_runs_and_locks_inputs(artifacts, tmp_path):
    from satquery.compare_heads import run_comparison

    features, targets = artifacts
    root = tmp_path / "comparison"
    report = run_comparison(
        features,
        targets,
        root,
        seeds=(17, 29),
        max_epochs=2,
        patience=2,
        batch_size=256,
        bootstrap_repeats=10,
    )
    assert report["passed"] and len(report["runs"]) == 4
    assert report["selection"]["test_used_for_selection"] is False
    assert set(report["summary"]) == {"linear", "mlp"}
    paths = list(root.glob("*/seed-*/head.pt"))
    assert len(paths) == 4
    mtimes = [p.stat().st_mtime_ns for p in paths]
    repeated = run_comparison(
        features,
        targets,
        root,
        seeds=(17, 29),
        max_epochs=2,
        patience=2,
        batch_size=256,
        bootstrap_repeats=10,
    )
    assert repeated == report
    assert [p.stat().st_mtime_ns for p in paths] == mtimes
    with pytest.raises(ValueError, match="configuration"):
        run_comparison(
            features,
            targets,
            root,
            seeds=(17,),
            max_epochs=2,
            patience=2,
            batch_size=256,
            bootstrap_repeats=10,
        )
