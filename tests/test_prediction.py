import json

import pytest
import torch


def test_head_shape_probabilities_and_gradients():
    from satquery.prediction import CoverageHead, coverage_loss

    model = CoverageHead()
    x = torch.randn(2, 225, 768)
    logits = model(x)
    assert logits.shape == (2, 225, 19)
    y = torch.zeros_like(logits)
    y[..., 2], y[..., 8] = 0.25, 0.75
    loss = coverage_loss(logits, y)
    loss.backward()
    assert model.linear.weight.grad is not None
    assert torch.isfinite(model.linear.weight.grad).all()
    predictions = model.predict(x)
    assert predictions.shape == y.shape and not predictions.requires_grad
    assert (predictions >= 0).all()
    torch.testing.assert_close(predictions.sum(-1), torch.ones(2, 225))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 767))
    with pytest.raises(ValueError):
        model(torch.full((1, 768), float("nan")))


def test_fractional_loss_matches_hand_computed_values():
    from satquery.prediction import coverage_loss

    # Uniform predictions over 19 classes yield log(19), including mixed targets.
    y = torch.zeros(2, 19)
    y[:, 0] = 0.25
    y[:, 1] = 0.75
    assert coverage_loss(torch.zeros(2, 19), y).item() == pytest.approx(2.944438979, abs=1e-6)
    for bad in [y * 0.5, -y, torch.full_like(y, float("nan"))]:
        with pytest.raises(ValueError):
            coverage_loss(torch.zeros(2, 19), bad)


def test_training_learns_mixtures_reproducibly_and_checkpoint_round_trip(tmp_path):
    from satquery.prediction import CoverageHead, load_head, save_head
    from satquery.train_prediction import train_head

    x = torch.zeros(32, 768)
    x[:16, 0], x[16:, 1] = 1, 1
    y = torch.zeros(32, 19)
    y[:16, 2], y[:16, 8] = 0.25, 0.75
    y[16:, 2], y[16:, 8] = 0.8, 0.2
    source = lambda: iter([(x, y)])
    model, history = train_head(source, epochs=120, learning_rate=0.1, batch_size=32, seed=7)
    assert history[-1]["loss"] < history[0]["loss"] * 0.4
    assert (model.predict(x) - y).abs().mean().item() < 0.005
    repeat, _ = train_head(source, epochs=120, learning_rate=0.1, batch_size=32, seed=7)
    torch.testing.assert_close(repeat.predict(x), model.predict(x), rtol=0, atol=0)
    path = tmp_path / "head.pt"
    save_head(path, model, {"purpose": "synthetic_test"})
    restored, metadata = load_head(path)
    assert isinstance(restored, CoverageHead) and metadata["purpose"] == "synthetic_test"
    torch.testing.assert_close(restored.predict(x), model.predict(x), rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_head(path, model, {})


@pytest.fixture
def artifacts(tmp_path):
    from satquery.preprocessing import sha256
    from satquery.targets import export_targets

    prepared, features, targets = [tmp_path / name for name in ("prepared", "features", "targets")]
    prepared.mkdir()
    features.mkdir()
    samples = [
        {
            "patch_id": f"p{i}",
            "s1_name": f"s{i}",
            "split": split,
            "crs": "EPSG:32633",
            "transform": [10, 0, 373200, 0, -10, 5353200],
            "bounds": [373200, 5352000, 374400, 5353200],
        }
        for i, split in enumerate(["train", "validation", "test"])
    ]

    def write(p, value):
        p.write_text(json.dumps(value))

    write(prepared / "batch.json", {"sample_count": 3, "samples": samples})
    maps = torch.full((3, 120, 120), 311)
    maps[0, :8, :8] = 999
    torch.save(maps, prepared / "reference_maps.pt")
    pb = {
        "directory": ".",
        "start_index": 0,
        "sample_count": 3,
        "sha256": {n: sha256(prepared / n) for n in ["batch.json", "reference_maps.pt"]},
    }
    write(prepared / "manifest.json", {"format_version": 1, "sample_count": 3, "batches": [pb]})
    tensors = {
        f"{m}_{k}": torch.full((3, 225, 768) if k == "encodings" else (3, 768), float(i))
        for i, m in enumerate(["optical", "SAR", "joint"])
        for k in ["encodings", "GAP"]
    }
    tensors["joint_encodings"][1] = 100
    tensors["joint_encodings"][2] = 200
    torch.save(tensors, features / "features.pt")
    write(
        features / "batch.json",
        {
            "sample_count": 3,
            "samples": samples,
            "spatial_grid": {
                "height": 15,
                "width": 15,
                "token_order": "row-major",
                "input_patch_pixels": [8, 8],
            },
        },
    )
    fm = {
        "format_version": 1,
        "sample_count": 3,
        "feature_dimension": 768,
        "model": {"kind": "fixture", "checkpoint_sha256": "fixture-checkpoint"},
        "channel_profile": {"optical": ["fixture"], "sar": ["fixture"]},
        "normalization_profile": "fixture-normalization",
        "source_manifest_sha256": sha256(prepared / "manifest.json"),
        "batches": [
            {
                "directory": ".",
                "start_index": 0,
                "sample_count": 3,
                "sha256": {n: sha256(features / n) for n in ["batch.json", "features.pt"]},
            }
        ],
    }
    write(features / "manifest.json", fm)
    export_targets(prepared, features, targets)
    return features, targets


def test_training_reader_excludes_held_out_and_incomplete_tokens(artifacts):
    from satquery.prediction_data import TrainingPairs

    features, targets = artifacts
    pairs = TrainingPairs(features, targets)
    batches = list(pairs)
    assert len(batches) == 1
    x, y = batches[0]
    assert x.shape == (224, 768) and y.shape == (224, 19)
    assert (x == 2).all()  # held-out features have deliberately different values
    assert (y[:, 8] == 1).all()


def test_reader_rejects_target_tampering_and_no_training_samples(artifacts):
    from satquery.prediction_data import TrainingPairs
    from satquery.train_prediction import train_head

    features, targets = artifacts
    with (targets / "targets.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        list(TrainingPairs(features, targets))
    with pytest.raises(ValueError, match="training tokens"):
        train_head(lambda: iter([]), epochs=1)


def test_training_and_prediction_commands(artifacts, tmp_path):
    import subprocess
    import sys

    features, targets = artifacts
    model = tmp_path / "model"
    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "satquery.train_prediction",
            "--features",
            str(features),
            "--targets",
            str(targets),
            "--output",
            str(model),
            "--epochs",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert train.returncode == 0, train.stderr
    output = tmp_path / "predictions"
    infer = subprocess.run(
        [
            sys.executable,
            "-m",
            "satquery.predict_cover",
            "--features",
            str(features),
            "--checkpoint",
            str(model / "head.pt"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert infer.returncode == 0, infer.stderr
    values = torch.load(output / "predictions.pt", weights_only=True)
    assert values["class_fractions"].shape == (3, 225, 19)
    torch.testing.assert_close(values["class_fractions"].sum(-1), torch.ones(3, 225))
    assert values["dominant_class"].shape == (3, 225)
    info = json.loads((output / "batch.json").read_text())
    assert [s["patch_id"] for s in info["samples"]] == ["p0", "p1", "p2"]
    assert json.loads((model / "training.json").read_text())["history"][-1]["token_count"] == 224


def test_prediction_rejects_different_feature_contract(artifacts, tmp_path):
    from satquery.predict_cover import export_predictions
    from satquery.prediction import CoverageHead, save_head

    features, _ = artifacts
    checkpoint = tmp_path / "head.pt"
    save_head(
        checkpoint,
        CoverageHead(),
        {"feature_contract": {"checkpoint_sha256": "different"}, "feature_key": "joint_encodings"},
    )
    with pytest.raises(ValueError, match="contract"):
        export_predictions(features, checkpoint, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_explicit_demo_mode_includes_held_out_samples(artifacts):
    from satquery.prediction_data import TrainingPairs

    x, y = next(iter(TrainingPairs(*artifacts, demo_fit_all=True)))
    assert x.shape == (674, 768) and y.shape == (674, 19)
    assert (x == 100).any() and (x == 200).any()
