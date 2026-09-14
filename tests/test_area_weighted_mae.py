import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from satquery import area_weighted_mae as metric
from satquery.preprocessing import sha256


def test_weights_use_truth_and_blocks_receive_equal_weight():
    # First block: .02*8 + .80*8 + .18*0 = 6.56 pp.
    # Second block: .70*20 + .30*10 = 17 pp; absent water gets zero weight.
    truth = [[0.02, 0.80, 0.18], [0.70, 0.30, 0]]
    pred = [[0.10, 0.72, 0.18], [0.50, 0.40, 0.10]]
    assert metric.area_weighted_mae(pred, truth) == pytest.approx(11.78)
    assert metric.area_weighted_mae(truth, truth) == 0
    assert metric.area_weighted_mae([[0, 1]], [[1, 0]]) == 100


@pytest.mark.parametrize(
    "pred,truth",
    [
        ([], []),
        ([[0.5, 0.5]], [[1, 0], [1, 0]]),
        ([[50, 50]], [[1, 0]]),
        ([[0.4, 0.4]], [[1, 0]]),
        ([[float("nan"), 0]], [[1, 0]]),
        ([[1.1, -0.1]], [[1, 0]]),
        ([[1, 0]], [[0, 0]]),
        ([[1, 0]], [[-1, 2]]),
    ],
)
def test_rejects_invalid_or_unnormalized_fractions(pred, truth):
    with pytest.raises(ValueError):
        metric.area_weighted_mae(pred, truth)


def make_comparison(tmp_path):
    runs = []
    for architecture, seed, predictions in [
        ("linear", 17, [[0.5, 0.4, 0.1], [0, 1, 0]]),
        ("linear", 29, [[0.7, 0.3, 0], [1, 0, 0]]),
        ("mlp", 17, [[0.6, 0.3, 0.1], [0.9, 0.1, 0]]),
    ]:
        folder = tmp_path / architecture / f"seed-{seed}"
        folder.mkdir(parents=True)
        torch.save(
            {
                "predicted_fractions": torch.tensor(predictions),
                "true_fractions": torch.tensor([[0.7, 0.3, 0], [1, 0, 0]]),
                "area_ids": ["a", "b"],
            },
            folder / "test-predictions.pt",
        )
        (folder / "head.pt").write_bytes(b"checkpoint unchanged")
        report = {
            "test_predictions_sha256": sha256(folder / "test-predictions.pt"),
            "checkpoint_sha256": sha256(folder / "head.pt"),
        }
        (folder / "test-report.json").write_text(json.dumps(report))
        runs.append(
            {
                "architecture": architecture,
                "seed": seed,
                "folder": str(folder.relative_to(tmp_path)),
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
        )
    manifest = {"runs": runs, "selection": {"architecture": "mlp", "seed": 17}}
    (tmp_path / "comparison.json").write_text(json.dumps(manifest))
    return tmp_path


def test_cli_scores_existing_predictions_and_summarizes_seeds(tmp_path):
    root = make_comparison(tmp_path)
    output = root / "weighted.json"
    before = {p: sha256(p) for p in root.rglob("*") if p.is_file()}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "satquery.area_weighted_mae",
            "--comparison",
            str(root),
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["summary"]["linear"]["mean"] == pytest.approx(29.25)
    assert report["summary"]["mlp"]["mean"] == pytest.approx(8.5)
    assert report["selected_checkpoint_score_pp"] == pytest.approx(8.5)
    assert all(sha256(p) == h for p, h in before.items())
    assert report["runs"][0]["block_count"] == 2
    assert report["runs"][0]["area_count"] == 2


def test_comparison_rejects_different_targets_even_with_valid_hash(tmp_path):
    root = make_comparison(tmp_path)
    folder = root / "mlp/seed-17"
    saved = torch.load(folder / "test-predictions.pt", weights_only=True)
    saved["true_fractions"] = saved["true_fractions"].flip(0)
    torch.save(saved, folder / "test-predictions.pt")
    report_path = folder / "test-report.json"
    report = json.loads(report_path.read_text())
    report["test_predictions_sha256"] = sha256(folder / "test-predictions.pt")
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="same.*targets"):
        metric.score_comparison(root)


def test_existing_evaluator_includes_reusable_metric():
    from satquery.evaluation import coverage_metrics

    truth = np.zeros((1, 19))
    truth[0, :2] = [0.7, 0.3]
    pred = np.zeros((1, 19))
    pred[0, :3] = [0.5, 0.4, 0.1]
    assert coverage_metrics(pred, truth, ["a"], bootstrap_repeats=2)[
        "area_weighted_mae_pp"
    ] == pytest.approx(17)
