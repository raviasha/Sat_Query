"""Ground-truth area-weighted coverage MAE, reusable without model inference.

Inputs are aligned N x C fractions in [0, 1], summing to one per row.
Each row represents one equally sized, fully labelled spatial block.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .preprocessing import sha256


def area_weighted_mae(predictions, truth):
    """Return mean_blocks(sum_classes(truth * abs(pred-truth))) * 100, in pp.

    Weights use true class coverage, including zero weight for absent classes.
    There is no division by the class count or by two. Blocks have equal weight.
    """
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if (
        p.ndim != 2
        or p.shape != y.shape
        or not p.size
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or (p < 0).any()
        or (p > 1).any()
        or (y < 0).any()
        or (y > 1).any()
        or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-5)
        or not np.allclose(y.sum(1), 1, rtol=0, atol=1e-5)
    ):
        raise ValueError("Expected matching nonempty N x C normalized finite fractions in [0, 1]")
    return float((y * np.abs(p - y)).sum(1).mean() * 100)


def _read_predictions(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    p, y = saved["predicted_fractions"], saved["true_fractions"]
    ids = saved["area_ids"]
    score = area_weighted_mae(p, y)
    if (
        not isinstance(ids, list)
        or len(ids) != len(y)
        or not all(isinstance(i, str) and i for i in ids)
    ):
        raise ValueError("Expected one nonempty string area ID per block")
    return (
        {
            "area_weighted_mae_pp": score,
            "block_count": len(y),
            "area_count": len(set(ids)),
            "predictions_sha256": sha256(path),
        },
        y,
        ids,
    )


def _definition():
    return {
        "format_version": 1,
        "metric": "area_weighted_mae_pp",
        "formula": "100 * mean_blocks(sum_classes(true_fraction * abs(predicted_fraction - true_fraction)))",
        "weight_source": "ground_truth_class_fraction",
        "block_weighting": "equal; assumes equally sized fully labelled blocks",
        "units": "percentage_points",
        "lower_is_better": True,
        "interpretation": "Emphasizes major land cover; rare/small features have less influence. Absent-class errors receive no direct weight. Not prediction confidence.",
    }


def score_predictions(path):
    """Score an existing safe test-predictions.pt export from any coverage head."""
    score, _, _ = _read_predictions(Path(path))
    return {**_definition(), **score, "predictions_path": str(path)}


def score_comparison(directory):
    """Score all saved comparison seeds, verifying hashes and identical test rows.

    Existing validation selection is reported, never changed by this test metric.
    Does not load features, fit models, or alter the source comparison artifacts.
    """
    root = Path(directory).resolve()
    comparison_path = root / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    results, reference_y, reference_ids = [], None, None
    for run in comparison["runs"]:
        folder = (root / run["folder"]).resolve()
        if not folder.is_relative_to(root):
            raise ValueError("Run folder must be inside comparison directory")
        path = folder / "test-predictions.pt"
        original = json.loads((folder / "test-report.json").read_text())
        if (
            sha256(path) != original["test_predictions_sha256"]
            or sha256(folder / "head.pt") != run["checkpoint_sha256"]
            or run["checkpoint_sha256"] != original["checkpoint_sha256"]
        ):
            raise ValueError(f"Saved prediction/checkpoint hash mismatch: {folder}")
        score, y, ids = _read_predictions(path)
        if reference_y is None:
            reference_y, reference_ids = y, ids
        elif ids != reference_ids or not torch.equal(y, reference_y):
            raise ValueError("Comparisons require the same ordered test blocks and targets")
        results.append(
            {
                "architecture": run["architecture"],
                "seed": run["seed"],
                "folder": run["folder"],
                "checkpoint_sha256": run["checkpoint_sha256"],
                **score,
            }
        )
    if not results:
        raise ValueError("Comparison contains no saved runs")
    summary = {}
    for architecture in dict.fromkeys(r["architecture"] for r in results):
        values = [r["area_weighted_mae_pp"] for r in results if r["architecture"] == architecture]
        summary[architecture] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "runs": len(values),
        }
    selected = comparison["selection"]
    selected_score = next(
        r["area_weighted_mae_pp"]
        for r in results
        if r["architecture"] == selected["architecture"] and r["seed"] == selected["seed"]
    )
    return {
        **_definition(),
        "comparison_sha256": sha256(comparison_path),
        "runs": results,
        "summary": summary,
        "existing_selection": selected,
        "selected_checkpoint_score_pp": selected_score,
        "selection_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="Safe saved test-predictions.pt file")
    source.add_argument("--comparison", type=Path, help="Completed compare_heads output directory")
    parser.add_argument("--output", type=Path, required=True, help="New metric report JSON path")
    args = parser.parse_args()
    # Avoid replacing checkpoints or original experiment JSONs through a typo.
    if args.output.suffix != ".json" or args.output.name in {
        "comparison.json",
        "test-report.json",
        "training.json",
        "selection.json",
        "config.json",
        "reliability.json",
        "splits.json",
    }:
        parser.error("Use a separate .json metric report, e.g. area-weighted-mae.json")
    report = (
        score_comparison(args.comparison)
        if args.comparison
        else score_predictions(args.predictions)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name("." + args.output.name + ".tmp")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pending.replace(args.output)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
