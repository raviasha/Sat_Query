"""Held-out coverage metrics, with uncertainty resampled by whole image area."""

from dataclasses import dataclass

import numpy as np
import torch

from .prediction_data import TrainingPairs, class_schema


@dataclass
class TokenSplit:
    split: str
    x: torch.Tensor
    y: torch.Tensor
    area_ids: np.ndarray
    patch_ids: tuple[str, ...]


def load_splits(features, targets, *, feature_key="joint_encodings", progress=None):
    """Read verified exports once; keep official area splits and eligible token identities."""
    pairs = TrainingPairs(features, targets, feature_key=feature_key)
    parts = {
        s: {"x": [], "y": [], "ids": [], "patches": []} for s in ("train", "validation", "test")
    }
    seen_sar = set()
    count = 0
    for samples, x, y, mask in pairs.iter_batches():
        for i, sample in enumerate(samples):
            split = sample.get("split")
            if split not in parts:
                raise ValueError(f"Unknown official split: {split}")
            if sample["s1_name"] in seen_sar:
                raise ValueError("Duplicate SAR area across data splits")
            seen_sar.add(sample["s1_name"])
            part = parts[split]
            part["patches"].append(sample["patch_id"])
            part["x"].append(x[i, mask[i]])
            part["y"].append(y[i, mask[i]])
            part["ids"].extend([sample["patch_id"]] * int(mask[i].sum()))
        count += len(samples)
        if progress:
            progress(f"Loaded verified feature/target pairs for {count} areas")
    result = {}
    for split, part in parts.items():
        if not part["ids"]:
            raise ValueError(f"No eligible tokens for {split}")
        result[split] = TokenSplit(
            split,
            torch.cat(part["x"]),
            torch.cat(part["y"]),
            np.asarray(part["ids"]),
            tuple(part["patches"]),
        )
    return result, {
        "feature_key": feature_key,
        "feature_contract": pairs.features.contract,
        "classes": class_schema(),
        "feature_manifest_sha256": pairs.features.digest,
        "target_manifest_sha256": pairs.digest,
        "fully_labeled_tokens_only": True,
        "input_sample_splits": {s: len(v.patch_ids) for s, v in result.items()},
        "eligible_tokens_by_split": {s: len(v.x) for s, v in result.items()},
    }


@torch.inference_mode()
def predict_rows(model, x, *, batch_size=2048):
    return torch.cat(
        [model.predict(x[i : i + batch_size]).cpu() for i in range(0, len(x), batch_size)]
    )


def coverage_metrics(predictions, truth, area_ids, *, bootstrap_repeats=500, seed=17):
    """Token-weighted metrics; bootstrap draws whole areas, retaining within-area dependence."""
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    ids = np.asarray(area_ids)
    if (
        p.ndim != 2
        or p.shape != y.shape
        or p.shape[1] != 19
        or not len(p)
        or ids.shape != (len(p),)
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or (p < 0).any()
        or (p > 1).any()
        or (y < 0).any()
        or (y > 1).any()
        or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
        or not np.allclose(y.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Expected aligned finite 19-class coverage distributions and area IDs")
    if not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    error = (p - y) * 100
    absolute = np.abs(error)
    unique, inverse = np.unique(ids, return_inverse=True)
    sizes = np.bincount(inverse)
    area_error = np.zeros((len(unique), 19))
    np.add.at(area_error, inverse, absolute)
    # A tied target has no unique dominant class; omit it from categorical metrics only.
    untied = (y == y.max(1, keepdims=True)).sum(1) == 1
    actual, predicted = y.argmax(1), p.argmax(1)
    confusion = np.zeros((19, 19), dtype=np.int64)
    np.add.at(confusion, (actual[untied], predicted[untied]), 1)
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap_repeats, 19))
    for i in range(bootstrap_repeats):
        selected = rng.integers(0, len(unique), len(unique))
        boot[i] = area_error[selected].sum(0) / sizes[selected].sum()
    intervals = np.quantile(boot, [0.025, 0.975], axis=0)
    records = []
    for c, schema in enumerate(class_schema()):
        present = y[:, c] > 0
        support = int(present.sum())
        areas = len(np.unique(ids[present]))
        tp = int(confusion[c, c])
        actual_n = int(confusion[c].sum())
        predicted_n = int(confusion[:, c].sum())
        records.append(
            {
                **schema,
                "mae_pp": float(absolute[:, c].mean()),
                "rmse_pp": float(np.sqrt(np.square(error[:, c]).mean())),
                "bias_pp": float(error[:, c].mean()),
                "present_mae_pp": float(absolute[present, c].mean()) if support else None,
                "present_tokens": support,
                "present_areas": areas,
                "mean_true_coverage_percent": float(y[:, c].mean() * 100),
                "within_5pp_fraction": float((absolute[:, c] <= 5 + 1e-7).mean()),
                "present_within_5pp_fraction": float((absolute[present, c] <= 5 + 1e-7).mean())
                if support
                else None,
                "dominant_precision": tp / predicted_n if predicted_n else None,
                "dominant_recall": tp / actual_n if actual_n else None,
                "dominant_f1": 2 * tp / (actual_n + predicted_n)
                if actual_n + predicted_n
                else None,
                "dominant_true_tokens": actual_n,
                "dominant_predicted_tokens": predicted_n,
                "mae_pp_area_bootstrap_95ci": intervals[:, c].tolist() if len(unique) > 1 else None,
                "support_status": "no_test_presence"
                if not areas
                else "limited_test_presence"
                if areas < 20
                else "at_least_20_test_areas",
            }
        )
    return {
        "token_count": len(y),
        "evaluated_area_count": len(unique),
        "coverage_mae_pp": float(absolute.mean()),
        "coverage_mae_pp_area_bootstrap_95ci": np.quantile(boot.mean(1), [0.025, 0.975]).tolist()
        if len(unique) > 1
        else None,
        "dominant_class_accuracy": float(np.trace(confusion) / untied.sum())
        if untied.any()
        else None,
        "dominant_evaluated_tokens": int(untied.sum()),
        "dominant_tied_tokens_excluded": int((~untied).sum()),
        "dominant_confusion_matrix": confusion.tolist(),
        "classes": records,
        "bootstrap": {
            "unit": "whole 1.2 km image area",
            "repeats": bootstrap_repeats,
            "seed": seed,
            "method": "percentile interval; dependence between different areas is not modeled",
        },
        "interpretation": "Coverage error and historical test reliability, not individual prediction confidence. Support flags use a descriptive 20-area threshold, not a statistical guarantee.",
    }
