"""Resumable linear/MLP comparison over the same verified frozen-feature splits."""

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from .evaluation import coverage_metrics, load_splits, predict_rows
from .prediction import load_head, save_head
from .prediction_data import class_schema
from .preprocessing import sha256
from .validation_training import fit_with_validation


def write_json(path, value):
    path = Path(path)
    pending = path.with_name("." + path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_completed(output, config):
    report = read_json(output / "comparison.json")
    if report["config"] != config:
        raise ValueError("Experiment configuration changed; choose a new output directory")
    for relative, expected in report["artifact_sha256"].items():
        if sha256(output / relative) != expected:
            raise ValueError(f"Experiment artifact hash mismatch: {relative}")
    return report


def run_comparison(
    features,
    targets,
    output,
    *,
    seeds=(17, 29, 43),
    device="cpu",
    max_epochs=60,
    patience=10,
    learning_rate=0.001,
    batch_size=1024,
    bootstrap_repeats=1000,
    progress=print,
    source_revision=None,
):
    """Complete seeds are reusable; an interrupted unfinished seed is fitted again.

    All epochs and the architecture recommendation are selected on validation data.
    Test metrics are calculated only after selection.json has been saved.
    """
    features, targets, output = map(Path, (features, targets, output))
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("Expected distinct integer seeds")
    config = {
        "format_version": 1,
        "seeds": list(seeds),
        "architectures": ["linear", "mlp"],
        "max_epochs": max_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "bootstrap_repeats": bootstrap_repeats,
        "feature_manifest_sha256": sha256(features / "manifest.json"),
        "target_manifest_sha256": sha256(targets / "manifest.json"),
        "source_revision": source_revision,
        "mlp_hidden": [256, 128],
        "mlp_dropout": 0.1,
        "selection_metric": "mean_validation_soft_target_cross_entropy",
        "device_type": torch.device(device).type,
        "torch_version": str(torch.__version__),
    }
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Experiment configuration changed; choose a new output directory")
    write_json(config_path, config)
    if (output / "comparison.json").exists():
        report = validate_completed(output, config)
        if progress:
            progress("REUSED_COMPLETE_COMPARISON " + str(output))
        return report

    splits, provenance = load_splits(features, targets, progress=progress)
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(splits[a].patch_ids) & set(splits[b].patch_ids):
            raise ValueError("Area identities overlap between splits")
    write_json(output / "splits.json", {k: list(v.patch_ids) for k, v in splits.items()})
    if progress:
        progress("VERIFIED_SPLITS " + json.dumps(provenance["eligible_tokens_by_split"]))
    runs = []
    for architecture in config["architectures"]:
        for seed in seeds:
            parent = output / architecture
            parent.mkdir(exist_ok=True)
            folder = parent / f"seed-{seed}"
            if folder.exists():
                fit = read_json(folder / "training.json")
                if (
                    fit["config"] != config
                    or fit["seed"] != seed
                    or fit["architecture"] != architecture
                ):
                    raise ValueError("Saved run configuration mismatch")
                if sha256(folder / "head.pt") != fit["checkpoint_sha256"]:
                    raise ValueError("Saved checkpoint hash mismatch")
                if progress:
                    progress(f"REUSED {architecture} seed {seed}")
            else:
                temporary = Path(tempfile.mkdtemp(prefix=f".seed-{seed}-", dir=parent))
                try:
                    model, fit = fit_with_validation(
                        splits["train"],
                        splits["validation"],
                        architecture=architecture,
                        seed=seed,
                        device=device,
                        max_epochs=max_epochs,
                        patience=patience,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        progress=(
                            lambda msg, a=architecture, s=seed: progress(f"{a} seed {s}: {msg}")
                        )
                        if progress
                        else None,
                    )
                    vp = predict_rows(model, splits["validation"].x)
                    validation_metrics = coverage_metrics(
                        vp.numpy(),
                        splits["validation"].y.numpy(),
                        splits["validation"].area_ids,
                        bootstrap_repeats=bootstrap_repeats,
                    )
                    metadata = {
                        **provenance,
                        **{k: v for k, v in fit.items() if k != "history"},
                        "training_mode": "train_split_only",
                        "config": config,
                    }
                    save_head(temporary / "head.pt", model, metadata)
                    restored, _ = load_head(temporary / "head.pt")
                    torch.testing.assert_close(
                        predict_rows(restored, splits["validation"].x[:256]),
                        vp[:256],
                        rtol=0,
                        atol=0,
                    )
                    fit = {
                        **metadata,
                        "history": fit["history"],
                        "validation_metrics": validation_metrics,
                        "checkpoint_sha256": sha256(temporary / "head.pt"),
                    }
                    write_json(temporary / "training.json", fit)
                    temporary.rename(folder)
                    del model, restored, vp
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if progress:
                        progress(f"SAVED {architecture} seed {seed} epoch {fit['selected_epoch']}")
                except BaseException:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
            runs.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "folder": str(folder.relative_to(output)),
                    "selected_epoch": fit["selected_epoch"],
                    "validation_loss": fit["best_validation_loss"],
                    "validation_mae_pp": fit["validation_metrics"]["coverage_mae_pp"],
                    "checkpoint_sha256": fit["checkpoint_sha256"],
                }
            )

    means = {
        a: float(np.mean([r["validation_loss"] for r in runs if r["architecture"] == a]))
        for a in config["architectures"]
    }
    winner = min(means, key=means.get)
    selected = min(
        (r for r in runs if r["architecture"] == winner), key=lambda r: r["validation_loss"]
    )
    selection = {
        "architecture": winner,
        "seed": selected["seed"],
        "checkpoint": selected["folder"] + "/head.pt",
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "mean_validation_loss": means,
        "test_used_for_selection": False,
        "rule": "lowest mean validation CE across seeds; then lowest validation CE seed",
    }
    # Persist the decision before any test evaluation. Do not revise it using test outcomes.
    write_json(output / "selection.json", selection)
    if progress:
        progress("VALIDATION_SELECTION " + json.dumps(selection))
    test = splits["test"]
    for run in runs:
        folder = output / run["folder"]
        report_path = folder / "test-report.json"
        if report_path.exists():
            report = read_json(report_path)
            if report["checkpoint_sha256"] != run["checkpoint_sha256"]:
                raise ValueError("Test report belongs to another checkpoint")
            if sha256(folder / "test-predictions.pt") != report["test_predictions_sha256"]:
                raise ValueError("Test prediction hash mismatch")
        else:
            model, metadata = load_head(folder / "head.pt")
            pred = predict_rows(model, test.x)
            metrics = coverage_metrics(
                pred.numpy(), test.y.numpy(), test.area_ids, bootstrap_repeats=bootstrap_repeats
            )
            pending = folder / ".test-predictions.pt.tmp"
            torch.save(
                {
                    "predicted_fractions": pred,
                    "true_fractions": test.y,
                    "area_ids": test.area_ids.tolist(),
                },
                pending,
            )
            reloaded = torch.load(pending, weights_only=True)
            torch.testing.assert_close(reloaded["predicted_fractions"], pred, rtol=0, atol=0)
            assert reloaded["area_ids"] == test.area_ids.tolist()
            pending.replace(folder / "test-predictions.pt")
            report = dict(
                metrics=metrics,
                checkpoint_sha256=run["checkpoint_sha256"],
                test_predictions_sha256=sha256(folder / "test-predictions.pt"),
                evaluation_scope="held_out_test_areas",
                **provenance,
            )
            write_json(report_path, report)
            del model, pred, reloaded
        metrics = report["metrics"]
        reliability = {
            "format_version": 1,
            "checkpoint_sha256": run["checkpoint_sha256"],
            "feature_contract": provenance["feature_contract"],
            "classes": class_schema(),
            "evaluation_scope": "held_out_test_areas",
            "prediction_confidence_available": False,
            "test_report_sha256": sha256(report_path),
            "class_metrics": metrics["classes"],
            "note": "Historical class errors and support; not calibrated prediction confidence.",
        }
        write_json(folder / "reliability.json", reliability)
        with (folder / "per-class-metrics.csv").open("w", newline="") as f:
            fields = [
                "index",
                "name",
                "mae_pp",
                "present_mae_pp",
                "present_areas",
                "dominant_precision",
                "dominant_recall",
                "dominant_f1",
                "support_status",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(metrics["classes"])
        run.update(
            test_mae_pp=metrics["coverage_mae_pp"],
            test_dominant_accuracy=metrics["dominant_class_accuracy"],
        )
        if progress:
            progress("TEST_RESULT " + json.dumps(run))
    summary = {}
    for architecture in config["architectures"]:
        summary[architecture] = {}
        for key in [
            "validation_loss",
            "validation_mae_pp",
            "test_mae_pp",
            "test_dominant_accuracy",
        ]:
            values = [r[key] for r in runs if r["architecture"] == architecture]
            summary[architecture][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    files = [output / "config.json", output / "splits.json", output / "selection.json"]
    for run in runs:
        files.extend(
            output / run["folder"] / n
            for n in [
                "head.pt",
                "training.json",
                "test-report.json",
                "test-predictions.pt",
                "reliability.json",
                "per-class-metrics.csv",
            ]
        )
    report = {
        "passed": True,
        "config": config,
        "provenance": provenance,
        "runs": runs,
        "summary": summary,
        "selection": selection,
        "artifact_sha256": {str(p.relative_to(output)): sha256(p) for p in files},
        "interpretation": "Same held-out areas reused for controlled comparison; not a new untouched benchmark. Seed SD describes training variation, not statistical uncertainty over new landscapes.",
    }
    write_json(output / "comparison.json", report)
    validate_completed(output, config)
    if progress:
        progress("COMPARISON_COMPLETE " + json.dumps({"summary": summary, "selection": selection}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_comparison(
        args.pipeline / "features",
        args.pipeline / "targets",
        args.output or args.pipeline / "experiments" / "linear-vs-mlp-v1",
        seeds=tuple(args.seeds),
        device=args.device,
        max_epochs=args.max_epochs,
        source_revision=args.source_revision,
        progress=lambda m: print(m, flush=True),
    )


if __name__ == "__main__":
    main()
