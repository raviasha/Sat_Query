"""Export patch coverage predictions from an independently trained small head."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from .prediction import load_head
from .prediction_data import FeatureBatches, class_schema
from .preprocessing import sha256


def export_predictions(
    features, checkpoint, output, *, batch_size=128, progress=None, reliability=None
):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output = output.resolve()
    checkpoint = Path(checkpoint).resolve()
    root = Path(features).resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside feature input")
    digest = sha256(checkpoint)
    model, metadata = load_head(checkpoint)
    reader = FeatureBatches(features, feature_key=metadata.get("feature_key", "joint_encodings"))
    if metadata.get("feature_contract") != reader.contract:
        raise ValueError("Feature contract differs from trained head")
    if metadata.get("classes") != class_schema():
        raise ValueError("Checkpoint class order mismatch")
    reliability_bytes = None
    if reliability is not None:
        reliability_bytes = Path(reliability).read_bytes()
        report = json.loads(reliability_bytes)
        if (
            report.get("format_version") != 1
            or report.get("checkpoint_sha256") != digest
            or report.get("feature_contract") != reader.contract
            or report.get("classes") != class_schema()
            or report.get("evaluation_scope") != "held_out_test_areas"
            or report.get("prediction_confidence_available") is not False
            or [c.get("index") for c in report.get("class_metrics", [])] != list(range(19))
        ):
            raise ValueError("Test reliability report does not match this prediction head")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    batches = []
    total = 0
    try:
        for number, (batch, info, x) in enumerate(reader):
            flat = x.reshape(-1, 768)
            predictions = torch.cat(
                [
                    model.predict(flat[start : start + batch_size])
                    for start in range(0, len(flat), batch_size)
                ]
            ).reshape(len(x), 225, 19)
            if not torch.isfinite(predictions).all() or not torch.allclose(
                predictions.sum(-1), torch.ones(len(x), 225), atol=1e-6, rtol=0
            ):
                raise ValueError("Invalid predicted fractions")
            relative = "." if len(reader.manifest["batches"]) == 1 else f"batch-{number:06d}"
            destination = temporary / relative
            destination.mkdir(exist_ok=True)
            values = {
                "class_fractions": predictions,
                "dominant_class": predictions.argmax(-1),
                "dominant_fraction": predictions.max(-1).values,
            }
            torch.save(values, destination / "predictions.pt")
            restored = torch.load(destination / "predictions.pt", weights_only=True)
            if any(not torch.equal(v, restored[k]) for k, v in values.items()):
                raise ValueError("Prediction save/reload mismatch")
            (destination / "batch.json").write_text(
                json.dumps(
                    {
                        **info,
                        "shapes": {k: list(v.shape) for k, v in values.items()},
                        "source_feature_batch": batch,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            batches.append(
                {
                    "directory": relative,
                    "sample_count": len(x),
                    "start_index": total,
                    "sha256": {
                        name: sha256(destination / name)
                        for name in ["predictions.pt", "batch.json"]
                    },
                }
            )
            total += len(x)
            if progress:
                progress(f"Predicted coverage for {total}/{reader.manifest['sample_count']} areas")
        if sha256(checkpoint) != digest:
            raise ValueError("Checkpoint changed during inference")
        manifest = {
            "format_version": 1,
            "sample_count": total,
            "token_count": total * 225,
            "classes": class_schema(),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "training_mode": metadata.get("training_mode"),
            "same_features_as_training": reader.digest == metadata.get("feature_manifest_sha256"),
            "feature_manifest_sha256": reader.digest,
            "feature_contract": reader.contract,
            "feature_key": reader.feature_key,
            "prediction_units": "fractions of area, not confidence",
            "batches": batches,
        }
        if reliability_bytes is not None:
            (temporary / "reliability.json").write_bytes(reliability_bytes)
            manifest["historical_class_reliability"] = {
                "file": "reliability.json",
                "sha256": sha256(temporary / "reliability.json"),
                "interpretation": "Historical error on held-out test areas; not per-prediction confidence",
            }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        (temporary / "validation.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "sample_count": total,
                    "token_count": total * 225,
                    "checks": [
                        "input_hashes",
                        "feature_contract",
                        "class_order",
                        "finite_fractions",
                        "fractions_sum_one",
                        "save_reload_equality",
                    ],
                    "manifest_sha256": sha256(temporary / "manifest.json"),
                },
                indent=2,
            )
            + "\n"
        )
        if output.exists():
            raise FileExistsError(f"Output appeared during prediction: {output}")
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--reliability", type=Path, help="Matching held-out test reliability report"
    )
    args = parser.parse_args(argv)
    try:
        export_predictions(
            args.features,
            args.checkpoint,
            args.output,
            batch_size=args.batch_size,
            reliability=args.reliability,
            progress=lambda text: print(text, flush=True),
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved coverage predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
