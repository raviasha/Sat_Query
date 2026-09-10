"""Train only the small prediction head; never load or update CROMA weights."""

import argparse
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import torch

from .prediction import CoverageHead, coverage_loss, load_head, save_head
from .prediction_data import FEATURE_KEYS, TrainingPairs, class_schema
from .preprocessing import sha256


def train_head(
    batch_source, *, epochs=100, learning_rate=0.01, batch_size=128, seed=17, progress=None
):
    """Fit CPU linear head from a repeatable callable yielding (features, fractions)."""
    for name, value in [("epochs", epochs), ("batch_size", batch_size), ("seed", seed)]:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < (0 if name == "seed" else 1)
        ):
            raise ValueError(f"Invalid {name}")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid learning rate")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CoverageHead()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for epoch in range(epochs):
        total, loss_sum = 0, 0.0
        for x, y in batch_source():
            x, y = x.detach().cpu(), y.detach().cpu()
            if x.ndim != 2 or y.shape != (len(x), 19):
                raise ValueError("Expected aligned token rows")
            order = torch.randperm(len(x), generator=generator)
            for start in range(0, len(x), batch_size):
                ids = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = coverage_loss(model(x[ids]), y[ids])
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(ids)
                total += len(ids)
        if not total:
            raise ValueError(
                "No eligible training tokens; provide train data or explicitly use --demo-fit-all"
            )
        history.append({"epoch": epoch + 1, "loss": loss_sum / total, "token_count": total})
        if progress and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
            progress(f"Epoch {epoch + 1}/{epochs}: loss={loss_sum / total:.6f}, tokens={total}")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Training produced nonfinite weights")
    return model.eval(), history


def evaluate_fit(model, batch_source):
    """Descriptive fit metrics on supplied rows; callers state their evaluation scope."""
    count = correct = 0
    absolute = cross_entropy = 0.0
    with torch.inference_mode():
        for x, y in batch_source():
            for start in range(0, len(x), 512):
                truth = y[start : start + 512]
                logits = model(x[start : start + 512])
                pred = logits.softmax(-1)
                n = len(truth)
                count += n
                absolute += float((pred - truth).abs().sum())
                cross_entropy += float(coverage_loss(logits, truth)) * n
                correct += int((pred.argmax(-1) == truth.argmax(-1)).sum())
    if not count:
        raise ValueError("No evaluation tokens")
    return {
        "token_count": count,
        "coverage_mae_percentage_points": absolute / (count * 19) * 100,
        "dominant_class_accuracy": correct / count,
        "soft_target_cross_entropy": cross_entropy / count,
    }


def train_from_exports(
    features,
    targets,
    output,
    *,
    feature_key="joint_encodings",
    demo_fit_all=False,
    epochs=100,
    learning_rate=0.01,
    batch_size=128,
    seed=17,
    progress=None,
):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output = output.resolve()
    if any(
        Path(p).resolve() == output or Path(p).resolve() in output.parents
        for p in [features, targets]
    ):
        raise ValueError("Output must be outside input directories")
    pairs = TrainingPairs(features, targets, feature_key=feature_key, demo_fit_all=demo_fit_all)
    model, history = train_head(
        lambda: iter(pairs),
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
    )
    metrics = evaluate_fit(model, lambda: iter(pairs))
    splits = Counter()
    for _, info, _ in pairs.features:
        splits.update(s.get("split", "unknown") for s in info["samples"])
    metadata = {
        "feature_key": feature_key,
        "feature_contract": pairs.features.contract,
        "classes": class_schema(),
        "training_mode": "demo_fit_all" if demo_fit_all else "train_split_only",
        "evaluation_scope": "training_data_only",
        "input_sample_splits": dict(splits),
        "feature_manifest_sha256": pairs.features.digest,
        "target_manifest_sha256": pairs.digest,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "loss": "unweighted_soft_target_cross_entropy",
        "croma_frozen": True,
        "fully_labeled_tokens_only": True,
        "torch_version": str(torch.__version__),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        save_head(temporary / "head.pt", model, metadata)
        restored, _ = load_head(temporary / "head.pt")
        for x, _ in pairs:
            torch.testing.assert_close(restored.predict(x), model.predict(x), rtol=0, atol=0)
        report = {
            **metadata,
            "history": history,
            "fit_metrics": metrics,
            "checkpoint_sha256": sha256(temporary / "head.pt"),
        }
        (temporary / "training.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(f"Output appeared during training: {output}")
        temporary.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-key", choices=FEATURE_KEYS, default="joint_encodings")
    parser.add_argument(
        "--demo-fit-all",
        action="store_true",
        help="Fit all splits for an explicit same-data demonstration",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        report = train_from_exports(
            args.features,
            args.targets,
            args.output,
            feature_key=args.feature_key,
            demo_fit_all=args.demo_fit_all,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
            progress=lambda text: print(text, flush=True),
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved prediction head to {args.output.resolve()}")
    print("Training-data fit:", json.dumps(report["fit_metrics"]))


if __name__ == "__main__":
    main()
