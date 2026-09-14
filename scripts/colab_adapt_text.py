"""Colab-friendly driver for reusable text adaptation over already saved features.

This script never extracts imagery or CROMA features.  Inputs are explicit paths;
new pair, model, and evaluation artifacts are written beneath ``--edu-p``.
"""

import argparse
import json
from pathlib import Path

from satquery.assistant.text_adaptation import (
    DEFAULT_EXCLUDED_ANNOTATION_IDS,
    OpenAIEmbedder,
    evaluate_retrieval,
    prepare_text_pairs,
    train_text_adapter,
)


def _splits(value):
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _destination(edu_root, name):
    root = Path(edu_root).resolve()
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Artifact name must be a relative path beneath EDU P")
    destination = (root / "text-adaptation" / relative).resolve()
    if root not in destination.parents:
        raise ValueError("Artifact path escapes EDU P")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edu-p", required=True, help="Supplied persistent EDU directory")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--annotations", required=True)
    prepare.add_argument("--features", required=True)
    prepare.add_argument("--name", default="text-pairs")
    prepare.add_argument("--feature-key", default="joint_encodings")
    prepare.add_argument("--splits", type=_splits, default=("train", "validation"))
    prepare.add_argument("--embedding-batch-size", type=int, default=128)
    prepare.add_argument("--exclude-id", action="append", default=[])

    train = commands.add_parser("train")
    train.add_argument("--prepared", required=True)
    train.add_argument("--name", default="text-adapter")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--seed", type=int, default=17)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--prepared", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--name", default="retrieval-test.json")
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_text_pairs(
            args.annotations,
            args.features,
            _destination(args.edu_p, args.name),
            requested_splits=args.splits,
            feature_key=args.feature_key,
            embedder=OpenAIEmbedder(),
            embedding_provenance={"kind": "openai_api", "authentication": "environment_only"},
            excluded_annotation_ids=DEFAULT_EXCLUDED_ANNOTATION_IDS + tuple(args.exclude_id),
            embedding_batch_size=args.embedding_batch_size,
        )
    elif args.command == "train":
        result = train_text_adapter(
            args.prepared,
            _destination(args.edu_p, args.name),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
        )
    else:
        result = evaluate_retrieval(
            args.checkpoint,
            args.prepared,
            split=args.split,
            output=_destination(args.edu_p, args.name),
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
