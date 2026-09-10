"""CLI for feature extraction, separate from BigEarthNet preprocessing."""

import argparse
from pathlib import Path

from .features import CromaFeatureExtractor, extract_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Prepared tensor export directory"
    )
    parser.add_argument("--output", type=Path, required=True, help="New feature output directory")
    parser.add_argument("--checkpoint", type=Path, help="Existing official CROMA Base checkpoint")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/models/croma"))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        print("Loading verified CROMA-Base checkpoint...", flush=True)
        extractor = CromaFeatureExtractor.from_pretrained(
            args.checkpoint, cache_dir=args.cache_dir, device=args.device
        )
        extract_features(
            args.input,
            args.output,
            extractor,
            batch_size=args.batch_size,
            progress=lambda text: print(text, flush=True),
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved CROMA features to {args.output.resolve()}")


if __name__ == "__main__":
    main()
