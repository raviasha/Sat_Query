"""CLI for matching CROMA spatial features to BigEarthNet coverage targets."""

import argparse
from pathlib import Path

from .targets import export_targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Prepared tensor export directory"
    )
    parser.add_argument(
        "--features", type=Path, required=True, help="Matching CROMA feature export"
    )
    parser.add_argument("--output", type=Path, required=True, help="New target output directory")
    args = parser.parse_args(argv)
    try:
        export_targets(
            args.input, args.features, args.output, progress=lambda text: print(text, flush=True)
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved aligned label targets to {args.output.resolve()}")


if __name__ == "__main__":
    main()
