"""Bounded-memory tensor export. Run with python -m satquery.prepare_croma."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch

from .preprocessing import (
    CROMA_REVISION,
    DEFAULT_PROFILE,
    NORMALIZATION_PROFILE,
    ChannelProfile,
    load_raw_patch,
    normalize_patch,
    read_records,
    sha256,
)


def _json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _save_checked(path: Path, value):
    torch.save(value, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in value:
            if not torch.equal(value[key], restored[key]):
                raise ValueError(f"Reload mismatch in {path.name}:{key}")
    elif not torch.equal(value, restored):
        raise ValueError(f"Reload mismatch in {path.name}")


def _write_batch(root, directory, records, profile, raw_only):
    raw_optical, raw_sar, maps, optical, sar, samples = [], [], [], [], [], []
    for record in records:
        patch = load_raw_patch(root, record, profile)
        raw_optical.append(patch["optical_images"])
        raw_sar.append(patch["SAR_images"])
        maps.append(patch["reference_map"])
        sample = patch["provenance"]
        if not raw_only:
            o, os = normalize_patch(patch["optical_images"])
            s, ss = normalize_patch(patch["SAR_images"])
            optical.append(o)
            sar.append(s)
            sample["normalization"] = {"optical": os, "SAR": ss}
        samples.append(sample)
    raw = {"optical_images": torch.stack(raw_optical), "SAR_images": torch.stack(raw_sar)}
    _save_checked(directory / "raw_inputs.pt", raw)
    _save_checked(directory / "reference_maps.pt", torch.stack(maps))
    files = ["raw_inputs.pt", "reference_maps.pt"]
    if not raw_only:
        normalized = {"optical_images": torch.stack(optical), "SAR_images": torch.stack(sar)}
        for key, tensor in normalized.items():
            if (
                tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all()
                or tensor.min() < 0
                or tensor.max() > 1
            ):
                raise ValueError(f"Invalid normalized {key}")
        _save_checked(directory / "croma_inputs.pt", normalized)
        files.append("croma_inputs.pt")
    _json(
        directory / "batch.json",
        {
            "sample_count": len(records),
            "samples": samples,
            "shapes": {key: list(value.shape) for key, value in raw.items()},
            "channel_profile": asdict(profile),
            "reference_map_shape": [len(records), 120, 120],
            "reference_map_dtype": "int64",
            "imagery_dtype": "float32",
        },
    )
    files.append("batch.json")
    return {name: sha256(directory / name) for name in files}


def export_inputs(
    root: Path,
    output: Path,
    profile: ChannelProfile = DEFAULT_PROFILE,
    *,
    metadata: Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    batch_size: int = 64,
    raw_only: bool = False,
    progress=None,
) -> dict:
    """Export selected metadata rows in bounded batches; never overwrite an output.

    One batch writes directly into output. Larger selections create batch-NNNNNN
    subdirectories indexed by manifest.json. Source images are never modified.
    """
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the input dataset directory")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if profile != DEFAULT_PROFILE and not raw_only:
        raise ValueError("Unverified custom channel profile: use raw_only=True")
    records = read_records(root, metadata=metadata, split=split, limit=limit)
    metadata_path = Path(metadata).resolve() if metadata else root / "metadata.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        batches = []
        for number, start in enumerate(range(0, len(records), batch_size)):
            selected = records[start : start + batch_size]
            relative = "." if len(records) <= batch_size else f"batch-{number:06d}"
            directory = temporary / relative
            directory.mkdir(exist_ok=True)
            hashes = _write_batch(root, directory, selected, profile, raw_only)
            batches.append(
                {
                    "directory": relative,
                    "sample_count": len(selected),
                    "start_index": start,
                    "sha256": hashes,
                }
            )
            if progress:
                progress(f"Prepared {start + len(selected)}/{len(records)} areas")
        manifest = {
            "format_version": 1,
            "sample_count": len(records),
            "batch_size": batch_size,
            "input_root": str(root),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "split_filter": split,
            "limit": limit,
            "channel_profile": asdict(profile),
            "normalization_profile": None if raw_only else NORMALIZATION_PROFILE,
            "croma_source_revision": CROMA_REVISION,
            "checkpoint_inference_run": False,
            "package_versions": {
                p: version(p)
                for p in (
                    "satquery-preprocessing",
                    "torch",
                    "rasterio",
                    "numpy",
                    "pandas",
                    "pyarrow",
                )
            },
            "batches": batches,
        }
        _json(temporary / "manifest.json", manifest)
        _json(
            temporary / "validation.json",
            {
                "passed": True,
                "sample_count": len(records),
                "checks": [
                    "explicit_band_order",
                    "metadata_pairing",
                    "source_grids",
                    "finite_imagery",
                    "integer_reference_maps",
                    "saved_tensor_reload_equality",
                ]
                + ([] if raw_only else ["normalized_float32_range_0_1"]),
                "channel_order_evidence_level": profile.evidence_level,
                "checkpoint_compatibility_tested": False,
                "manifest_sha256": sha256(temporary / "manifest.json"),
            },
        )
        if output.exists():
            raise FileExistsError(f"Output appeared during export: {output}")
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="BigEarthNet v2 root")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--metadata", type=Path, help="Alternative metadata parquet selection")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64, help="Maximum areas per saved batch")
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = export_inputs(
            args.input,
            args.output,
            metadata=args.metadata,
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            raw_only=args.raw_only,
            progress=print,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {manifest['sample_count']} areas to {args.output.resolve()}")


if __name__ == "__main__":
    main()
