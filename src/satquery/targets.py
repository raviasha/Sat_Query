"""Land-cover coverage targets aligned with CROMA's spatial feature tokens."""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import tempfile
from pathlib import Path

import torch
from rasterio.crs import CRS

from .preprocessing import sha256

TAXONOMY_SOURCE = "https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf"
# Official v2 Table 1, fixed 19-class order; absent classes keep their columns.
CLASSES = (
    ("Urban fabric", (111, 112)),
    ("Industrial or commercial units", (121,)),
    ("Arable land", (211, 212, 213)),
    ("Permanent crops", (221, 222, 223, 241)),
    ("Pastures", (231,)),
    ("Complex cultivation patterns", (242,)),
    (
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        (243,),
    ),
    ("Agro-forestry areas", (244,)),
    ("Broad-leaved forest", (311,)),
    ("Coniferous forest", (312,)),
    ("Mixed forest", (313,)),
    ("Natural grassland and sparsely vegetated areas", (321, 333)),
    ("Moors, heathland and sclerophyllous vegetation", (322, 323)),
    ("Transitional woodland, shrub", (324,)),
    ("Beaches, dunes, sands", (331,)),
    ("Inland wetlands", (411, 412)),
    ("Coastal wetlands", (421, 422)),
    ("Inland waters", (511, 512)),
    ("Marine waters", (521, 522, 523)),
)
UNLABELED_CODES = (0, 999, 122, 123, 124, 131, 132, 133, 141, 142, 332, 334, 335, 423)


def patch_label_targets(reference_maps: torch.Tensor) -> dict[str, torch.Tensor]:
    """Count all 64 pixels per token; retain unlabeled area without renormalizing."""
    if (
        not isinstance(reference_maps, torch.Tensor)
        or reference_maps.ndim != 3
        or tuple(reference_maps.shape[1:]) != (120, 120)
        or len(reference_maps) < 1
        or reference_maps.dtype
        not in (
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
    ):
        raise ValueError("Reference maps must be nonempty integer (N,120,120) tensors")
    maps = reference_maps.detach().to(device="cpu", dtype=torch.int64)
    known = torch.tensor([c for _, codes in CLASSES for c in codes] + list(UNLABELED_CODES))
    unknown = maps.unique()[~torch.isin(maps.unique(), known)]
    if len(unknown):
        raise ValueError(f"Unknown reference-map codes: {unknown.tolist()}")
    lookup = torch.full((1000,), 19, dtype=torch.int64)
    for index, (_, codes) in enumerate(CLASSES):
        lookup[list(codes)] = index
    blocks = lookup[maps].reshape(-1, 15, 8, 15, 8).permute(0, 1, 3, 2, 4).reshape(-1, 225, 64)
    histogram = torch.zeros(len(maps), 225, 20, dtype=torch.int64)
    histogram.scatter_add_(2, blocks, torch.ones_like(blocks))
    counts, unlabeled = histogram[:, :, :19].contiguous(), histogram[:, :, 19].contiguous()
    return {
        "class_counts": counts,
        "class_fractions": counts.to(torch.float32) / 64,
        "unlabeled_counts": unlabeled,
        "unlabeled_fraction": unlabeled.to(torch.float32) / 64,
        "fully_labeled_mask": unlabeled == 0,
        "has_labels_mask": unlabeled < 64,
    }


def _bounds(samples):
    result = []
    for sample in samples:
        transform = sample.get("transform", [])
        if len(transform) != 6 or not all(math.isfinite(v) for v in transform):
            raise ValueError("Missing or invalid sample transform")
        a, b, c, d, e, f = transform
        crs = CRS.from_user_input(sample["crs"])
        if (
            not crs.is_projected
            or not math.isclose(crs.linear_units_factor[1], 1)
            or not all(
                math.isclose(x, y, abs_tol=1e-9) for x, y in zip((a, b, d, e), (10, 0, 0, -10))
            )
        ):
            raise ValueError("Targets require a projected north-up 10 metre pixel grid")
        expected = [c, f - 1200, c + 1200, f]
        bounds = sample.get("bounds", [])
        if len(bounds) != 4 or any(
            not math.isclose(x, y, abs_tol=1e-6, rel_tol=0) for x, y in zip(bounds, expected)
        ):
            raise ValueError("Sample bounds do not match the 120x120 ground grid")
        result.append(
            [
                [c + col * 80, f - (row + 1) * 80, c + (col + 1) * 80, f - row * 80]
                for row in range(15)
                for col in range(15)
            ]
        )
    return torch.tensor(result, dtype=torch.float64)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _read_checked(root, batch, name):
    directory = batch["directory"]
    if not isinstance(directory, str) or Path(directory).is_absolute():
        raise ValueError("Expected relative batch directory")
    path = (root / directory / name).resolve()
    if root not in path.parents:
        raise ValueError("Batch path escapes source directory")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != batch.get("sha256", {}).get(name):
        raise ValueError(f"Input hash mismatch: {path}")
    return path, data


def export_targets(prepared: Path, features: Path, output: Path, *, progress=None) -> dict:
    """Export per-token targets only after checking exact feature/reference alignment."""
    prepared, features, output = (
        Path(prepared).resolve(),
        Path(features).resolve(),
        Path(output).absolute(),
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output = output.resolve()
    if any(root == output or root in output.parents for root in (prepared, features)):
        raise ValueError("Output must be outside both source directories")
    paths = [prepared / "manifest.json", features / "manifest.json"]
    source_bytes = [p.read_bytes() for p in paths]
    digests = [hashlib.sha256(data).hexdigest() for data in source_bytes]
    pm, fm = [json.loads(data) for data in source_bytes]
    if any(m.get("format_version") != 1 or not m.get("batches") for m in (pm, fm)):
        raise ValueError("Expected version 1 prepared and feature manifests with batches")
    if fm.get("source_manifest_sha256") != digests[0]:
        raise ValueError("Features were not generated from this exact prepared manifest")
    if pm["sample_count"] != fm["sample_count"] or len(pm["batches"]) != len(fm["batches"]):
        raise ValueError("Prepared and feature sample/batch counts differ")
    if fm.get("feature_dimension") != 768:
        raise ValueError("Expected CROMA-Base feature dimension 768")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    batches, seen = [], set()
    total = complete = partial = empty = 0
    try:
        for number, (pb, fb) in enumerate(zip(pm["batches"], fm["batches"])):
            n = pb["sample_count"]
            if (
                not isinstance(n, int)
                or n < 1
                or n != fb["sample_count"]
                or pb["start_index"] != total
                or fb["start_index"] != total
            ):
                raise ValueError("Inconsistent source batch counts or order")
            _, pinfo_bytes = _read_checked(prepared, pb, "batch.json")
            _, finfo_bytes = _read_checked(features, fb, "batch.json")
            pinfo, finfo = json.loads(pinfo_bytes), json.loads(finfo_bytes)
            samples = pinfo["samples"]
            if (
                pinfo["sample_count"] != n
                or finfo["sample_count"] != n
                or len(samples) != n
                or finfo["samples"] != samples
            ):
                raise ValueError("Feature and reference sample metadata/order differ")
            for sample in samples:
                if (
                    not isinstance(sample.get("patch_id"), str)
                    or not sample["patch_id"]
                    or not isinstance(sample.get("s1_name"), str)
                    or not sample["s1_name"]
                    or sample["patch_id"] in seen
                ):
                    raise ValueError("Missing or duplicate sample identity")
                seen.add(sample["patch_id"])
            grid = finfo.get("spatial_grid", {})
            if any(
                grid.get(k) != v
                for k, v in {
                    "height": 15,
                    "width": 15,
                    "token_order": "row-major",
                    "input_patch_pixels": [8, 8],
                }.items()
            ):
                raise ValueError("Feature grid must be row-major 15x15 with 8x8 input patches")
            bounds = _bounds(samples)
            map_path, map_bytes = _read_checked(prepared, pb, "reference_maps.pt")
            reference_maps = torch.load(
                io.BytesIO(map_bytes), map_location="cpu", weights_only=True
            )
            targets = patch_label_targets(reference_maps)
            if targets["class_counts"].shape[0] != n:
                raise ValueError("Reference map count differs from sample metadata")
            feature_path, feature_bytes = _read_checked(features, fb, "features.pt")
            tensors = torch.load(io.BytesIO(feature_bytes), map_location="cpu", weights_only=True)
            expected = {
                f"{modality}_{kind}"
                for modality in ("optical", "SAR", "joint")
                for kind in ("encodings", "GAP")
            }
            if not isinstance(tensors, dict) or set(tensors) != expected:
                raise ValueError("Expected all six CROMA feature tensors")
            for key, value in tensors.items():
                shape = (n, 225, 768) if key.endswith("encodings") else (n, 768)
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != shape
                    or value.dtype != torch.float32
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid feature tensor: {key}")
            del tensors, feature_bytes, map_bytes, reference_maps
            targets["bounds"] = bounds
            if not torch.equal(
                targets["class_counts"].sum(-1) + targets["unlabeled_counts"],
                torch.full((n, 225), 64),
            ):
                raise ValueError("Target counts do not sum to 64")
            relative = "." if len(pm["batches"]) == 1 else f"batch-{number:06d}"
            destination = temporary / relative
            destination.mkdir(exist_ok=True)
            torch.save(targets, destination / "targets.pt")
            reloaded = torch.load(destination / "targets.pt", weights_only=True)
            if any(not torch.equal(value, reloaded[key]) for key, value in targets.items()):
                raise ValueError("Target save/reload mismatch")
            _json(
                destination / "batch.json",
                {
                    "sample_count": n,
                    "samples": samples,
                    "spatial_grid": grid,
                    "shapes": {k: list(v.shape) for k, v in targets.items()},
                    "feature_file": str(feature_path),
                    "feature_file_sha256": fb["sha256"]["features.pt"],
                    "reference_map_file": str(map_path),
                    "reference_map_sha256": pb["sha256"]["reference_maps.pt"],
                    "bounds_order": ["left", "bottom", "right", "top"],
                    "bounds_units": "metres in sample CRS",
                },
            )
            batches.append(
                {
                    "directory": relative,
                    "sample_count": n,
                    "start_index": total,
                    "feature_batch": fb,
                    "prepared_batch": pb,
                    "sha256": {
                        name: sha256(destination / name) for name in ("targets.pt", "batch.json")
                    },
                }
            )
            total += n
            complete += int(targets["fully_labeled_mask"].sum())
            empty += int((~targets["has_labels_mask"]).sum())
            partial += int((targets["has_labels_mask"] & ~targets["fully_labeled_mask"]).sum())
            if progress:
                progress(f"Prepared label targets for {total}/{pm['sample_count']} areas")
        if total != pm["sample_count"]:
            raise ValueError("Total source sample count differs from batches")
        if any(sha256(path) != digest for path, digest in zip(paths, digests)):
            raise ValueError("Source manifest changed during target export")
        manifest = {
            "format_version": 1,
            "target_profile": "bigearthnet19_full_patch_fractions_v1",
            "sample_count": total,
            "token_count": total * 225,
            "pixels_per_token": 64,
            "class_fraction_denominator": "all 64 pixels, including unlabeled",
            "classes": [
                {"index": i, "name": name, "clc_codes": list(codes)}
                for i, (name, codes) in enumerate(CLASSES)
            ],
            "unlabeled_codes": list(UNLABELED_CODES),
            "taxonomy_source": TAXONOMY_SOURCE,
            "taxonomy_source_table": 1,
            "target_implementation_sha256": sha256(Path(__file__)),
            "prepared_manifest_path": str(paths[0]),
            "prepared_manifest_sha256": digests[0],
            "feature_manifest_path": str(paths[1]),
            "feature_manifest_sha256": digests[1],
            "batches": batches,
        }
        _json(temporary / "manifest.json", manifest)
        _json(
            temporary / "validation.json",
            {
                "passed": True,
                "sample_count": total,
                "token_count": total * 225,
                "fully_labeled_tokens": complete,
                "partially_labeled_tokens": partial,
                "unlabeled_tokens": empty,
                "checks": [
                    "source_hashes",
                    "feature_sample_alignment",
                    "row_major_grid",
                    "ground_bounds",
                    "official_class_codes",
                    "feature_shapes_and_finiteness",
                    "64_pixel_totals",
                    "save_reload_equality",
                ],
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
