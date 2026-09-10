"""Resumable selective BigEarthNet v2 download, suitable for Colab + Drive."""

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from safetensors.numpy import load as load_safetensors

from .preprocessing import DEFAULT_PROFILE, sha256
from .remote_lmdb import HTTPRanges, LMDBReader

LMDB_REVISION = "118d1b6285c080ba8e4078414e1b8a243b18c9bd"
LMDB_URL = f"https://huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB/resolve/{LMDB_REVISION}/BENv2.lmdb/data.mdb"
LMDB_SIZE = 155372892160
ARCHIVE_REVISION = "3cf3a5910a5302d449fdb8e570e5b78de24fe07f"
BASE = f"https://huggingface.co/datasets/torchgeo/bigearthnet/resolve/{ARCHIVE_REVISION}/V2/"
MAPS_SHA256 = "23311e4efee2622052a7102a164473a5150f1e3bc47625c4a897b1c21063efd1"
METADATA_SHA256 = "408911df2da7092da9ecc72071972a808ec486ba09f6cb048f7716793d14ded6"
EXCLUDED_DEMO_IDS = [f"S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_{n}" for n in (39, 40, 41)]


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def download_file(url, size, digest, path, progress=print):
    """Resume a sequential cache file; verify publisher hash before using it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != digest:
            raise ValueError(f"Cached checksum mismatch: {path}")
        return path
    partial = path.with_suffix(path.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    if start > size:
        raise ValueError("Partial file is larger than pinned source")
    remote = HTTPRanges(url, size)
    with partial.open("ab") as stream:
        while start < size:
            data = remote.read(start, min(8 * 1024 * 1024, size - start))
            stream.write(data)
            stream.flush()
            start += len(data)
            progress(f"{path.name}: {start / 1024**2:.1f}/{size / 1024**2:.1f} MiB")
    if sha256(partial) != digest:
        raise ValueError(f"Download checksum mismatch: {partial}")
    partial.replace(path)
    return path


def select_records(metadata, counts, *, seed=17):
    """Deterministic country-balanced selection within existing official splits."""
    if any(
        k not in ("train", "validation", "test") or not isinstance(v, int) or v < 0
        for k, v in counts.items()
    ):
        raise ValueError("Invalid split counts")
    frame = (
        metadata[~metadata.patch_id.isin(EXCLUDED_DEMO_IDS)]
        .drop_duplicates("patch_id")
        .drop_duplicates("s1_name")
    )
    chosen = []
    for split, count in counts.items():
        groups = [
            g.sample(frac=1, random_state=seed).reset_index(drop=True)
            for _, g in frame[frame.split == split].groupby("country", sort=True)
        ]
        if sum(len(g) for g in groups) < count:
            raise ValueError(f"Insufficient {split} records")
        indices = [0] * len(groups)
        rows = []
        while len(rows) < count:
            for i, group in enumerate(groups):
                if indices[i] < len(group):
                    rows.append(group.iloc[indices[i]])
                    indices[i] += 1
                    if len(rows) == count:
                        break
        chosen.extend(rows)
    if not chosen:
        raise ValueError("Select at least one area")
    result = pd.DataFrame(chosen).reset_index(drop=True)
    for column in ["patch_id", "s1_name"]:
        if (
            not result[column]
            .map(lambda s: isinstance(s, str) and re.fullmatch("[A-Za-z0-9_]+", s) is not None)
            .all()
        ):
            raise ValueError("Unsafe dataset identifier")
    return result


def extract_reference_maps(archive, patch_ids, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    wanted = {p + "_reference_map.tif": p for p in patch_ids}
    found = {}
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in wanted:
                continue
            if not member.isfile() or member.size > 1024 * 1024:
                raise ValueError("Invalid reference-map member")
            patch = wanted[name]
            dest = output / name
            if patch in found:
                raise ValueError("Duplicate reference map")
            with tar.extractfile(member) as source:
                dest.write_bytes(source.read())
            found[patch] = dest
            if len(found) == len(wanted):
                break
    if len(found) != len(wanted):
        raise ValueError(f"Missing {len(wanted) - len(found)} reference maps")
    return found


def write_patch(root, record, reference, optical_bytes, sar_bytes):
    root = Path(root)
    patch, s1 = record["patch_id"], record["s1_name"]
    optical, sar = load_safetensors(optical_bytes), load_safetensors(sar_bytes)
    if set(optical) != set(DEFAULT_PROFILE.optical) or set(sar) != set(DEFAULT_PROFILE.sar):
        raise ValueError("Unexpected band set in mirror")
    with rasterio.open(reference) as ref:
        if ref.shape != (120, 120) or ref.res != (10, 10) or not ref.crs:
            raise ValueError("Invalid reference grid")
        transform, crs = ref.transform, ref.crs
    files = []
    for sensor, identifier, arrays in [
        ("BigEarthNet-S2", patch, optical),
        ("BigEarthNet-S1", s1, sar),
    ]:
        tile = identifier.rsplit("_", 2 if sensor.endswith("S2") else 3)[0]
        folder = root / sensor / tile / identifier
        folder.mkdir(parents=True, exist_ok=True)
        for band, array in arrays.items():
            n = (
                20
                if band in ("B01", "B09")
                else (60 if band in ("B05", "B06", "B07", "B8A", "B11", "B12") else 120)
            )
            dtype = np.dtype("uint16" if sensor.endswith("S2") else "float32")
            if array.shape != (n, n) or array.dtype != dtype or not np.isfinite(array).all():
                raise ValueError("Invalid native band array")
            p = folder / f"{identifier}_{band}.tif"
            with rasterio.open(
                p,
                "w",
                driver="GTiff",
                height=n,
                width=n,
                count=1,
                dtype=dtype,
                crs=crs,
                transform=transform * Affine.scale(120 / n),
                nodata=0 if sensor.endswith("S2") else None,
            ) as dst:
                dst.write(array, 1)
            files.append({"path": p.relative_to(root).as_posix(), "sha256": sha256(p)})
    p = root / "Reference_Maps" / patch.rsplit("_", 2)[0] / patch / f"{patch}_reference_map.tif"
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference, p)
    files.append({"path": p.relative_to(root).as_posix(), "sha256": sha256(p)})
    return {
        "patch_id": patch,
        "s1_name": s1,
        "files": files,
        "optical_record_sha256": hashlib.sha256(optical_bytes).hexdigest(),
        "sar_record_sha256": hashlib.sha256(sar_bytes).hexdigest(),
    }


def download_subset(
    destination, work, *, counts=None, seed=17, shard_size=100, workers=4, progress=print
):
    counts = counts or {"train": 600, "validation": 200, "test": 200}
    if shard_size < 1 or not 1 <= workers <= 8:
        raise ValueError("Invalid shard size or workers")
    destination, work = Path(destination).resolve(), Path(work).resolve()
    if destination == work or destination in work.parents or work in destination.parents:
        raise ValueError("Use separate persistent destination and temporary work directories")
    destination.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    cache = destination / "source-cache"
    metadata_path = download_file(
        BASE + "metadata.parquet", 3616349, METADATA_SHA256, cache / "metadata.parquet", progress
    )
    selected = select_records(pd.read_parquet(metadata_path), counts, seed=seed)
    config = {
        "format_version": 1,
        "counts": counts,
        "seed": seed,
        "shard_size": shard_size,
        "selection": "country-balanced within official splits, unique S1 pairs, original demo excluded",
        "patch_ids": selected.patch_id.tolist(),
        "metadata_sha256": METADATA_SHA256,
        "lmdb_url": LMDB_URL,
        "lmdb_revision": LMDB_REVISION,
        "mirror_status": "unofficial pre-conversion",
        "reference_archive_url": BASE + "Reference_Maps.tar.gzaa",
        "reference_archive_sha256": MAPS_SHA256,
        "imagery_storage": "GeoTIFF reconstructed from native band arrays using original reference-map georeferencing",
    }
    if (destination / "selection.json").exists():
        if json.loads((destination / "selection.json").read_text()) != config:
            raise ValueError("Selection differs from existing download; use a new destination")
    else:
        _json(destination / "selection.json", config)
    selected_metadata = destination / "metadata.parquet.tmp"
    selected.to_parquet(selected_metadata, index=False)
    selected_metadata.replace(destination / "metadata.parquet")
    pieces = [
        selected.iloc[start : start + shard_size] for start in range(0, len(selected), shard_size)
    ]
    completed = []
    todo = []
    for index, part in enumerate(pieces):
        stem = f"shard-{index:06d}"
        receipt = destination / (stem + ".json")
        archive = destination / (stem + ".zip")
        if receipt.exists():
            info = json.loads(receipt.read_text())
            if (
                info["patch_ids"] != part.patch_id.tolist()
                or not archive.exists()
                or sha256(archive) != info["sha256"]
            ):
                raise ValueError(f"Completed shard failed verification: {stem}")
            completed.append(info)
        else:
            todo.append((index, part))
    if todo:
        maps_archive = download_file(
            BASE + "Reference_Maps.tar.gzaa",
            326839377,
            MAPS_SHA256,
            cache / "reference-maps.tar.gz",
            progress,
        )
        progress("Extracting only the selected original reference maps...")
        maps = extract_reference_maps(
            maps_archive, [p for _, part in todo for p in part.patch_id], work / "reference-maps"
        )
        remote = HTTPRanges(LMDB_URL, LMDB_SIZE)
        db = LMDBReader(remote.read, LMDB_SIZE)
        for index, part in todo:
            stem = f"shard-{index:06d}"
            local = Path(tempfile.mkdtemp(prefix=stem + "-", dir=work))
            records = {}

            def fetch(record, local=local):
                optical = db.get(record["patch_id"])
                sar = db.get(record["s1_name"])
                return write_patch(local, record, maps[record["patch_id"]], optical, sar)

            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    pending = [pool.submit(fetch, r) for r in part.to_dict("records")]
                    for future in as_completed(pending):
                        rec = future.result()
                        records[rec["patch_id"]] = rec
                        if len(records) % 10 == 0 or len(records) == len(part):
                            progress(f"{stem}: {len(records)}/{len(part)} paired areas")
                part.to_parquet(local / "metadata.parquet", index=False)
                _json(
                    local / "download-provenance.json",
                    {"source": config, "records": [records[p] for p in part.patch_id]},
                )
                zip_path = work / (stem + ".zip")
                with zipfile.ZipFile(
                    zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
                ) as zipped:
                    for path in sorted(local.rglob("*")):
                        if path.is_file():
                            zipped.write(path, path.relative_to(local))
                digest = sha256(zip_path)
                remote_zip = destination / (stem + ".zip")
                partial = destination / (stem + ".zip.part")
                shutil.copyfile(zip_path, partial)
                if sha256(partial) != digest:
                    raise ValueError("Drive copy checksum mismatch")
                partial.replace(remote_zip)
                receipt = {
                    "archive": remote_zip.name,
                    "sha256": digest,
                    "bytes": remote_zip.stat().st_size,
                    "sample_count": len(part),
                    "patch_ids": part.patch_id.tolist(),
                    "file_count": len(part) * 15,
                }
                _json(destination / (stem + ".json"), receipt)
                completed.append(receipt)
                zip_path.unlink()
                progress(f"Saved and verified {stem} in {destination}")
            finally:
                shutil.rmtree(local)
    result = {
        "format_version": 1,
        "complete": True,
        "sample_count": len(selected),
        "counts": counts,
        "countries": selected.country.value_counts().to_dict(),
        "classes_present": sorted({label for labels in selected.labels for label in labels}),
        "selection_sha256": sha256(destination / "selection.json"),
        "metadata_sha256": sha256(destination / "metadata.parquet"),
        "shards": sorted(completed, key=lambda r: r["archive"]),
    }
    _json(destination / "download.json", result)
    progress(
        f"COMPLETE: {len(selected)} matched areas, {len(selected) * 15} TIFFs in {len(completed)} ZIP shards"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    download_subset(
        args.destination,
        args.work,
        counts={"train": args.train, "validation": args.validation, "test": args.test},
        seed=args.seed,
        shard_size=args.shard_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
