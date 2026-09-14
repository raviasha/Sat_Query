"""Strict, label-free satellite bundle ingestion for assistant inference."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.warp import reproject

from ..preprocessing import DEFAULT_PROFILE, NATIVE_SIZE, normalize_patch, sha256

MAX_UPLOAD_SIZE = 64 * 1024 * 1024
MAX_EXPANDED_SIZE = 128 * 1024 * 1024
MAX_MEMBER_SIZE = 64 * 1024 * 1024
MAX_MEMBERS = 64
MAX_OBSERVATIONS = 4
_GRID_ATOL = 1e-6


@dataclass(frozen=True)
class Observation:
    """One normalized sensor observation on the fixed CROMA input grid."""

    id: str
    modality: str
    sensor: str
    acquired: date
    normalized: torch.Tensor
    preview: np.ndarray
    grid: dict[str, Any]
    normalization: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InputBundle:
    """An explicitly validated single, cross-modal, or temporal input."""

    kind: str
    observations: tuple[Observation, ...]
    grid: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe input facts without extraction paths or pixel values."""
        return {
            "kind": self.kind,
            "observation_count": len(self.observations),
            "observations": [
                {
                    "id": item.id,
                    "modality": item.modality,
                    "sensor": item.sensor,
                    "acquired": item.acquired.isoformat(),
                    "bands": list(item.metadata["bands"]),
                    **(
                        {"radiometry": item.metadata["radiometry"]}
                        if item.modality == "SAR"
                        else {}
                    ),
                }
                for item in self.observations
            ],
            "grid": dict(self.grid),
        }


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError("ZIP member path must be a nonempty portable relative path")
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return path.as_posix()


def _validate_zip(
    archive: zipfile.ZipFile,
) -> tuple[list[zipfile.ZipInfo], dict[str, zipfile.ZipInfo]]:
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ValueError(f"ZIP has too many members; maximum is {MAX_MEMBERS}")
    files: dict[str, zipfile.ZipInfo] = {}
    folded: set[str] = set()
    expanded = 0
    for info in infos:
        name = _safe_member_name(info.filename.rstrip("/") if info.is_dir() else info.filename)
        folded_name = name.casefold()
        if folded_name in folded:
            raise ValueError(f"Duplicate ZIP member path: {name}")
        folded.add(folded_name)
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if stat.S_ISLNK(mode) or (not info.is_dir() and file_type not in (0, stat.S_IFREG)):
            raise ValueError(f"ZIP members must be regular files: {name}")
        if info.flag_bits & 0x1:
            raise ValueError("Encrypted ZIP members are not supported")
        if info.is_dir():
            continue
        if info.file_size > MAX_MEMBER_SIZE:
            raise ValueError(f"ZIP member is too large: {name}")
        expanded += info.file_size
        if expanded > MAX_EXPANDED_SIZE:
            raise ValueError("ZIP expanded size exceeds 128 MiB")
        files[name] = info
    if not files:
        raise ValueError("ZIP contains no regular files")
    return infos, files


def _parse_date(value: Any, identifier: str) -> date:
    if not isinstance(value, str):
        raise TypeError(f"{identifier}: acquired must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{identifier}: acquired must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{identifier}: acquired must use YYYY-MM-DD")
    return parsed


def _validate_manifest(
    raw: bytes, files: dict[str, zipfile.ZipInfo]
) -> tuple[str, list[dict[str, Any]]]:
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request.json must contain valid UTF-8 JSON") from exc
    if not isinstance(request, dict) or set(request) != {"schema_version", "observations"}:
        raise ValueError("request.json must contain only schema_version and observations")
    schema_version = request["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError("Unsupported request schema_version; expected 1")
    observations = request["observations"]
    if not isinstance(observations, list) or not observations:
        raise ValueError("request observations must be a nonempty list")
    if len(observations) > MAX_OBSERVATIONS:
        raise ValueError(f"Too many observations; maximum is {MAX_OBSERVATIONS}")

    ids: set[str] = set()
    referenced: set[str] = set()
    band_sources: set[tuple[str, int]] = set()
    parsed: list[dict[str, Any]] = []
    for number, value in enumerate(observations):
        if not isinstance(value, dict):
            raise TypeError(f"observation {number}: expected an object")
        identifier = value.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or len(identifier) > 128
            or identifier in ids
        ):
            raise ValueError("observation id must be nonempty, unique, and at most 128 characters")
        ids.add(identifier)
        modality = value.get("modality")
        if modality not in ("optical", "SAR"):
            raise ValueError(f"{identifier}: modality must be optical or SAR")
        expected_sensor = "Sentinel-2" if modality == "optical" else "Sentinel-1"
        if value.get("sensor") != expected_sensor:
            raise ValueError(
                f"{identifier}: unsupported sensor; {modality} requires {expected_sensor} (RGB is unsupported)"
            )
        required = {"id", "modality", "sensor", "acquired", "bands"}
        if modality == "SAR":
            required.add("radiometry")
        if set(value) != required:
            raise ValueError(f"{identifier}: expected exactly fields {sorted(required)}")
        if modality == "SAR" and value["radiometry"] != "db":
            raise ValueError(f"{identifier}: SAR radiometry must be 'db'")
        acquired = _parse_date(value.get("acquired"), identifier)
        expected_bands = DEFAULT_PROFILE.optical if modality == "optical" else DEFAULT_PROFILE.sar
        bands = value.get("bands")
        if not isinstance(bands, dict) or set(bands) != set(expected_bands):
            raise ValueError(
                f"{identifier}: bands must be exactly {list(expected_bands)}; single-band SAR is unsupported"
            )
        normalized_bands: dict[str, dict[str, Any]] = {}
        for band in expected_bands:
            entry = bands[band]
            if (
                not isinstance(entry, dict)
                or set(entry) != {"file", "band"}
                or not isinstance(entry["band"], int)
                or isinstance(entry["band"], bool)
                or entry["band"] < 1
            ):
                raise ValueError(
                    f"{identifier}: band {band} must specify a file and positive band index"
                )
            relative = _safe_member_name(entry["file"])
            if Path(relative).suffix.lower() not in (".tif", ".tiff"):
                raise ValueError(f"{identifier}: only TIFF band files are accepted")
            if relative not in files:
                raise ValueError(f"{identifier}: referenced band file is missing: {relative}")
            source_key = (relative, entry["band"])
            if source_key in band_sources:
                raise ValueError(f"Duplicate referenced TIFF band: {relative} band {entry['band']}")
            referenced.add(relative)
            band_sources.add(source_key)
            normalized_bands[band] = {"file": relative, "band": entry["band"]}
        parsed.append(
            {
                "id": identifier,
                "modality": modality,
                "sensor": expected_sensor,
                "acquired": acquired,
                "bands": normalized_bands,
                **({"radiometry": "db"} if modality == "SAR" else {}),
            }
        )

    extras = set(files) - referenced - {"request.json"}
    if extras:
        raise ValueError(f"ZIP contains unreferenced files: {sorted(extras)}")
    if len(parsed) == 1:
        kind = "single"
    elif len(parsed) == 2 and {item["modality"] for item in parsed} == {"optical", "SAR"}:
        kind = "cross_modal"
    elif len(parsed) == 2 and parsed[0]["modality"] == parsed[1]["modality"]:
        if parsed[0]["acquired"] >= parsed[1]["acquired"]:
            raise ValueError("Temporal observation dates must be distinct and in ascending order")
        kind = "temporal"
    else:
        raise ValueError(
            "Unsupported observation configuration; use one scene, one optical plus one SAR, "
            "or two ordered dates of the same modality"
        )
    return kind, parsed


def _extract(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo], root: Path) -> None:
    total = 0
    for info in infos:
        name = _safe_member_name(info.filename.rstrip("/") if info.is_dir() else info.filename)
        target = root.joinpath(*PurePosixPath(name).parts)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with archive.open(info) as source, target.open("xb") as output:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                total += len(block)
                if written > MAX_MEMBER_SIZE or total > MAX_EXPANDED_SIZE:
                    raise ValueError("ZIP expanded size exceeds configured limits")
                output.write(block)
        if written != info.file_size:
            raise ValueError(f"ZIP member size changed while extracting: {name}")


def _metre_projected(crs) -> bool:
    if crs is None or not crs.is_projected:
        return False
    try:
        return bool(np.isclose(crs.linear_units_factor[1], 1.0, rtol=0, atol=1e-12))
    except (AttributeError, TypeError):
        return crs.linear_units == "metre"


def _grid_from_dataset(src, identifier: str, band: str) -> dict[str, Any]:
    transform = src.transform
    if (
        src.shape != (120, 120)
        or not _metre_projected(src.crs)
        or not np.allclose(
            [transform.a, transform.b, transform.d, transform.e],
            [10, 0, 0, -10],
            rtol=0,
            atol=_GRID_ATOL,
        )
    ):
        raise ValueError(
            f"{identifier}: invalid projected north-up 120x120 10 metre grid in {band}"
        )
    return {
        "crs": str(src.crs),
        "transform": [float(value) for value in list(transform)[:6]],
        "bounds": [float(value) for value in src.bounds],
        "width": 120,
        "height": 120,
        "pixel_size_metres": 10.0,
    }


def _same_grid(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["crs"] == right["crs"]
        and np.allclose(left["transform"], right["transform"], rtol=0, atol=_GRID_ATOL)
        and np.allclose(left["bounds"], right["bounds"], rtol=0, atol=1e-5)
        and left["width"] == right["width"]
        and left["height"] == right["height"]
    )


def _read_observation(root: Path, record: dict[str, Any]) -> Observation:
    identifier, modality = record["id"], record["modality"]
    band_names = DEFAULT_PROFILE.optical if modality == "optical" else DEFAULT_PROFILE.sar
    target_band = "B02" if modality == "optical" else "VV"
    target_path = root / record["bands"][target_band]["file"]
    target_index = record["bands"][target_band]["band"]
    try:
        with rasterio.open(target_path, driver="GTiff") as target:
            if target.driver != "GTiff" or target.count < target_index:
                raise ValueError(f"{identifier}: {target_band} is not a readable TIFF band")
            grid = _grid_from_dataset(target, identifier, target_band)
            target_crs, target_transform = target.crs, target.transform

        arrays: list[torch.Tensor] = []
        sources: list[dict[str, Any]] = []
        for band in band_names:
            relative = record["bands"][band]["file"]
            band_index = record["bands"][band]["band"]
            path = root / relative
            with rasterio.open(path, driver="GTiff") as src:
                if src.driver != "GTiff" or src.count < band_index:
                    raise ValueError(f"{identifier}: {band} is not a readable TIFF band")
                native_size = NATIVE_SIZE[band] if modality == "optical" else 120
                if src.shape not in {(native_size, native_size), (120, 120)}:
                    raise ValueError(f"{identifier}: unsupported native shape for {band}")
                resolution = 1200 / src.width
                if (
                    src.crs != target_crs
                    or not np.allclose(src.bounds, grid["bounds"], rtol=0, atol=1e-5)
                    or not np.allclose(
                        [src.transform.a, src.transform.b, src.transform.d, src.transform.e],
                        [resolution, 0, 0, -resolution],
                        rtol=0,
                        atol=_GRID_ATOL,
                    )
                ):
                    raise ValueError(f"{identifier}: grid or footprint mismatch in {band}")
                if not (src.read_masks(band_index) == 255).all():
                    raise ValueError(f"{identifier}: masked or nodata pixels in {band}")
                values = src.read(band_index).astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"{identifier}: nonfinite pixels in {band}")
                if src.shape != (120, 120):
                    aligned = np.empty((120, 120), dtype=np.float32)
                    reproject(
                        source=values,
                        destination=aligned,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )
                    values = aligned
                if not np.isfinite(values).all():
                    raise ValueError(f"{identifier}: nonfinite resampled pixels in {band}")
                arrays.append(torch.from_numpy(values))
                sources.append(
                    {
                        "file": relative,
                        "band": band_index,
                        "sha256": sha256(path),
                        "native_shape": list(src.shape),
                        "dtype": src.dtypes[band_index - 1],
                    }
                )
    except rasterio.errors.RasterioError as exc:
        raise ValueError(f"{identifier}: cannot read TIFF: {exc}") from exc

    normalized, normalization = normalize_patch(torch.stack(arrays))
    if modality == "optical":
        indices = [band_names.index(name) for name in ("B04", "B03", "B02")]
        preview = (normalized[indices].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    else:
        channels = normalized.numpy()
        preview = np.stack([channels[0], channels[1], channels.mean(axis=0)], axis=-1)
        preview = (preview * 255).round().astype(np.uint8)
    metadata = {
        "bands": list(band_names),
        "sources": sources,
        **({"radiometry": "db"} if modality == "SAR" else {}),
    }
    return Observation(
        id=identifier,
        modality=modality,
        sensor=record["sensor"],
        acquired=record["acquired"],
        normalized=normalized,
        preview=np.ascontiguousarray(preview),
        grid=grid,
        normalization=normalization,
        metadata=metadata,
    )


def load_bundle(path: Path, destination: Path) -> InputBundle:
    """Validate and atomically extract a bounded request ZIP into ``destination``."""
    source = Path(path)
    destination = Path(destination).absolute()
    source_stat = source.lstat()
    if not stat.S_ISREG(source_stat.st_mode) or source.is_symlink():
        raise ValueError("Bundle path must be a regular ZIP file")
    if source_stat.st_size > MAX_UPLOAD_SIZE:
        raise ValueError("ZIP upload exceeds 64 MiB")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        try:
            with zipfile.ZipFile(source) as archive:
                infos, files = _validate_zip(archive)
                request_info = files.get("request.json")
                if request_info is None:
                    raise ValueError("ZIP must contain request.json at its root")
                if request_info.file_size > 1024 * 1024:
                    raise ValueError("request.json is too large")
                kind, records = _validate_manifest(archive.read(request_info), files)
                _extract(archive, infos, temporary)
        except zipfile.BadZipFile as exc:
            raise ValueError("Bundle is not a valid ZIP file") from exc

        observations = tuple(_read_observation(temporary, record) for record in records)
        common = observations[0].grid
        if any(not _same_grid(common, item.grid) for item in observations[1:]):
            raise ValueError("Observation pair must share the same CRS, extent, and 10 metre grid")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Extraction destination appeared during loading: {destination}")
        os.rename(temporary, destination)
        return InputBundle(kind=kind, observations=observations, grid=dict(common))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
