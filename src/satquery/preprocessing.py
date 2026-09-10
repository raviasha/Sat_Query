"""BigEarthNet v2 raster loading, independent of export, model weights and training."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.warp import reproject
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChannelProfile:
    optical: tuple[str, ...]
    sar: tuple[str, ...]
    evidence: str
    evidence_level: str


DEFAULT_PROFILE = ChannelProfile(
    optical=("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"),
    sar=("VV", "VH"),
    evidence=(
        "https://github.com/zhu-xlab/SSL4EO-S12/blob/"
        "2156913c5d8e5a2c572a5b000f0d5eaed6fc3192/"
        "src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py"
    ),
    evidence_level="source_dataset_verified_croma_order_inferred",
)
CROMA_REVISION = "59505a6bcadbf36ba20767270154bf9f3067c5e7"
NORMALIZATION_PROFILE = "croma_readme_patch_8bit_v1"
NATIVE_SIZE = {
    band: (20 if band in ("B01", "B09") else 120 if band in ("B02", "B03", "B04", "B08") else 60)
    for band in DEFAULT_PROFILE.optical
}


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    s1_name: str
    split: str
    country: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(
    root: Path, *, metadata: Path | None = None, split: str | None = None, limit: int | None = None
) -> list[PatchRecord]:
    """Read ordered pairings; selection never depends on filesystem enumeration."""
    path = Path(metadata) if metadata is not None else Path(root) / "metadata.parquet"
    frame = pd.read_parquet(path)
    fields = ("patch_id", "s1_name", "split", "country")
    if not set(fields).issubset(frame.columns):
        raise ValueError(f"{path}: required metadata columns are {fields}")
    if frame[list(fields)].isna().any().any():
        raise ValueError(f"{path}: null metadata identifiers or attributes")
    if frame["patch_id"].duplicated().any() or frame["s1_name"].duplicated().any():
        raise ValueError(f"{path}: duplicate optical or SAR IDs")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    if split is not None:
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        frame = frame.loc[frame["split"] == split]
    if limit is not None:
        frame = frame.head(limit)
    if frame.empty:
        raise ValueError(f"{path}: empty sample selection")
    records = []
    for values in frame[list(fields)].itertuples(index=False, name=None):
        record = PatchRecord(*values)
        for identifier in (record.patch_id, record.s1_name):
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
                raise ValueError(f"Unsafe patch identifier: {identifier!r}")
        records.append(record)
    return records


def _band_paths(folder: Path, bands: tuple[str, ...], patch_id: str) -> list[Path]:
    found = {}
    for path in folder.glob("*.tif"):
        band = path.stem.rsplit("_", 1)[-1]
        if band in found:
            raise ValueError(f"{patch_id}: duplicate band {band} in {folder}")
        found[band] = path
    missing, extra = set(bands) - found.keys(), found.keys() - set(bands)
    if missing or extra:
        raise ValueError(
            f"{patch_id}: missing bands {sorted(missing)}, unexpected bands "
            f"{sorted(extra)} in {folder}"
        )
    return [found[band] for band in bands]


def load_raw_patch(
    root: Path, record: PatchRecord, profile: ChannelProfile = DEFAULT_PROFILE
) -> dict:
    """Return aligned float32 imagery and integer CLC map for one 1.2 km area."""
    root = Path(root)
    if (
        set(profile.optical) != set(DEFAULT_PROFILE.optical)
        or len(profile.optical) != 12
        or set(profile.sar) != {"VV", "VH"}
        or len(profile.sar) != 2
    ):
        raise ValueError("Channel profile must contain exactly 12 optical and 2 SAR bands")
    for identifier in (record.patch_id, record.s1_name):
        if not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
            raise ValueError(f"Unsafe patch identifier: {identifier!r}")
    optical_folder = root / "BigEarthNet-S2" / record.patch_id.rsplit("_", 2)[0] / record.patch_id
    sar_folder = root / "BigEarthNet-S1" / record.s1_name.rsplit("_", 3)[0] / record.s1_name
    optical = _band_paths(optical_folder, profile.optical, record.patch_id)
    sar = _band_paths(sar_folder, profile.sar, record.patch_id)
    map_path = (
        root
        / "Reference_Maps"
        / record.patch_id.rsplit("_", 2)[0]
        / record.patch_id
        / f"{record.patch_id}_reference_map.tif"
    )
    target_path = optical[profile.optical.index("B02")]
    try:
        with rasterio.open(target_path) as target:
            if (
                target.crs is None
                or not target.crs.is_projected
                or target.crs.linear_units != "metre"
                or target.shape != (120, 120)
                or not np.allclose(
                    [
                        target.transform.a,
                        target.transform.b,
                        target.transform.d,
                        target.transform.e,
                    ],
                    [10, 0, 0, -10],
                    rtol=0,
                    atol=1e-6,
                )
            ):
                raise ValueError(f"{record.patch_id}: invalid B02 target grid in {target_path}")
            crs, bounds, transform = target.crs, target.bounds, target.transform
        arrays, sources = [], []
        for path, size, is_map in (
            [(p, NATIVE_SIZE[b], False) for p, b in zip(optical, profile.optical)]
            + [(p, 120, False) for p in sar]
            + [(map_path, 120, True)]
        ):
            with rasterio.open(path) as src:
                expected_res = 1200 / size
                if (
                    src.count != 1
                    or src.shape != (size, size)
                    or src.crs != crs
                    or not np.allclose(src.bounds, bounds, rtol=0, atol=1e-5)
                    or not np.allclose(
                        [src.transform.a, src.transform.b, src.transform.d, src.transform.e],
                        [expected_res, 0, 0, -expected_res],
                        rtol=0,
                        atol=1e-6,
                    )
                ):
                    raise ValueError(f"{record.patch_id}: grid mismatch in {path}")
                original = src.read(1)
                if not np.isfinite(original).all():
                    raise ValueError(f"{record.patch_id}: nonfinite pixels in {path}")
                if is_map:
                    if not np.issubdtype(original.dtype, np.integer):
                        raise ValueError(f"{record.patch_id}: noninteger class map in {path}")
                    # Keep unlabeled/nodata class codes exactly as distributed.
                    array = original.astype(np.int64)
                else:
                    if not (src.read_masks(1) == 255).all():
                        raise ValueError(f"{record.patch_id}: invalid/nodata pixels in {path}")
                    array = original.astype(np.float32)
                    if size != 120:
                        # GDAL raster reads can resample in the integer source dtype,
                        # rounding before out_dtype is applied. Convert first instead.
                        destination = np.empty((120, 120), dtype=np.float32)
                        reproject(
                            source=array,
                            destination=destination,
                            src_transform=src.transform,
                            src_crs=crs,
                            dst_transform=transform,
                            dst_crs=crs,
                            resampling=Resampling.bilinear,
                        )
                        array = destination
                    if not np.isfinite(array).all():
                        raise ValueError(f"{record.patch_id}: nonfinite resampled pixels in {path}")
                arrays.append(torch.from_numpy(array))
                sources.append(
                    {
                        "path": str(path.relative_to(root)),
                        "sha256": sha256(path),
                        "native_shape": list(src.shape),
                        "dtype": src.dtypes[0],
                        "nodata": src.nodata,
                    }
                )
    except rasterio.errors.RasterioError as exc:
        raise ValueError(f"{record.patch_id}: cannot read raster: {exc}") from exc
    return {
        "optical_images": torch.stack(arrays[:12]),
        "SAR_images": torch.stack(arrays[12:14]),
        "reference_map": arrays[14],
        "provenance": {
            **asdict(record),
            "crs": str(crs),
            "transform": list(transform)[:6],
            "bounds": list(bounds),
            "sources": sources,
        },
    }


def normalize_patch(image: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Apply CROMA README's N=1 normalization; never mix statistics across patches."""
    if image.ndim != 3 or image.shape[0] == 0 or image.shape[1] * image.shape[2] < 2:
        raise ValueError("Expected (channels, height, width) with at least two pixels")
    image = image.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(image).all():
        raise ValueError("Normalization requires finite pixels")
    means, stds, constants, channels = [], [], [], []
    for index, channel in enumerate(image):
        # Repeated fractional float32 values can have a nonzero computed std
        # from reduction roundoff, although the channel is exactly constant.
        if channel.min() == channel.max():
            mean, std = channel[0, 0], channel.new_tensor(0)
        else:
            mean, std = channel.mean(), channel.std(correction=1)
        if not torch.isfinite(mean) or not torch.isfinite(std):
            raise ValueError("Normalization statistics must be finite")
        lo, hi = mean - 2 * std, mean + 2 * std
        if std == 0 or hi == lo:
            constants.append(index)
            values = torch.full_like(channel, 0.5 * 255)
        else:
            values = ((channel - lo) / (hi - lo) * 255).clamp(0, 255)
        channels.append(values.to(torch.uint8).float() / 255)
        means.append(mean.item())
        stds.append(std.item())
    return torch.stack(channels), {
        "profile": NORMALIZATION_PROFILE,
        "mean": means,
        "sample_std": stds,
        "constant_channels": constants,
    }


class BigEarthNetCromaDataset(Dataset):
    """Lazy per-area loader, directly usable with torch.utils.data.DataLoader."""

    def __init__(
        self,
        root: Path,
        *,
        metadata: Path | None = None,
        split: str | None = None,
        limit: int | None = None,
        profile: ChannelProfile = DEFAULT_PROFILE,
    ):
        if profile != DEFAULT_PROFILE:
            raise ValueError("Unverified custom channel profile; use load_raw_patch for raw data")
        self.root, self.profile = Path(root), profile
        self.records = read_records(self.root, metadata=metadata, split=split, limit=limit)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        raw = load_raw_patch(self.root, record, self.profile)
        optical, _ = normalize_patch(raw["optical_images"])
        sar, _ = normalize_patch(raw["SAR_images"])
        return {
            "optical_images": optical,
            "SAR_images": sar,
            "reference_map": raw["reference_map"],
            "patch_id": record.patch_id,
            "s1_name": record.s1_name,
        }
