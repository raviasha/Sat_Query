from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio import Affine

from satquery.preprocessing import (
    DEFAULT_PROFILE,
    BigEarthNetCromaDataset,
    load_raw_patch,
    normalize_patch,
    read_records,
)

# Independent fixture sequence: catches filename sorting moving B8A to the end.
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S2 = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_61_39"
S1 = "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39"


def write_tif(path, values, *, shift=0, nodata=None, crs="EPSG:32633"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=Affine(
            1200 / values.shape[1], 0, 373200 + shift, 0, -1200 / values.shape[0], 5353200
        ),
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)


@pytest.fixture
def sample(tmp_path):
    optical = tmp_path / "BigEarthNet-S2" / S2.rsplit("_", 2)[0] / S2
    sar = tmp_path / "BigEarthNet-S1" / S1.rsplit("_", 3)[0] / S1
    maps = tmp_path / "Reference_Maps" / S2.rsplit("_", 2)[0] / S2
    for i, band in enumerate(BANDS):
        n = 20 if band in ("B01", "B09") else 120 if band in ("B02", "B03", "B04", "B08") else 60
        write_tif(optical / f"{S2}_{band}.tif", np.full((n, n), i + 1, np.uint16))
    for band, value in [("VV", -12.25), ("VH", -18.75)]:
        write_tif(sar / f"{S1}_{band}.tif", np.full((120, 120), value, np.float32))
    write_tif(maps / f"{S2}_reference_map.tif", np.full((120, 120), 311, np.uint16))
    pd.DataFrame(
        [{"patch_id": S2, "s1_name": S1, "split": "test", "country": "Austria"}]
    ).to_parquet(tmp_path / "metadata.parquet")
    return tmp_path


def test_band_order_sar_precision_and_maps(sample):
    patch = load_raw_patch(sample, read_records(sample)[0])
    assert patch["optical_images"].shape == (12, 120, 120)
    assert patch["optical_images"][:, 50, 50].tolist() == list(range(1, 13))
    assert patch["SAR_images"][:, 0, 0].tolist() == [-12.25, -18.75]
    assert patch["reference_map"].unique().tolist() == [311]
    assert patch["reference_map"].dtype == torch.int64


def test_metadata_order_filter_and_duplicate(sample):
    p = sample / "metadata.parquet"
    df = pd.read_parquet(p)
    other = df.copy()
    other["patch_id"] = S2 + "_other"
    other["s1_name"] = S1 + "_other"
    other["split"] = "train"
    pd.concat([other, df]).to_parquet(p, index=False)
    assert [r.patch_id for r in read_records(sample)] == [S2 + "_other", S2]
    assert read_records(sample, split="test")[0].patch_id == S2
    pd.concat([df, df]).to_parquet(p, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_records(sample)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "shift", "crs", "nan", "nodata"])
def test_invalid_source_rejected(sample, problem):
    path = next(sample.rglob("*_B05.tif"))
    values = np.full((60, 60), 5.0, np.float32)
    if problem == "missing":
        path.unlink()
    elif problem == "duplicate":
        path.with_name("duplicate_B05.tif").write_bytes(path.read_bytes())
    elif problem == "shift":
        write_tif(path, values, shift=10)
    elif problem == "crs":
        write_tif(path, values, crs="EPSG:32632")
    elif problem == "nan":
        values[0, 0] = np.nan
        write_tif(path, values)
    else:
        values[0, 0] = -9999
        write_tif(path, values, nodata=-9999)
    with pytest.raises(ValueError, match=S2):
        load_raw_patch(sample, read_records(sample)[0])


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_native_grid_unchanged_and_bilinear_ramp(sample, dtype):
    native = np.arange(14400, dtype=np.float32).reshape(120, 120)
    write_tif(next(sample.rglob("*_B02.tif")), native)
    ramp = np.broadcast_to(np.arange(60, dtype=dtype), (60, 60)).copy()
    write_tif(next(sample.rglob("*_B05.tif")), ramp)
    patch = load_raw_patch(sample, read_records(sample)[0])
    np.testing.assert_array_equal(patch["optical_images"][1].numpy(), native)
    # Target column 20 has center x=205m; source centers x=190m,210m => 9.75.
    assert patch["optical_images"][4, 20, 20].item() == pytest.approx(9.75)


def test_normalize_values_constant_and_invalid():
    out, stats = normalize_patch(torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]))
    torch.testing.assert_close(out, torch.tensor([[[53.0, 102.0], [152.0, 201.0]]]) / 255)
    constant, stats = normalize_patch(torch.full((2, 120, 120), -7.0))
    assert stats["constant_channels"] == [0, 1]
    torch.testing.assert_close(constant, torch.full_like(constant, 127 / 255))
    with pytest.raises(ValueError, match="finite"):
        normalize_patch(torch.tensor([[[float("nan"), 1.0], [2.0, 3.0]]]))


def test_dataset_normalization_independent_of_batch(sample):
    dataset = BigEarthNetCromaDataset(sample)
    one = dataset[0]
    assert one["patch_id"] == S2
    assert one["optical_images"].shape == (12, 120, 120)
    assert one["SAR_images"].dtype == torch.float32
    many = torch.stack([dataset[0]["optical_images"] for _ in range(3)])
    torch.testing.assert_close(many[2], one["optical_images"])


def test_normalize_clips_outliers():
    x = torch.zeros((1, 120, 120))
    x[0, 0, 0] = 100
    x[0, 0, 1] = -100
    out, _ = normalize_patch(x)
    assert out[0, 0, :2].tolist() == [1.0, 0.0]


@pytest.mark.parametrize("value", [0.1, 1.1, -12.37])
def test_normalize_fractional_constant(value):
    out, stats = normalize_patch(torch.full((1, 120, 120), value))
    assert stats["constant_channels"] == [0]
    torch.testing.assert_close(out, torch.full_like(out, 127 / 255))


def test_dataset_rejects_unverified_channel_order(sample):
    reversed_profile = replace(DEFAULT_PROFILE, sar=("VH", "VV"))
    with pytest.raises(ValueError, match="profile"):
        BigEarthNetCromaDataset(sample, profile=reversed_profile)


def test_reference_map_preserves_unlabeled_zero(sample):
    values = np.full((120, 120), 311, np.uint16)
    values[0, 0] = 0
    write_tif(next(sample.rglob("*_reference_map.tif")), values, nodata=0)
    patch = load_raw_patch(sample, read_records(sample)[0])
    assert patch["reference_map"][0, 0].item() == 0
    assert patch["reference_map"].unique().tolist() == [0, 311]
