import json
import stat
import warnings
import zipfile
from datetime import date

import numpy as np
import pytest
import rasterio
import torch
from rasterio import Affine

from satquery.preprocessing import DEFAULT_PROFILE, NATIVE_SIZE, normalize_patch


def _write_tif(path, values, *, transform=None, crs="EPSG:32633", nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = transform or Affine(
        1200 / values.shape[1], 0, 373200, 0, -1200 / values.shape[0], 5353200
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)


def _observation_files(root, identifier, modality):
    bands = DEFAULT_PROFILE.optical if modality == "optical" else DEFAULT_PROFILE.sar
    result = {}
    for index, band in enumerate(bands):
        size = NATIVE_SIZE[band] if modality == "optical" else 120
        y, x = np.mgrid[:size, :size]
        values = (index * 17 + x * (index + 1) + y * (index + 2)).astype(np.float32)
        if modality == "SAR":
            values = values / 100 - 20
        relative = f"imagery/{identifier}-{band}.tif"
        _write_tif(root / relative, values)
        result[band] = {"file": relative, "band": 1}
    return result


def _request(root, observations):
    value = {"schema_version": 1, "observations": observations}
    (root / "request.json").write_text(json.dumps(value))
    return value


def _zip_tree(root, output):
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return output


def _observation(root, identifier, modality, acquired="2026-01-02"):
    return {
        "id": identifier,
        "modality": modality,
        "sensor": "Sentinel-2" if modality == "optical" else "Sentinel-1",
        "acquired": acquired,
        "bands": _observation_files(root, identifier, modality),
        **({"radiometry": "db"} if modality == "SAR" else {}),
    }


def _bundle(tmp_path, observations):
    source = tmp_path / "source"
    source.mkdir(parents=True)
    records = [_observation(source, item[0], item[1], item[2]) for item in observations]
    _request(source, records)
    return _zip_tree(source, tmp_path / "bundle.zip"), records


def test_load_optical_bundle_resamples_in_profile_order_and_has_safe_summary(tmp_path):
    from satquery.assistant.inputs import load_bundle

    archive, request_observations = _bundle(tmp_path, [("optical-a", "optical", "2026-01-02")])
    bundle = load_bundle(archive, tmp_path / "expanded")

    assert bundle.kind == "single"
    assert len(bundle.observations) == 1
    observation = bundle.observations[0]
    assert observation.id == "optical-a"
    assert observation.acquired == date(2026, 1, 2)
    assert observation.normalized.shape == (12, 120, 120)
    assert observation.normalized.dtype == torch.float32
    assert observation.preview.shape == (120, 120, 3)
    assert observation.preview.dtype == np.uint8
    assert bundle.grid == {
        "crs": "EPSG:32633",
        "transform": [10.0, 0.0, 373200.0, 0.0, -10.0, 5353200.0],
        "bounds": [373200.0, 5352000.0, 374400.0, 5353200.0],
        "width": 120,
        "height": 120,
        "pixel_size_metres": 10.0,
    }

    raw = []
    with zipfile.ZipFile(archive) as zipped:
        for band in DEFAULT_PROFILE.optical:
            extracted = tmp_path / f"expected-{band}.tif"
            extracted.write_bytes(zipped.read(request_observations[0]["bands"][band]["file"]))
            with rasterio.open(extracted) as src:
                values = src.read(
                    1, out_shape=(120, 120), resampling=rasterio.enums.Resampling.bilinear
                )
            raw.append(torch.from_numpy(values.astype(np.float32)))
    expected, _ = normalize_patch(torch.stack(raw))
    torch.testing.assert_close(observation.normalized, expected)

    summary = bundle.summary()
    assert summary["kind"] == "single"
    assert summary["observations"][0]["bands"] == list(DEFAULT_PROFILE.optical)
    serialized = json.dumps(summary)
    assert str(tmp_path) not in serialized
    assert "normalized" not in serialized and "preview" not in serialized


def test_loads_multiband_tiff_indices_in_declared_channel_order(tmp_path):
    from satquery.assistant.inputs import load_bundle

    source = tmp_path / "source"
    source.mkdir()
    path = source / "imagery" / "optical-stack.tif"
    path.parent.mkdir()
    y, x = np.mgrid[:120, :120]
    stack = np.stack([(index + 1) * x + (13 - index) * y for index in range(12)]).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=120,
        width=120,
        count=12,
        dtype="float32",
        crs="EPSG:32633",
        transform=Affine(10, 0, 373200, 0, -10, 5353200),
    ) as dst:
        dst.write(stack)
    observation = {
        "id": "stacked-optical",
        "modality": "optical",
        "sensor": "Sentinel-2",
        "acquired": "2026-01-02",
        "bands": {
            band: {"file": "imagery/optical-stack.tif", "band": index + 1}
            for index, band in enumerate(DEFAULT_PROFILE.optical)
        },
    }
    _request(source, [observation])
    bundle = load_bundle(_zip_tree(source, tmp_path / "bundle.zip"), tmp_path / "expanded")
    expected, _ = normalize_patch(torch.from_numpy(stack))
    torch.testing.assert_close(bundle.observations[0].normalized, expected, rtol=0, atol=0)


def test_multiband_manifest_rejects_out_of_range_and_duplicate_band_indices(tmp_path):
    from satquery.assistant.inputs import load_bundle

    source = tmp_path / "source"
    source.mkdir()
    record = _observation(source, "sar-a", "SAR")
    record["bands"]["VV"]["band"] = 2
    _request(source, [record])
    with pytest.raises(ValueError, match="readable TIFF band"):
        load_bundle(_zip_tree(source, tmp_path / "out-of-range.zip"), tmp_path / "out")

    source = tmp_path / "duplicate-source"
    source.mkdir()
    record = _observation(source, "sar-a", "SAR")
    record["bands"]["VH"] = dict(record["bands"]["VV"])
    _request(source, [record])
    with pytest.raises(ValueError, match="Duplicate referenced TIFF band"):
        load_bundle(
            _zip_tree(source, tmp_path / "duplicate-source.zip"), tmp_path / "duplicate-out"
        )


def test_load_sar_cross_modal_and_temporal_configurations(tmp_path):
    from satquery.assistant.inputs import load_bundle

    sar_zip, _ = _bundle(tmp_path / "sar-case", [("sar-a", "SAR", "2026-01-02")])
    sar = load_bundle(sar_zip, tmp_path / "sar-expanded")
    assert sar.kind == "single"
    assert sar.observations[0].normalized.shape == (2, 120, 120)
    assert sar.observations[0].metadata["radiometry"] == "db"

    cross_zip, _ = _bundle(
        tmp_path / "cross-case",
        [("optical-a", "optical", "2026-01-02"), ("sar-a", "SAR", "2026-01-03")],
    )
    cross = load_bundle(cross_zip, tmp_path / "cross-expanded")
    assert cross.kind == "cross_modal"
    assert [item.modality for item in cross.observations] == ["optical", "SAR"]

    temporal_zip, _ = _bundle(
        tmp_path / "temporal-case",
        [("before", "SAR", "2026-01-02"), ("after", "SAR", "2026-02-02")],
    )
    temporal = load_bundle(temporal_zip, tmp_path / "temporal-expanded")
    assert temporal.kind == "temporal"
    assert [item.id for item in temporal.observations] == ["before", "after"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["observations"][0].update(sensor="RGB"), "sensor"),
        (lambda value: value["observations"][0]["bands"].pop("VH"), "bands"),
        (lambda value: value["observations"][0].update(radiometry="linear"), "radiometry"),
        (lambda value: value.update(schema_version=2), "schema"),
    ],
)
def test_manifest_rejects_unknown_sensor_single_band_sar_and_wrong_schema(
    tmp_path, mutate, message
):
    from satquery.assistant.inputs import load_bundle

    source = tmp_path / "source"
    source.mkdir()
    request = _request(source, [_observation(source, "sar-a", "SAR")])
    mutate(request)
    (source / "request.json").write_text(json.dumps(request))
    archive = _zip_tree(source, tmp_path / "bundle.zip")
    with pytest.raises(ValueError, match=message):
        load_bundle(archive, tmp_path / "expanded")


def test_rejects_mismatched_grids_masked_pixels_and_bad_temporal_order(tmp_path):
    from satquery.assistant.inputs import load_bundle

    source = tmp_path / "shifted"
    source.mkdir()
    observation = _observation(source, "optical-a", "optical")
    path = source / observation["bands"]["B05"]["file"]
    values = np.ones((60, 60), dtype=np.float32)
    _write_tif(path, values, transform=Affine(20, 0, 373210, 0, -20, 5353200))
    _request(source, [observation])
    with pytest.raises(ValueError, match="grid|footprint"):
        load_bundle(_zip_tree(source, tmp_path / "shifted.zip"), tmp_path / "shifted-out")

    source = tmp_path / "masked"
    source.mkdir()
    observation = _observation(source, "sar-a", "SAR")
    path = source / observation["bands"]["VV"]["file"]
    values = np.ones((120, 120), dtype=np.float32)
    values[0, 0] = -9999
    _write_tif(path, values, nodata=-9999)
    _request(source, [observation])
    with pytest.raises(ValueError, match="masked|nodata"):
        load_bundle(_zip_tree(source, tmp_path / "masked.zip"), tmp_path / "masked-out")

    archive, _ = _bundle(
        tmp_path / "dates",
        [("later", "optical", "2026-02-02"), ("earlier", "optical", "2026-01-02")],
    )
    with pytest.raises(ValueError, match="date|order"):
        load_bundle(archive, tmp_path / "dates-out")


@pytest.mark.parametrize("attack", ["traversal", "absolute", "symlink", "duplicate", "oversize"])
def test_rejects_malicious_or_excessive_zip_members(tmp_path, monkeypatch, attack):
    from satquery.assistant import inputs

    archive = tmp_path / "attack.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        if attack == "traversal":
            zipped.writestr("../outside.tif", b"x")
        elif attack == "absolute":
            zipped.writestr("/outside.tif", b"x")
        elif attack == "symlink":
            info = zipfile.ZipInfo("linked.tif")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zipped.writestr(info, "target.tif")
        elif attack == "duplicate":
            zipped.writestr("request.json", b"{}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                zipped.writestr("request.json", b"{}")
        else:
            monkeypatch.setattr(inputs, "MAX_EXPANDED_SIZE", 8)
            zipped.writestr("request.json", b"0123456789")
    with pytest.raises(ValueError, match="path|regular|duplicate|size|large|expanded"):
        inputs.load_bundle(archive, tmp_path / "expanded")
    assert not (tmp_path / "outside.tif").exists()
    assert not (tmp_path / "expanded").exists()
