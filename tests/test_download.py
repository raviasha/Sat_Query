import io
import tarfile

import lmdb
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from safetensors.numpy import save


def test_lmdb_reader_finds_exact_keys_and_overflow_data(tmp_path):
    from satquery.remote_lmdb import LMDBReader

    env = lmdb.open(str(tmp_path / "db"), map_size=64 * 1024 * 1024)
    expected = {f"patch_{i:05d}".encode(): (str(i).encode() * 3000) for i in range(3000)}
    with env.begin(write=True) as txn:
        for k, v in expected.items():
            txn.put(k, v)
        txn.put(b"inline", b"small")
    env.close()
    path = tmp_path / "db/data.mdb"

    def read(start, size):
        with path.open("rb") as stream:
            stream.seek(start)
            return stream.read(size)

    reader = LMDBReader(read, path.stat().st_size)
    for key in [b"patch_00000", b"patch_01001", b"patch_02999"]:
        assert reader.get(key) == expected[key]
    assert reader.get(b"inline") == b"small"
    with pytest.raises(KeyError):
        reader.get(b"missing")


def test_lmdb_reader_rejects_invalid_header():
    from satquery.remote_lmdb import LMDBReader

    with pytest.raises(ValueError):
        LMDBReader(lambda offset, size: b"\0" * size, 8192)


def test_selection_preserves_splits_and_country_balance():
    from satquery.download_bigearthnet import select_records

    rows = []
    for split in ["train", "validation", "test"]:
        for country in ["Austria", "Finland"]:
            for n in range(10):
                rows.append(
                    {
                        "patch_id": f"{split}_{country}_{n}",
                        "s1_name": f"S1_{split}_{country}_{n}",
                        "split": split,
                        "country": country,
                        "labels": ["Arable land"],
                    }
                )
    df = pd.DataFrame(rows)
    a = select_records(df, {"train": 6, "validation": 2, "test": 2}, seed=17)
    b = select_records(df, {"train": 6, "validation": 2, "test": 2}, seed=17)
    assert a.patch_id.tolist() == b.patch_id.tolist()
    assert a.groupby("split").size().to_dict() == {"test": 2, "train": 6, "validation": 2}
    assert a.groupby("country").size().to_dict() == {"Austria": 5, "Finland": 5}
    assert not a.patch_id.duplicated().any()


def test_reference_extraction_ignores_unselected_members(tmp_path):
    from satquery.download_bigearthnet import extract_reference_maps

    archive = tmp_path / "maps.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        for name, body in [
            ("Reference_Maps/tile/A_1_2/A_1_2_reference_map.tif", b"map"),
            ("../../unwanted", b"no"),
        ]:
            entry = tarfile.TarInfo(name)
            entry.size = len(body)
            t.addfile(entry, io.BytesIO(body))
    result = extract_reference_maps(archive, ["A_1_2"], tmp_path / "out")
    assert result["A_1_2"].read_bytes() == b"map"
    assert not (tmp_path / "unwanted").exists()


def test_reconstruction_preserves_native_values_and_georeferencing(tmp_path):
    from satquery.download_bigearthnet import write_patch

    reference = tmp_path / "reference.tif"
    with rasterio.open(
        reference,
        "w",
        driver="GTiff",
        height=120,
        width=120,
        count=1,
        dtype="uint16",
        crs="EPSG:32633",
        transform=from_origin(373200, 5353200, 10, 10),
        nodata=0,
    ) as d:
        d.write(np.full((120, 120), 311, dtype="uint16"), 1)
    shapes = {
        "B01": 20,
        "B09": 20,
        "B02": 120,
        "B03": 120,
        "B04": 120,
        "B08": 120,
        "B05": 60,
        "B06": 60,
        "B07": 60,
        "B8A": 60,
        "B11": 60,
        "B12": 60,
    }
    optical = {k: np.full((n, n), 321, dtype="uint16") for k, n in shapes.items()}
    sar = {k: np.full((120, 120), -12.345, dtype="float32") for k in ["VV", "VH"]}
    rec = {"patch_id": "S2_TEST_1_2", "s1_name": "S1_TEST_TILE_1_2"}
    output = tmp_path / "raw"
    record = write_patch(output, rec, reference, save(optical), save(sar))
    assert len(record["files"]) == 15
    with rasterio.open(next(output.rglob("*_B01.tif"))) as ds:
        assert ds.shape == (20, 20) and ds.res == (60, 60)
        assert ds.bounds == rasterio.coords.BoundingBox(373200, 5352000, 374400, 5353200)
        assert (ds.read(1) == 321).all()
    with rasterio.open(next(output.rglob("*_VV.tif"))) as ds:
        assert np.array_equal(ds.read(1), sar["VV"]) and ds.nodata is None


def test_range_requires_exact_response(monkeypatch):
    import satquery.remote_lmdb as module

    class Response(io.BytesIO):
        status = 200

        def __init__(self, value):
            super().__init__(value)
            self.headers = {}

    monkeypatch.setattr(module, "urlopen", lambda *a, **k: Response(b"whole-file"))
    with pytest.raises(ValueError, match="exact requested"):
        module.HTTPRanges("https://example.org/file", 100, attempts=1).read(5, 3)


def test_download_resumes_partial_cache_and_checks_hash(tmp_path, monkeypatch):
    import hashlib

    import satquery.download_bigearthnet as module

    body = b"original-data"
    requests = []

    class LocalRanges:
        def __init__(self, url, size):
            pass

        def read(self, start, length):
            requests.append((start, length))
            return body[start : start + length]

    monkeypatch.setattr(module, "HTTPRanges", LocalRanges)
    path = tmp_path / "source"
    path.with_suffix(".part").write_bytes(body[:4])
    digest = hashlib.sha256(body).hexdigest()
    module.download_file("unused", len(body), digest, path, lambda _: None)
    assert path.read_bytes() == body and requests == [(4, len(body) - 4)]
    module.download_file("unused", len(body), digest, path, lambda _: None)
    assert len(requests) == 1
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        module.download_file("unused", len(body), digest, path)


def test_subset_resume_repairs_metadata_without_refetching_shards(tmp_path, monkeypatch):
    import json
    import shutil
    import zipfile

    import satquery.download_bigearthnet as module

    source = tmp_path / "source"
    source.mkdir()
    ids = ["S2_TEST_1_2", "S2_TEST_1_3"]
    s1ids = ["S1_TEST_TILE_1_2", "S1_TEST_TILE_1_3"]
    frame = pd.DataFrame(
        {
            "patch_id": ids,
            "s1_name": s1ids,
            "split": ["train"] * 2,
            "country": ["Austria"] * 2,
            "labels": [["Arable land"]] * 2,
        }
    )
    frame.to_parquet(source / "metadata.parquet")
    reference = source / "reference.tif"
    with rasterio.open(
        reference,
        "w",
        driver="GTiff",
        height=120,
        width=120,
        count=1,
        dtype="uint16",
        crs="EPSG:32633",
        transform=from_origin(373200, 5353200, 10, 10),
    ) as dst:
        dst.write(np.full((120, 120), 211, dtype="uint16"), 1)
    with tarfile.open(source / "maps.tar.gz", "w:gz") as tar:
        for patch in ids:
            tar.add(reference, arcname=f"{patch}_reference_map.tif")
    shapes = {
        "B01": 20,
        "B09": 20,
        "B02": 120,
        "B03": 120,
        "B04": 120,
        "B08": 120,
        "B05": 60,
        "B06": 60,
        "B07": 60,
        "B8A": 60,
        "B11": 60,
        "B12": 60,
    }
    optical = save({b: np.full((n, n), 321, dtype="uint16") for b, n in shapes.items()})
    sar = save({b: np.full((120, 120), -12.5, dtype="float32") for b in ["VV", "VH"]})
    env = lmdb.open(str(source / "db"), map_size=4 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for key in ids:
            txn.put(key.encode(), optical)
        for key in s1ids:
            txn.put(key.encode(), sar)
    env.close()
    database = (source / "db/data.mdb").read_bytes()
    monkeypatch.setattr(module, "LMDB_SIZE", len(database))

    class LocalRanges:
        def __init__(self, url, size):
            pass

        def read(self, start, length):
            return database[start : start + length]

    monkeypatch.setattr(module, "HTTPRanges", LocalRanges)

    def cached(url, size, digest, path, progress):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            source / ("metadata.parquet" if url.endswith("metadata.parquet") else "maps.tar.gz"),
            path,
        )
        return path

    monkeypatch.setattr(module, "download_file", cached)
    drive, work = tmp_path / "drive", tmp_path / "work"
    result = module.download_subset(
        drive, work, counts={"train": 2}, shard_size=1, progress=lambda _: None
    )
    assert result["complete"] and result["sample_count"] == 2 and len(result["shards"]) == 2
    for shard in result["shards"]:
        with zipfile.ZipFile(drive / shard["archive"]) as archive:
            assert len([n for n in archive.namelist() if n.endswith(".tif")]) == 15
            assert "metadata.parquet" in archive.namelist()
    original_hashes = [s["sha256"] for s in result["shards"]]
    (drive / "metadata.parquet").write_bytes(b"interrupted-write")
    monkeypatch.setattr(module, "LMDBReader", lambda *a: pytest.fail("Re-fetched completed shards"))
    resumed = module.download_subset(
        drive, work, counts={"train": 2}, shard_size=1, progress=lambda _: None
    )
    assert [s["sha256"] for s in resumed["shards"]] == original_hashes
    assert set(pd.read_parquet(drive / "metadata.parquet").patch_id) == set(ids)
    assert json.loads((drive / "download.json").read_text())["complete"]
