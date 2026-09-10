import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from satquery.prepare_croma import export_inputs

REAL = Path(__file__).resolve().parents[1] / "data/bigearthnet-v2/small-sample"


@pytest.fixture
def real_data():
    if not (REAL / "metadata.parquet").exists():
        pytest.skip("Local downloaded sample not installed")
    return REAL


@pytest.mark.parametrize("batch_size", [2, 64])
def test_export_three_real_samples_in_bounded_batches(real_data, tmp_path, batch_size):
    output = tmp_path / "prepared"
    manifest = export_inputs(real_data, output, batch_size=batch_size)
    assert manifest["sample_count"] == 3
    ids, raw_parts, parts = [], [], []
    for batch in manifest["batches"]:
        directory = output / batch["directory"]
        info = json.loads((directory / "batch.json").read_text())
        ids.extend(x["patch_id"] for x in info["samples"])
        tensors = torch.load(directory / "croma_inputs.pt", weights_only=True)
        raw = torch.load(directory / "raw_inputs.pt", weights_only=True)
        maps = torch.load(directory / "reference_maps.pt", weights_only=True)
        assert maps.dtype == torch.int64
        assert tensors["optical_images"].shape[1:] == (12, 120, 120)
        assert tensors["SAR_images"].shape[1:] == (2, 120, 120)
        for t in tensors.values():
            assert t.dtype == torch.float32 and torch.isfinite(t).all()
            assert 0 <= t.min() <= t.max() <= 1
            assert len(t) <= batch_size
        parts.append(tensors["optical_images"])
        raw_parts.append(raw["SAR_images"])
    assert ids == pd.read_parquet(real_data / "metadata.parquet").patch_id.tolist()
    assert torch.cat(parts).shape == (3, 12, 120, 120)
    # Regression: reading SAR as integers would discard these fractional values.
    raw_sar = torch.cat(raw_parts)
    assert (raw_sar != raw_sar.round()).any()
    assert json.loads((output / "validation.json").read_text())["passed"]


def test_export_batch_composition_does_not_change_tensors(real_data, tmp_path):
    export_inputs(real_data, tmp_path / "all", batch_size=64)
    export_inputs(real_data, tmp_path / "one", limit=1)
    all_samples = torch.load(tmp_path / "all/croma_inputs.pt", weights_only=True)
    first = torch.load(tmp_path / "one/croma_inputs.pt", weights_only=True)
    for key in first:
        torch.testing.assert_close(first[key][0], all_samples[key][0], rtol=0, atol=0)


def test_output_safety_and_failure_cleanup(real_data, tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        export_inputs(real_data, output)
    assert (output / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="outside"):
        export_inputs(real_data, real_data / "generated")
    bad_metadata = tmp_path / "bad.parquet"
    df = pd.read_parquet(real_data / "metadata.parquet").iloc[:1].copy()
    df["s1_name"] = "missing_SAR_patch"
    df.to_parquet(bad_metadata)
    with pytest.raises(ValueError):
        export_inputs(real_data, tmp_path / "failed", metadata=bad_metadata)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed-*"))


def test_cli_runs_with_selected_metadata(real_data, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "satquery.prepare_croma",
            "--input",
            str(real_data),
            "--output",
            str(tmp_path / "cli"),
            "--limit",
            "1",
            "--split",
            "test",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads((tmp_path / "cli/manifest.json").read_text())["sample_count"] == 1
