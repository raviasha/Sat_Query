"""Catch class remapping, spatial permutation, denominator and pairing errors."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from satquery.preprocessing import sha256


def test_patch_percentages_and_row_major_order():
    from satquery.targets import patch_label_targets

    maps = torch.full((2, 120, 120), 999, dtype=torch.int64)
    maps[0, :4, :8] = 311  # 32 broad-leaved forest pixels
    maps[0, 4:6, :8] = 312  # 16 coniferous, 16 unlabeled
    maps[0, :8, 8:16] = 211  # token 1: arable land
    maps[0, 8:16, :8] = 231  # token 15: pastures
    maps[1, 112:, 112:] = 523  # token 224, second sample: marine waters
    result = patch_label_targets(maps)
    assert result["class_fractions"].shape == (2, 225, 19)
    assert result["class_counts"][0, 0, 8].item() == 32
    assert result["class_counts"][0, 0, 9].item() == 16
    assert result["class_fractions"][0, 0, 8].item() == 0.5
    assert result["class_fractions"][0, 0, 9].item() == 0.25
    assert result["unlabeled_fraction"][0, 0].item() == 0.25
    assert not result["fully_labeled_mask"][0, 0]
    assert result["has_labels_mask"][0, 0]
    assert result["class_fractions"][0, 1, 2].item() == 1
    assert result["class_fractions"][0, 15, 4].item() == 1
    assert result["class_fractions"][1, 224, 18].item() == 1
    assert result["class_counts"][1, 0].sum() == 0
    assert not result["has_labels_mask"][1, 0]
    assert result["unlabeled_counts"][1, 0] == 64
    assert torch.equal(
        result["class_fractions"].sum(-1) + result["unlabeled_fraction"], torch.ones(2, 225)
    )


@pytest.mark.parametrize(
    "codes,index",
    [
        ([111, 112], 0),
        ([121], 1),
        ([211, 212, 213], 2),
        ([221, 222, 223, 241], 3),
        ([231], 4),
        ([242], 5),
        ([243], 6),
        ([244], 7),
        ([311], 8),
        ([312], 9),
        ([313], 10),
        ([321, 333], 11),
        ([322, 323], 12),
        ([324], 13),
        ([331], 14),
        ([411, 412], 15),
        ([421, 422], 16),
        ([511, 512], 17),
        ([521, 522, 523], 18),
    ],
)
def test_official_19_class_mapping(codes, index):
    from satquery.targets import patch_label_targets

    maps = torch.stack([torch.full((120, 120), c) for c in codes])
    fractions = patch_label_targets(maps)["class_fractions"]
    assert (fractions[:, :, index] == 1).all()
    assert (fractions.sum(-1) == 1).all()


def test_unlabeled_and_excluded_clc_codes_are_not_training_classes():
    from satquery.targets import patch_label_targets

    codes = [0, 999, 122, 123, 124, 131, 132, 133, 141, 142, 332, 334, 335, 423]
    result = patch_label_targets(torch.stack([torch.full((120, 120), c) for c in codes]))
    assert result["class_counts"].sum() == 0
    assert (result["unlabeled_counts"] == 64).all()
    assert not result["has_labels_mask"].any()
    assert not result["fully_labeled_mask"].any()


@pytest.mark.parametrize(
    "maps",
    [
        torch.zeros(120, 120, dtype=torch.int64),
        torch.zeros(0, 120, 120, dtype=torch.int64),
        torch.zeros(1, 119, 120, dtype=torch.int64),
        torch.zeros(1, 120, 120),
        torch.full((1, 120, 120), 255),
        torch.full((1, 120, 120), -1),
        torch.zeros(1, 120, 120, dtype=torch.bool),
    ],
)
def test_invalid_maps_rejected(maps):
    from satquery.targets import patch_label_targets

    with pytest.raises(ValueError):
        patch_label_targets(maps)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def exports(tmp_path):
    """Real serialized inputs, distinct labels per sample and two source batches."""
    prepared, features = tmp_path / "prepared", tmp_path / "features"
    prepared.mkdir()
    features.mkdir()
    pbatches, fbatches = [], []
    for number, code in enumerate([311, 231]):
        name = f"batch-{number:06d}"
        p, f = prepared / name, features / name
        p.mkdir()
        f.mkdir()
        samples = [
            {
                "patch_id": f"patch-{number}",
                "s1_name": f"sar-{number}",
                "split": "test",
                "crs": "EPSG:32633",
                "transform": [10, 0, 373200, 0, -10, 5353200],
                "bounds": [373200, 5352000, 374400, 5353200],
            }
        ]
        write_json(p / "batch.json", {"sample_count": 1, "samples": samples})
        torch.save(torch.full((1, 120, 120), code), p / "reference_maps.pt")
        pb = {
            "directory": name,
            "sample_count": 1,
            "start_index": number,
            "sha256": {n: sha256(p / n) for n in ["batch.json", "reference_maps.pt"]},
        }
        pbatches.append(pb)
        write_json(
            f / "batch.json",
            {
                "sample_count": 1,
                "samples": samples,
                "source_batch": pb,
                "spatial_grid": {
                    "height": 15,
                    "width": 15,
                    "token_order": "row-major",
                    "input_patch_pixels": [8, 8],
                },
            },
        )
        torch.save(
            {
                f"{modality}_{kind}": torch.full(
                    (1, 225, 768) if kind == "encodings" else (1, 768), float(number)
                )
                for modality in ["optical", "SAR", "joint"]
                for kind in ["encodings", "GAP"]
            },
            f / "features.pt",
        )
        fbatches.append(
            {
                "directory": name,
                "sample_count": 1,
                "start_index": number,
                "sha256": {n: sha256(f / n) for n in ["batch.json", "features.pt"]},
            }
        )
    write_json(
        prepared / "manifest.json", {"format_version": 1, "sample_count": 2, "batches": pbatches}
    )
    write_json(
        features / "manifest.json",
        {
            "format_version": 1,
            "sample_count": 2,
            "feature_dimension": 768,
            "source_manifest_sha256": sha256(prepared / "manifest.json"),
            "batches": fbatches,
        },
    )
    return prepared, features


def test_export_alignment_bounds_and_cli(exports, tmp_path):
    prepared, features = exports
    output = tmp_path / "targets"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "satquery.prepare_targets",
            "--input",
            str(prepared),
            "--features",
            str(features),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sample_count"] == 2 and manifest["token_count"] == 450
    assert len(manifest["classes"]) == 19
    a = torch.load(output / "batch-000000/targets.pt", weights_only=True)
    b = torch.load(output / "batch-000001/targets.pt", weights_only=True)
    assert (a["class_fractions"][:, :, 8] == 1).all()
    assert (b["class_fractions"][:, :, 4] == 1).all()
    assert a["bounds"][0, 0].tolist() == [373200, 5353120, 373280, 5353200]
    assert a["bounds"][0, 1].tolist() == [373280, 5353120, 373360, 5353200]
    assert a["bounds"][0, 15].tolist() == [373200, 5353040, 373280, 5353120]
    assert a["bounds"][0, 224].tolist() == [374320, 5352000, 374400, 5352080]
    info = json.loads((output / "batch-000001/batch.json").read_text())
    assert info["samples"][0]["patch_id"] == "patch-1"
    assert info["samples"][0]["split"] == "test"
    assert Path(info["feature_file"]).resolve() == (features / "batch-000001/features.pt").resolve()
    assert json.loads((output / "validation.json").read_text())["passed"]


@pytest.mark.parametrize(
    "problem", ["hash", "source", "sample", "grid", "resolution", "unknown", "feature_shape"]
)
def test_export_rejects_misaligned_or_invalid_sources(exports, tmp_path, problem):
    from satquery.targets import export_targets

    p, f = exports
    pm, fm = (
        json.loads((p / "manifest.json").read_text()),
        json.loads((f / "manifest.json").read_text()),
    )
    pdir, fdir = p / "batch-000000", f / "batch-000000"
    if problem == "hash":
        with (pdir / "reference_maps.pt").open("ab") as stream:
            stream.write(b"changed")
    if problem == "source":
        fm["source_manifest_sha256"] = "wrong"
    if problem in ["sample", "grid", "resolution"]:
        info = json.loads((fdir / "batch.json").read_text())
        if problem == "sample":
            info["samples"][0]["patch_id"] = "wrong"
        if problem == "grid":
            info["spatial_grid"]["token_order"] = "column-major"
        if problem == "resolution":
            info["samples"][0]["transform"][0] = 20
            pi = json.loads((pdir / "batch.json").read_text())
            pi["samples"] = info["samples"]
            write_json(pdir / "batch.json", pi)
            pm["batches"][0]["sha256"]["batch.json"] = sha256(pdir / "batch.json")
        write_json(fdir / "batch.json", info)
        fm["batches"][0]["sha256"]["batch.json"] = sha256(fdir / "batch.json")
    if problem == "unknown":
        torch.save(torch.full((1, 120, 120), 255), pdir / "reference_maps.pt")
        pm["batches"][0]["sha256"]["reference_maps.pt"] = sha256(pdir / "reference_maps.pt")
    if problem == "feature_shape":
        torch.save({"joint_encodings": torch.zeros(1, 224, 768)}, fdir / "features.pt")
        fm["batches"][0]["sha256"]["features.pt"] = sha256(fdir / "features.pt")
    write_json(p / "manifest.json", pm)
    if problem != "source":
        fm["source_manifest_sha256"] = sha256(p / "manifest.json")
    write_json(f / "manifest.json", fm)
    with pytest.raises(ValueError):
        export_targets(p, f, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_output_protection_and_manifest_change(exports, tmp_path):
    from satquery.targets import export_targets

    p, f = exports
    with pytest.raises(FileExistsError):
        export_targets(p, f, p)
    with pytest.raises(ValueError):
        export_targets(p, f, p / "targets")
    with pytest.raises(ValueError):
        export_targets(p, f, f / "targets")

    def mutate(_):
        with (p / "manifest.json").open("a") as stream:
            stream.write(" ")

    with pytest.raises(ValueError, match="changed"):
        export_targets(p, f, tmp_path / "changed", progress=mutate)
    assert not (tmp_path / "changed").exists()
