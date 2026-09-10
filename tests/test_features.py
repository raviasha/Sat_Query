import json

import pytest
import torch
from torch import nn

from satquery.features import CromaFeatureExtractor, extract_features
from satquery.preprocessing import DEFAULT_PROFILE, NORMALIZATION_PROFILE, sha256


class TinyEncoder(nn.Module):
    """Actual small neural encoder used to test wrapper behavior, not pretrained quality."""

    encoder_dim = 4

    def __init__(self):
        super().__init__()
        self.optical = nn.Conv2d(12, 4, kernel_size=8, stride=8)
        self.sar = nn.Conv2d(2, 4, kernel_size=8, stride=8)
        self.dropout = nn.Dropout(0.8)

    def forward(self, SAR_images, optical_images):
        o = self.dropout(self.optical(optical_images).flatten(2).transpose(1, 2))
        s = self.dropout(self.sar(SAR_images).flatten(2).transpose(1, 2))
        j = o + s
        return {
            "optical_encodings": o,
            "SAR_encodings": s,
            "joint_encodings": j,
            "optical_GAP": o.mean(1),
            "SAR_GAP": s.mean(1),
            "joint_GAP": j.mean(1),
        }


@pytest.fixture
def extractor():
    torch.manual_seed(10)
    return CromaFeatureExtractor(TinyEncoder(), device="cpu")


def test_tensor_extraction_eval_inference_and_order(extractor):
    optical = torch.stack([torch.full((12, 120, 120), n / 4) for n in [1, 2, 3]])
    sar = optical[:, :2].clone()
    outputs = extractor(optical, sar, batch_size=2)
    expected = extractor(optical, sar, batch_size=1)
    assert not extractor.model.training
    for key, tensor in outputs.items():
        assert tensor.shape == ((3, 225, 4) if "encodings" in key else (3, 4))
        assert not tensor.requires_grad
        assert tensor.device.type == "cpu" and torch.isfinite(tensor).all()
        torch.testing.assert_close(tensor, expected[key])
    assert not torch.equal(outputs["joint_encodings"][0], outputs["joint_encodings"][2])


@pytest.mark.parametrize("problem", ["channels", "dtype", "range", "nan", "pairing", "empty"])
def test_reject_invalid_tensors(extractor, problem):
    o, s = torch.zeros(2, 12, 120, 120), torch.zeros(2, 2, 120, 120)
    if problem == "channels":
        o = o[:, :11]
    if problem == "dtype":
        o = o.double()
    if problem == "range":
        o[0, 0, 0, 0] = 256
    if problem == "nan":
        s[0, 0, 0, 0] = float("nan")
    if problem == "pairing":
        s = s[:1]
    if problem == "empty":
        o, s = o[:0], s[:0]
    with pytest.raises(ValueError):
        extractor(o, s)


@pytest.fixture
def prepared(tmp_path):
    from dataclasses import asdict

    root = tmp_path / "prepared"
    root.mkdir()
    data = {"optical_images": torch.rand(3, 12, 120, 120), "SAR_images": torch.rand(3, 2, 120, 120)}
    torch.save(data, root / "croma_inputs.pt")
    info = {
        "sample_count": 3,
        "samples": [
            {
                "patch_id": f"patch_{n}",
                "s1_name": f"sar_{n}",
                "crs": "EPSG:32633",
                "transform": [10, 0, 373200, 0, -10, 5353200],
            }
            for n in range(3)
        ],
    }
    (root / "batch.json").write_text(json.dumps(info))
    manifest = {
        "format_version": 1,
        "sample_count": 3,
        "channel_profile": asdict(DEFAULT_PROFILE),
        "normalization_profile": NORMALIZATION_PROFILE,
        "batches": [
            {
                "directory": ".",
                "sample_count": 3,
                "start_index": 0,
                "sha256": {n: sha256(root / n) for n in ["batch.json", "croma_inputs.pt"]},
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_feature_export_and_input_integrity(prepared, tmp_path, extractor):
    output = tmp_path / "features"
    result = extract_features(prepared, output, extractor, batch_size=2)
    assert result["sample_count"] == 3
    saved = torch.load(output / "features.pt", weights_only=True)
    assert saved["joint_encodings"].shape == (3, 225, 4)
    info = json.loads((output / "batch.json").read_text())
    assert [x["patch_id"] for x in info["samples"]] == ["patch_0", "patch_1", "patch_2"]
    assert info["spatial_grid"]["token_order"] == "row-major"
    assert json.loads((output / "validation.json").read_text())["passed"]
    assert not json.loads((output / "validation.json").read_text())["checkpoint_inference_run"]
    with pytest.raises(FileExistsError):
        extract_features(prepared, output, extractor)
    # Input mutation must fail before a result is published.
    with (prepared / "croma_inputs.pt").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        extract_features(prepared, tmp_path / "bad", extractor)
    assert not (tmp_path / "bad").exists()


def test_wrong_profile_and_nested_output_rejected(prepared, tmp_path, extractor):
    with pytest.raises(ValueError, match="outside"):
        extract_features(prepared, prepared / "features", extractor)
    p = prepared / "manifest.json"
    m = json.loads(p.read_text())
    m["channel_profile"]["sar"] = ["VH", "VV"]
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="channel"):
        extract_features(prepared, tmp_path / "wrong", extractor)


def test_source_manifest_change_during_inference_rejected(prepared, tmp_path, extractor):
    path = prepared / "manifest.json"

    def change_manifest(_):
        m = json.loads(path.read_text())
        m["channel_profile"]["sar"] = ["VH", "VV"]
        path.write_text(json.dumps(m))

    with pytest.raises(ValueError, match="manifest changed"):
        extract_features(prepared, tmp_path / "changed", extractor, progress=change_manifest)
    assert not (tmp_path / "changed").exists()
