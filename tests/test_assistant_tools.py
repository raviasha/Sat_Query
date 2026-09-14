import json
from datetime import date

import numpy as np
import pytest
import torch
from rasterio.warp import transform
from torch import nn

from satquery.prediction import CoverageHead, save_head
from satquery.prediction_data import class_schema
from satquery.preprocessing import (
    CROMA_REVISION,
    DEFAULT_PROFILE,
    NORMALIZATION_PROFILE,
    sha256,
)

GRID = {
    "crs": "EPSG:32633",
    "transform": [10.0, 0.0, 100.0, 0.0, -10.0, 1300.0],
    "bounds": [100.0, 100.0, 1300.0, 1300.0],
    "width": 120,
    "height": 120,
    "pixel_size_metres": 10.0,
}


def _scene(identifier="scene", acquired=date(2026, 1, 2), modality="optical", coverage=None):
    from satquery.assistant.runtime import SceneResult

    if coverage is None:
        coverage = np.zeros((15, 15, 19), dtype=np.float32)
        coverage[..., 2] = 1
    return SceneResult(
        id=identifier,
        modality=modality,
        acquired=acquired,
        coverage=coverage,
        preview=np.zeros((120, 120, 3), dtype=np.uint8),
        grid=dict(GRID),
        provenance={
            "feature_contract": {"checkpoint_sha256": "same"},
            "feature_key": f"{modality}_encodings",
            "head_sha256": "same-head",
        },
    )


class _FakeCroma(nn.Module):
    encoder_dim = 768

    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        self.calls = []

    def forward(self, optical_images=None, SAR_images=None):
        assert not torch.is_grad_enabled()
        source = optical_images if optical_images is not None else SAR_images
        self.calls.append((optical_images, SAR_images))
        values = torch.zeros(len(source), 225, 768, device=source.device)
        if self.modality == "optical":
            return {"optical_encodings": values}
        if self.modality == "SAR":
            return {"SAR_encodings": values}
        return {
            "optical_encodings": values,
            "SAR_encodings": values,
            "joint_encodings": values,
        }


def _runtime_with_head(tmp_path, capability, monkeypatch):
    import satquery.assistant.runtime as runtime_module
    from satquery.assistant.runtime import CoverageRuntime

    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "croma.pt"
    checkpoint.write_bytes(b"fixture-croma")
    monkeypatch.setattr(runtime_module, "CHECKPOINT_SHA256", sha256(checkpoint))
    head_path = tmp_path / f"{capability}-head.pt"
    runtime = CoverageRuntime(checkpoint, {capability: head_path})
    model = CoverageHead()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    feature_key = {
        "optical": "optical_encodings",
        "SAR": "SAR_encodings",
        "joint": "joint_encodings",
    }[capability]
    save_head(
        head_path,
        model,
        {
            "feature_key": feature_key,
            "feature_contract": runtime.feature_contract,
            "classes": class_schema(),
        },
    )
    fake = _FakeCroma("both" if capability == "joint" else capability)
    runtime._create_model = lambda modality: fake
    return runtime, fake


def test_runtime_is_lazy_uses_matching_modality_and_returns_coverage(tmp_path, monkeypatch):
    from satquery.assistant.inputs import InputBundle, Observation

    runtime, fake = _runtime_with_head(tmp_path, "optical", monkeypatch)
    assert runtime.available_capabilities == ["optical"]
    assert runtime._models == {}
    observation = Observation(
        id="o1",
        modality="optical",
        sensor="Sentinel-2",
        acquired=date(2026, 1, 2),
        normalized=torch.zeros(12, 120, 120),
        preview=np.zeros((120, 120, 3), dtype=np.uint8),
        grid=dict(GRID),
        normalization={"profile": NORMALIZATION_PROFILE},
        metadata={"bands": list(DEFAULT_PROFILE.optical)},
    )
    results = runtime.analyze(InputBundle("single", (observation,), dict(GRID)))
    assert len(results) == 1
    assert results[0].coverage.shape == (15, 15, 19)
    np.testing.assert_allclose(results[0].coverage.sum(-1), 1, atol=1e-6)
    assert results[0].provenance["checkpoint_sha256"] == sha256(tmp_path / "croma.pt")
    assert len(fake.calls) == 1 and fake.calls[0][0].shape == (1, 12, 120, 120)
    assert fake.calls[0][1] is None


def test_runtime_requires_matching_head_and_temporal_runs_one_model_batch(tmp_path, monkeypatch):
    import satquery.assistant.runtime as runtime_module
    from satquery.assistant.inputs import InputBundle, Observation
    from satquery.assistant.runtime import CapabilityError, CoverageRuntime

    checkpoint = tmp_path / "croma.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(runtime_module, "CHECKPOINT_SHA256", sha256(checkpoint))
    empty = CoverageRuntime(checkpoint, {})
    obs = Observation(
        "s1",
        "SAR",
        "Sentinel-1",
        date(2026, 1, 1),
        torch.zeros(2, 120, 120),
        np.zeros((120, 120, 3), np.uint8),
        dict(GRID),
        {"profile": NORMALIZATION_PROFILE},
        {"bands": ["VV", "VH"], "radiometry": "db"},
    )
    with pytest.raises(CapabilityError, match="SAR"):
        empty.analyze(InputBundle("single", (obs,), dict(GRID)))

    runtime, fake = _runtime_with_head(tmp_path / "temporal", "SAR", monkeypatch)
    after = Observation(**{**obs.__dict__, "id": "s2", "acquired": date(2026, 2, 1)})
    results = runtime.analyze(InputBundle("temporal", (obs, after), dict(GRID)))
    assert [item.id for item in results] == ["s1", "s2"]
    assert len(fake.calls) == 1 and fake.calls[0][1].shape == (2, 2, 120, 120)


def test_runtime_rejects_unpinned_checkpoint_before_model_loading(tmp_path):
    from satquery.assistant.runtime import CoverageRuntime

    checkpoint = tmp_path / "untrusted.pt"
    checkpoint.write_bytes(b"not-the-official-checkpoint")
    runtime = CoverageRuntime(checkpoint, {})
    with pytest.raises(ValueError, match="pinned official"):
        _ = runtime.feature_contract
    assert runtime._models == {}


def test_cross_modal_feeds_both_inputs_and_records_three_head_comparison(tmp_path, monkeypatch):
    from satquery.assistant.inputs import InputBundle, Observation

    runtime, fake = _runtime_with_head(tmp_path, "joint", monkeypatch)
    for capability in ("optical", "SAR"):
        path = tmp_path / f"{capability}-head.pt"
        save_head(
            path,
            CoverageHead(),
            {
                "feature_key": f"{capability}_encodings",
                "feature_contract": runtime.feature_contract,
                "classes": class_schema(),
            },
        )
        runtime.heads[capability] = path
    optical = Observation(
        "o1",
        "optical",
        "Sentinel-2",
        date(2026, 1, 1),
        torch.zeros(12, 120, 120),
        np.zeros((120, 120, 3), np.uint8),
        dict(GRID),
        {"profile": NORMALIZATION_PROFILE},
        {"bands": list(DEFAULT_PROFILE.optical)},
    )
    sar = Observation(
        "s1",
        "SAR",
        "Sentinel-1",
        date(2026, 1, 1),
        torch.zeros(2, 120, 120),
        np.zeros((120, 120, 3), np.uint8),
        dict(GRID),
        {"profile": NORMALIZATION_PROFILE},
        {"bands": list(DEFAULT_PROFILE.sar), "radiometry": "db"},
    )
    results = runtime.analyze(InputBundle("cross_modal", (optical, sar), dict(GRID)))
    assert len(results) == 1 and results[0].modality == "joint"
    optical_arg, sar_arg = fake.calls[0]
    assert optical_arg.shape == (1, 12, 120, 120)
    assert sar_arg.shape == (1, 2, 120, 120)
    assert results[0].provenance["joint_encoder_inputs"] == {"optical": "o1", "SAR": "s1"}
    assert set(results[0].provenance["sensor_comparison"]["top_scene_coverage"]) == {
        "optical",
        "SAR",
        "joint",
    }
    assert (
        "no fusion improvement is claimed"
        in results[0].provenance["sensor_comparison"]["interpretation"]
    )


def test_cached_feature_path_selects_index_without_loading_croma(tmp_path, monkeypatch):
    import satquery.assistant.runtime as runtime_module
    from satquery.assistant.runtime import CoverageRuntime

    root = tmp_path / "features"
    root.mkdir()
    checkpoint = tmp_path / "croma.pt"
    checkpoint.write_bytes(b"fixture-croma")
    monkeypatch.setattr(runtime_module, "CHECKPOINT_SHA256", sha256(checkpoint))
    runtime = CoverageRuntime(checkpoint, {"joint": tmp_path / "joint-head.pt"})
    contract = runtime.feature_contract
    tensors = {"joint_encodings": torch.stack([torch.zeros(225, 768), torch.ones(225, 768)])}
    torch.save(tensors, root / "features.pt")
    samples = [
        {
            "patch_id": f"p{i}",
            "s1_name": f"s{i}",
            "acquired": f"2026-01-0{i + 1}",
            "crs": GRID["crs"],
            "transform": GRID["transform"],
            "bounds": GRID["bounds"],
        }
        for i in range(2)
    ]
    info = {
        "sample_count": 2,
        "samples": samples,
        "spatial_grid": {
            "height": 15,
            "width": 15,
            "token_order": "row-major",
            "input_patch_pixels": [8, 8],
        },
    }
    (root / "batch.json").write_text(json.dumps(info))
    batch = {
        "directory": ".",
        "sample_count": 2,
        "start_index": 0,
        "sha256": {name: sha256(root / name) for name in ("batch.json", "features.pt")},
    }
    manifest = {
        "format_version": 1,
        "sample_count": 2,
        "feature_dimension": 768,
        "model": {
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "croma_source_revision": CROMA_REVISION,
            "inference_implementation_sha256": contract["inference_implementation_sha256"],
        },
        "channel_profile": contract["channel_profile"],
        "normalization_profile": NORMALIZATION_PROFILE,
        "batches": [batch],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    model = CoverageHead()
    save_head(
        tmp_path / "joint-head.pt",
        model,
        {"feature_key": "joint_encodings", "feature_contract": contract, "classes": class_schema()},
    )

    selected = runtime.analyze_cached(root, selected_index=1, capability="joint")
    assert selected.id == "p1"
    assert selected.coverage.shape == (15, 15, 19)
    assert selected.preview is None
    assert runtime._models == {}
    assert selected.provenance["source"] == "cached_features"


def test_coverage_alias_pooling_and_area_are_spatial_and_json_safe():
    from satquery.assistant.tools import execute_task

    values = np.zeros((15, 15, 19), dtype=np.float32)
    values[..., 2] = 1
    values[0, 0, 2] = 0.4
    values[0, 0, 8:11] = 0.2
    result = execute_task("coverage", [_scene(coverage=values)], class_name="forest")
    measurement = result["measurements"][0]
    assert measurement["class_indices"] == [8, 9, 10]
    assert measurement["estimated_area_m2"] == pytest.approx(0.6 * 6400)
    assert result["evidence"][0]["fraction_grid"][0][0] == pytest.approx(0.6)
    assert result["trace"]["selected_tool"] == "coverage"
    assert "confidence" not in json.dumps(result).lower()
    json.dumps(result, allow_nan=False)


def test_presence_does_not_claim_absence_and_locate_uses_affine_cell_bounds():
    from satquery.assistant.tools import execute_task

    values = np.zeros((15, 15, 19), dtype=np.float32)
    values[..., 2] = 1
    values[0, 0, 2] = 0.4
    values[0, 0, 17] = 0.3
    values[0, 0, 18] = 0.3
    scene = _scene(coverage=values)
    present = execute_task("presence", [scene], class_name="water", threshold=0.7)
    assert present["measurements"][0]["detected"] is False
    assert "does not establish absence" in present["answer"]
    assert "heuristic" in " ".join(present["limitations"]).lower()

    located = execute_task("locate", [scene], class_name="water", threshold=0.5)
    geojson = located["evidence"][0]["geojson"]
    assert "crs" not in geojson
    feature = geojson["features"][0]
    assert feature["properties"]["projected_crs"] == "EPSG:32633"
    assert feature["properties"]["projected_bounds_m"] == [100.0, 1220.0, 180.0, 1300.0]
    ring = feature["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert all(-180 <= longitude <= 180 and -90 <= latitude <= 90 for longitude, latitude in ring)
    x, y = transform("EPSG:4326", "EPSG:32633", [p[0] for p in ring], [p[1] for p in ring])
    np.testing.assert_allclose(
        np.column_stack([x, y]),
        [[100, 1220], [180, 1220], [180, 1300], [100, 1300], [100, 1220]],
        rtol=0,
        atol=1e-4,
    )
    assert feature["properties"]["estimate"] == pytest.approx(0.6)


def test_describe_change_and_abstention_contracts():
    from satquery.assistant.tools import execute_task

    described = execute_task("describe", [_scene()])
    assert described["measurements"][0]["top_land_cover"][0]["name"] == "Arable land"

    unsupported = execute_task("coverage", [_scene()], class_name="school")
    assert unsupported["abstained"] is True
    assert unsupported["measurements"] == []

    before_values = np.zeros((15, 15, 19), dtype=np.float32)
    before_values[..., 2] = 1
    after_values = before_values.copy()
    after_values[0, 0, 2] = 0.25
    after_values[0, 0, 17] = 0.75
    before = _scene("before", date(2026, 1, 1), coverage=before_values)
    after = _scene("after", date(2026, 2, 1), coverage=after_values)
    change = execute_task("change", [before, after], class_name="water")
    assert change["measurements"]["before_area_m2"] == 0
    assert change["measurements"]["after_area_m2"] == pytest.approx(0.75 * 6400)
    assert change["evidence"]["class_delta_grid"][0][0] == pytest.approx(0.75)
    assert "provisional" in " ".join(change["limitations"]).lower()
    assert "unvalidated" in " ".join(change["limitations"]).lower()

    mismatch = _scene("after", date(2026, 2, 1), modality="SAR", coverage=after_values)
    with pytest.raises(ValueError, match="same modality"):
        execute_task("change", [before, mismatch], class_name="water")

    different_head = _scene("after", date(2026, 2, 1), coverage=after_values)
    different_head.provenance["head_sha256"] = "different-head"
    with pytest.raises(ValueError, match="same trained coverage head"):
        execute_task("change", [before, different_head], class_name="water")


def test_tools_reject_geographic_crs_for_fixed_square_metre_areas():
    from satquery.assistant.tools import execute_task

    scene = _scene()
    scene.grid["crs"] = "EPSG:4326"
    with pytest.raises(ValueError, match="projected metre"):
        execute_task("coverage", [scene], class_name="water")


def test_change_requires_datetime_date_instances():
    from satquery.assistant.tools import execute_task

    before = _scene("before", "2026-01-01")
    after = _scene("after", "2026-02-01")
    with pytest.raises(TypeError, match="datetime.date"):
        execute_task("change", [before, after], class_name="water")


def test_spatial_outputs_qualify_attention_context_for_80_m_anchors():
    from satquery.assistant.tools import execute_task

    one = _scene()
    before = _scene("before", date(2026, 1, 1))
    after = _scene("after", date(2026, 2, 1))
    results = [
        execute_task("coverage", [one], class_name="forest"),
        execute_task("presence", [one], class_name="forest"),
        execute_task("describe", [one]),
        execute_task("locate", [one], class_name="forest"),
        execute_task("change", [before, after], class_name="forest"),
    ]
    for result in results:
        limitations = " ".join(result["limitations"]).lower()
        assert "80 m" in limitations
        assert "spatial anchor" in limitations
        assert "neighboring" in limitations
        assert "whole-scene context" in limitations
        assert "attention" in limitations


@pytest.mark.parametrize("task", ["segment", "detect", "chat"])
def test_unknown_task_abstains_instead_of_guessing(task):
    from satquery.assistant.tools import execute_task

    result = execute_task(task, [_scene()], class_name="water")
    assert result["abstained"] is True
    assert result["trace"]["selected_tool"] is None
