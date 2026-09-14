import io
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin


class _Bundle:
    def summary(self):
        return {
            "kind": "single",
            "observation_count": 1,
            "observations": [
                {
                    "id": "observation-1",
                    "modality": "optical",
                    "sensor": "Sentinel-2",
                    "acquired": "2026-01-02",
                    "bands": ["B01", "B02"],
                }
            ],
            "grid": {"width": 120, "height": 120},
        }


class _Runtime:
    def __init__(self):
        self.available_capabilities = ["optical", "joint"]
        self.bundles = []
        self.demo_calls = []

    def analyze(self, bundle):
        self.bundles.append(bundle)
        return [object()]

    def analyze_cached(self, features, *, selected_index, capability):
        self.demo_calls.append((Path(features), selected_index, capability))
        return object()


def _provenance_scene(*, source="live_croma_inference"):
    return SimpleNamespace(
        id="observation-1",
        modality="joint",
        preview=None,
        provenance={
            "source": source,
            "model": "CROMA-Base",
            "model_mode": "both",
            "feature_key": "joint_encodings",
            "feature_contract": {
                "checkpoint_sha256": "a" * 64,
                "croma_source_revision": "croma-base-2024-02",
                "inference_implementation_sha256": "b" * 64,
                "normalization_profile": "per-image-mean-plus-minus-2std-uint8-v1",
                "channel_profile": {
                    "optical": ["B01", "B02"],
                    "sar": ["VV", "VH"],
                    "evidence": "source-contract",
                    "evidence_level": "verified",
                },
                "private_path": "/Users/private-owner/models/head.pt",
            },
            "checkpoint_sha256": "a" * 64,
            "head_sha256": "c" * 64,
            "head_training_mode": "demo_fit_all",
            "head_evaluation_scope": "training_data_only",
            "joint_encoder_inputs": {"optical": "optical-1", "SAR": "sar-1"},
            "fusion": "Both optical and SAR tensors fed the CROMA joint encoder.",
            "sensor_comparison": {
                "top_scene_coverage": {
                    "optical": [{"name": "Inland waters", "estimated_fraction": 0.4}],
                    "SAR": [{"name": "Inland waters", "estimated_fraction": 0.3}],
                    "joint": [{"name": "Inland waters", "estimated_fraction": 0.5}],
                },
                "interpretation": "Descriptive sensor estimates only.",
            },
            "secret": "sk-do-not-persist",
            "raw_metadata": {"owner": "private-owner"},
        },
    )


class _Controller:
    openai_configured = False
    model = "gpt-4.1-mini-2025-04-14"

    def __init__(self):
        self.calls = []

    def answer(self, question, scenes, summary, *, provider):
        self.calls.append((question, scenes, summary, provider))
        return {
            "provider": provider,
            "language_model": None,
            "question": question,
            "abstained": False,
            "answer": "Estimated forest coverage — observation-1: 432000.0 m² (30.00%).",
            "deterministic_answer": "Estimated forest coverage — observation-1: 432000.0 m² (30.00%).",
            "llm_wording": None,
            "measurements": [
                {
                    "scene_id": "observation-1",
                    "class_name": "forest",
                    "estimated_fraction": 0.3,
                    "estimated_area_m2": 432000.0,
                }
            ],
            "evidence": [
                {
                    "scene_id": "observation-1",
                    "geojson": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [1, 2]},
                                "properties": {"estimate": 0.3},
                            }
                        ],
                    },
                }
            ],
            "limitations": ["Coverage is estimated."],
            "trace": {"selected_tool": "coverage", "function_calls": []},
        }


def _advanced_zip(manifest="{}"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("request.json", manifest)
    return stream.getvalue()


def _geotiff(*, count=1, base=1.0):
    values = np.stack(
        [np.full((120, 120), base + index, dtype=np.float32) for index in range(count)]
    )
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=120,
            height=120,
            count=count,
            dtype="float32",
            crs="EPSG:32633",
            transform=from_origin(500000, 5200000, 10, 10),
        ) as dataset:
            dataset.write(values)
        return memory.read()


def _app(tmp_path, *, loader=None, demo=False, demo_fit_all=False):
    from satquery.assistant.web import create_app

    runtime = _Runtime()
    controller = _Controller()
    seen = []

    def fake_loader(path, destination):
        seen.append((Path(path), Path(destination), Path(path).read_bytes()))
        destination.mkdir()
        return _Bundle()

    demo_features = tmp_path / "features" if demo else None
    if demo_features:
        demo_features.mkdir()
    app = create_app(
        runtime=runtime,
        controller=controller,
        output_dir=tmp_path / "results",
        demo_features=demo_features,
        demo_fit_all=demo_fit_all,
        bundle_loader=loader or fake_loader,
    )
    return app, runtime, controller, seen


def test_status_and_browser_explain_capabilities_and_demo_fit_all(tmp_path):
    app, _, _, _ = _app(tmp_path, demo=True, demo_fit_all=True)
    client = TestClient(app)

    status = client.get("/api/status")
    page = client.get("/")

    assert status.status_code == 200
    assert status.json()["capabilities"] == ["optical", "joint"]
    assert status.json()["providers"] == {"local": True, "openai": False}
    assert len(status.json()["classes"]) == 19
    assert status.json()["classes"][0] == {"index": 0, "name": "Urban fabric"}
    assert status.json()["demo"] == {
        "available": True,
        "capability": "joint",
        "manual_demo_fit_all_label": True,
    }
    assert page.status_code == 200
    assert "Guided GeoTIFF upload" in page.text
    assert "Advanced ZIP" in page.text
    assert "demo_fit_all" in page.text
    assert "OpenAI tool router" in page.text
    assert "Optional OpenAI wording" not in page.text
    assert 'id="training-disclosure"' in page.text
    assert 'id="scene-provenance"' in page.text
    assert page.text.count('aria-label="Demo sample index"') == 1
    assert page.headers["content-security-policy"].startswith("default-src 'self'")
    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "report.training_disclosure" in script.text
    assert "report.scene_provenance" in script.text
    assert "Loaded head training mode:" in script.text
    assert "Manual CLI demo label:" in script.text
    assert 'setText(byId("training-disclosure")' in script.text


def test_advanced_upload_creates_durable_opaque_report_and_geojson(tmp_path):
    app, runtime, controller, seen = _app(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/analyze",
        data={"question": "How much forest?", "provider": "local"},
        files={"archive": ("request.zip", _advanced_zip(), "application/zip")},
    )

    assert response.status_code == 200
    report = response.json()
    assert len(report["result_id"]) == 32
    assert report["input_summary"]["kind"] == "single"
    assert report["downloads"]["report"].endswith("/report.json")
    assert report["downloads"]["geojson"].endswith("/evidence.geojson")
    assert runtime.bundles and controller.calls[0][3] == "local"
    assert not seen[0][0].exists() and not seen[0][1].exists()

    downloaded = client.get(report["downloads"]["report"])
    geojson = client.get(report["downloads"]["geojson"])
    assert downloaded.status_code == 200 and downloaded.json()["result_id"] == report["result_id"]
    assert geojson.status_code == 200
    assert geojson.json()["features"][0]["properties"]["estimate"] == 0.3
    assert (tmp_path / "results" / report["result_id"] / "report.json").is_file()


def test_upload_report_preserves_allowlisted_loaded_scene_provenance(tmp_path):
    app, runtime, _, _ = _app(tmp_path, demo_fit_all=True)
    runtime.analyze = lambda bundle: [_provenance_scene()]
    client = TestClient(app)

    response = client.post(
        "/api/analyze",
        data={"question": "Describe", "provider": "local"},
        files={"archive": ("request.zip", _advanced_zip(), "application/zip")},
    )

    assert response.status_code == 200
    report = response.json()
    provenance = report["scene_provenance"][0]
    assert provenance["checkpoint_sha256"] == "a" * 64
    assert provenance["head_sha256"] == "c" * 64
    assert provenance["feature_key"] == "joint_encodings"
    assert provenance["feature_contract"]["inference_implementation_sha256"] == "b" * 64
    assert provenance["head_training_mode"] == "demo_fit_all"
    assert provenance["joint_encoder_inputs"] == {"optical": "optical-1", "SAR": "sar-1"}
    assert provenance["sensor_comparison"]["top_scene_coverage"]["joint"][0] == {
        "name": "Inland waters",
        "estimated_fraction": 0.5,
    }
    assert report["training_disclosure"] == {
        "head_training_modes": ["demo_fit_all"],
        "contains_demo_fit_all": True,
        "manual_demo_fit_all_label": False,
    }
    downloaded = client.get(report["downloads"]["report"]).json()
    assert downloaded["scene_provenance"] == report["scene_provenance"]
    serialized = json.dumps(report)
    assert "/Users/" not in serialized
    assert "sk-do-not-persist" not in serialized
    assert "raw_metadata" not in serialized


def test_cached_demo_disclosure_comes_from_loaded_head_provenance(tmp_path):
    app, runtime, _, _ = _app(tmp_path, demo=True, demo_fit_all=False)
    runtime.analyze_cached = lambda *args, **kwargs: _provenance_scene(source="cached_features")
    client = TestClient(app)

    response = client.post(
        "/api/demo",
        data={"question": "Describe", "provider": "local", "index": "0"},
    )

    assert response.status_code == 200
    report = response.json()
    assert report["training_disclosure"]["contains_demo_fit_all"] is True
    assert report["training_disclosure"]["manual_demo_fit_all_label"] is False
    assert report["input_summary"]["head_training_modes"] == ["demo_fit_all"]
    assert report["input_summary"]["demo_fit_all"] is True
    assert (
        client.get(report["downloads"]["report"]).json()["scene_provenance"]
        == report["scene_provenance"]
    )


def test_result_download_rejects_paths_and_unknown_identifiers(tmp_path):
    app, _, _, _ = _app(tmp_path)
    client = TestClient(app)

    assert client.get("/api/results/../../pyproject.toml/report.json").status_code == 404
    assert client.get("/api/results/not-an-id/report.json").status_code == 404
    assert client.get(f"/api/results/{'a' * 32}/report.json").status_code == 404

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "report.json").write_text('{"secret": true}')
    (tmp_path / "results" / ("b" * 32)).symlink_to(outside, target_is_directory=True)
    assert client.get(f"/api/results/{'b' * 32}/report.json").status_code == 404


def test_api_rejects_cross_origin_and_oversized_requests_before_processing(tmp_path):
    app, runtime, _, _ = _app(tmp_path)
    client = TestClient(app)

    cross_origin = client.get("/api/status", headers={"Origin": "https://attacker.example"})
    too_large = client.post(
        "/api/analyze",
        headers={"Content-Length": str(65 * 1024 * 1024)},
        data={"question": "Describe", "provider": "local"},
        files={"archive": ("request.zip", b"small", "application/zip")},
    )

    assert cross_origin.status_code == 403
    assert "access-control-allow-origin" not in cross_origin.headers
    assert too_large.status_code == 413
    assert runtime.bundles == []


def test_chunked_multipart_aggregate_limit_rejects_before_form_parse_and_closes_spool(
    tmp_path, monkeypatch
):
    import satquery.assistant.web as web_module

    created_spools = []
    original_spool = tempfile.SpooledTemporaryFile

    def tracked_spool(*args, **kwargs):
        spool = original_spool(*args, **kwargs)
        created_spools.append(spool)
        return spool

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(web_module.tempfile, "SpooledTemporaryFile", tracked_spool)
    form_calls = []
    original_form = web_module.Request.form

    async def tracked_form(request, *args, **kwargs):
        form_calls.append(request)
        return await original_form(request, *args, **kwargs)

    monkeypatch.setattr(web_module.Request, "form", tracked_form)
    app, runtime, _, _ = _app(tmp_path)
    app.state.request_size_limit = 512
    app.state.request_spool_memory_size = 64
    client = TestClient(app)
    boundary = "satquery-stream-boundary"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="question"\r\n\r\n'
            "Describe\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="archive"; filename="request.zip"\r\n'
            "Content-Type: application/zip\r\n\r\n"
        ).encode()
        + b"x" * 1024
        + f"\r\n--{boundary}--\r\n".encode()
    )

    def chunks():
        for start in range(0, len(body), 73):
            yield body[start : start + 73]

    request = client.build_request(
        "POST",
        "/api/analyze",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        content=chunks(),
    )
    assert "content-length" not in request.headers
    assert request.headers["transfer-encoding"] == "chunked"
    response = client.send(request)

    assert response.status_code == 413
    assert response.json() == {"detail": "request is too large"}
    assert runtime.bundles == []
    assert form_calls == []
    assert created_spools and all(spool.closed for spool in created_spools)
    assert not list(tmp_path.rglob("satquery-body-*"))


def test_manifest_loader_type_error_is_sanitized_422(tmp_path):
    from satquery.assistant.inputs import load_bundle

    app, runtime, _, _ = _app(tmp_path, loader=load_bundle)
    response = TestClient(app).post(
        "/api/analyze",
        data={"question": "Describe", "provider": "local"},
        files={
            "archive": (
                "request.zip",
                _advanced_zip('{"schema_version":1,"observations":[1]}'),
                "application/zip",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "observation 0: expected an object"
    assert runtime.bundles == []


def test_manifest_loader_value_error_is_sanitized_422(tmp_path):
    error = ValueError("bad grid")

    def malformed_loader(path, destination):
        raise error

    app, runtime, _, _ = _app(tmp_path, loader=malformed_loader)
    response = TestClient(app).post(
        "/api/analyze",
        data={"question": "Describe", "provider": "local"},
        files={"archive": ("request.zip", _advanced_zip(), "application/zip")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == str(error)
    assert runtime.bundles == []


def test_demo_uses_configured_cached_features_and_validates_index(tmp_path):
    app, runtime, controller, _ = _app(tmp_path, demo=True)
    client = TestClient(app)

    response = client.post(
        "/api/demo",
        data={"question": "Describe this scene", "provider": "local", "index": "2"},
    )

    assert response.status_code == 200
    assert runtime.demo_calls == [(tmp_path / "features", 2, "joint")]
    assert controller.calls[0][2]["source"] == "configured_cached_features"
    assert (
        client.post(
            "/api/demo", data={"question": "Describe", "provider": "local", "index": "-1"}
        ).status_code
        == 422
    )


def test_demo_validation_error_never_exposes_configured_feature_paths(tmp_path):
    class LeakyRuntime(_Runtime):
        def analyze_cached(self, features, *, selected_index, capability):
            raise ValueError("Input hash mismatch: /Users/private-owner/secret-demo/features.pt")

    app, _, _, _ = _app(tmp_path, demo=True)
    app.state.services.runtime = LeakyRuntime()

    response = TestClient(app).post(
        "/api/demo",
        data={"question": "Describe", "provider": "local", "index": "0"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Configured cached feature demo could not be read or validated."
    )
    assert "/Users/" not in response.text and "features.pt" not in response.text


def test_openai_provider_without_configuration_is_actionable(tmp_path):
    from satquery.assistant.controller import AssistantController

    app, runtime, _, _ = _app(tmp_path)
    app.state.services.controller = AssistantController(openai_client=None)
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        data={"question": "Describe", "provider": "openai"},
        files={"archive": ("request.zip", _advanced_zip(), "application/zip")},
    )

    assert response.status_code == 502
    assert "OPENAI_API_KEY" in response.json()["detail"]
    assert runtime.bundles == []


def test_configured_paths_reject_symlinks(tmp_path):
    from satquery.assistant.web import create_app

    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(ValueError, match="output_dir"):
        create_app(
            runtime=_Runtime(),
            controller=_Controller(),
            output_dir=linked_output,
        )


def test_cli_exposes_model_demo_and_artifact_configuration():
    from satquery.assistant.web import build_parser

    parsed = build_parser().parse_args(
        [
            "--croma-checkpoint",
            "croma.pt",
            "--optical-head",
            "optical.pt",
            "--sar-head",
            "sar.pt",
            "--joint-head",
            "joint.pt",
            "--demo-features",
            "features",
            "--demo-preview",
            "preview.png",
            "--demo-fit-all",
            "--output-dir",
            "artifacts",
        ]
    )

    assert parsed.croma_checkpoint == "croma.pt"
    assert (parsed.optical_head, parsed.sar_head, parsed.joint_head) == (
        "optical.pt",
        "sar.pt",
        "joint.pt",
    )
    assert parsed.demo_features == "features" and parsed.demo_fit_all is True
    assert parsed.output_dir == "artifacts"


def test_guided_stacked_optical_upload_builds_explicit_band_index_manifest(tmp_path):
    manifests = []

    def loader(path, destination):
        from satquery.assistant.inputs import load_bundle

        with zipfile.ZipFile(path) as archive:
            manifests.append(json.loads(archive.read("request.json")))
            assert archive.read("observation-1/scene.tif")[:4] in (b"II*\x00", b"MM\x00*")
        return load_bundle(path, destination)

    app, _, _, _ = _app(tmp_path, loader=loader)
    response = TestClient(app).post(
        "/api/analyze-files",
        data={
            "question": "Describe the scene",
            "provider": "local",
            "modality1": "optical",
            "acquired1": "2026-01-02",
            "stack_order1": "true",
        },
        files=[("files1", ("scene.tif", _geotiff(count=12), "image/tiff"))],
    )

    assert response.status_code == 200
    observation = manifests[0]["observations"][0]
    assert observation["sensor"] == "Sentinel-2"
    assert list(observation["bands"]) == [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
    assert [entry["band"] for entry in observation["bands"].values()] == list(range(1, 13))
    assert {entry["file"] for entry in observation["bands"].values()} == {"observation-1/scene.tif"}


def test_guided_separate_sar_bands_require_explicit_db_and_use_names(tmp_path):
    manifests = []

    def loader(path, destination):
        from satquery.assistant.inputs import load_bundle

        with zipfile.ZipFile(path) as archive:
            manifests.append(json.loads(archive.read("request.json")))
        return load_bundle(path, destination)

    app, _, _, _ = _app(tmp_path, loader=loader)
    client = TestClient(app)
    files = [
        ("files1", ("tile_VV.tif", _geotiff(base=-12), "image/tiff")),
        ("files1", ("tile_VH.tif", _geotiff(base=-18), "image/tiff")),
    ]
    missing_db = client.post(
        "/api/analyze-files",
        data={
            "question": "Locate water",
            "provider": "local",
            "modality1": "SAR",
            "acquired1": "2026-02-03",
        },
        files=files,
    )
    accepted = client.post(
        "/api/analyze-files",
        data={
            "question": "Locate water",
            "provider": "local",
            "modality1": "SAR",
            "acquired1": "2026-02-03",
            "radiometry1": "db",
        },
        files=files,
    )

    assert missing_db.status_code == 422
    assert "radiometry" in missing_db.json()["detail"]
    assert accepted.status_code == 200
    observation = manifests[0]["observations"][0]
    assert observation["sensor"] == "Sentinel-1" and observation["radiometry"] == "db"
    assert observation["bands"] == {
        "VV": {"file": "observation-1/tile_VV.tif", "band": 1},
        "VH": {"file": "observation-1/tile_VH.tif", "band": 1},
    }


def test_guided_upload_rejects_missing_bands_and_unsafe_filenames(tmp_path):
    app, _, _, _ = _app(tmp_path)
    client = TestClient(app)
    fields = {
        "question": "Describe",
        "provider": "local",
        "modality1": "optical",
        "acquired1": "2026-01-02",
    }

    missing = client.post(
        "/api/analyze-files",
        data=fields,
        files=[("files1", ("tile_B01.tif", b"one-band", "image/tiff"))],
    )
    unsafe = client.post(
        "/api/analyze-files",
        data={**fields, "stack_order1": "true"},
        files=[("files1", ("../scene.tif", b"stack", "image/tiff"))],
    )

    assert missing.status_code == 422 and "exactly" in missing.json()["detail"]
    assert unsafe.status_code == 422 and "filename" in unsafe.json()["detail"]
