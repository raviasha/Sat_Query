"""Local FastAPI application for bounded SatQuery analysis."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from PIL import Image
    from starlette.datastructures import UploadFile
    from starlette.middleware.trustedhost import TrustedHostMiddleware
except ImportError as exc:  # pragma: no cover - exercised only in a clean base installation
    raise RuntimeError(
        "SatQuery's browser app requires the assistant extra; install "
        "'satquery-preprocessing[assistant]'."
    ) from exc

from ..prediction_data import class_schema
from ..preprocessing import DEFAULT_PROFILE
from .controller import MAX_QUESTION_LENGTH, AssistantController, ProviderError
from .inputs import MAX_MEMBERS, MAX_UPLOAD_SIZE, load_bundle
from .runtime import CapabilityError, CoverageRuntime

MAX_REQUEST_SIZE = MAX_UPLOAD_SIZE + 256 * 1024
REQUEST_SPOOL_MEMORY_SIZE = 1024 * 1024
MAX_FORM_FIELDS = 32
_RESULT_ID = re.compile(r"^[0-9a-f]{32}$")
_TIFF_SUFFIX = re.compile(r"\.tiff?$", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+ -]{0,127}$")
_SENSITIVE_TEXT = re.compile(
    r"(?:sk-[A-Za-z0-9]|api[_-]?key|authorization:|bearer\s|/Users/|/home/|/tmp/|"
    r"[A-Za-z]:\\\\Users\\\\)",
    re.IGNORECASE,
)


class _BoundedRequestBodyMiddleware:
    """Count and spool upload bodies before multipart parsing, then replay bounded chunks."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") not in {"POST", "PUT", "PATCH"}
            or scope.get("path") not in {"/api/analyze", "/api/analyze-files", "/api/demo"}
        ):
            await self.app(scope, receive, send)
            return
        state = scope["app"].state
        maximum = int(state.request_size_limit)
        memory_size = int(state.request_spool_memory_size)
        with tempfile.SpooledTemporaryFile(
            max_size=memory_size, mode="w+b", prefix="satquery-body-"
        ) as spool:
            total = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body = message.get("body", b"")
                total += len(body)
                if total > maximum:
                    response = JSONResponse(
                        status_code=413, content={"detail": "request is too large"}
                    )
                    await response(scope, receive, send)
                    return
                spool.write(body)
                if not message.get("more_body", False):
                    break
            spool.seek(0)
            remaining = total

            async def replay_receive():
                nonlocal remaining
                if remaining:
                    body = spool.read(min(1024 * 1024, remaining))
                    remaining -= len(body)
                    return {
                        "type": "http.request",
                        "body": body,
                        "more_body": remaining > 0,
                    }
                return await receive()

            await self.app(scope, replay_receive, send)


@dataclass
class _Services:
    runtime: Any
    controller: Any
    output_dir: Path
    demo_features: Path | None
    demo_capability: str
    demo_index: int
    demo_preview: Path | None
    demo_fit_all: bool
    bundle_loader: Callable[[Path, Path], Any]
    inference_lock: asyncio.Lock


def _safe_error(message: str, temporary_root: Path | None = None) -> str:
    if temporary_root is not None:
        message = message.replace(str(temporary_root), "[temporary upload]")
        message = message.replace(str(temporary_root.resolve()), "[temporary upload]")
    return message[:1000]


def _validate_question_provider(question: Any, provider: Any) -> tuple[str, str]:
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"question must contain 1 to {MAX_QUESTION_LENGTH} characters")
    if provider not in ("local", "openai"):
        raise ValueError("provider must be local or openai")
    return question.strip(), provider


def _clean_filename(filename: Any, *, suffix: str | None = None) -> str:
    if (
        not isinstance(filename, str)
        or not filename
        or len(filename) > 255
        or "\x00" in filename
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise ValueError("upload filename must be a simple portable filename")
    if suffix is not None and not filename.casefold().endswith(suffix):
        raise ValueError(f"upload filename must end in {suffix}")
    return filename


async def _copy_upload(upload: UploadFile, target: Path, *, maximum: int) -> int:
    written = 0
    with target.open("xb") as destination:
        while True:
            block = await upload.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > maximum:
                raise ValueError(f"uploaded data exceeds {maximum // (1024 * 1024)} MiB")
            destination.write(block)
    if written == 0:
        raise ValueError("uploaded files must not be empty")
    return written


def _parse_date(value: Any, number: int) -> str:
    if not isinstance(value, str):
        raise ValueError(  # noqa: TRY004 - malformed form data is a client value error
            f"observation {number} requires an acquisition date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"observation {number} acquisition date must use YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"observation {number} acquisition date must use YYYY-MM-DD")
    return value


def _upload_values(form: Any, key: str) -> list[UploadFile]:
    values = form.getlist(key)
    if not values or any(not isinstance(value, UploadFile) for value in values):
        raise ValueError(f"{key} must contain one or more TIFF files")
    return values


async def _stage_observation(
    form: Any,
    number: int,
    staging: Path,
    *,
    remaining: int,
) -> tuple[dict[str, Any], int]:
    if remaining <= 0:
        raise ValueError("guided upload exceeds 64 MiB")
    modality = form.get(f"modality{number}")
    if modality not in ("optical", "SAR"):
        raise ValueError(f"observation {number} modality must be optical or SAR")
    acquired = _parse_date(form.get(f"acquired{number}"), number)
    radiometry = form.get(f"radiometry{number}")
    if modality == "SAR" and radiometry != "db":
        raise ValueError(f"observation {number} SAR radiometry must be explicitly declared as db")
    if modality == "optical" and radiometry not in (None, ""):
        raise ValueError(f"observation {number} optical input must not declare SAR radiometry")
    uploads = _upload_values(form, f"files{number}")
    if len(uploads) > MAX_MEMBERS - 1:
        raise ValueError(f"too many files; maximum is {MAX_MEMBERS - 1}")
    stack_order = form.get(f"stack_order{number}") == "true"
    if stack_order and len(uploads) != 1:
        raise ValueError("standard channel-order acknowledgement requires exactly one stacked TIFF")

    expected = list(DEFAULT_PROFILE.optical if modality == "optical" else DEFAULT_PROFILE.sar)
    directory = staging / f"observation-{number}"
    directory.mkdir()
    saved: list[tuple[str, Path]] = []
    consumed = 0
    folded: set[str] = set()
    for upload in uploads:
        filename = _clean_filename(upload.filename)
        if not _TIFF_SUFFIX.search(filename):
            raise ValueError("guided uploads accept only .tif or .tiff files")
        if filename.casefold() in folded:
            raise ValueError("guided upload filenames must be unique ignoring case")
        folded.add(filename.casefold())
        target = directory / filename
        size = await _copy_upload(upload, target, maximum=remaining - consumed)
        consumed += size
        saved.append((filename, target))

    archive_prefix = f"observation-{number}"
    if stack_order:
        filename = saved[0][0]
        bands = {
            band: {"file": f"{archive_prefix}/{filename}", "band": index}
            for index, band in enumerate(expected, start=1)
        }
    else:
        if len(saved) != len(expected):
            raise ValueError(
                f"observation {number} requires exactly {len(expected)} named band files, "
                "or one acknowledged standard-order stack"
            )
        found: dict[str, str] = {}
        for filename, _ in saved:
            matches = [
                band
                for band in expected
                if re.search(rf"_{re.escape(band)}\.tiff?$", filename, re.IGNORECASE)
            ]
            if len(matches) != 1 or matches[0] in found:
                raise ValueError(
                    f"observation {number} filenames must end in one unique expected band name"
                )
            found[matches[0]] = filename
        if set(found) != set(expected):
            raise ValueError(f"observation {number} is missing one or more required bands")
        bands = {band: {"file": f"{archive_prefix}/{found[band]}", "band": 1} for band in expected}

    observation = {
        "id": f"observation-{number}",
        "modality": modality,
        "sensor": "Sentinel-2" if modality == "optical" else "Sentinel-1",
        "acquired": acquired,
        "bands": bands,
    }
    if modality == "SAR":
        observation["radiometry"] = "db"
    return observation, consumed


async def _guided_archive(form: Any, root: Path) -> Path:
    staging = root / "guided"
    staging.mkdir()
    observation1, consumed = await _stage_observation(form, 1, staging, remaining=MAX_UPLOAD_SIZE)
    observations = [observation1]
    has_second = bool(form.getlist("files2")) or any(
        form.get(name) not in (None, "") for name in ("modality2", "acquired2", "radiometry2")
    )
    if has_second:
        observation2, second_size = await _stage_observation(
            form, 2, staging, remaining=MAX_UPLOAD_SIZE - consumed
        )
        consumed += second_size
        observations.append(observation2)
    manifest = {"schema_version": 1, "observations": observations}
    (staging / "request.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )
    archive_path = root / "guided-request.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(staging / "request.json", "request.json")
        for path in sorted(staging.glob("observation-*/*")):
            archive.write(path, path.relative_to(staging).as_posix())
    if archive_path.stat().st_size > MAX_UPLOAD_SIZE:
        raise ValueError("assembled ZIP exceeds 64 MiB")
    return archive_path


def _evidence_geojson(result: dict[str, Any]) -> dict[str, Any]:
    features = []
    for item in result.get("evidence", []):
        if isinstance(item, dict):
            geojson = item.get("geojson")
            if isinstance(geojson, dict) and isinstance(geojson.get("features"), list):
                features.extend(geojson["features"])
    return {
        "type": "FeatureCollection",
        "name": "SatQuery coarse spatial evidence",
        "features": features,
    }


def _safe_identifier(value: Any) -> str | None:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value):
        return None
    if _SENSITIVE_TEXT.search(value):
        return None
    return value


def _safe_digest(value: Any) -> str | None:
    return value if isinstance(value, str) and _SHA256.fullmatch(value) else None


def _safe_text(value: Any, *, maximum: int = 500) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 and character not in "\n\t" for character in value)
        or _SENSITIVE_TEXT.search(value)
    ):
        return None
    return value


def _public_feature_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("checkpoint_sha256", "inference_implementation_sha256"):
        if digest := _safe_digest(value.get(key)):
            result[key] = digest
    for key in ("croma_source_revision", "normalization_profile"):
        if identifier := _safe_identifier(value.get(key)):
            result[key] = identifier
    profile = value.get("channel_profile")
    if isinstance(profile, dict):
        public_profile: dict[str, Any] = {}
        for key in ("optical", "sar"):
            bands = profile.get(key)
            if (
                isinstance(bands, list)
                and len(bands) <= 16
                and all(_safe_identifier(band) is not None for band in bands)
            ):
                public_profile[key] = bands
        for key in ("evidence", "evidence_level"):
            if identifier := _safe_identifier(profile.get(key)):
                public_profile[key] = identifier
        if public_profile:
            result["channel_profile"] = public_profile
    return result


def _public_sensor_comparison(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    coverage = value.get("top_scene_coverage")
    public_coverage: dict[str, Any] = {}
    if isinstance(coverage, dict):
        class_names = {item["name"] for item in class_schema()}
        for modality in ("optical", "SAR", "joint"):
            entries = coverage.get(modality)
            if not isinstance(entries, list) or len(entries) > 3:
                continue
            selected = []
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("name") not in class_names:
                    continue
                fraction = entry.get("estimated_fraction")
                if (
                    isinstance(fraction, bool)
                    or not isinstance(fraction, (int, float))
                    or not 0 <= fraction <= 1
                ):
                    continue
                selected.append({"name": entry["name"], "estimated_fraction": float(fraction)})
            if selected:
                public_coverage[modality] = selected
    result: dict[str, Any] = {}
    if public_coverage:
        result["top_scene_coverage"] = public_coverage
    if interpretation := _safe_text(value.get("interpretation")):
        result["interpretation"] = interpretation
    return result or None


def _public_scene_provenance(scenes: list[Any]) -> list[dict[str, Any]]:
    records = []
    for scene in scenes[:4]:
        provenance = getattr(scene, "provenance", None)
        if not isinstance(provenance, dict):
            continue
        record: dict[str, Any] = {}
        for output_key, value in (
            ("scene_id", getattr(scene, "id", None)),
            ("modality", getattr(scene, "modality", None)),
            ("source", provenance.get("source")),
            ("model", provenance.get("model")),
            ("model_mode", provenance.get("model_mode")),
            ("feature_key", provenance.get("feature_key")),
            ("head_training_mode", provenance.get("head_training_mode")),
            ("head_evaluation_scope", provenance.get("head_evaluation_scope")),
        ):
            if identifier := _safe_identifier(value):
                record[output_key] = identifier
        for key in (
            "checkpoint_sha256",
            "head_sha256",
            "feature_manifest_sha256",
        ):
            if digest := _safe_digest(provenance.get(key)):
                record[key] = digest
        contract = _public_feature_contract(provenance.get("feature_contract"))
        if contract:
            record["feature_contract"] = contract
        observations = provenance.get("source_observations")
        if (
            isinstance(observations, list)
            and len(observations) <= 4
            and all(_safe_identifier(item) is not None for item in observations)
        ):
            record["source_observations"] = observations
        selected_index = provenance.get("selected_index")
        if (
            isinstance(selected_index, int)
            and not isinstance(selected_index, bool)
            and 0 <= selected_index <= 1_000_000
        ):
            record["selected_index"] = selected_index
        if prediction_units := _safe_text(provenance.get("prediction_units"), maximum=100):
            record["prediction_units"] = prediction_units
        encoder_inputs = provenance.get("joint_encoder_inputs")
        if isinstance(encoder_inputs, dict):
            selected_inputs = {
                key: identifier
                for key in ("optical", "SAR")
                if (identifier := _safe_identifier(encoder_inputs.get(key))) is not None
            }
            if selected_inputs:
                record["joint_encoder_inputs"] = selected_inputs
        if fusion := _safe_text(provenance.get("fusion")):
            record["fusion"] = fusion
        if comparison := _public_sensor_comparison(provenance.get("sensor_comparison")):
            record["sensor_comparison"] = comparison
        records.append(record)
    return records


def _training_disclosure(
    provenance: list[dict[str, Any]], *, manual_demo_fit_all: bool
) -> dict[str, Any]:
    modes = sorted(
        {
            item["head_training_mode"]
            for item in provenance
            if item.get("head_training_mode") in {"demo_fit_all", "train_split_only"}
        }
    )
    return {
        "head_training_modes": modes,
        "contains_demo_fit_all": "demo_fit_all" in modes,
        "manual_demo_fit_all_label": bool(manual_demo_fit_all),
    }


def _artifact_path(output_dir: Path, result_id: str, filename: str) -> Path:
    if not _RESULT_ID.fullmatch(result_id):
        raise HTTPException(status_code=404, detail="result not found")
    directory = output_dir / result_id
    if not directory.is_dir() or directory.is_symlink():
        raise HTTPException(status_code=404, detail="result not found")
    path = directory / filename
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve().parent != directory.resolve()
        or output_dir not in path.resolve().parents
    ):
        raise HTTPException(status_code=404, detail="result artifact not found")
    return path


def _persist_result(
    services: _Services,
    result: dict[str, Any],
    summary: dict[str, Any],
    scenes: list[Any],
    *,
    fallback_preview: Path | None = None,
) -> dict[str, Any]:
    result_id = uuid.uuid4().hex
    artifact_dir = services.output_dir / result_id
    artifact_dir.mkdir()
    report = dict(result)
    report["result_id"] = result_id
    report["input_summary"] = summary
    report["scene_provenance"] = _public_scene_provenance(scenes)
    report["training_disclosure"] = _training_disclosure(
        report["scene_provenance"],
        manual_demo_fit_all=(
            services.demo_fit_all and summary.get("source") == "configured_cached_features"
        ),
    )
    base = f"/api/results/{result_id}"
    downloads = {"report": f"{base}/report.json", "geojson": f"{base}/evidence.geojson"}
    report["downloads"] = downloads
    try:
        geojson = _evidence_geojson(report)
        (artifact_dir / "evidence.geojson").write_text(
            json.dumps(geojson, allow_nan=False, separators=(",", ":")), encoding="utf-8"
        )
        preview = next(
            (
                getattr(scene, "preview", None)
                for scene in scenes
                if getattr(scene, "preview", None) is not None
            ),
            None,
        )
        if preview is not None:
            Image.fromarray(preview).save(artifact_dir / "preview.png", format="PNG")
        elif fallback_preview is not None:
            shutil.copyfile(fallback_preview, artifact_dir / "preview.png")
        if (artifact_dir / "preview.png").is_file():
            downloads["preview"] = f"{base}/preview.png"
        (artifact_dir / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise
    return report


async def _analyze_bundle(
    services: _Services,
    bundle: Any,
    question: str,
    provider: str,
) -> dict[str, Any]:
    if provider == "openai" and not services.controller.openai_configured:
        raise ProviderError(
            "OpenAI is not configured; set OPENAI_API_KEY on the server or choose local mode."
        )
    async with services.inference_lock:
        scenes = await asyncio.to_thread(services.runtime.analyze, bundle)
        result = await asyncio.to_thread(
            services.controller.answer,
            question,
            scenes,
            bundle.summary(),
            provider=provider,
        )
        return await asyncio.to_thread(_persist_result, services, result, bundle.summary(), scenes)


def create_app(
    *,
    runtime: Any,
    controller: Any,
    output_dir: Path,
    demo_features: Path | None = None,
    demo_capability: str = "joint",
    demo_index: int = 0,
    demo_preview: Path | None = None,
    demo_fit_all: bool = False,
    bundle_loader: Callable[[Path, Path], Any] = load_bundle,
) -> FastAPI:
    """Create a localhost-oriented app with injected runtime/controller dependencies."""
    raw_output = Path(output_dir).expanduser()
    if raw_output.is_symlink():
        raise ValueError("output_dir must be a regular directory")
    output = raw_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("output_dir must be a regular directory")
    raw_features = Path(demo_features).expanduser() if demo_features is not None else None
    if raw_features is not None and raw_features.is_symlink():
        raise ValueError("demo_features must be a regular directory")
    features = raw_features.resolve() if raw_features is not None else None
    if features is not None and (not features.is_dir() or features.is_symlink()):
        raise ValueError("demo_features must be a regular directory")
    raw_preview = Path(demo_preview).expanduser() if demo_preview is not None else None
    if raw_preview is not None and raw_preview.is_symlink():
        raise ValueError("demo_preview must be a regular file")
    preview = raw_preview.resolve() if raw_preview is not None else None
    if preview is not None and (not preview.is_file() or preview.is_symlink()):
        raise ValueError("demo_preview must be a regular file")
    if demo_capability not in ("optical", "SAR", "joint"):
        raise ValueError("demo_capability must be optical, SAR, or joint")
    if not isinstance(demo_index, int) or isinstance(demo_index, bool) or demo_index < 0:
        raise ValueError("demo_index must be a nonnegative integer")

    app = FastAPI(title="SatQuery", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )
    app.state.services = _Services(
        runtime=runtime,
        controller=controller,
        output_dir=output,
        demo_features=features,
        demo_capability=demo_capability,
        demo_index=demo_index,
        demo_preview=preview,
        demo_fit_all=bool(demo_fit_all),
        bundle_loader=bundle_loader,
        inference_lock=asyncio.Lock(),
    )
    app.state.request_size_limit = MAX_REQUEST_SIZE
    app.state.request_spool_memory_size = REQUEST_SPOOL_MEMORY_SIZE
    app.add_middleware(_BoundedRequestBodyMiddleware)

    @app.middleware("http")
    async def local_request_boundary(request: Request, call_next):
        if request.url.path.startswith("/api/") or request.url.path == "/api/status":
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > app.state.request_size_limit:
                        return JSONResponse(
                            status_code=413, content={"detail": "request is too large"}
                        )
                except ValueError:
                    return JSONResponse(
                        status_code=400, content={"detail": "invalid Content-Length"}
                    )
            origin = request.headers.get("origin")
            host = request.headers.get("host", "")
            if origin:
                parsed = urlparse(origin)
                if parsed.scheme not in ("http", "https") or parsed.netloc != host:
                    return JSONResponse(
                        status_code=403, content={"detail": "cross-origin API access denied"}
                    )
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse(
                    status_code=403, content={"detail": "cross-origin API access denied"}
                )
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.get("/api/status")
    async def status():
        services = app.state.services
        return {
            "status": "ready",
            "capabilities": list(services.runtime.available_capabilities),
            "providers": {
                "local": True,
                "openai": bool(services.controller.openai_configured),
            },
            "classes": [{"index": item["index"], "name": item["name"]} for item in class_schema()],
            "openai_model": services.controller.model,
            "demo": {
                "available": services.demo_features is not None,
                "capability": services.demo_capability,
                "manual_demo_fit_all_label": services.demo_fit_all,
            },
            "limits": {
                "question_characters": MAX_QUESTION_LENGTH,
                "upload_bytes": MAX_UPLOAD_SIZE,
            },
        }

    async def parse_form(request: Request):
        try:
            return await request.form(
                max_files=MAX_MEMBERS,
                max_fields=MAX_FORM_FIELDS,
                max_part_size=MAX_UPLOAD_SIZE,
            )
        except Exception as exc:
            raise HTTPException(status_code=413, detail="multipart request exceeds limits") from exc

    async def process_archive(archive_path: Path, root: Path, question: str, provider: str):
        services = app.state.services
        extraction = root / "extracted"
        try:
            bundle = await asyncio.to_thread(services.bundle_loader, archive_path, extraction)
            return await _analyze_bundle(services, bundle, question, provider)
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        except (CapabilityError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=_safe_error(str(exc), temporary_root=root)
            ) from None

    @app.post("/api/analyze")
    async def analyze(request: Request):
        form = await parse_form(request)
        try:
            question, provider = _validate_question_provider(
                form.get("question"), form.get("provider")
            )
            upload = form.get("archive")
            if not isinstance(upload, UploadFile):
                raise ValueError(  # noqa: TRY004 - malformed form data is a client value error
                    "archive must be a ZIP file"
                )
            _clean_filename(upload.filename, suffix=".zip")
            with tempfile.TemporaryDirectory(prefix="satquery-request-") as raw_root:
                root = Path(raw_root)
                archive_path = root / "request.zip"
                await _copy_upload(upload, archive_path, maximum=MAX_UPLOAD_SIZE)
                return await process_archive(archive_path, root, question, provider)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=_safe_error(str(exc))) from None

    @app.post("/api/analyze-files")
    async def analyze_files(request: Request):
        form = await parse_form(request)
        try:
            question, provider = _validate_question_provider(
                form.get("question"), form.get("provider")
            )
            with tempfile.TemporaryDirectory(prefix="satquery-guided-") as raw_root:
                root = Path(raw_root)
                archive_path = await _guided_archive(form, root)
                return await process_archive(archive_path, root, question, provider)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=_safe_error(str(exc))) from None

    @app.post("/api/demo")
    async def demo(request: Request):
        form = await parse_form(request)
        services = app.state.services
        if services.demo_features is None:
            raise HTTPException(status_code=404, detail="cached feature demo is not configured")
        try:
            question, provider = _validate_question_provider(
                form.get("question"), form.get("provider")
            )
            if provider == "openai" and not services.controller.openai_configured:
                raise ProviderError(
                    "OpenAI is not configured; set OPENAI_API_KEY on the server or choose local mode."
                )
            raw_index = form.get("index", str(services.demo_index))
            if (
                not isinstance(raw_index, str)
                or not raw_index.isascii()
                or not raw_index.isdecimal()
            ):
                raise ValueError("demo index must be a nonnegative integer")
            index = int(raw_index)
            if index > 1_000_000:
                raise ValueError("demo index is too large")
            async with services.inference_lock:
                scene = await asyncio.to_thread(
                    services.runtime.analyze_cached,
                    services.demo_features,
                    selected_index=index,
                    capability=services.demo_capability,
                )
                summary = {
                    "source": "configured_cached_features",
                    "selected_index": index,
                    "capability": services.demo_capability,
                }
                provenance = _public_scene_provenance([scene])
                disclosure = _training_disclosure(
                    provenance, manual_demo_fit_all=services.demo_fit_all
                )
                summary["head_training_modes"] = disclosure["head_training_modes"]
                summary["demo_fit_all"] = disclosure["contains_demo_fit_all"]
                summary["manual_demo_fit_all_label"] = services.demo_fit_all
                result = await asyncio.to_thread(
                    services.controller.answer,
                    question,
                    [scene],
                    summary,
                    provider=provider,
                )
                return await asyncio.to_thread(
                    _persist_result,
                    services,
                    result,
                    summary,
                    [scene],
                    fallback_preview=services.demo_preview,
                )
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        except CapabilityError:
            raise HTTPException(
                status_code=422,
                detail="Demo capability is unavailable or incompatible with configured models.",
            ) from None
        except IndexError:
            raise HTTPException(
                status_code=422,
                detail="Demo sample index is outside the configured feature dataset.",
            ) from None
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Configured cached feature demo could not be read or validated.",
            ) from None

    @app.get("/api/results/{result_id}/report.json")
    async def report_download(result_id: str):
        return FileResponse(
            _artifact_path(output, result_id, "report.json"),
            media_type="application/json",
            filename=f"satquery-{result_id}-report.json",
        )

    @app.get("/api/results/{result_id}/evidence.geojson")
    async def evidence_download(result_id: str):
        return FileResponse(
            _artifact_path(output, result_id, "evidence.geojson"),
            media_type="application/geo+json",
            filename=f"satquery-{result_id}-evidence.geojson",
        )

    @app.get("/api/results/{result_id}/preview.png")
    async def preview_download(result_id: str):
        return FileResponse(
            _artifact_path(output, result_id, "preview.png"), media_type="image/png"
        )

    return app


def _regular_file(parser: argparse.ArgumentParser, value: str, label: str) -> Path:
    raw_path = Path(value).expanduser()
    if raw_path.is_symlink():
        parser.error(f"{label} must be a regular file: {value}")
    path = raw_path.resolve()
    if not path.is_file() or path.is_symlink():
        parser.error(f"{label} must be a regular file: {value}")
    return path


def _regular_directory(parser: argparse.ArgumentParser, value: str, label: str) -> Path:
    raw_path = Path(value).expanduser()
    if raw_path.is_symlink():
        parser.error(f"{label} must be a regular directory: {value}")
    path = raw_path.resolve()
    if not path.is_dir() or path.is_symlink():
        parser.error(f"{label} must be a regular directory: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the local SatQuery browser application")
    parser.add_argument("--croma-checkpoint", required=True)
    parser.add_argument("--optical-head")
    parser.add_argument("--sar-head")
    parser.add_argument("--joint-head")
    parser.add_argument("--demo-features")
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--demo-capability", choices=("optical", "SAR", "joint"), default="joint")
    parser.add_argument("--demo-preview")
    parser.add_argument("--demo-fit-all", action="store_true")
    parser.add_argument("--output-dir", default="satquery-results")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint = _regular_file(parser, args.croma_checkpoint, "--croma-checkpoint")
    heads = {}
    for capability, value in (
        ("optical", args.optical_head),
        ("SAR", args.sar_head),
        ("joint", args.joint_head),
    ):
        if value:
            heads[capability] = _regular_file(parser, value, f"--{capability.lower()}-head")
    if not heads:
        parser.error("configure at least one of --optical-head, --sar-head, or --joint-head")
    demo_features = (
        _regular_directory(parser, args.demo_features, "--demo-features")
        if args.demo_features
        else None
    )
    demo_preview = (
        _regular_file(parser, args.demo_preview, "--demo-preview") if args.demo_preview else None
    )
    if args.demo_index < 0:
        parser.error("--demo-index must be nonnegative")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be from 1 through 65535")
    runtime = CoverageRuntime(checkpoint, heads, device=args.device)
    available = set(runtime.available_capabilities)
    incompatible = set(heads) - available
    if incompatible:
        parser.error(
            "configured heads are incompatible with the pinned runtime: "
            + ", ".join(sorted(incompatible))
        )
    if demo_features is not None and args.demo_capability not in available:
        parser.error(f"--demo-capability {args.demo_capability} requires a compatible head")
    app = create_app(
        runtime=runtime,
        controller=AssistantController(),
        output_dir=Path(args.output_dir),
        demo_features=demo_features,
        demo_capability=args.demo_capability,
        demo_index=args.demo_index,
        demo_preview=demo_preview,
        demo_fit_all=args.demo_fit_all,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
