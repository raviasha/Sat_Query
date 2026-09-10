"""Reusable CROMA inference on already-prepared tensors; no TIFF preprocessing."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch

from .preprocessing import CROMA_REVISION, DEFAULT_PROFILE, NORMALIZATION_PROFILE, sha256

CHECKPOINT_REVISION = "0dd28e3d633bd6715856ae9890e8c49360040598"
CHECKPOINT_SHA256 = "0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63"
CHECKPOINT_URL = (
    f"https://huggingface.co/antofuller/CROMA/resolve/{CHECKPOINT_REVISION}/CROMA_base.pt"
)
FEATURE_KEYS = (
    "optical_encodings",
    "SAR_encodings",
    "joint_encodings",
    "optical_GAP",
    "SAR_GAP",
    "joint_GAP",
)


class CromaFeatureExtractor:
    """Run an initialized CROMA-compatible model in eval/inference mode."""

    def __init__(self, model: torch.nn.Module, *, device: str = "cpu", provenance=None):
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps, or cuda")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.dimension = model.encoder_dim
        self.provenance = provenance or {
            "kind": "caller_supplied_model",
            "pretrained_verified": False,
        }

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Path | None = None,
        *,
        cache_dir: Path = Path("data/models/croma"),
        device: str = "cpu",
    ):
        """Load the pinned official Base checkpoint, checking its publisher SHA-256."""
        if checkpoint is None:
            cached = Path(cache_dir) / "CROMA_base.pt"
            if cached.is_file() and sha256(cached) == CHECKPOINT_SHA256:
                checkpoint = cached
            else:
                from huggingface_hub import hf_hub_download

                checkpoint = Path(
                    hf_hub_download(
                        "antofuller/CROMA",
                        "CROMA_base.pt",
                        revision=CHECKPOINT_REVISION,
                        local_dir=cache_dir,
                    )
                )
        checkpoint = Path(checkpoint)
        digest = sha256(checkpoint)
        if digest != CHECKPOINT_SHA256:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        from ._vendor.croma import PretrainedCROMA

        model = PretrainedCROMA(
            pretrained_path=str(checkpoint), size="base", modality="both", image_resolution=120
        )
        return cls(
            model,
            device=device,
            provenance={
                "kind": "CROMA-Base",
                "pretrained_verified": True,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": digest,
                "checkpoint_url": CHECKPOINT_URL,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "croma_source_revision": CROMA_REVISION,
                "inference_implementation_sha256": sha256(
                    Path(__file__).parent / "_vendor/croma.py"
                ),
            },
        )

    def __call__(
        self, optical_images: torch.Tensor, SAR_images: torch.Tensor, *, batch_size: int = 4
    ) -> dict[str, torch.Tensor]:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for name, value, channels in [
            ("optical_images", optical_images, 12),
            ("SAR_images", SAR_images, 2),
        ]:
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 4
                or tuple(value.shape[1:]) != (channels, 120, 120)
                or len(value) < 1
                or value.dtype != torch.float32
            ):
                raise ValueError(f"{name} must be nonempty float32 (N,{channels},120,120)")
            if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError(f"{name} must be finite and normalized to [0,1]")
        if len(optical_images) != len(SAR_images):
            raise ValueError("Optical and SAR batch lengths must match")
        pieces = {key: [] for key in FEATURE_KEYS}
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(optical_images), batch_size):
                optical = optical_images[start : start + batch_size].to(self.device)
                sar = SAR_images[start : start + batch_size].to(self.device)
                result = self.model(optical_images=optical, SAR_images=sar)
                if set(result) != set(FEATURE_KEYS):
                    raise ValueError("Model did not return all six CROMA feature keys")
                for key, value in result.items():
                    shape = (
                        (len(optical), 225, self.dimension)
                        if key.endswith("encodings")
                        else (len(optical), self.dimension)
                    )
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                        raise ValueError(f"Unexpected model output shape for {key}")
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Nonfinite model output: {key}")
                    pieces[key].append(value.detach().to(device="cpu", dtype=torch.float32))
        return {key: torch.cat(value, dim=0) for key, value in pieces.items()}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _within(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("Expected a relative batch path")
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Batch path escapes input directory")
    return path


def extract_features(
    root: Path,
    output: Path,
    extractor: CromaFeatureExtractor,
    *,
    batch_size: int = 4,
    progress=None,
) -> dict:
    """Consume preprocessing exports; retain sample order and per-batch provenance."""
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the input directory")
    source_bytes = (root / "manifest.json").read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    if source.get("format_version") != 1 or not source.get("batches"):
        raise ValueError("Expected preprocessing manifest version 1 with batches")
    profile = source.get("channel_profile", {})
    if profile.get("optical") != list(DEFAULT_PROFILE.optical) or profile.get("sar") != list(
        DEFAULT_PROFILE.sar
    ):
        raise ValueError("Unexpected input channel order")
    if source.get("normalization_profile") != NORMALIZATION_PROFILE:
        raise ValueError("Expected normalized CROMA preprocessing inputs, not raw tensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    seen, batches, total = set(), [], 0
    try:
        for number, batch in enumerate(source["batches"]):
            folder = _within(root, batch["directory"])
            for name in ("batch.json", "croma_inputs.pt"):
                if sha256(_within(folder, name)) != batch.get("sha256", {}).get(name):
                    raise ValueError(f"Input hash mismatch: {folder / name}")
            info = json.loads((folder / "batch.json").read_text())
            samples = info["samples"]
            if (
                not samples
                or len(samples) != batch["sample_count"]
                or len(samples) != info["sample_count"]
                or batch["start_index"] != total
            ):
                raise ValueError("Inconsistent sample counts/order in input manifest")
            for sample in samples:
                if (
                    not sample.get("patch_id")
                    or not isinstance(sample.get("patch_id"), str)
                    or sample["patch_id"] in seen
                ):
                    raise ValueError("Missing or duplicate sample ID")
                if not sample.get("s1_name") or not isinstance(sample.get("s1_name"), str):
                    raise ValueError("Missing SAR sample ID")
                seen.add(sample["patch_id"])
            inputs = torch.load(folder / "croma_inputs.pt", map_location="cpu", weights_only=True)
            if not isinstance(inputs, dict) or set(inputs) != {"optical_images", "SAR_images"}:
                raise ValueError("Expected optical_images and SAR_images tensor dictionary")
            if len(inputs["optical_images"]) != len(samples):
                raise ValueError("Tensor count does not match sample metadata")
            features = extractor(
                inputs["optical_images"], inputs["SAR_images"], batch_size=batch_size
            )
            relative = "." if len(source["batches"]) == 1 else f"batch-{number:06d}"
            destination = temporary / relative
            destination.mkdir(exist_ok=True)
            torch.save(features, destination / "features.pt")
            reloaded = torch.load(destination / "features.pt", weights_only=True)
            if any(not torch.equal(value, reloaded[key]) for key, value in features.items()):
                raise ValueError("Saved features did not reload identically")
            _write_json(
                destination / "batch.json",
                {
                    "sample_count": len(samples),
                    "samples": samples,
                    "shapes": {key: list(value.shape) for key, value in features.items()},
                    "dtype": "float32",
                    "source_batch": batch,
                    "spatial_grid": {
                        "height": 15,
                        "width": 15,
                        "token_order": "row-major",
                        "token_index": "row * 15 + column",
                        "input_patch_pixels": [8, 8],
                        "note": "Tokens are spatially anchored but include whole-image context through attention. Ground coordinates come from each sample transform.",
                    },
                },
            )
            batches.append(
                {
                    "directory": relative,
                    "sample_count": len(samples),
                    "start_index": total,
                    "sha256": {
                        name: sha256(destination / name) for name in ("features.pt", "batch.json")
                    },
                }
            )
            total += len(samples)
            if progress:
                progress(f"Extracted features for {total}/{source['sample_count']} areas")
        if total != source["sample_count"]:
            raise ValueError("Total sample count does not match input manifest")
        result = {
            "format_version": 1,
            "sample_count": total,
            "model": extractor.provenance,
            "feature_dimension": extractor.dimension,
            "device": str(extractor.device),
            "inference_batch_size": batch_size,
            "eval_mode": True,
            "inference_mode": True,
            "source_manifest_path": str(root / "manifest.json"),
            "source_manifest_sha256": source_digest,
            "channel_profile": profile,
            "normalization_profile": NORMALIZATION_PROFILE,
            "torch_version": version("torch"),
            "batches": batches,
        }
        _write_json(temporary / "manifest.json", result)
        _write_json(
            temporary / "validation.json",
            {
                "passed": True,
                "sample_count": total,
                "model_inference_run": True,
                "checkpoint_inference_run": extractor.provenance.get("pretrained_verified", False),
                "pretrained_checkpoint_verified": extractor.provenance.get(
                    "pretrained_verified", False
                ),
                "checks": [
                    "input_hashes",
                    "sample_pairing",
                    "tensor_shapes_and_range",
                    "six_output_keys",
                    "finite_features",
                    "feature_reload_equality",
                ],
                "channel_semantics_independently_validated": False,
                "manifest_sha256": sha256(temporary / "manifest.json"),
            },
        )
        if sha256(root / "manifest.json") != source_digest:
            raise ValueError("Input manifest changed during inference")
        if output.exists():
            raise FileExistsError(f"Output appeared during extraction: {output}")
        temporary.rename(output)
        return result
    except BaseException:
        shutil.rmtree(temporary)
        raise
