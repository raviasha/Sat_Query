"""Lazy CROMA and coverage-head runtime for validated assistant inputs."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..features import CHECKPOINT_SHA256, CHECKPOINT_URL
from ..prediction import load_head
from ..prediction_data import FeatureBatches, class_schema
from ..preprocessing import CROMA_REVISION, DEFAULT_PROFILE, NORMALIZATION_PROFILE, sha256
from .inputs import InputBundle

_CAPABILITY_KEYS = {
    "optical": "optical_encodings",
    "SAR": "SAR_encodings",
    "joint": "joint_encodings",
}
_MODEL_MODALITIES = {"optical": "optical", "SAR": "SAR", "joint": "both"}


class CapabilityError(RuntimeError):
    """Raised when a request needs an installed, compatible head that is unavailable."""


@dataclass(frozen=True)
class SceneResult:
    """Spatial coverage estimates and display context for one analyzed scene."""

    id: str
    modality: str
    acquired: date | None
    coverage: np.ndarray
    preview: np.ndarray | None
    grid: dict[str, Any]
    provenance: dict[str, Any]


def _channel_profile() -> dict[str, Any]:
    return {
        "optical": list(DEFAULT_PROFILE.optical),
        "sar": list(DEFAULT_PROFILE.sar),
        "evidence": DEFAULT_PROFILE.evidence,
        "evidence_level": DEFAULT_PROFILE.evidence_level,
    }


class CoverageRuntime:
    """Load CROMA modes and compatible coverage heads only when a request needs them."""

    def __init__(self, checkpoint: Path, heads: dict[str, Path], device: str = "cpu"):
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps, or cuda")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if not isinstance(heads, dict) or any(key not in _CAPABILITY_KEYS for key in heads):
            raise ValueError("heads may contain only optical, SAR, and joint paths")
        self.checkpoint = Path(checkpoint)
        self.heads = {key: Path(path) for key, path in heads.items()}
        self.device = torch.device(device)
        self._models: dict[str, torch.nn.Module] = {}
        self._loaded_heads: dict[str, tuple[torch.nn.Module, dict[str, Any], str]] = {}

    @property
    def feature_contract(self) -> dict[str, Any]:
        """The exact checkpoint, source, channel, and normalization inference contract."""
        return {
            "checkpoint_sha256": self._checkpoint_digest(),
            "croma_source_revision": CROMA_REVISION,
            "inference_implementation_sha256": sha256(
                Path(__file__).parents[1] / "_vendor" / "croma.py"
            ),
            "channel_profile": _channel_profile(),
            "normalization_profile": NORMALIZATION_PROFILE,
        }

    def _checkpoint_digest(self) -> str:
        if not self.checkpoint.is_file() or self.checkpoint.is_symlink():
            raise ValueError("CROMA checkpoint must be a regular file")
        digest = sha256(self.checkpoint)
        if digest != CHECKPOINT_SHA256:
            raise ValueError("CROMA checkpoint hash does not match the pinned official checkpoint")
        return digest

    @property
    def available_capabilities(self) -> list[str]:
        """Return installed heads that match this runtime's complete feature contract."""
        contract = self.feature_contract
        available = []
        for capability in ("optical", "SAR", "joint"):
            if capability not in self.heads or not self.heads[capability].is_file():
                continue
            try:
                self._head(
                    capability,
                    contract=contract,
                    feature_key=_CAPABILITY_KEYS[capability],
                )
            except CapabilityError:
                continue
            available.append(capability)
        return available

    def _create_model(self, modality: str) -> torch.nn.Module:
        from .._vendor.croma import PretrainedCROMA

        return PretrainedCROMA(
            pretrained_path=str(self.checkpoint),
            size="base",
            modality=modality,
            image_resolution=120,
        )

    def _model(self, capability: str) -> torch.nn.Module:
        modality = _MODEL_MODALITIES[capability]
        if modality not in self._models:
            before = self._checkpoint_digest()
            model = self._create_model(modality)
            if self._checkpoint_digest() != before:
                raise ValueError("CROMA checkpoint changed while loading")
            self._models[modality] = model.to(device=self.device, dtype=torch.float32).eval()
        return self._models[modality]

    def _head(
        self,
        capability: str,
        *,
        contract: dict[str, Any],
        feature_key: str,
    ) -> tuple[torch.nn.Module, dict[str, Any], str]:
        path = self.heads.get(capability)
        if path is None or not path.is_file():
            raise CapabilityError(
                f"{capability} coverage is unavailable because no matching head is installed"
            )
        digest = sha256(path)
        cached = self._loaded_heads.get(capability)
        if cached is None or cached[2] != digest:
            try:
                model, metadata = load_head(path)
            except (
                EOFError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                pickle.UnpicklingError,
            ) as exc:
                raise CapabilityError(f"{capability} head is unreadable or unsupported") from exc
            cached = (model.to(device=self.device, dtype=torch.float32).eval(), metadata, digest)
            self._loaded_heads[capability] = cached
        model, metadata, loaded_digest = cached
        if metadata.get("feature_key") != feature_key:
            raise CapabilityError(
                f"{capability} head uses {metadata.get('feature_key')!r}, expected {feature_key!r}"
            )
        if metadata.get("feature_contract") != contract:
            raise CapabilityError(f"{capability} head does not match the CROMA feature contract")
        if metadata.get("classes") != class_schema():
            raise CapabilityError(f"{capability} head does not match the 19-class schema")
        return model, metadata, loaded_digest

    @staticmethod
    def _check_features(features: torch.Tensor, feature_key: str, count: int) -> torch.Tensor:
        if (
            not isinstance(features, torch.Tensor)
            or tuple(features.shape) != (count, 225, 768)
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            raise ValueError(f"CROMA returned invalid {feature_key} features")
        return features

    def _predict(
        self,
        capability: str,
        features: torch.Tensor,
        contract: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        feature_key = _CAPABILITY_KEYS[capability]
        features = self._check_features(features, feature_key, len(features))
        head, metadata, head_digest = self._head(
            capability, contract=contract, feature_key=feature_key
        )
        with torch.inference_mode():
            predictions = head.predict(features.to(self.device)).detach().to("cpu", torch.float32)
        if sha256(self.heads[capability]) != head_digest:
            raise ValueError("Coverage head changed during inference")
        if (
            tuple(predictions.shape) != (len(features), 225, 19)
            or not torch.isfinite(predictions).all()
            or (predictions < 0).any()
            or not torch.allclose(
                predictions.sum(-1), torch.ones(len(features), 225), atol=1e-6, rtol=0
            )
        ):
            raise ValueError("Coverage head returned invalid class fractions")
        return predictions.reshape(-1, 15, 15, 19).numpy(), metadata

    @staticmethod
    def _comparison_summary(values: np.ndarray) -> list[dict[str, Any]]:
        means = values.mean(axis=(0, 1))
        order = np.argsort(-means, kind="stable")[:3]
        classes = class_schema()
        return [
            {"name": classes[index]["name"], "estimated_fraction": float(means[index])}
            for index in order
        ]

    @staticmethod
    def _validate_observation(item, bundle_grid: dict[str, Any]) -> None:
        expected = {
            "optical": ("Sentinel-2", (12, 120, 120), list(DEFAULT_PROFILE.optical)),
            "SAR": ("Sentinel-1", (2, 120, 120), list(DEFAULT_PROFILE.sar)),
        }
        if item.modality not in expected:
            raise ValueError("Observation modality must be optical or SAR")
        sensor, shape, bands = expected[item.modality]
        if (
            item.sensor != sensor
            or not isinstance(item.metadata, dict)
            or item.metadata.get("bands") != bands
        ):
            raise ValueError(f"{item.id}: sensor or channel profile does not match {item.modality}")
        if item.modality == "SAR" and item.metadata.get("radiometry") != "db":
            raise ValueError(f"{item.id}: SAR radiometry must be db")
        if (
            not isinstance(item.normalized, torch.Tensor)
            or tuple(item.normalized.shape) != shape
            or item.normalized.dtype != torch.float32
            or not torch.isfinite(item.normalized).all()
            or item.normalized.min() < 0
            or item.normalized.max() > 1
        ):
            raise ValueError(f"{item.id}: normalized tensor violates the CROMA input contract")
        if (
            not isinstance(item.normalization, dict)
            or item.normalization.get("profile") != NORMALIZATION_PROFILE
        ):
            raise ValueError(f"{item.id}: normalization profile does not match CROMA")
        if item.grid != bundle_grid:
            raise ValueError(f"{item.id}: observation grid differs from the bundle grid")

    def analyze(self, bundle: InputBundle) -> list[SceneResult]:
        """Analyze validated live inputs, returning one result or an ordered temporal pair."""
        if not isinstance(bundle, InputBundle):
            raise TypeError("bundle must be an InputBundle")
        expected_count = 2 if bundle.kind in ("cross_modal", "temporal") else 1
        if (
            bundle.kind not in ("single", "cross_modal", "temporal")
            or len(bundle.observations) != expected_count
        ):
            raise ValueError("InputBundle kind and observation count are inconsistent")
        for item in bundle.observations:
            self._validate_observation(item, bundle.grid)
        if bundle.kind == "temporal":
            before, after = (item.acquired for item in bundle.observations)
            if not isinstance(before, date) or not isinstance(after, date) or before >= after:
                raise ValueError(
                    "Temporal observations must have distinct dates in ascending order"
                )
        contract = self.feature_contract

        if bundle.kind == "cross_modal":
            by_modality = {item.modality: item for item in bundle.observations}
            if set(by_modality) != {"optical", "SAR"}:
                raise ValueError("Cross-modal bundle requires one optical and one SAR observation")
            optical, sar = by_modality["optical"], by_modality["SAR"]
            self._head("joint", contract=contract, feature_key=_CAPABILITY_KEYS["joint"])
            model = self._model("joint")
            with torch.inference_mode():
                outputs = model(
                    optical_images=optical.normalized[None].to(self.device),
                    SAR_images=sar.normalized[None].to(self.device),
                )
            feature_key = _CAPABILITY_KEYS["joint"]
            features = self._check_features(outputs.get(feature_key), feature_key, 1)
            coverage, metadata = self._predict("joint", features, contract)
            provenance: dict[str, Any] = self._provenance("joint", metadata, contract)
            provenance.update(
                {
                    "source_observations": [optical.id, sar.id],
                    "joint_encoder_inputs": {"optical": optical.id, "SAR": sar.id},
                    "fusion": "Both optical and SAR tensors fed the CROMA joint encoder.",
                }
            )
            if set(self.available_capabilities) == {"optical", "SAR", "joint"}:
                comparison: dict[str, Any] = {}
                for capability in ("optical", "SAR", "joint"):
                    key = _CAPABILITY_KEYS[capability]
                    values, _ = self._predict(
                        capability,
                        self._check_features(outputs.get(key), key, 1),
                        contract,
                    )
                    comparison[capability] = self._comparison_summary(values[0])
                provenance["sensor_comparison"] = {
                    "top_scene_coverage": comparison,
                    "interpretation": (
                        "Descriptive optical-only, SAR-only, and fused estimates; this is not an "
                        "accuracy comparison and no fusion improvement is claimed."
                    ),
                }
            return [
                SceneResult(
                    id=f"{optical.id}+{sar.id}",
                    modality="joint",
                    acquired=max(optical.acquired, sar.acquired),
                    coverage=coverage[0],
                    preview=optical.preview,
                    grid=dict(bundle.grid),
                    provenance=provenance,
                )
            ]

        modality = bundle.observations[0].modality
        if any(item.modality != modality for item in bundle.observations):
            raise ValueError("Single and temporal bundles must use one modality")
        capability = modality
        feature_key = _CAPABILITY_KEYS[capability]
        self._head(capability, contract=contract, feature_key=feature_key)
        inputs = torch.stack([item.normalized for item in bundle.observations]).to(self.device)
        model = self._model(capability)
        kwargs = {"optical_images": inputs} if modality == "optical" else {"SAR_images": inputs}
        with torch.inference_mode():
            outputs = model(**kwargs)
        features = self._check_features(outputs.get(feature_key), feature_key, len(inputs))
        coverage, metadata = self._predict(capability, features, contract)
        provenance = self._provenance(capability, metadata, contract)
        return [
            SceneResult(
                id=item.id,
                modality=item.modality,
                acquired=item.acquired,
                coverage=coverage[index],
                preview=item.preview,
                grid=dict(bundle.grid),
                provenance={**provenance, "source_observations": [item.id]},
            )
            for index, item in enumerate(bundle.observations)
        ]

    def _provenance(
        self,
        capability: str,
        metadata: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "source": "live_croma_inference",
            "model": "CROMA-Base",
            "model_mode": _MODEL_MODALITIES[capability],
            "feature_key": _CAPABILITY_KEYS[capability],
            "feature_contract": contract,
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "checkpoint_source": CHECKPOINT_URL,
            "head_sha256": sha256(self.heads[capability]),
            "head_training_mode": metadata.get("training_mode"),
            "head_evaluation_scope": metadata.get("evaluation_scope"),
            "prediction_units": "estimated class fractions of cell area",
        }

    def analyze_cached(
        self,
        features: Path,
        *,
        selected_index: int,
        capability: str = "joint",
    ) -> SceneResult:
        """Run a supplied head on one verified saved feature sample without loading CROMA."""
        if capability not in _CAPABILITY_KEYS:
            raise CapabilityError("Cached capability must be optical, SAR, or joint")
        if (
            not isinstance(selected_index, int)
            or isinstance(selected_index, bool)
            or selected_index < 0
        ):
            raise ValueError("selected_index must be a nonnegative integer")
        feature_key = _CAPABILITY_KEYS[capability]
        reader = FeatureBatches(features, feature_key=feature_key)
        if reader.contract != self.feature_contract:
            raise CapabilityError("Cached features do not match the pinned CROMA runtime contract")
        offset = 0
        for _, info, batch_features in reader:
            if selected_index >= offset + len(batch_features):
                offset += len(batch_features)
                continue
            local = selected_index - offset
            chosen = batch_features[local : local + 1]
            coverage, metadata = self._predict(capability, chosen, reader.contract)
            sample = info["samples"][local]
            transform = sample.get("transform")
            bounds = sample.get("bounds")
            if (
                not isinstance(transform, list)
                or len(transform) != 6
                or not isinstance(bounds, list)
                or len(bounds) != 4
            ):
                raise ValueError("Cached sample lacks geospatial transform and bounds")
            acquired = sample.get("acquired")
            try:
                acquired_date = date.fromisoformat(acquired) if acquired is not None else None
            except (TypeError, ValueError) as exc:
                raise ValueError("Cached sample acquired date is invalid") from exc
            provenance = {
                **self._provenance(capability, metadata, reader.contract),
                "source": "cached_features",
                "feature_manifest_sha256": reader.digest,
                "selected_index": selected_index,
            }
            return SceneResult(
                id=sample["patch_id"],
                modality=capability,
                acquired=acquired_date,
                coverage=coverage[0],
                preview=None,
                grid={
                    "crs": sample["crs"],
                    "transform": [float(value) for value in transform],
                    "bounds": [float(value) for value in bounds],
                    "width": 120,
                    "height": 120,
                    "pixel_size_metres": 10.0,
                },
                provenance=provenance,
            )
        raise IndexError(f"selected_index {selected_index} is outside the feature dataset")
