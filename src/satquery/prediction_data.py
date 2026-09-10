"""Verified streaming readers for saved features and aligned coverage targets."""

import hashlib
import io
import json
from pathlib import Path

import torch

from .preprocessing import sha256
from .targets import CLASSES

FEATURE_KEYS = ("joint_encodings", "optical_encodings", "SAR_encodings")


def class_schema():
    return [
        {"index": i, "name": name, "clc_codes": list(codes)}
        for i, (name, codes) in enumerate(CLASSES)
    ]


def feature_contract(manifest):
    model = manifest.get("model", {})
    if (
        not model.get("checkpoint_sha256")
        or not manifest.get("channel_profile")
        or not manifest.get("normalization_profile")
    ):
        raise ValueError("Feature manifest lacks checkpoint/channel/normalization contract")
    return {
        "checkpoint_sha256": model["checkpoint_sha256"],
        "croma_source_revision": model.get("croma_source_revision"),
        "inference_implementation_sha256": model.get("inference_implementation_sha256"),
        "channel_profile": manifest["channel_profile"],
        "normalization_profile": manifest["normalization_profile"],
    }


def read_manifest(root):
    root = Path(root).resolve()
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format_version") != 1 or not manifest.get("batches"):
        raise ValueError("Expected manifest version 1 with batches")
    return root, manifest, hashlib.sha256(raw).hexdigest()


def checked_file(root, batch, name):
    directory = batch["directory"]
    if not isinstance(directory, str) or Path(directory).is_absolute():
        raise ValueError("Expected relative batch directory")
    path = (root / directory / name).resolve()
    if root not in path.parents:
        raise ValueError("Batch path escapes input directory")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != batch.get("sha256", {}).get(name):
        raise ValueError(f"Input hash mismatch: {path}")
    return raw


class FeatureBatches:
    def __init__(self, root, *, feature_key="joint_encodings"):
        if feature_key not in FEATURE_KEYS:
            raise ValueError("Expected spatial feature key")
        self.root, self.manifest, self.digest = read_manifest(root)
        if self.manifest.get("feature_dimension") != 768:
            raise ValueError("Expected 768-dimensional CROMA features")
        self.feature_key = feature_key
        self.contract = feature_contract(self.manifest)

    def check_unchanged(self):
        if sha256(self.root / "manifest.json") != self.digest:
            raise ValueError("Feature manifest changed during operation")

    def __iter__(self):
        self.check_unchanged()
        total, seen = 0, set()
        for batch in self.manifest["batches"]:
            info = json.loads(checked_file(self.root, batch, "batch.json"))
            n = batch["sample_count"]
            if (
                not isinstance(n, int)
                or n < 1
                or batch["start_index"] != total
                or info["sample_count"] != n
                or len(info["samples"]) != n
            ):
                raise ValueError("Inconsistent feature sample counts/order")
            for sample in info["samples"]:
                if (
                    not isinstance(sample.get("patch_id"), str)
                    or not sample["patch_id"]
                    or not isinstance(sample.get("s1_name"), str)
                    or not sample["s1_name"]
                    or sample["patch_id"] in seen
                ):
                    raise ValueError("Missing or duplicate sample identity")
                seen.add(sample["patch_id"])
            if any(
                info.get("spatial_grid", {}).get(k) != v
                for k, v in {
                    "height": 15,
                    "width": 15,
                    "token_order": "row-major",
                    "input_patch_pixels": [8, 8],
                }.items()
            ):
                raise ValueError("Expected row-major 15x15 feature grid")
            tensors = torch.load(
                io.BytesIO(checked_file(self.root, batch, "features.pt")),
                weights_only=True,
                map_location="cpu",
            )
            x = tensors[self.feature_key]
            if (
                tuple(x.shape) != (n, 225, 768)
                or x.dtype != torch.float32
                or not torch.isfinite(x).all()
            ):
                raise ValueError("Invalid spatial feature tensor")
            total += n
            yield batch, info, x.detach()
        if total != self.manifest["sample_count"]:
            raise ValueError("Feature total count mismatch")
        self.check_unchanged()


class TrainingPairs:
    """Yield fully labeled tokens from official train samples, one source batch at a time."""

    def __init__(self, features, targets, *, feature_key="joint_encodings", demo_fit_all=False):
        self.demo_fit_all = demo_fit_all
        self.features = FeatureBatches(features, feature_key=feature_key)
        self.root, self.manifest, self.digest = read_manifest(targets)
        if (
            self.manifest.get("target_profile") != "bigearthnet19_full_patch_fractions_v1"
            or self.manifest.get("classes") != class_schema()
            or self.manifest.get("feature_manifest_sha256") != self.features.digest
            or self.manifest.get("sample_count") != self.features.manifest["sample_count"]
            or len(self.manifest["batches"]) != len(self.features.manifest["batches"])
        ):
            raise ValueError("Targets do not match feature manifest or class schema")

    def iter_batches(self):
        """Yield verified samples, spatial features, fractions and eligibility masks."""
        if sha256(self.root / "manifest.json") != self.digest:
            raise ValueError("Target manifest changed")
        for (fb, fi, x), tb in zip(self.features, self.manifest["batches"], strict=True):
            ti = json.loads(checked_file(self.root, tb, "batch.json"))
            if (
                tb["start_index"] != fb["start_index"]
                or tb["sample_count"] != fb["sample_count"]
                or tb.get("feature_batch") != fb
                or ti["samples"] != fi["samples"]
                or ti["sample_count"] != fb["sample_count"]
                or ti["feature_file_sha256"] != fb["sha256"]["features.pt"]
            ):
                raise ValueError("Feature/target sample alignment mismatch")
            values = torch.load(
                io.BytesIO(checked_file(self.root, tb, "targets.pt")),
                weights_only=True,
                map_location="cpu",
            )
            y, mask = values["class_fractions"], values["fully_labeled_mask"]
            if (
                tuple(y.shape) != (len(x), 225, 19)
                or y.dtype != torch.float32
                or tuple(mask.shape) != (len(x), 225)
                or mask.dtype != torch.bool
                or not torch.isfinite(y).all()
                or (y < 0).any()
                or (y > 1).any()
                or not torch.equal(mask, values["unlabeled_counts"] == 0)
                or not torch.allclose(
                    y.sum(-1) + values["unlabeled_fraction"],
                    torch.ones_like(mask, dtype=torch.float32),
                    atol=1e-6,
                    rtol=0,
                )
            ):
                raise ValueError("Invalid coverage targets or mask")
            yield fi["samples"], x, y, mask
        if sha256(self.root / "manifest.json") != self.digest:
            raise ValueError("Target manifest changed")

    def __iter__(self):
        for samples, x, y, mask in self.iter_batches():
            selected = (
                torch.tensor([self.demo_fit_all or s.get("split") == "train" for s in samples])[
                    :, None
                ]
                & mask
            )
            if selected.any():
                yield x[selected], y[selected]
