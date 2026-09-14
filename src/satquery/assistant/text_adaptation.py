"""Experimental scene-level CROMA-to-text adaptation with strict split quarantine.

The adapter aligns mean-pooled scene features with frozen text embeddings.  It is
not an open-ended captioning or visual-question-answering model, and every score
returned by this module is a cosine similarity rather than calibrated confidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from satquery.prediction_data import FeatureBatches

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512
DEFAULT_EXCLUDED_ANNOTATION_IDS = (17046, 2970546)
ALLOWED_SPLITS = ("train", "validation", "test")
SCORE_KIND = "cosine_similarity_not_calibrated_confidence"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _semantic_record_payload(manifest: Mapping[str, object]) -> dict:
    return {
        "feature_key": manifest.get("feature_key"),
        "requested_splits": manifest.get("requested_splits"),
        "image_records": manifest.get("image_records"),
        "caption_records": manifest.get("caption_records"),
    }


def _write_json_exclusive(path: Path, value: object) -> None:
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(serialized)


def _safe_provenance(value: Mapping[str, object] | None) -> dict:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("kind"), str)
        or not value["kind"].strip()
    ):
        raise TypeError("Embedding provenance with a nonempty kind is required")
    safe_metadata_names = {
        "authentication",
        "privatekeysource",
        "tokencount",
    }
    credential_components = {
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "token",
    }
    item_count = 0

    def key_components(key: str) -> tuple[str, ...]:
        separated_acronyms = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
        separated_words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated_acronyms)
        return tuple(
            component.casefold() for component in re.findall(r"[A-Za-z0-9]+", separated_words)
        )

    def is_credential_key(key: str) -> bool:
        normalized = "".join(character for character in key.casefold() if character.isalnum())
        if normalized in safe_metadata_names:
            return False
        components = key_components(key)
        if any(component in credential_components for component in components):
            return True
        return any(
            first in {"api", "private"} and second == "key"
            for first, second in pairwise(components)
        )

    def validate(item: object, depth: int) -> None:
        nonlocal item_count
        item_count += 1
        if item_count > 64 or depth > 6:
            raise ValueError("Embedding provenance must be bounded")
        if isinstance(item, Mapping):
            if len(item) > 32:
                raise ValueError("Embedding provenance must be bounded")
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError("Embedding provenance object keys must be strings")
                if is_credential_key(key):
                    raise ValueError("Embedding provenance must not contain credentials")
                validate(child, depth + 1)
        elif isinstance(item, list):
            if len(item) > 32:
                raise ValueError("Embedding provenance must be bounded")
            for child in item:
                validate(child, depth + 1)
        elif isinstance(item, str):
            if len(item) > 1024:
                raise ValueError("Embedding provenance must be bounded")
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Embedding provenance must be JSON-safe")
        elif item is not None and not isinstance(item, (bool, int)):
            raise TypeError("Embedding provenance must contain JSON-safe values")

    validate(value, 0)
    encoded = json.dumps(dict(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode()) > 4096:
        raise ValueError("Embedding provenance must be bounded")
    serialized = json.loads(encoded)
    serialized["kind"] = serialized["kind"].strip()
    return serialized


def _cache_key(text: str, model: str, dimensions: int) -> str:
    payload = json.dumps(
        {"content": text, "model": model, "dimensions": dimensions},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_cache(path: Path, model: str, dimensions: int) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    if not path.exists():
        return values
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
            vector = torch.tensor(record["embedding"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid embedding cache record {number}") from exc
        expected = _cache_key(record.get("content"), model, dimensions)
        if (
            record.get("cache_key") != expected
            or record.get("model") != model
            or record.get("dimensions") != dimensions
            or tuple(vector.shape) != (dimensions,)
            or not torch.isfinite(vector).all()
        ):
            raise ValueError(f"Embedding cache contract mismatch at record {number}")
        if expected in values and not torch.equal(values[expected], vector):
            raise ValueError("Conflicting duplicate embedding cache entry")
        values[expected] = vector
    return values


def _embed_requested(
    texts: Sequence[str],
    cache_path: Path,
    embedder: Callable,
    *,
    model: str,
    dimensions: int,
    batch_size: int,
) -> tuple[torch.Tensor, int, int]:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("embedding_batch_size must be positive")
    cache = _load_cache(cache_path, model, dimensions)
    keys = [_cache_key(text, model, dimensions) for text in texts]
    hits = sum(key in cache for key in keys)
    missing_texts: list[str] = []
    missing_keys: list[str] = []
    already_requested = set(cache)
    for text, key in zip(texts, keys, strict=True):
        if key not in already_requested:
            missing_texts.append(text)
            missing_keys.append(key)
            already_requested.add(key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(missing_texts), batch_size):
        batch_texts = missing_texts[start : start + batch_size]
        batch_keys = missing_keys[start : start + batch_size]
        raw = embedder(batch_texts, model=model, dimensions=dimensions)
        vectors = torch.as_tensor(raw, dtype=torch.float32)
        if (
            tuple(vectors.shape) != (len(batch_texts), dimensions)
            or not torch.isfinite(vectors).all()
        ):
            raise ValueError("Embedder returned invalid vectors")
        with cache_path.open("a") as stream:
            for text, key, vector in zip(batch_texts, batch_keys, vectors, strict=True):
                record = {
                    "cache_key": key,
                    "content": text,
                    "model": model,
                    "dimensions": dimensions,
                    "embedding": vector.tolist(),
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                cache[key] = vector.clone()
    return torch.stack([cache[key] for key in keys]), hits, len(missing_texts)


@dataclass(frozen=True)
class PreparedTextPairs:
    image_features: torch.Tensor
    caption_embeddings: torch.Tensor
    caption_image_indices: torch.Tensor
    image_records: tuple[dict, ...]
    caption_records: tuple[dict, ...]
    manifest: dict


def _read_verified_annotations(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    report_path, parquet_path = root / "report.json", root / "annotations.parquet"
    if not report_path.is_file() or not parquet_path.is_file():
        raise ValueError("Expected match_annotations report.json and annotations.parquet")
    report = json.loads(report_path.read_text())
    required_report = ("source_revision", "source_sha256", "selected_metadata_sha256")
    if (
        report.get("format_version") != 1
        or report.get("passed") is not True
        or any(not report.get(key) for key in required_report)
    ):
        raise ValueError("Annotation source is not a verified match_annotations output")
    parquet_digest = _sha256(parquet_path)
    if report.get("files", {}).get("annotations.parquet") != parquet_digest:
        raise ValueError("Verified annotation output hash mismatch")
    frame = pd.read_parquet(parquet_path)
    required_columns = {
        "ID",
        "patch_id",
        "s1_name",
        "split",
        "type",
        "output",
        "image_split",
        "use_partition",
        "training_eligible",
    }
    if not required_columns.issubset(frame.columns):
        raise ValueError("Annotation table lacks verified matching fields")
    if frame.ID.duplicated().any():
        raise ValueError("Duplicate annotation ID")
    source_hashes = {
        "annotations_parquet_sha256": parquet_digest,
        "annotation_report_sha256": _sha256(report_path),
        "annotation_source_sha256": report["source_sha256"],
        "selected_metadata_sha256": report["selected_metadata_sha256"],
    }
    return frame, report, source_hashes


def _feature_scenes(features: FeatureBatches) -> tuple[list[dict], torch.Tensor]:
    records: list[dict] = []
    means: list[torch.Tensor] = []
    for _, info, spatial in features:
        records.extend(copy.deepcopy(info["samples"]))
        means.extend(spatial.mean(dim=1).unbind())
    if not records:
        raise ValueError("No feature scenes")
    return records, torch.stack(means)


def _eligible_split(row: Mapping[str, object]) -> str | None:
    split = row["split"]
    partition = row["use_partition"]
    image_split = row["image_split"]
    if partition in ("bench_holdout", "split_conflict_holdout") or split == "bench":
        return None
    if split not in ALLOWED_SPLITS or partition != split or image_split != split:
        return None
    if split == "train" and row["training_eligible"] is not True:
        return None
    if split != "train" and row["training_eligible"] is not False:
        return None
    return str(split)


def prepare_text_pairs(
    annotations: str | Path,
    features: str | Path,
    output: str | Path,
    *,
    requested_splits: Sequence[str] = ("train", "validation"),
    feature_key: str = "joint_encodings",
    embedder: Callable,
    embedding_provenance: Mapping[str, object] | None,
    excluded_annotation_ids: Iterable[int | str] = DEFAULT_EXCLUDED_ANNOTATION_IDS,
    embedding_model: str = EMBEDDING_MODEL,
    embedding_dimensions: int = EMBEDDING_DIMENSIONS,
    embedding_batch_size: int = 128,
) -> dict:
    """Prepare integrity-bound scene/caption pairs and a resumable embedding cache."""
    annotation_root = Path(annotations).resolve()
    feature_root = Path(features).resolve()
    output = Path(output).absolute()
    if (output / "manifest.json").exists() or (output / "pairs.pt").exists():
        raise FileExistsError(f"Prepared output already complete or ambiguous: {output}")
    if embedding_model != EMBEDDING_MODEL or embedding_dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError("Expected text-embedding-3-small with 512 dimensions")
    splits = tuple(requested_splits)
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(s not in ALLOWED_SPLITS for s in splits)
    ):
        raise ValueError("requested_splits must be unique train/validation/test values")
    provenance = _safe_provenance(embedding_provenance)
    excluded = tuple(excluded_annotation_ids)
    excluded_keys = {str(item) for item in excluded}
    frame, annotation_report, source_hashes = _read_verified_annotations(annotation_root)
    feature_reader = FeatureBatches(feature_root, feature_key=feature_key)
    feature_records, pooled = _feature_scenes(feature_reader)
    source_hashes["feature_manifest_sha256"] = feature_reader.digest

    identities = {
        record["patch_id"]: (record["s1_name"], i, record)
        for i, record in enumerate(feature_records)
    }
    for row in frame.to_dict(orient="records"):
        identity = identities.get(row["patch_id"])
        if identity is None or identity[0] != row["s1_name"]:
            raise ValueError(f"Feature/annotation SAR identity mismatch for {row['patch_id']}")
        if identity[2].get("split") != row["image_split"]:
            raise ValueError(f"Feature/annotation image split mismatch for {row['patch_id']}")

    quarantine = Counter()
    candidates: list[dict] = []
    for row in frame.to_dict(orient="records"):
        if str(row["ID"]) in excluded_keys:
            quarantine["excluded_annotation_id"] += 1
            continue
        if row["type"] != "captioning":
            quarantine["non_captioning"] += 1
            continue
        split = _eligible_split(row)
        if split is None:
            if row["split"] == "bench" or row["use_partition"] in (
                "bench_holdout",
                "split_conflict_holdout",
            ):
                quarantine["bench_or_conflict"] += 1
            else:
                quarantine["ineligible_split_contract"] += 1
            continue
        if split not in splits:
            quarantine["unrequested_split"] += 1
            continue
        caption = row["output"].strip() if isinstance(row["output"], str) else ""
        if not caption:
            raise ValueError("Caption text must be nonempty")
        candidates.append(
            {
                "annotation_id": row["ID"],
                "patch_id": row["patch_id"],
                "s1_name": row["s1_name"],
                "split": split,
                "use_partition": row["use_partition"],
                "training_eligible": bool(row["training_eligible"]),
                "caption": caption,
                "source_feature_index": identities[row["patch_id"]][1],
            }
        )

    def annotation_order(record: dict) -> tuple[int, str]:
        return record["source_feature_index"], str(record["annotation_id"])

    candidates.sort(key=annotation_order)
    selected: list[dict] = []
    seen_captions: set[tuple[str, str]] = set()
    for record in candidates:
        key = record["patch_id"], " ".join(record["caption"].split()).casefold()
        if key in seen_captions:
            quarantine["same_image_duplicate_caption"] += 1
            continue
        seen_captions.add(key)
        selected.append(record)
    if not selected:
        raise ValueError("No eligible captions for requested splits")

    output.mkdir(parents=True, exist_ok=True)
    cache_contract_path = output / "embedding-cache-manifest.json"
    cache_contract = {
        "format_version": 1,
        "embedding": {"model": embedding_model, "dimensions": embedding_dimensions},
        "embedding_provenance": provenance,
        "key": "sha256(canonical content+model+dimensions)",
        "requested_splits": list(splits),
        "excluded_annotation_ids": sorted(excluded, key=str),
        "feature_key": feature_key,
        "source_hashes": source_hashes,
    }
    if cache_contract_path.exists():
        if json.loads(cache_contract_path.read_text()) != cache_contract:
            raise ValueError("Embedding cache provenance or model contract changed during resume")
    else:
        _write_json_exclusive(cache_contract_path, cache_contract)
    embeddings, hits, misses = _embed_requested(
        [record["caption"] for record in selected],
        output / "embedding-cache.jsonl",
        embedder,
        model=embedding_model,
        dimensions=embedding_dimensions,
        batch_size=embedding_batch_size,
    )

    def check_sources_unchanged() -> None:
        if (
            _sha256(annotation_root / "annotations.parquet")
            != source_hashes["annotations_parquet_sha256"]
            or _sha256(annotation_root / "report.json") != source_hashes["annotation_report_sha256"]
        ):
            raise ValueError("Annotation source changed during pair preparation")
        feature_reader.check_unchanged()
        for batch in feature_reader.manifest["batches"]:
            directory = batch["directory"]
            if not isinstance(directory, str) or Path(directory).is_absolute():
                raise ValueError("Feature source changed during pair preparation")
            folder = (feature_reader.root / directory).resolve()
            if folder != feature_reader.root and feature_reader.root not in folder.parents:
                raise ValueError("Feature source changed during pair preparation")
            for name in ("batch.json", "features.pt"):
                if _sha256(folder / name) != batch.get("sha256", {}).get(name):
                    raise ValueError("Feature source changed during pair preparation")

    check_sources_unchanged()
    source_indices = sorted({record["source_feature_index"] for record in selected})
    compact_index = {source: i for i, source in enumerate(source_indices)}
    image_features = pooled[source_indices]
    image_records = []
    for source in source_indices:
        record = copy.deepcopy(feature_records[source])
        record["source_feature_index"] = source
        image_records.append(record)
    caption_image_indices = torch.tensor(
        [compact_index[record["source_feature_index"]] for record in selected], dtype=torch.int64
    )
    semantic_payload = {
        "feature_key": feature_key,
        "requested_splits": list(splits),
        "image_records": image_records,
        "caption_records": selected,
    }
    semantic_records_sha256 = _canonical_sha256(semantic_payload)
    tensor_path = output / "pairs.pt"
    with tempfile.NamedTemporaryFile(dir=output, prefix=".pairs-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            torch.save(
                {
                    "image_features": image_features,
                    "caption_embeddings": embeddings,
                    "caption_image_indices": caption_image_indices,
                    "semantic_records_sha256": semantic_records_sha256,
                },
                stream,
            )
        temporary.replace(tensor_path)
    finally:
        temporary.unlink(missing_ok=True)

    try:
        check_sources_unchanged()
    except BaseException:
        tensor_path.unlink(missing_ok=True)
        raise

    counts_by_split = Counter(record["split"] for record in selected)
    image_counts_by_split = Counter(record["split"] for record in image_records)
    manifest = {
        "format_version": 1,
        "artifact_kind": "experimental_scene_level_image_text_pairs",
        "feature_key": feature_key,
        "feature_pooling": "mean_over_15x15_spatial_tokens",
        "embedding": {"model": embedding_model, "dimensions": embedding_dimensions},
        "embedding_provenance": provenance,
        "embedding_cache": {
            "key": "sha256(canonical content+model+dimensions)",
            "hits": hits,
            "misses": misses,
            "requested_caption_count": len(selected),
            "manifest_sha256": _sha256(cache_contract_path),
            "entries_sha256": _sha256(output / "embedding-cache.jsonl"),
        },
        "requested_splits": list(splits),
        "counts_by_split": {split: counts_by_split[split] for split in splits},
        "image_counts_by_split": {split: image_counts_by_split[split] for split in splits},
        "source_annotation_counts_by_split": dict(
            sorted(Counter(str(value) for value in frame.split).items())
        ),
        "source_image_counts_by_split": dict(
            sorted(Counter(str(record["split"]) for record in feature_records).items())
        ),
        "quarantine_counts": dict(sorted(quarantine.items())),
        "excluded_annotation_ids": sorted(excluded, key=str),
        "annotation_source": {
            "source_revision": annotation_report["source_revision"],
            "source_sha256": annotation_report["source_sha256"],
        },
        "source_hashes": source_hashes,
        "feature_contract": feature_reader.contract,
        "image_records": image_records,
        "caption_records": selected,
        "semantic_records_sha256": semantic_records_sha256,
        "tensor_shapes": {
            "image_features": list(image_features.shape),
            "caption_embeddings": list(embeddings.shape),
            "caption_image_indices": list(caption_image_indices.shape),
        },
        "pairs_sha256": _sha256(tensor_path),
        "limitations": [
            "Experimental scene-level image-text adapter",
            "No open-ended captioning or VQA quality claim",
            "Embedding similarities are not calibrated confidence",
        ],
    }
    _write_json_exclusive(output / "manifest.json", manifest)
    return manifest


def load_prepared_pairs(root: str | Path) -> PreparedTextPairs:
    root = Path(root).resolve()
    manifest_path, tensor_path = root / "manifest.json", root / "pairs.pt"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format_version") != 1
        or manifest.get("artifact_kind") != "experimental_scene_level_image_text_pairs"
        or _sha256(tensor_path) != manifest.get("pairs_sha256")
    ):
        raise ValueError("Prepared text-pair artifact integrity mismatch")
    values = torch.load(tensor_path, map_location="cpu", weights_only=True)
    image_features = values["image_features"]
    caption_embeddings = values["caption_embeddings"]
    caption_indices = values["caption_image_indices"]
    semantic_records_sha256 = _canonical_sha256(_semantic_record_payload(manifest))
    if (
        manifest.get("semantic_records_sha256") != semantic_records_sha256
        or values.get("semantic_records_sha256") != semantic_records_sha256
    ):
        raise ValueError("Prepared text-pair semantic record digest mismatch")
    if (
        tuple(image_features.shape) != tuple(manifest["tensor_shapes"]["image_features"])
        or tuple(caption_embeddings.shape) != tuple(manifest["tensor_shapes"]["caption_embeddings"])
        or tuple(caption_indices.shape) != tuple(manifest["tensor_shapes"]["caption_image_indices"])
        or image_features.shape[1:] != (768,)
        or caption_embeddings.shape[1:] != (EMBEDDING_DIMENSIONS,)
        or caption_indices.dtype != torch.int64
        or not torch.isfinite(image_features).all()
        or not torch.isfinite(caption_embeddings).all()
        or caption_indices.min().item() < 0
        or caption_indices.max().item() >= len(image_features)
    ):
        raise ValueError("Invalid prepared text-pair tensors")
    _validate_prepared_semantics(
        manifest, caption_indices, len(image_features), len(caption_embeddings)
    )
    return PreparedTextPairs(
        image_features=image_features.detach(),
        caption_embeddings=caption_embeddings.detach(),
        caption_image_indices=caption_indices.detach(),
        image_records=tuple(manifest["image_records"]),
        caption_records=tuple(manifest["caption_records"]),
        manifest=manifest,
    )


def _validate_prepared_semantics(
    manifest: Mapping[str, object],
    caption_indices: torch.Tensor,
    image_count: int,
    caption_count: int,
) -> None:
    requested_splits = manifest.get("requested_splits")
    images = manifest.get("image_records")
    captions = manifest.get("caption_records")
    if (
        not isinstance(requested_splits, list)
        or not requested_splits
        or len(set(requested_splits)) != len(requested_splits)
        or any(split not in ALLOWED_SPLITS for split in requested_splits)
        or not isinstance(images, list)
        or len(images) != image_count
        or not isinstance(captions, list)
        or len(captions) != caption_count
    ):
        raise ValueError("Invalid prepared split/record semantics")
    seen_patches: set[str] = set()
    seen_sar: set[str] = set()
    seen_feature_indices: set[int] = set()
    for record in images:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("patch_id"), str)
            or not record["patch_id"]
            or not isinstance(record.get("s1_name"), str)
            or not record["s1_name"]
            or record.get("split") not in requested_splits
            or record["patch_id"] in seen_patches
            or record["s1_name"] in seen_sar
            or not isinstance(record.get("source_feature_index"), int)
            or record["source_feature_index"] in seen_feature_indices
        ):
            raise ValueError("Invalid prepared image split/identity semantics")
        seen_patches.add(record["patch_id"])
        seen_sar.add(record["s1_name"])
        seen_feature_indices.add(record["source_feature_index"])
    seen_annotations: set[str] = set()
    for record, owner_index in zip(captions, caption_indices.tolist(), strict=True):
        owner = images[owner_index]
        annotation_id = record.get("annotation_id") if isinstance(record, dict) else None
        annotation_key = str(annotation_id) if isinstance(annotation_id, (int, str)) else ""
        if (
            not isinstance(record, dict)
            or not annotation_key.strip()
            or annotation_key in seen_annotations
            or record.get("patch_id") != owner["patch_id"]
            or record.get("s1_name") != owner["s1_name"]
            or record.get("source_feature_index") != owner["source_feature_index"]
            or record.get("split") != owner["split"]
            or record.get("use_partition") != owner["split"]
            or record.get("training_eligible") is not (owner["split"] == "train")
            or not isinstance(record.get("caption"), str)
            or not record["caption"].strip()
        ):
            raise ValueError("Invalid prepared caption split/eligibility semantics")
        seen_annotations.add(annotation_key)


class TextProjection(nn.Module):
    """Small 768→256→512 projection with unit-normalized outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(768, 256)
        self.activation = nn.GELU()
        self.output = nn.Linear(256, EMBEDDING_DIMENSIONS)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1:] != (768,) or not torch.isfinite(features).all():
            raise ValueError("Expected finite 768-dimensional scene features")
        return F.normalize(self.output(self.activation(self.input(features))), dim=-1)


def _split_indices(pairs: PreparedTextPairs, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    image_indices = torch.tensor(
        [i for i, record in enumerate(pairs.image_records) if record["split"] == split],
        dtype=torch.int64,
    )
    allowed = set(image_indices.tolist())
    caption_indices = torch.tensor(
        [i for i, scene in enumerate(pairs.caption_image_indices.tolist()) if scene in allowed],
        dtype=torch.int64,
    )
    return image_indices, caption_indices


def _per_image_cosine_loss(
    projected_images: torch.Tensor,
    caption_targets: torch.Tensor,
    caption_image_indices: torch.Tensor,
) -> torch.Tensor:
    losses = []
    normalized_targets = F.normalize(caption_targets, dim=-1)
    for local_index in range(len(projected_images)):
        selected = caption_image_indices == local_index
        if selected.any():
            losses.append(
                1 - (projected_images[local_index] * normalized_targets[selected]).sum(-1).mean()
            )
    if not losses:
        raise ValueError("No captions for image loss")
    return torch.stack(losses).mean()


def train_text_adapter(
    prepared: str | Path,
    output: str | Path,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 10,
    seed: int = 17,
) -> dict:
    """Fit on train only and select the saved checkpoint using validation only."""
    if not isinstance(epochs, int) or epochs < 1 or not isinstance(patience, int) or patience < 1:
        raise ValueError("epochs and patience must be positive integers")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    pairs = load_prepared_pairs(prepared)
    train_images, train_captions = _split_indices(pairs, "train")
    validation_images, validation_captions = _split_indices(pairs, "validation")
    if not len(train_images) or not len(train_captions):
        raise ValueError("Training split has no eligible image-caption pairs")
    if not len(validation_images) or not len(validation_captions):
        raise ValueError("Validation split is required for checkpoint selection")

    train_x = pairs.image_features[train_images]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    normalized = (pairs.image_features - mean) / std
    image_to_local_train = {int(scene): i for i, scene in enumerate(train_images.tolist())}
    image_to_local_validation = {
        int(scene): i for i, scene in enumerate(validation_images.tolist())
    }
    train_caption_scenes = torch.tensor(
        [image_to_local_train[int(pairs.caption_image_indices[i])] for i in train_captions],
        dtype=torch.int64,
    )
    validation_caption_scenes = torch.tensor(
        [
            image_to_local_validation[int(pairs.caption_image_indices[i])]
            for i in validation_captions
        ],
        dtype=torch.int64,
    )

    torch.manual_seed(seed)
    model = TextProjection()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = _per_image_cosine_loss(
            model(normalized[train_images]),
            pairs.caption_embeddings[train_captions],
            train_caption_scenes,
        )
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = _per_image_cosine_loss(
                model(normalized[validation_images]),
                pairs.caption_embeddings[validation_captions],
                validation_caption_scenes,
            )
        value = float(validation_loss)
        history.append(
            {"epoch": epoch, "train_loss": float(train_loss.detach()), "validation_loss": value}
        )
        if value < best_loss - 1e-8:
            best_loss, best_epoch = value, epoch
            best_state = {
                key: tensor.detach().clone() for key, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)

    training_bank = []
    for index in train_captions.tolist():
        record = copy.deepcopy(pairs.caption_records[index])
        if (
            record["split"] != "train"
            or record["use_partition"] != "train"
            or record["training_eligible"] is not True
        ):
            raise ValueError("Non-training caption reached retrieval bank")
        record["embedding"] = pairs.caption_embeddings[index].tolist()
        training_bank.append(record)
    metadata = {
        "format_version": 1,
        "artifact_kind": "experimental_scene_level_image_text_adapter",
        "model": {"input_dimensions": 768, "hidden_dimensions": 256, "output_dimensions": 512},
        "embedding": pairs.manifest["embedding"],
        "feature_key": pairs.manifest["feature_key"],
        "feature_contract": pairs.manifest["feature_contract"],
        "feature_normalization": {
            "fit_split": "train",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "source_hashes": pairs.manifest["source_hashes"],
        "prepared_manifest_sha256": _sha256(Path(prepared).resolve() / "manifest.json"),
        "selection_split": "validation",
        "training_caption_bank": training_bank,
        "limitations": [
            "Experimental scene-level image-text adapter",
            "No open-ended captioning or VQA quality claim",
            "Scores are similarities, not calibrated confidence",
        ],
    }
    output.mkdir(parents=True)
    checkpoint_path = output / "best.pt"
    with checkpoint_path.open("xb") as stream:
        torch.save({"state_dict": best_state, "metadata": metadata}, stream)
    result = {
        "format_version": 1,
        "status": "fixture_or_caller_data_training_not_a_production_adaptation_claim",
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "selection_split": "validation",
        "stopped_early": len(history) < epochs,
        "history": history,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_hashes": pairs.manifest["source_hashes"],
    }
    _write_json_exclusive(output / "training.json", result)
    return result


def load_text_adapter(checkpoint: str | Path) -> tuple[TextProjection, dict]:
    values = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    metadata = values.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("artifact_kind") != "experimental_scene_level_image_text_adapter"
        or metadata.get("embedding")
        != {"model": EMBEDDING_MODEL, "dimensions": EMBEDDING_DIMENSIONS}
        or metadata.get("feature_normalization", {}).get("fit_split") != "train"
    ):
        raise ValueError("Invalid text adapter checkpoint contract")
    model = TextProjection()
    model.load_state_dict(values["state_dict"])
    model.eval()
    return model, metadata


def _project_one(model: TextProjection, metadata: dict, feature: torch.Tensor) -> torch.Tensor:
    feature = torch.as_tensor(feature, dtype=torch.float32)
    if tuple(feature.shape) == (225, 768):
        feature = feature.mean(0)
    if tuple(feature.shape) != (768,):
        raise ValueError("Expected one 768-vector or 225x768 spatial scene")
    normalizer = metadata["feature_normalization"]
    mean = torch.tensor(normalizer["mean"], dtype=torch.float32)
    std = torch.tensor(normalizer["std"], dtype=torch.float32)
    with torch.no_grad():
        return model((feature - mean) / std)


def image_query_similarity(
    model: TextProjection,
    metadata: dict,
    image_feature: torch.Tensor,
    query_embedding: Sequence[float] | torch.Tensor,
) -> dict:
    query = torch.as_tensor(query_embedding, dtype=torch.float32)
    if tuple(query.shape) != (EMBEDDING_DIMENSIONS,) or not torch.isfinite(query).all():
        raise ValueError("Expected finite 512-dimensional query embedding")
    score = float((_project_one(model, metadata, image_feature) * F.normalize(query, dim=0)).sum())
    return {"similarity": score, "score_kind": SCORE_KIND}


def top_k_training_captions(
    model: TextProjection,
    metadata: dict,
    image_feature: torch.Tensor,
    *,
    k: int = 5,
) -> list[dict]:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be positive")
    bank = metadata.get("training_caption_bank", [])
    if any(record.get("split") != "train" for record in bank):
        raise ValueError("Retrieval bank contains held-out captions")
    if not bank:
        return []
    projected = _project_one(model, metadata, image_feature)
    embeddings = F.normalize(
        torch.tensor([record["embedding"] for record in bank], dtype=torch.float32), dim=-1
    )
    scores = embeddings @ projected
    order = sorted(
        range(len(bank)), key=lambda i: (-float(scores[i]), str(bank[i]["annotation_id"]))
    )
    return [
        {
            "annotation_id": bank[i]["annotation_id"],
            "patch_id": bank[i]["patch_id"],
            "caption": bank[i]["caption"],
            "similarity": float(scores[i]),
            "score_kind": SCORE_KIND,
        }
        for i in order[:k]
    ]


def evaluate_retrieval(
    checkpoint: str | Path,
    prepared: str | Path,
    *,
    split: str = "test",
    output: str | Path | None = None,
) -> dict:
    """Report held-out retrieval ranks; never use these metrics for checkpoint selection."""
    if split not in ("validation", "test"):
        raise ValueError("Retrieval evaluation is restricted to validation or test")
    model, metadata = load_text_adapter(checkpoint)
    pairs = load_prepared_pairs(prepared)
    if metadata["prepared_manifest_sha256"] != _sha256(Path(prepared).resolve() / "manifest.json"):
        raise ValueError("Checkpoint prepared manifest provenance differs")
    if metadata["source_hashes"] != pairs.manifest["source_hashes"]:
        raise ValueError("Checkpoint and prepared-pair provenance differ")
    images, captions = _split_indices(pairs, split)
    if not len(images) or not len(captions):
        report = {
            "format_version": 1,
            "status": "absent",
            "split": split,
            "counts": {"images": len(images), "captions": len(captions)},
            "unsupported_or_absent": "No eligible held-out image-caption pairs for this split",
            "selection_use": "report_only_never_checkpoint_promotion",
        }
    else:
        normalizer = metadata["feature_normalization"]
        mean = torch.tensor(normalizer["mean"], dtype=torch.float32)
        std = torch.tensor(normalizer["std"], dtype=torch.float32)
        with torch.no_grad():
            image_vectors = model((pairs.image_features[images] - mean) / std)
        text_vectors = F.normalize(pairs.caption_embeddings[captions], dim=-1)
        similarities = image_vectors @ text_vectors.T
        image_local = {int(scene): i for i, scene in enumerate(images.tolist())}
        caption_scene_local = [
            image_local[int(pairs.caption_image_indices[index])] for index in captions.tolist()
        ]
        ranks = []
        image_ranks = []
        for local_image, scene_index in enumerate(images.tolist()):
            order = torch.argsort(similarities[local_image], descending=True).tolist()
            positives = {i for i, owner in enumerate(caption_scene_local) if owner == local_image}
            rank = min(order.index(positive) + 1 for positive in positives)
            image_ranks.append(rank)
            ranks.append(
                {
                    "patch_id": pairs.image_records[scene_index]["patch_id"],
                    "best_caption_rank": rank,
                }
            )
        text_ranks = []
        for local_caption, owner in enumerate(caption_scene_local):
            order = torch.argsort(similarities[:, local_caption], descending=True).tolist()
            text_ranks.append(order.index(owner) + 1)
        report = {
            "format_version": 1,
            "status": "reported",
            "split": split,
            "counts": {"images": len(images), "captions": len(captions)},
            "candidate_set": {
                "image_count": len(images),
                "caption_count": len(captions),
                "scope": f"eligible {split} records only",
                "caption_ids": [pairs.caption_records[i]["annotation_id"] for i in captions],
            },
            "metrics": {
                "image_to_text_mean_rank": sum(image_ranks) / len(image_ranks),
                "image_to_text_recall_at_1": sum(rank == 1 for rank in image_ranks)
                / len(image_ranks),
                "text_to_image_mean_rank": sum(text_ranks) / len(text_ranks),
                "text_to_image_recall_at_1": sum(rank == 1 for rank in text_ranks)
                / len(text_ranks),
            },
            "ranks": ranks,
            "selection_use": "report_only_never_checkpoint_promotion",
            "limitations": {
                "scores": "similarities, not calibrated confidence",
                "scope": "experimental scene-level retrieval; no captioning or VQA quality claim",
            },
            "source_hashes": pairs.manifest["source_hashes"],
        }
    if output is not None:
        output = Path(output).absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_json_exclusive(output, report)
    return report


class OpenAIEmbedder:
    """OpenAI SDK adapter; authentication is read by the SDK from the environment."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI embedding support is not installed; install "
                "satquery-preprocessing[assistant]"
            ) from exc

        self.client = OpenAI()

    def __call__(self, texts: Sequence[str], *, model: str, dimensions: int) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=list(texts), model=model, dimensions=dimensions
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]


def _parse_splits(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--annotations", required=True)
    prepare.add_argument("--features", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--feature-key", default="joint_encodings")
    prepare.add_argument("--splits", type=_parse_splits, default=("train", "validation"))
    prepare.add_argument("--exclude-id", action="append", default=[])
    prepare.add_argument("--embedding-batch-size", type=int, default=128)
    train = subparsers.add_parser("train")
    train.add_argument("--prepared", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--seed", type=int, default=17)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prepared", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        excluded = DEFAULT_EXCLUDED_ANNOTATION_IDS + tuple(args.exclude_id)
        result = prepare_text_pairs(
            args.annotations,
            args.features,
            args.output,
            requested_splits=args.splits,
            feature_key=args.feature_key,
            embedder=OpenAIEmbedder(),
            embedding_provenance={"kind": "openai_api", "authentication": "environment_only"},
            excluded_annotation_ids=excluded,
            embedding_batch_size=args.embedding_batch_size,
        )
    elif args.command == "train":
        result = train_text_adapter(
            args.prepared,
            args.output,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
        )
    else:
        result = evaluate_retrieval(
            args.checkpoint, args.prepared, split=args.split, output=args.output
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
