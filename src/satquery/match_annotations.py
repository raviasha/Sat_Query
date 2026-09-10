"""Match official BigEarthNet.txt records to existing optical/SAR areas without changing labels."""

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .preprocessing import sha256

SOURCE_REVISION = "72d865f2146f0a85b720f7f3ca1cdbaeafc3d316"
SOURCE_SHA256 = "d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554"
SOURCE_SIZE = 466819745
SOURCE_URL = f"https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/{SOURCE_REVISION}/BigEarthNet.txt.parquet"


def match_annotations(
    metadata,
    source,
    output,
    *,
    source_revision,
    expected_source_sha256=None,
    batch_size=65536,
    progress=None,
):
    """Export all matching text, an image index, and conservative whole-area split eligibility."""
    metadata, source = Path(metadata).resolve(), Path(source).resolve()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("Source revision is required")
    source_digest, metadata_digest = sha256(source), sha256(metadata)
    if expected_source_sha256 and source_digest != expected_source_sha256:
        raise ValueError("Text source checksum mismatch")
    images = pd.read_parquet(metadata)
    required = {"patch_id", "s1_name", "split"}
    if not required.issubset(images.columns) or images.empty:
        raise ValueError("Expected selected image metadata")
    for key in ("patch_id", "s1_name"):
        if (
            images[key].duplicated().any()
            or not images[key].map(lambda x: isinstance(x, str) and bool(x)).all()
        ):
            raise ValueError(f"Duplicate or invalid {key}")
    if not images.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown image split")
    wanted = dict(zip(images.patch_id, images.s1_name, strict=True))
    image_splits = dict(zip(images.patch_id, images.split, strict=True))
    image_order = {p: i for i, p in enumerate(images.patch_id)}
    parquet = pq.ParquetFile(source)
    if not {"ID", "patch_id", "s1_name", "input", "output", "type", "split"}.issubset(
        parquet.schema_arrow.names
    ):
        raise ValueError("Unexpected text source schema")
    added_fields = {"image_split", "image_index", "use_partition", "training_eligible"}
    if added_fields.intersection(parquet.schema_arrow.names):
        raise ValueError("Source collides with output annotation fields")
    rows, scanned = [], 0
    keys = pa.array(list(wanted))
    for batch in parquet.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        rows.extend(table.filter(pc.is_in(table["patch_id"], value_set=keys)).to_pylist())
        scanned += len(batch)
        if progress and (scanned % (batch_size * 20) == 0 or scanned == parquet.metadata.num_rows):
            progress(
                f"Scanned {scanned:,}/{parquet.metadata.num_rows:,} text records; matched {len(rows):,}"
            )
    seen, by_image = set(), {p: [] for p in wanted}
    for row in rows:
        if row["ID"] in seen:
            raise ValueError("Duplicate annotation ID")
        seen.add(row["ID"])
        if row["s1_name"] != wanted[row["patch_id"]]:
            raise ValueError(f"SAR pairing mismatch for {row['patch_id']}")
        if row["split"] not in ("train", "validation", "test", "bench"):
            raise ValueError("Unknown text split")
        if row["type"] not in ("binary", "mcq", "captioning", "bounding box"):
            raise ValueError("Unknown annotation type")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ("input", "output")):
            raise ValueError("Empty or invalid annotation text")
        row.update(
            image_split=image_splits[row["patch_id"]], image_index=image_order[row["patch_id"]]
        )
        by_image[row["patch_id"]].append(row)
    conflicts, benchmarks, missing, scenes = [], [], [], []
    for patch, annotations in by_image.items():
        splits = {r["split"] for r in annotations}
        if not annotations:
            partition = "missing"
            missing.append(patch)
        elif "bench" in splits:
            partition = "bench_holdout"
            benchmarks.append(patch)
        elif splits == {image_splits[patch]}:
            partition = image_splits[patch]
        else:
            partition = "split_conflict_holdout"
            conflicts.append(patch)
        for row in annotations:
            row.update(use_partition=partition, training_eligible=partition == "train")
        scenes.append(
            {
                "patch_id": patch,
                "s1_name": wanted[patch],
                "image_index": image_order[patch],
                "image_split": image_splits[patch],
                "annotation_splits": sorted(splits),
                "use_partition": partition,
                "annotation_count": len(annotations),
                "annotation_types": dict(Counter(r["type"] for r in annotations)),
                "annotations": annotations,
            }
        )
    # Retain publisher's exact fields and strings; add only image linkage and eligibility.
    rows.sort(key=lambda r: (r["image_index"], r["ID"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        schema = parquet.schema_arrow
        for name, kind in [
            ("image_split", pa.string()),
            ("image_index", pa.int64()),
            ("use_partition", pa.string()),
            ("training_eligible", pa.bool_()),
        ]:
            schema = schema.append(pa.field(name, kind))
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, temporary / "annotations.parquet", compression="zstd")
        if pq.read_table(temporary / "annotations.parquet").to_pylist() != rows:
            raise ValueError("Saved annotations did not reload identically")
        with (temporary / "image-annotations.jsonl").open("w") as f:
            for scene in scenes:
                f.write(json.dumps(scene, ensure_ascii=False, allow_nan=False) + "\n")
        report = {
            "format_version": 1,
            "passed": True,
            "source_revision": source_revision,
            "source_sha256": source_digest,
            "matching_implementation_sha256": sha256(Path(__file__)),
            "selected_metadata_sha256": metadata_digest,
            "source_record_count": scanned,
            "image_count": len(images),
            "matched_image_count": len(images) - len(missing),
            "missing_image_count": len(missing),
            "missing_patch_ids": missing,
            "annotation_count": len(rows),
            "counts_by_type": dict(Counter(r["type"] for r in rows)),
            "counts_by_category": dict(Counter(str(r.get("category")) for r in rows)),
            "counts_by_image_split": dict(Counter(r["image_split"] for r in rows)),
            "counts_by_annotation_split": dict(Counter(r["split"] for r in rows)),
            "counts_by_use_partition": dict(Counter(r["use_partition"] for r in rows)),
            "training_eligible_annotation_count": sum(r["training_eligible"] for r in rows),
            "benchmark_patch_ids": benchmarks,
            "split_conflict_patch_ids": conflicts,
            "annotation_scope": "Original 1.2 km image area or explicitly referenced region; not separate labels for each 80 m token",
            "split_policy": "Train only when the image and every linked annotation are train; any benchmark annotation holds out the entire image; other disagreements are retained but quarantined",
            "files": {
                name: sha256(temporary / name)
                for name in ("annotations.parquet", "image-annotations.jsonl")
            },
        }
        if sha256(metadata) != metadata_digest or sha256(source) != source_digest:
            raise ValueError("Input changed during matching")
        (temporary / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = match_annotations(
        args.metadata,
        args.source,
        args.output,
        source_revision=SOURCE_REVISION,
        expected_source_sha256=SOURCE_SHA256,
        progress=print,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
