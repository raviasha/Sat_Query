"""Exercise the Colab orchestration on a real, tiny local export."""

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

from satquery.preprocessing import sha256

ROOT = Path(__file__).resolve().parents[1]


def test_configurable_matching_reuses_export_and_rejects_changed_source(tmp_path, monkeypatch):
    import satquery.match_annotations as matching

    pipeline = tmp_path / "pipeline-2"
    images = tmp_path / "images"
    images.mkdir()
    batch = pipeline / "features" / "batch-0"
    batch.mkdir(parents=True)
    samples = [
        {"patch_id": "p1", "s1_name": "s1", "split": "train"},
        {"patch_id": "p2", "s1_name": "s2", "split": "test"},
    ]
    pd.DataFrame(samples).to_parquet(images / "metadata.parquet")
    (images / "download.json").write_text(
        json.dumps({"metadata_sha256": sha256(images / "metadata.parquet")})
    )
    (batch / "batch.json").write_text(json.dumps({"samples": samples}))
    # The linker inspects the batch index, not the tensor contents.
    (batch / "features.pt").write_bytes(b"fixture tensor bytes")
    manifest = pipeline / "features" / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "directory": "batch-0",
                        "start_index": 0,
                        "sha256": {
                            name: sha256(batch / name) for name in ["batch.json", "features.pt"]
                        },
                    }
                ]
            }
        )
    )
    (pipeline / "stage-2-report.json").write_text(
        json.dumps({"feature_manifest_sha256": sha256(manifest)})
    )
    source = pipeline / "annotations" / "source" / "fixture.parquet"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            dict(
                ID=1,
                **samples[0],
                input="Forest?",
                output="yes",
                type="binary",
                category="presence",
            )
        ]
    ).to_parquet(source)
    monkeypatch.setattr(matching, "SOURCE_SHA256", sha256(source))
    settings = {
        "P": pipeline,
        "IMAGE_ROOT": images,
        "TEXT_SOURCE": source,
        "OUTPUT": pipeline / "annotations" / "matched",
    }
    script = str(ROOT / "scripts" / "colab_match_annotations.py")
    runpy.run_path(script, init_globals=settings)
    links_path = settings["OUTPUT"] / "feature-links.jsonl"
    links = [json.loads(line) for line in links_path.read_text().splitlines()]
    assert [(r["patch_id"], r["index_in_feature_batch"], r["annotation_count"]) for r in links] == [
        ("p1", 0, 1),
        ("p2", 1, 0),
    ]
    before = links_path.read_bytes()
    runpy.run_path(script, init_globals=settings)
    assert links_path.read_bytes() == before
    source.write_bytes(b"damaged source")
    with pytest.raises((AssertionError, ValueError), match="source|Source|checksum"):
        runpy.run_path(script, init_globals=settings)
