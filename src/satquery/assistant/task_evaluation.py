"""Strict, score-by-score evaluation for explicit task records."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

REQUIRED_FIELDS = {"id", "dataset", "split", "task", "answer", "abstention"}
SUPPORTED_TASKS = frozenset({"vqa", "area", "change"})


def _read_jsonl(path: str | Path, label: str) -> list[dict]:
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {label} JSONL line {number}") from exc
        if not isinstance(record, dict) or not REQUIRED_FIELDS.issubset(record):
            raise ValueError(f"{label} line {number} lacks required explicit fields")
        for field in ("id", "dataset", "split", "task"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"Invalid {label} {field} at line {number}")
        if record["task"] not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task at {label} line {number}: {record['task']!r}; "
                f"expected one of {sorted(SUPPORTED_TASKS)}"
            )
        if not isinstance(record["abstention"], bool):
            raise TypeError(f"Invalid {label} abstention at line {number}")
        if record["abstention"] and record["answer"] is not None:
            raise ValueError(f"Abstaining {label} must use a null answer")
        if not record["abstention"] and record["answer"] is None:
            raise ValueError(f"Non-abstaining {label} requires an answer")
        records.append(record)
    if not records:
        raise ValueError(f"No {label} records; required records are missing")
    return records


def _validate_unique(records: list[dict], label: str) -> dict[tuple[str, str, str, str], dict]:
    seen_ids: dict[tuple[str, str], str] = {}
    keyed = {}
    for record in records:
        dataset_id = record["dataset"], record["id"]
        previous_split = seen_ids.get(dataset_id)
        if previous_split is not None and previous_split != record["split"]:
            raise ValueError(f"{label} ID leakage across splits: {record['id']}")
        seen_ids[dataset_id] = record["split"]
        key = record["dataset"], record["split"], record["task"], record["id"]
        if key in keyed:
            raise ValueError(f"duplicate {label} record: {key}")
        keyed[key] = record
    return keyed


def _normalize_vqa(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_fraction_map(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} class_fractions must be a nonempty object")
    result = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} class fraction name is invalid")
        number = _finite_number(raw, f"{label} class fraction")
        if number < 0 or number > 1:
            raise ValueError(f"{label} class fractions must use 0..1 scene fractions")
        result[name] = number
    return result


def evaluate_task_records(references: str | Path, predictions: str | Path) -> dict:
    reference_records = _read_jsonl(references, "reference")
    prediction_records = _read_jsonl(predictions, "prediction")
    reference_by_key = _validate_unique(reference_records, "reference")
    prediction_by_key = _validate_unique(prediction_records, "prediction")
    if set(reference_by_key) != set(prediction_by_key):
        missing = sorted(set(reference_by_key) - set(prediction_by_key))
        extra = sorted(set(prediction_by_key) - set(reference_by_key))
        raise ValueError(
            f"Strict ID/dataset/task/split matching failed; missing={missing}, extra={extra}"
        )

    per_task = Counter()
    abstentions = 0
    vqa_total = vqa_answered = vqa_correct = 0
    numeric_errors: dict[str, list[float]] = defaultdict(list)
    fraction_errors: list[float] = []
    change_total = change_correct = 0
    for key in sorted(reference_by_key):
        reference, prediction = reference_by_key[key], prediction_by_key[key]
        task = reference["task"]
        per_task[task] += 1
        if prediction["abstention"]:
            abstentions += 1
        if task == "vqa":
            vqa_total += 1
            if not prediction["abstention"]:
                vqa_answered += 1
                vqa_correct += _normalize_vqa(reference["answer"]) == _normalize_vqa(
                    prediction["answer"]
                )

        reference_fraction = reference.get("class_fractions")
        prediction_fraction = prediction.get("class_fractions")
        if not prediction["abstention"] and (reference_fraction is None) != (
            prediction_fraction is None
        ):
            raise ValueError(f"class_fractions must be present on both matched records: {key}")
        if reference_fraction is not None and not prediction["abstention"]:
            expected = _validate_fraction_map(reference_fraction, "reference")
            actual = _validate_fraction_map(prediction_fraction, "prediction")
            if set(expected) != set(actual):
                raise ValueError(f"class_fractions class mismatch: {key}")
            fraction_errors.extend(abs(expected[name] - actual[name]) for name in sorted(expected))

        reference_changes = reference.get("change_labels")
        prediction_changes = prediction.get("change_labels")
        if not prediction["abstention"] and (reference_changes is None) != (
            prediction_changes is None
        ):
            raise ValueError(f"change_labels must be present on both matched records: {key}")
        if reference_changes is not None and not prediction["abstention"]:
            if not isinstance(reference_changes, list) or not isinstance(prediction_changes, list):
                raise ValueError("change_labels must be lists")
            if any(not isinstance(item, str) for item in reference_changes + prediction_changes):
                raise ValueError("change_labels must contain strings")
            change_total += 1
            change_correct += set(reference_changes) == set(prediction_changes)

        if task == "area" and not prediction["abstention"]:
            expected = _finite_number(reference["answer"], "reference area")
            actual = _finite_number(prediction["answer"], "prediction area")
            reference_unit, prediction_unit = reference.get("unit"), prediction.get("unit")
            if (
                not isinstance(reference_unit, str)
                or not reference_unit
                or prediction_unit != reference_unit
            ):
                raise ValueError(f"Explicit matching area unit required: {key}")
            numeric_errors[reference_unit].append(abs(expected - actual))

    total = len(reference_records)
    report = {
        "format_version": 1,
        "supported_tasks": sorted(SUPPORTED_TASKS),
        "matching": {
            "reference_count": total,
            "prediction_count": len(prediction_records),
            "matched_count": total,
        },
        "per_task_counts": dict(sorted(per_task.items())),
        "abstention_count": abstentions,
        "coverage": (total - abstentions) / total,
        "vqa": {
            "metric": "provisional_normalized_exact_match_not_official_scorer",
            "normalization": "Unicode NFKC, casefold, punctuation-to-space, whitespace collapse",
            "count": vqa_total,
            "answered_count": vqa_answered,
            "coverage": vqa_answered / vqa_total if vqa_total else None,
            "answered_exact_match": vqa_correct / vqa_answered if vqa_answered else None,
        },
        "numerical_area_mae": {
            unit: {"count": len(errors), "mae": sum(errors) / len(errors)}
            for unit, errors in sorted(numeric_errors.items())
        },
        "class_fraction_mae": (
            {
                "count": len(fraction_errors),
                "unit": "fraction_of_scene",
                "mae": sum(fraction_errors) / len(fraction_errors),
                "percentage_point_mae": 100 * sum(fraction_errors) / len(fraction_errors),
            }
            if fraction_errors
            else {
                "count": 0,
                "unit": "fraction_of_scene",
                "mae": None,
                "percentage_point_mae": None,
            }
        ),
        "change_labels": {
            "count": change_total,
            "metric": "exact_label_set_match",
            "exact_set_match": change_correct / change_total if change_total else None,
        },
        "unsupported_or_absent": {
            "official_vqa_scorer": "not integrated",
        },
    }
    return report


def benchmark_compatibility_manifest() -> dict:
    return {
        "format_version": 1,
        "status": "compatibility_description_only",
        "benchmarks": {
            "VRSBench": {
                "expected_modality": "remote-sensing image plus language task record",
                "status": "unsupported",
                "current_gap": "No VRSBench dataset adapter or benchmark model is integrated",
            },
            "RSVQA": {
                "expected_modality": "remote-sensing image plus question and answer",
                "status": "unsupported",
                "current_gap": "No RSVQA dataset adapter or official scorer is integrated",
            },
            "CDVQA": {
                "expected_modality": "paired temporal imagery plus question",
                "status": "unsupported",
                "current_gap": "No temporal-image encoder, CDVQA adapter, or official scorer is integrated",
            },
            "SAC": {
                "expected_modality": (
                    "pre-georeferenced co-registered Cartosat-2S optical and RISAT SAR "
                    "evaluation inputs"
                ),
                "status": "unsupported",
                "current_gap": (
                    "No direct Cartosat-2S/RISAT sensor profiles, hidden evaluation adapter, "
                    "or SAC evaluation results are integrated"
                ),
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compatibility-output")
    args = parser.parse_args(argv)
    output = Path(args.output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(
            evaluate_task_records(args.references, args.predictions),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    if args.compatibility_output:
        compatibility = Path(args.compatibility_output).absolute()
        compatibility.parent.mkdir(parents=True, exist_ok=True)
        with compatibility.open("x") as stream:
            json.dump(benchmark_compatibility_manifest(), stream, indent=2, allow_nan=False)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
