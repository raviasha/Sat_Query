"""Deterministic, JSON-safe tools over spatial coverage estimates."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.warp import transform

from ..prediction_data import class_schema
from .runtime import SceneResult

TASKS = ("coverage", "presence", "describe", "locate", "change")
CELL_AREA_M2 = 80 * 80
SCENE_AREA_M2 = 15 * 15 * CELL_AREA_M2
ALIASES = {
    "forest": (8, 9, 10),
    "water": (17, 18),
    "farmland": (2, 3, 4, 5, 6, 7),
    "built-up": (0, 1),
    "built up": (0, 1),
    "builtup": (0, 1),
}


def _abstain(task: Any, reason: str, class_name: Any = None) -> dict[str, Any]:
    return {
        "abstained": True,
        "answer": f"I cannot answer this request: {reason}",
        "measurements": [],
        "evidence": [],
        "limitations": [
            "Only the fixed 19-class land-cover schema and documented aliases are supported."
        ],
        "trace": {
            "selected_tool": None,
            "model": "CROMA-Base coverage head",
            "parameters": {"requested_task": task, "requested_class": class_name},
        },
    }


def _resolve_class(name: Any) -> tuple[str, list[int]] | None:
    if not isinstance(name, str) or not name.strip():
        return None
    key = name.strip().casefold()
    if key in ALIASES:
        display = "built-up" if key in ("built up", "builtup") else key
        return display, list(ALIASES[key])
    for item in class_schema():
        if item["name"].casefold() == key:
            return item["name"], [item["index"]]
    return None


def _validate_threshold(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 1
    ):
        raise ValueError("threshold must be a finite number from 0 to 1")
    return float(value)


def _validate_scene(scene: SceneResult) -> np.ndarray:
    if not isinstance(scene, SceneResult):
        raise TypeError("scenes must contain SceneResult values")
    values = np.asarray(scene.coverage)
    if (
        values.shape != (15, 15, 19)
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=-1), 1, rtol=0, atol=1e-5)
    ):
        raise ValueError(f"{scene.id}: expected finite 15x15x19 class fractions summing to one")
    grid = scene.grid
    if (
        not isinstance(grid, dict)
        or grid.get("width") != 120
        or grid.get("height") != 120
        or not isinstance(grid.get("transform"), list)
        or len(grid["transform"]) != 6
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in grid["transform"]
        )
    ):
        raise ValueError(f"{scene.id}: invalid scene grid")
    a, b, _, d, e, _ = grid["transform"]
    if not np.allclose([a, b, d, e], [10, 0, 0, -10], rtol=0, atol=1e-6):
        raise ValueError(f"{scene.id}: tools require a north-up 10 metre grid")
    try:
        crs = CRS.from_user_input(grid.get("crs"))
    except (CRSError, TypeError, ValueError) as exc:
        raise ValueError(f"{scene.id}: invalid grid CRS") from exc
    if not crs.is_projected or not math.isclose(
        crs.linear_units_factor[1], 1.0, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"{scene.id}: area tools require a projected metre CRS")
    return values.astype(np.float64, copy=False)


def _scene_limitations(scenes: list[SceneResult]) -> list[str]:
    limitations = [
        "Coverage values are model-estimated fractions of 80 m cells and are not surveyed ground truth."
    ]
    scopes = sorted(
        {
            scene.provenance.get("head_evaluation_scope")
            for scene in scenes
            if scene.provenance.get("head_evaluation_scope")
        }
    )
    if scopes:
        limitations.append(f"Head evaluation scope: {', '.join(scopes)}.")
    return limitations


def _trace(task: str, scenes: list[SceneResult], parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_tool": task,
        "model": "CROMA-Base coverage head",
        "scene_ids": [scene.id for scene in scenes],
        "parameters": parameters,
    }


def _pooled(values: np.ndarray, indices: list[int]) -> np.ndarray:
    return values[..., indices].sum(axis=-1)


def _fraction_grid(values: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in values]


def _coverage(
    scenes: list[SceneResult],
    arrays: list[np.ndarray],
    display: str,
    indices: list[int],
) -> dict[str, Any]:
    measurements, evidence, phrases = [], [], []
    for scene, values in zip(scenes, arrays, strict=True):
        pooled = _pooled(values, indices)
        area = float(pooled.sum() * CELL_AREA_M2)
        fraction = area / SCENE_AREA_M2
        measurements.append(
            {
                "scene_id": scene.id,
                "class_name": display,
                "class_indices": indices,
                "estimated_fraction": fraction,
                "estimated_area_m2": area,
                "scene_area_m2": SCENE_AREA_M2,
            }
        )
        evidence.append({"scene_id": scene.id, "fraction_grid": _fraction_grid(pooled)})
        phrases.append(f"{scene.id}: {area:.1f} m² ({fraction * 100:.2f}%)")
    return {
        "abstained": False,
        "answer": f"Estimated {display} coverage — " + "; ".join(phrases) + ".",
        "measurements": measurements,
        "evidence": evidence,
        "limitations": _scene_limitations(scenes),
        "trace": _trace("coverage", scenes, {"class_name": display, "class_indices": indices}),
    }


def _presence(
    scenes: list[SceneResult],
    arrays: list[np.ndarray],
    display: str,
    indices: list[int],
    threshold: float,
) -> dict[str, Any]:
    measurements, evidence, phrases = [], [], []
    for scene, values in zip(scenes, arrays, strict=True):
        pooled = _pooled(values, indices)
        selected = pooled >= threshold
        count = int(selected.sum())
        detected = count > 0
        measurements.append(
            {
                "scene_id": scene.id,
                "class_name": display,
                "class_indices": indices,
                "threshold": threshold,
                "detected": detected,
                "qualifying_cell_count": count,
            }
        )
        evidence.append(
            {
                "scene_id": scene.id,
                "estimated_fraction_grid": _fraction_grid(pooled),
                "threshold_grid": selected.tolist(),
            }
        )
        if detected:
            phrases.append(f"{scene.id}: {count} coarse cells met the threshold")
        else:
            phrases.append(
                f"{scene.id}: no coarse cells met the threshold; this does not establish absence"
            )
    limitations = _scene_limitations(scenes)
    limitations.append(
        f"The {threshold:g} cutoff is heuristic and unvalidated; it is an estimated area-fraction threshold, not confidence."
    )
    return {
        "abstained": False,
        "answer": f"Threshold estimate for {display} — " + "; ".join(phrases) + ".",
        "measurements": measurements,
        "evidence": evidence,
        "limitations": limitations,
        "trace": _trace(
            "presence",
            scenes,
            {"class_name": display, "class_indices": indices, "threshold": threshold},
        ),
    }


def _cell_polygon(scene: SceneResult, row: int, column: int) -> list[list[float]]:
    a, _, c, _, e, f = [float(value) for value in scene.grid["transform"]]
    left = c + column * 8 * a
    right = left + 8 * a
    top = f + row * 8 * e
    bottom = top + 8 * e
    return [
        [left, bottom],
        [right, bottom],
        [right, top],
        [left, top],
        [left, bottom],
    ]


def _wgs84_polygon(scene: SceneResult, projected: list[list[float]]) -> list[list[float]]:
    longitudes, latitudes = transform(
        scene.grid["crs"],
        "EPSG:4326",
        [point[0] for point in projected],
        [point[1] for point in projected],
    )
    return [
        [float(longitude), float(latitude)] for longitude, latitude in zip(longitudes, latitudes)
    ]


def _locate(
    scenes: list[SceneResult],
    arrays: list[np.ndarray],
    display: str,
    indices: list[int],
    threshold: float,
) -> dict[str, Any]:
    measurements, evidence, phrases = [], [], []
    for scene, values in zip(scenes, arrays, strict=True):
        pooled = _pooled(values, indices)
        selected = np.argwhere(pooled >= threshold)
        features = []
        for row, column in selected:
            projected = _cell_polygon(scene, int(row), int(column))
            xs, ys = zip(*projected)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_wgs84_polygon(scene, projected)],
                    },
                    "properties": {
                        "scene_id": scene.id,
                        "row": int(row),
                        "column": int(column),
                        "estimate": float(pooled[row, column]),
                        "cell_area_m2": CELL_AREA_M2,
                        "projected_crs": scene.grid["crs"],
                        "projected_bounds_m": [min(xs), min(ys), max(xs), max(ys)],
                    },
                }
            )
        measurements.append(
            {
                "scene_id": scene.id,
                "class_name": display,
                "class_indices": indices,
                "threshold": threshold,
                "qualifying_cell_count": len(features),
            }
        )
        evidence.append(
            {
                "scene_id": scene.id,
                "geojson": {
                    "type": "FeatureCollection",
                    "name": f"{display} coarse evidence cells",
                    "features": features,
                },
            }
        )
        if features:
            phrases.append(f"{scene.id}: {len(features)} coarse 80 m cells met the threshold")
        else:
            phrases.append(
                f"{scene.id}: no coarse cells met the threshold; this does not establish absence"
            )
    limitations = _scene_limitations(scenes)
    limitations.extend(
        [
            f"The {threshold:g} cutoff is heuristic and unvalidated; it is an estimated area-fraction threshold, not confidence.",
            "GeoJSON polygons are coarse 80 m model cells, not exact object boundaries or boxes.",
        ]
    )
    return {
        "abstained": False,
        "answer": f"Estimated locations for {display} — " + "; ".join(phrases) + ".",
        "measurements": measurements,
        "evidence": evidence,
        "limitations": limitations,
        "trace": _trace(
            "locate",
            scenes,
            {"class_name": display, "class_indices": indices, "threshold": threshold},
        ),
    }


def _describe(scenes: list[SceneResult], arrays: list[np.ndarray]) -> dict[str, Any]:
    schema = class_schema()
    measurements, evidence, phrases = [], [], []
    for scene, values in zip(scenes, arrays, strict=True):
        means = values.mean(axis=(0, 1))
        order = np.argsort(-means, kind="stable")
        top = [
            {
                "name": schema[index]["name"],
                "class_index": int(index),
                "estimated_fraction": float(means[index]),
                "estimated_area_m2": float(means[index] * SCENE_AREA_M2),
            }
            for index in order[:3]
        ]
        dominant = values.argmax(axis=-1)
        measurements.append({"scene_id": scene.id, "top_land_cover": top})
        evidence.append({"scene_id": scene.id, "dominant_class_index_grid": dominant.tolist()})
        phrases.append(
            f"{scene.id}: "
            + ", ".join(f"{item['name']} {item['estimated_fraction'] * 100:.2f}%" for item in top)
        )
    return {
        "abstained": False,
        "answer": "Top estimated land-cover proportions — " + "; ".join(phrases) + ".",
        "measurements": measurements,
        "evidence": evidence,
        "limitations": _scene_limitations(scenes),
        "trace": _trace("describe", scenes, {"top_count": 3}),
    }


def _aligned_temporal(scenes: list[SceneResult]) -> None:
    if len(scenes) != 2:
        raise ValueError("change requires exactly two scenes")
    before, after = scenes
    if before.modality != after.modality:
        raise ValueError("change requires two scenes of the same modality")
    if before.acquired is None or after.acquired is None or before.acquired >= after.acquired:
        raise ValueError("change requires distinct scene dates in ascending order")
    if before.grid != after.grid:
        raise ValueError("change requires exactly aligned scene grids")
    head_hashes = [scene.provenance.get("head_sha256") for scene in scenes]
    if (
        any(not isinstance(value, str) or not value for value in head_hashes)
        or len(set(head_hashes)) != 1
    ):
        raise ValueError("change requires the same trained coverage head")
    contracts = [scene.provenance.get("feature_contract") for scene in scenes]
    feature_keys = [scene.provenance.get("feature_key") for scene in scenes]
    if (
        any(not isinstance(value, dict) or not value for value in contracts)
        or contracts[0] != contracts[1]
        or any(not isinstance(value, str) or not value for value in feature_keys)
        or feature_keys[0] != feature_keys[1]
    ):
        raise ValueError("change requires the same nonempty model and feature contract")


def _change(
    scenes: list[SceneResult],
    arrays: list[np.ndarray],
    resolved: tuple[str, list[int]] | None,
) -> dict[str, Any]:
    _aligned_temporal(scenes)
    before, after = arrays
    schema = class_schema()
    if resolved is not None:
        display, indices = resolved
        before_values, after_values = _pooled(before, indices), _pooled(after, indices)
        delta = after_values - before_values
        before_area = float(before_values.sum() * CELL_AREA_M2)
        after_area = float(after_values.sum() * CELL_AREA_M2)
        measurements: Any = {
            "class_name": display,
            "class_indices": indices,
            "before_area_m2": before_area,
            "after_area_m2": after_area,
            "net_area_change_m2": after_area - before_area,
        }
        evidence: Any = {"class_delta_grid": _fraction_grid(delta)}
        answer = (
            f"Provisional {display} change estimate: {before_area:.1f} m² before, "
            f"{after_area:.1f} m² after, net {after_area - before_area:+.1f} m²."
        )
        parameters = {"class_name": display, "class_indices": indices}
    else:
        before_means, after_means = before.mean(axis=(0, 1)), after.mean(axis=(0, 1))
        records = [
            {
                "name": item["name"],
                "class_index": item["index"],
                "before_area_m2": float(before_means[item["index"]] * SCENE_AREA_M2),
                "after_area_m2": float(after_means[item["index"]] * SCENE_AREA_M2),
                "net_area_change_m2": float(
                    (after_means[item["index"]] - before_means[item["index"]]) * SCENE_AREA_M2
                ),
            }
            for item in schema
        ]
        measurements = {"classes": records}
        evidence = {
            "class_order": [item["name"] for item in schema],
            "per_cell_class_delta": (after - before).tolist(),
        }
        largest = max(
            records, key=lambda item: (abs(item["net_area_change_m2"]), -item["class_index"])
        )
        answer = (
            "Provisional change estimate across all classes; largest absolute net change: "
            f"{largest['name']} {largest['net_area_change_m2']:+.1f} m²."
        )
        parameters = {"class_name": None, "class_indices": list(range(19))}
    limitations = _scene_limitations(scenes)
    limitations.append(
        "Temporal deltas are provisional and unvalidated model-estimate differences, not confirmed physical change."
    )
    return {
        "abstained": False,
        "answer": answer,
        "measurements": measurements,
        "evidence": evidence,
        "limitations": limitations,
        "trace": _trace("change", scenes, parameters),
    }


def execute_task(
    task: str,
    scenes: list[SceneResult],
    class_name: str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Execute one allowlisted spatial measurement without interpreting free-form objects."""
    if task not in TASKS:
        return _abstain(task, f"unsupported task {task!r}", class_name)
    if not isinstance(scenes, (list, tuple)) or not scenes:
        raise ValueError("scenes must be a nonempty sequence")
    scene_list = list(scenes)
    arrays = [_validate_scene(scene) for scene in scene_list]
    resolved = _resolve_class(class_name)
    if task in ("coverage", "presence", "locate") and resolved is None:
        return _abstain(task, f"unsupported or missing class {class_name!r}", class_name)
    if task == "change" and class_name is not None and resolved is None:
        return _abstain(task, f"unsupported class {class_name!r}", class_name)

    if task == "coverage":
        result = _coverage(scene_list, arrays, *resolved)
    elif task == "presence":
        result = _presence(scene_list, arrays, *resolved, _validate_threshold(threshold))
    elif task == "locate":
        result = _locate(scene_list, arrays, *resolved, _validate_threshold(threshold))
    elif task == "describe":
        result = _describe(scene_list, arrays)
    else:
        result = _change(scene_list, arrays, resolved)
    json.dumps(result, allow_nan=False)
    return result
