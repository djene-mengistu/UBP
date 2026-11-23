"""Label-map segmentation metrics for 2D or N-dimensional volumes."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from scipy import ndimage


def _validated_labels(array: Any, name: str, num_classes: int) -> np.ndarray:
    labels = np.asarray(array)
    if labels.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if not np.issubdtype(labels.dtype, np.number) or np.issubdtype(labels.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain numeric integer labels")
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"{name} contains non-finite labels")
    if not np.all(labels == np.floor(labels)):
        raise ValueError(f"{name} contains non-integer labels")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"{name} labels must be in [0, {num_classes - 1}]")
    return labels


def _validated_spacing(spacing: Sequence[float] | None, ndim: int) -> tuple[float, ...] | None:
    if spacing is None:
        return None
    try:
        values = tuple(float(value) for value in spacing)
    except (TypeError, ValueError) as exc:
        raise ValueError("spacing must contain one positive finite value per array dimension") from exc
    if len(values) != ndim or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("spacing must contain one positive finite value per array dimension")
    return values


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(mask.ndim, 1), border_value=0)


def _hd95(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...] | None) -> float:
    """Compute HD95 inside the masks' tight union box plus one voxel of context."""
    union = prediction | target
    coordinates = np.nonzero(union)
    slices = tuple(
        slice(max(int(axis.min()) - 1, 0), min(int(axis.max()) + 2, union.shape[index]))
        for index, axis in enumerate(coordinates)
    )
    prediction = prediction[slices]
    target = target[slices]
    pred_surface = _surface(prediction)
    target_surface = _surface(target)
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate((distance_to_target[pred_surface], distance_to_prediction[target_surface]))
    return float(np.percentile(distances, 95))


def segmentation_metrics(
    prediction: Any,
    target: Any,
    num_classes: int,
    spacing: Sequence[float] | None = None,
    compute_hd95: bool = True,
) -> list[dict[str, Any]]:
    """Return foreground-class Dice and symmetric HD95 for one label volume.

    HD95 is in voxel-coordinate units when ``spacing`` is omitted, and in
    caller-supplied physical units otherwise. Classes with both masks empty
    receive Dice 1 and undefined HD95; exactly-one-empty classes receive Dice
    0 and undefined HD95 with an explicit status. Set ``compute_hd95=False``
    to skip distance transforms; present classes then use ``not_computed``.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2, including background")
    if not isinstance(compute_hd95, (bool, np.bool_)):
        raise ValueError("compute_hd95 must be boolean")
    pred = _validated_labels(prediction, "prediction", int(num_classes))
    truth = _validated_labels(target, "target", int(num_classes))
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same shape")
    sample_spacing = _validated_spacing(spacing, pred.ndim)

    results: list[dict[str, Any]] = []
    for class_id in range(1, int(num_classes)):
        pred_mask = pred == class_id
        target_mask = truth == class_id
        pred_present = bool(np.any(pred_mask))
        gt_present = bool(np.any(target_mask))
        if not pred_present and not gt_present:
            dice, hd95, status = 1.0, None, "both_empty"
        elif not pred_present:
            dice, hd95, status = 0.0, None, "missing_prediction"
        elif not gt_present:
            dice, hd95, status = 0.0, None, "false_positive"
        else:
            intersection = int(np.count_nonzero(pred_mask & target_mask))
            dice = 2.0 * intersection / (int(pred_mask.sum()) + int(target_mask.sum()))
            if compute_hd95:
                hd95 = _hd95(pred_mask, target_mask, sample_spacing)
                status = "ok"
            else:
                hd95, status = None, "not_computed"
        results.append({
            "class_id": class_id,
            "dice": float(dice),
            "hd95": hd95,
            "hd95_status": status,
            "gt_present": gt_present,
            "pred_present": pred_present,
        })
    return results


def aggregate_metrics(case_results: Sequence[Sequence[dict[str, Any]]], num_classes: int) -> dict[str, Any]:
    """Aggregate per-case results returned by :func:`segmentation_metrics`.

    Dice means include empty cases (both-empty Dice is 1, one-sided failures
    are 0). HD95 means include only ``ok`` cases; all other outcomes remain
    visible in status counts and denominators.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2, including background")
    expected = set(range(1, int(num_classes)))
    by_class: dict[int, list[dict[str, Any]]] = {class_id: [] for class_id in expected}
    for case_index, case in enumerate(case_results):
        found: set[int] = set()
        for record in case:
            class_id = record.get("class_id")
            if class_id not in expected or class_id in found:
                raise ValueError(f"case {case_index} has an invalid or duplicate class_id")
            if record.get("hd95_status") not in {"ok", "both_empty", "missing_prediction", "false_positive", "not_computed"}:
                raise ValueError(f"case {case_index} has an invalid hd95_status")
            found.add(class_id)
            by_class[class_id].append(record)
        if found != expected:
            raise ValueError(f"case {case_index} must contain each foreground class exactly once")

    per_class: dict[int, dict[str, Any]] = {}
    dice_means: list[float] = []
    for class_id, records in by_class.items():
        dice_values = [float(record["dice"]) for record in records]
        valid_hd95 = [float(record["hd95"]) for record in records if record["hd95_status"] == "ok" and record.get("hd95") is not None]
        statuses = {
            status: sum(record["hd95_status"] == status for record in records)
            for status in ("ok", "both_empty", "missing_prediction", "false_positive", "not_computed")
        }
        mean_dice = float(np.mean(dice_values)) if dice_values else None
        if mean_dice is not None:
            dice_means.append(mean_dice)
        per_class[class_id] = {
            "mean_dice": mean_dice,
            "dice_denominator": len(dice_values),
            "mean_hd95": float(np.mean(valid_hd95)) if valid_hd95 else None,
            "valid_hd95_count": len(valid_hd95),
            "hd95_denominator": len(records),
            "hd95_status_counts": statuses,
        }
    return {
        "per_class": per_class,
        "overall_foreground_macro_dice": float(np.mean(dice_means)) if dice_means else None,
        "case_count": len(case_results),
    }
