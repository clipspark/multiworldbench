"""RGB-distribution consistency across revisited camera views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from .common import aggregate_numeric_cases, arithmetic_mean


def _rgb(value: Any) -> np.ndarray:
    if isinstance(value, (str, Path)):
        value = np.asarray(Image.open(value).convert("RGB"), dtype=np.uint8)
    else:
        value = np.asarray(value)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    return value.astype(np.uint8, copy=False)


def cropped_pixels(value: Any, crop=(0.08, 0.88, 0.08, 0.92)) -> np.ndarray:
    image = _rgb(value)
    y0, y1, x0, x1 = crop
    if not (0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1):
        raise ValueError("crop fractions must satisfy 0 <= start < end <= 1")
    h, w = image.shape[:2]
    pixels = image[int(y0*h):int(y1*h), int(x0*w):int(x1*w)].reshape(-1, 3)
    if len(pixels) == 0:
        raise ValueError("crop produced no pixels")
    keep = ((pixels.max(1) - pixels.min(1)) > 3) | (
        (pixels.mean(1) > 8) & (pixels.mean(1) < 248)
    )
    return pixels[keep] if keep.any() else pixels


def rgb_histogram(value: Any, bins: int = 16) -> np.ndarray:
    if type(bins) is not int or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    pixels = cropped_pixels(value)
    histogram, _ = np.histogramdd(
        pixels.astype(np.float32),
        bins=(bins, bins, bins),
        range=((0, 256), (0, 256), (0, 256)),
    )
    histogram = histogram.ravel().astype(float)
    return histogram / max(float(histogram.sum()), 1.0)


def sqrt_js_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("histograms must be one-dimensional with equal shape")
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("histograms cannot be negative")
    left = left / max(float(left.sum()), 1e-12)
    right = right / max(float(right.sum()), 1e-12)
    middle = (left + right) / 2.0

    def kl(source, target):
        mask = source > 0
        return float(np.sum(source[mask] * np.log2(
            source[mask] / np.maximum(target[mask], 1e-12)
        )))

    return float(np.sqrt(max(0.0, 0.5 * kl(left, middle) + 0.5 * kl(right, middle))))


def image_distance(left: Any, right: Any, *, bins: int = 16) -> float:
    return sqrt_js_distance(rgb_histogram(left, bins), rgb_histogram(right, bins))


def score_case(
    revisit_pairs: Iterable[Mapping[str, Any]], *, threshold: float = 0.25,
    bins: int = 16,
) -> dict[str, Any]:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    pairs = []
    for pair in revisit_pairs:
        required = ("gt_initial", "gt_return", "generated_initial", "generated_return")
        missing = [key for key in required if key not in pair]
        if missing:
            raise ValueError(f"revisit pair missing fields: {missing}")
        gt_distance = image_distance(pair["gt_initial"], pair["gt_return"], bins=bins)
        generated_distance = image_distance(
            pair["generated_initial"], pair["generated_return"], bins=bins
        )
        error = abs(generated_distance - gt_distance)
        pairs.append({
            "pair_id": pair.get("pair_id"),
            "gt_rgb_js_distance": gt_distance,
            "generated_rgb_js_distance": generated_distance,
            "absolute_distance_error": error,
            "excess_distance": max(0.0, generated_distance - gt_distance),
        })
    if not pairs:
        raise ValueError("at least one revisit pair is required")
    mean_error = arithmetic_mean(row["absolute_distance_error"] for row in pairs)
    score = 100.0 * max(0.0, 1.0 - mean_error / threshold)
    return {
        "metric_id": "style_consistency",
        "score": score,
        "status": "numeric",
        "pairs": pairs,
        "mean_absolute_distance_error": mean_error,
        "threshold": threshold,
        "histogram_bins_per_channel": bins,
        "distance": "sqrt Jensen-Shannon",
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "metric_id": "style_consistency",
        "aggregation": "arithmetic mean across cases",
        **aggregate_numeric_cases(rows),
    }
