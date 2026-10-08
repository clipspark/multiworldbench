"""Spatial geometry after a shared building-reveal event.

The reveal preconditions are computed once for the shared case. Building
identity is reused from cross_view_structural_coherence; this metric performs no
second VLM identity judgment.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .building_reveal import validate_preconditions
from .common import aggregate_numeric_cases, binary, score_0_100


def projected_area_consistency(predicted_pixels: Any, gt_pixels: Any) -> float:
    """Return 0--100 agreement between segmented projected areas."""
    predicted = float(predicted_pixels)
    ground_truth = float(gt_pixels)
    if not math.isfinite(predicted) or not math.isfinite(ground_truth):
        raise ValueError("segmented pixel areas must be finite")
    if predicted < 0 or ground_truth <= 0:
        raise ValueError("predicted area must be non-negative and GT area positive")
    if predicted == 0:
        return 0.0
    return 100.0 * min(predicted, ground_truth) / max(predicted, ground_truth)


def camera_angle_consistency(
    predicted_angle_deg: Any,
    gt_angle_deg: Any,
    tolerance_deg: Any = 30.0,
) -> float:
    """Linearly score wrapped camera-angle error, reaching zero at tolerance."""
    predicted = float(predicted_angle_deg)
    ground_truth = float(gt_angle_deg)
    tolerance = float(tolerance_deg)
    if not all(math.isfinite(v) for v in (predicted, ground_truth, tolerance)):
        raise ValueError("camera angles and tolerance must be finite")
    if tolerance <= 0:
        raise ValueError("camera angle tolerance must be positive")
    difference = abs((predicted - ground_truth + 180.0) % 360.0 - 180.0)
    return 100.0 * max(0.0, 1.0 - difference / tolerance)


def _structural_evidence(evidence: Mapping[str, Any]) -> tuple[dict[str, int], int]:
    structural = evidence.get("cross_view_structural_coherence")
    if not isinstance(structural, Mapping):
        raise ValueError("cross_view_structural_coherence result is required")
    if structural.get("metric_id") != "cross_view_structural_coherence":
        raise ValueError("invalid cross-view structural result")
    preconditions = validate_preconditions(structural.get("common_preconditions"))
    questions = structural.get("questions")
    if not isinstance(questions, Mapping) or "same_building_gt" not in questions:
        raise ValueError("cross-view same_building_gt result is required")
    identity = binary(questions["same_building_gt"], "same_building_gt")
    return preconditions, identity


def score_case(evidence: Mapping[str, Any]) -> dict[str, Any]:
    preconditions, identity = _structural_evidence(evidence)

    if "projected_area_score_0_100" in evidence:
        area = score_0_100(evidence["projected_area_score_0_100"], "projected_area_score_0_100")
    elif "predicted_area_pixels" in evidence and "gt_area_pixels" in evidence:
        area = projected_area_consistency(
            evidence["predicted_area_pixels"], evidence["gt_area_pixels"]
        )
    else:
        area = None

    if "camera_angle_score_0_100" in evidence:
        angle = score_0_100(evidence["camera_angle_score_0_100"], "camera_angle_score_0_100")
    elif "predicted_camera_angle_deg" in evidence and "gt_camera_angle_deg" in evidence:
        angle = camera_angle_consistency(
            evidence["predicted_camera_angle_deg"],
            evidence["gt_camera_angle_deg"],
            evidence.get("camera_angle_tolerance_deg", 30.0),
        )
    else:
        angle = None

    components = {
        **{key: value * 100.0 for key, value in preconditions.items()},
        "same_building_gt_from_cross_view": identity * 100.0,
        "projected_area_consistency": area,
        "camera_angle_consistency": angle,
    }
    if 0 in preconditions.values():
        score = 0.0
        status = "failed_precondition"
    elif identity == 0:
        score = 0.0
        status = "failed_cross_view_identity"
    elif area is None or angle is None:
        score = None
        status = "not_evaluable"
    else:
        score = min(area, angle)
        status = "evaluable"
    return {
        "metric_id": "spatial_reasoning",
        "score": score,
        "status": status,
        "common_preconditions": preconditions,
        "dependency": {
            "metric_id": "cross_view_structural_coherence",
            "question": "same_building_gt",
            "value": identity,
        },
        "components": components,
        "aggregation": (
            "shared reveal preconditions gate the case; same_building_gt is reused from "
            "cross-view structural coherence; then take the minimum of area and angle"
        ),
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summary = aggregate_numeric_cases(rows)
    return {"metric_id": "spatial_reasoning", **summary}
