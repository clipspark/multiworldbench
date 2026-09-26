"""Spatial reasoning after an occluder is removed.

The actor B starts aimed at a building whose view is blocked by dirt.  B removes
the dirt and must recover the correct building at the expected spatial scale and
viewpoint.  Binary event/identity checks gate two continuous geometry checks.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .common import aggregate_numeric_cases, binary, majority_vote, score_0_100

OUTPUT_KEYS = ("same_building_gt",)
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(OUTPUT_KEYS),
    "properties": {
        "same_building_gt": {"type": "integer", "enum": [0, 1]},
    },
}
PROMPT = (
    "Judge the target BUILDING after the occluding dirt has been removed. "
    "Image 1 is B native ground truth and Image 2 is generated B at the same "
    "time. Decide only whether they show the same physical building. Use its "
    "geometry, openings, roofline, attached parts, and block layout. Allow "
    "viewpoint, perspective, distance, lighting, and small texture differences. "
    "If the building is absent, still occluded, or not recognizable, answer 0. "
    'Return ONLY {"same_building_gt":0 or 1} with no explanation.'
)


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"VLM output must contain exactly {OUTPUT_KEYS}")
    return {"same_building_gt": binary(value["same_building_gt"], "same_building_gt")}


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


def _identity_vote(evidence: Mapping[str, Any]) -> int:
    if "same_building_gt_frames" in evidence:
        rows = list(evidence["same_building_gt_frames"])
        if not rows:
            raise ValueError("same_building_gt_frames must not be empty")
        votes = [
            validate_output(row)["same_building_gt"] if isinstance(row, Mapping)
            else binary(row, "same_building_gt")
            for row in rows
        ]
        return majority_vote(votes)
    if "same_building_gt" not in evidence:
        raise ValueError("same_building_gt evidence is required")
    return binary(evidence["same_building_gt"], "same_building_gt")


def score_case(evidence: Mapping[str, Any]) -> dict[str, Any]:
    removed = binary(evidence.get("occluder_removed"), "occluder_removed")
    revealed = binary(evidence.get("target_revealed"), "target_revealed")
    identity = _identity_vote(evidence)
    gates = {
        "occluder_removed": removed,
        "target_revealed": revealed,
        "same_building_gt": identity,
    }

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
        **{key: value * 100.0 for key, value in gates.items()},
        "projected_area_consistency": area,
        "camera_angle_consistency": angle,
    }
    if 0 in gates.values():
        score = 0.0
        status = "failed_gate"
    elif area is None or angle is None:
        score = None
        status = "not_evaluable"
    else:
        score = min(components.values())
        status = "evaluable"
    return {
        "metric_id": "spatial_reasoning",
        "score": score,
        "status": status,
        "components": components,
        "aggregation": "three binary gates, then minimum across all five subdimensions",
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summary = aggregate_numeric_cases(rows)
    return {"metric_id": "spatial_reasoning", **summary}
