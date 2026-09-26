"""VLM building-identity consistency across camera revisits."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary

OUTPUT_KEYS = ("gt_same_building", "generated_same_building")
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(OUTPUT_KEYS),
    "properties": {key: {"type": "integer", "enum": [0, 1]} for key in OUTPUT_KEYS},
}
PROMPT = (
    "Judge BUILDING identity across a Minecraft camera revisit. Images 1 and 2 "
    "are the initial visit and revisit in native ground truth. Images 3 and 4 "
    "are the corresponding generated views. gt_same_building asks whether GT "
    "Images 1 and 2 show the same physical building. generated_same_building "
    "asks whether generated Images 3 and 4 show the same physical building. "
    "Use distinctive geometry, openings, roofline, attached parts, and block "
    "layout. Allow viewpoint, visible side, perspective, distance, uniform scale, "
    "occlusion, lighting, and small texture differences. If either required "
    "building is absent, hidden, or unrecognizable, answer 0. Return ONLY JSON "
    'with exactly {"gt_same_building":0 or 1,"generated_same_building":0 or 1}. '
    "No explanation or percentage."
)


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"VLM output must contain exactly {OUTPUT_KEYS}")
    return {key: binary(value[key], key) for key in OUTPUT_KEYS}


def score_case(pair_measurements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = [validate_output(value) for value in pair_measurements]
    if not pairs:
        raise ValueError("at least one revisit pair is required")
    if not all(pair["gt_same_building"] == 1 for pair in pairs):
        return {
            "metric_id": "id_consistency",
            "score": None,
            "status": "not_evaluable",
            "reason": "GT identity validity gate failed",
            "pairs": pairs,
        }
    score = int(all(pair["generated_same_building"] == 1 for pair in pairs))
    return {
        "metric_id": "id_consistency",
        "score": score,
        "score_0_100": score * 100,
        "status": "binary",
        "pairs": pairs,
        "aggregation": "strict all-pairs pass after GT validity gate",
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    scores = [binary(row["score"], "score") for row in rows
              if row.get("score") is not None]
    fraction = arithmetic_mean(scores) if scores else None
    return {
        "metric_id": "id_consistency",
        "score": fraction,
        "score_0_100": None if fraction is None else fraction * 100.0,
        "eligible_cases": len(scores),
        "total_cases": len(rows),
        "coverage": len(scores) / len(rows) if rows else 0.0,
    }
