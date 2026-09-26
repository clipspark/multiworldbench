"""Two-question VLM judge for cross-view building identity."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary, majority_vote

OUTPUT_KEYS = ("same_building_gt", "same_building_ab")
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(OUTPUT_KEYS),
    "properties": {key: {"type": "integer", "enum": [0, 1]} for key in OUTPUT_KEYS},
}
PROMPT = (
    "Judge the target BUILDING in three Minecraft images. "
    "Image 1 is B native ground truth, Image 2 is generated B at the same time, "
    "and Image 3 is generated A at the same time. Answer exactly two independent "
    "binary identity questions. same_building_gt: is the building in generated B "
    "the same physical building as in GT B? same_building_ab: are the buildings "
    "seen by generated A and generated B the same physical building? Use distinctive "
    "geometry, openings, roofline, attached parts, and block layout. Allow viewpoint, "
    "visible side, perspective, distance, uniform scale, occlusion, lighting, and "
    "small texture differences. If the required building is absent, hidden, or not "
    "recognizable, answer 0 for that question. Return ONLY JSON with exactly "
    '{"same_building_gt":0 or 1,"same_building_ab":0 or 1}. '
    "No explanation, orientation score, third question, or percentage."
)


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"VLM output must contain exactly {OUTPUT_KEYS}")
    return {key: binary(value[key], key) for key in OUTPUT_KEYS}


def score_case(frame_measurements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    frames = [validate_output(value) for value in frame_measurements]
    if not frames:
        raise ValueError("at least one checked frame is required")
    voted = {
        key: majority_vote(frame[key] for frame in frames)
        for key in OUTPUT_KEYS
    }
    score = min(voted.values())
    return {
        "metric_id": "cross_view_structural_coherence",
        "score": score,
        "score_0_100": score * 100,
        "status": "binary",
        "questions": voted,
        "frames": frames,
        "aggregation": "majority per question, then minimum of the two questions",
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    scores = [binary(row["score"], "score") for row in rows]
    fraction = arithmetic_mean(scores) if scores else None
    return {
        "metric_id": "cross_view_structural_coherence",
        "score": fraction,
        "score_0_100": None if fraction is None else fraction * 100.0,
        "eligible_cases": len(scores),
        "total_cases": len(rows),
    }
