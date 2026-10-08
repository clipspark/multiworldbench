"""Cross-view building identity after a shared reveal event."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from . import building_reveal
from .common import arithmetic_mean, binary

OUTPUT_KEYS = building_reveal.OUTPUT_KEYS
OUTPUT_SCHEMA = building_reveal.OUTPUT_SCHEMA
PROMPT = building_reveal.PROMPT


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    return building_reveal.validate_output(value)


def score_case(frame_measurements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    voted = building_reveal.vote_frames(frame_measurements)
    preconditions = voted["preconditions"]
    questions = voted["identity"]
    score = min((*preconditions.values(), *questions.values()))
    return {
        "metric_id": "cross_view_structural_coherence",
        "score": score,
        "score_0_100": score * 100,
        "status": "binary" if all(preconditions.values()) else "failed_precondition",
        "common_preconditions": preconditions,
        "questions": questions,
        "frames": voted["frames"],
        "aggregation": (
            "majority vote per shared precondition and identity question; "
            "failed precondition scores zero, otherwise minimum of the two identity questions"
        ),
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
