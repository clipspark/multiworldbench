"""Shared-state synchronization scored by four equally weighted questions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary

OUTPUT_KEYS = (
    "actor_placement_success",
    "same_placement_state",
    "same_target_location",
    "same_expected_material",
)
OUTPUT_SCHEMA = {key: "0 or 1" for key in OUTPUT_KEYS}
MAX_SCORE = 4


def build_prompt(case: Mapping[str, Any]) -> str:
    roles = ", ".join(case["active_roles"])
    expected = ", ".join(
        f"{key}:{value}" for key, value in case["expected_after"].items()
    )
    return (
        "Judge a planned Minecraft shared-world block placement from four images. "
        "Images 1 and 2 are initial A and B; Images 3 and 4 are generated A and B "
        f"at the same post-action frame. Active role(s): {roles}; expected placed material: {expected}. "
        "Answer four independent binary questions. "
        "actor_placement_success: did every active role complete its intended placement? "
        "same_placement_state: do A and B observe the same placement result, including the same number of newly placed blocks? "
        "same_target_location: do both views place each result at the corresponding intended physical location? "
        "same_expected_material: are the placed blocks seen from A and B the same material, and is that material the expected material? "
        "Give one point for each yes. If evidence is hidden or unclear, answer 0. Return ONLY JSON with exactly "
        '{"actor_placement_success":0 or 1,"same_placement_state":0 or 1,'
        '"same_target_location":0 or 1,"same_expected_material":0 or 1}.'
    )


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    if set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"expected exactly {OUTPUT_KEYS}, got {tuple(value)}")
    return {key: binary(value[key], key) for key in OUTPUT_KEYS}


def score_case(evidence: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in OUTPUT_KEYS if evidence.get(key) is None]
    if missing:
        return {
            "metric_id": "shared_state_synchronization",
            "score": None,
            "score_0_4": None,
            "score_0_100": None,
            "status": "not_evaluable",
            "missing": missing,
            "questions": {key: evidence.get(key) for key in OUTPUT_KEYS},
        }
    questions = {key: binary(evidence[key], key) for key in OUTPUT_KEYS}
    score = sum(questions.values())
    return {
        "metric_id": "shared_state_synchronization",
        "score": score,
        "score_0_4": score,
        "score_0_100": score / MAX_SCORE * 100.0,
        "max_score": MAX_SCORE,
        "status": "four_point_additive",
        "questions": questions,
        "aggregation": "one_point_per_question",
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    mean_score = arithmetic_mean(scores) if scores else None
    return {
        "metric_id": "shared_state_synchronization",
        "score": mean_score,
        "score_0_4": mean_score,
        "score_0_100": None if mean_score is None else mean_score / MAX_SCORE * 100.0,
        "max_score": MAX_SCORE,
        "eligible_cases": len(scores),
        "total_cases": len(rows),
        "coverage": len(scores) / len(rows) if rows else 0.0,
        "aggregation": "mean_case_score_0_4",
    }
