"""Shared-state synchronization with operation-specific normalized scores."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary

COMMON_KEYS = (
    "action_success",
    "same_block_count",
    "same_target_location",
)
MATERIAL_KEY = "same_expected_material"
PLACEMENT_KEYS = COMMON_KEYS + (MATERIAL_KEY,)
DESTRUCTION_KEYS = COMMON_KEYS


def output_keys(operation: str) -> tuple[str, ...]:
    if operation == "place":
        return PLACEMENT_KEYS
    if operation == "destroy":
        return DESTRUCTION_KEYS
    raise ValueError(f"unsupported operation: {operation!r}")


def output_schema(operation: str) -> dict[str, str]:
    return {key: "0 or 1" for key in output_keys(operation)}


def build_prompt(case: Mapping[str, Any]) -> str:
    operation = str(case["operation"])
    roles = ", ".join(case["active_roles"])
    verb = "placement" if operation == "place" else "destruction"
    change = "added" if operation == "place" else "removed"
    expected_count = int(case["expected_change_count"])
    if expected_count < 0:
        raise ValueError("expected_change_count must be non-negative")
    marker = case.get("reference_marker", {})
    marker_material = marker.get("center_material", "colored")
    questions = (
        f"action_success: did the intended shared-world {verb} produce the expected final state? "
        f"same_block_count: do A and B observe the same number of blocks {change}, "
        f"and is that number exactly the expected count ({expected_count})? "
        "same_target_location: do both views show the state change on the colored target square(s) "
        f"({marker_material}) embedded in the ground cross? "
    )
    if operation == "place":
        expected = ", ".join(f"{key}:{value}" for key, value in case["expected_after"].items())
        questions += (
            "same_expected_material: are the placed full blocks seen from A and B the same material, "
            f"and is that material the expected material ({expected})? "
        )
    keys = output_keys(operation)
    schema = "{" + ",".join(f'"{key}":0 or 1' for key in keys) + "}"
    count = len(keys)
    return (
        "Judge a planned Minecraft shared-world action from four images. Images 1 and 2 are initial A and B; "
        f"Images 3 and 4 are generated A and B at the same post-action frame. Operation: {operation}; "
        f"active role(s): {roles}. Answer {count} independent binary questions. "
        "Count only full blocks fixed to the world grid. Ignore floating dropped-item entities, players, tools, particles, and HUD icons. "
        + questions
        + f"Give one point for each yes. If evidence is hidden or unclear, answer 0. Return ONLY JSON with exactly {schema}."
    )


def validate_output(value: Mapping[str, Any], operation: str) -> dict[str, int]:
    keys = output_keys(operation)
    if set(value) != set(keys):
        raise ValueError(f"expected exactly {keys}, got {tuple(value)}")
    return {key: binary(value[key], key) for key in keys}


def score_case(evidence: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(evidence.get("operation", ""))
    keys = output_keys(operation)
    missing = [key for key in keys if evidence.get(key) is None]
    if missing:
        return {
            "metric_id": "shared_state_synchronization",
            "operation": operation,
            "score": None,
            "score_0_1": None,
            "score_0_100": None,
            "status": "not_evaluable",
            "missing": missing,
            "questions": {key: evidence.get(key) for key in keys},
        }
    questions = {key: binary(evidence[key], key) for key in keys}
    raw_points = sum(questions.values())
    max_points = len(keys)
    normalized = raw_points / max_points
    return {
        "metric_id": "shared_state_synchronization",
        "operation": operation,
        "score": normalized,
        "score_0_1": normalized,
        "score_0_100": normalized * 100.0,
        "raw_points": raw_points,
        "max_points": max_points,
        "status": "normalized_operation_score",
        "questions": questions,
        "aggregation": f"sum_divided_by_{max_points}",
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    groups: dict[str, list[float]] = {"place": [], "destroy": []}
    for row in rows:
        operation = row.get("operation")
        if operation in groups and row.get("score") is not None:
            groups[operation].append(float(row["score"]))
    operation_scores = {
        operation: {
            "score": arithmetic_mean(scores) if scores else None,
            "eligible_cases": len(scores),
            "total_cases": sum(1 for row in rows if row.get("operation") == operation),
        }
        for operation, scores in groups.items()
    }
    available = [item["score"] for item in operation_scores.values() if item["score"] is not None]
    score = arithmetic_mean(available) if available else None
    eligible = sum(item["eligible_cases"] for item in operation_scores.values())
    return {
        "metric_id": "shared_state_synchronization",
        "score": score,
        "score_0_1": score,
        "score_0_100": None if score is None else score * 100.0,
        "operation_scores": operation_scores,
        "eligible_cases": eligible,
        "total_cases": len(rows),
        "coverage": eligible / len(rows) if rows else 0.0,
        "aggregation": "macro_mean_of_place_and_destroy_normalized_scores",
    }
