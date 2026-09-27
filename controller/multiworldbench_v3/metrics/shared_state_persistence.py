"""Score whether an off-screen shared-world change survives observer handoff."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary

OUTPUT_KEYS = (
    "operator_observed_expected_change",
    "returning_observer_observed_same_change",
)
OUTPUT_SCHEMA = {key: "0 or 1" for key in OUTPUT_KEYS}


def build_prompt(case: Mapping[str, Any]) -> str:
    operation = str(case["operation"])
    if operation not in {"place", "destroy"}:
        raise ValueError(f"unsupported operation: {operation!r}")
    expected_count = int(case["expected_change_count"])
    if expected_count != 1:
        raise ValueError("shared-state persistence expects exactly one changed block")
    expected_after = str(case["expected_after"]["target_cell"])
    marker = str(case["reference_marker"]["center_material"])
    verb = "placed" if operation == "place" else "destroyed"
    schema = "{" + ",".join(f'\"{key}\":0 or 1' for key in OUTPUT_KEYS) + "}"
    return (
        "Judge persistence of one Minecraft shared-world change from four images. "
        "Image 1 is A's initial view and Image 2 is B's initial view of the same colored cross center. "
        "Image 3 is B immediately after B performs the action, before B turns away. "
        "Image 4 is A after B has turned away and A has returned to the target. "
        f"Operation: {operation}; exactly {expected_count} block must be {verb}; expected target state: "
        f"{expected_after}; cross-center marker: {marker}. "
        "Answer operator_observed_expected_change=1 only when Image 3 visibly shows the exact expected "
        "change at the marked target. Answer returning_observer_observed_same_change=1 only when Image 4 "
        "shows that same correct changed state at the same marked target. Count only full world-grid blocks; "
        "ignore dropped items, particles, tools, players, and HUD icons. Hidden, ambiguous, or wrong evidence is 0. "
        f"Return ONLY JSON with exactly {schema}."
    )


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    if set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"expected exactly {OUTPUT_KEYS}, got {tuple(value)}")
    return {key: binary(value[key], key) for key in OUTPUT_KEYS}


def _majority(values: list[int]) -> int:
    if not values:
        raise ValueError("at least one frame judgment is required")
    counts = Counter(values)
    if counts[0] == counts[1]:
        return 0
    return 1 if counts[1] > counts[0] else 0


def score_case(frame_outputs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [validate_output(row) for row in frame_outputs]
    questions = {
        key: _majority([row[key] for row in rows])
        for key in OUTPUT_KEYS
    }
    score = questions[OUTPUT_KEYS[0]] * questions[OUTPUT_KEYS[1]]
    return {
        "metric_id": "shared_state_persistence",
        "score": score,
        "score_0_1": score,
        "score_0_100": score * 100.0,
        "status": "passed" if score else "failed",
        "questions": questions,
        "frame_outputs": rows,
        "check_frame_count": len(rows),
        "aggregation": "majority_per_question_then_strict_product",
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
        "metric_id": "shared_state_persistence",
        "score": score,
        "score_0_1": score,
        "score_0_100": None if score is None else score * 100.0,
        "operation_scores": operation_scores,
        "eligible_cases": eligible,
        "total_cases": len(rows),
        "coverage": eligible / len(rows) if rows else 0.0,
        "aggregation": "macro_mean_of_place_and_destroy_case_pass_rates",
    }
