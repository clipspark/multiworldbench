"""Score whether an off-screen shared-world change survives observer handoff."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary

OUTPUT_KEYS = (
    "operator_expected_change_occurred",
    "returning_observer_expected_change_occurred",
)
OUTPUT_SCHEMA = {key: "0 or 1" for key in OUTPUT_KEYS}


def build_prompt(case: Mapping[str, Any]) -> str:
    operation = str(case["operation"])
    if operation not in {"place", "destroy"}:
        raise ValueError(f"unsupported operation: {operation!r}")
    expected_count = int(case["expected_change_count"])
    if expected_count != 1:
        raise ValueError("shared-state persistence expects exactly one changed block")
    expected_before = str(case["expected_before"]["target_cell"])
    expected_after = str(case["expected_after"]["target_cell"])
    if operation == "place":
        transition = (
            "the cross-center target changes from empty air to exactly one full coarse-dirt block"
        )
    else:
        transition = (
            "the one full one-meter coarse-dirt cube at the cross center disappears, leaving the target cell "
            "empty/exposed. Judge the full cube, not whether any brown pixels remain: a small rotating or "
            "floating brown collectible item is merely a drop produced by destruction and MUST be ignored. "
            "A dropped item is much smaller than a 1x1x1 block, does not occupy the target cell, and does not "
            "need to disappear. If the full cube is gone and the ground/cross center is visible, the after "
            "state is air and the destroy transition MUST receive 1 even when that small item remains"
        )
    schema = "{" + ",".join(f'\"{key}\":0 or 1' for key in OUTPUT_KEYS) + "}"
    return (
        "Judge whether an expected Minecraft world-state change occurred by comparing BEFORE and AFTER images. "
        "There are two independent temporal pairs from the same generated video. Image 1 is operator B BEFORE the "
        "operation and Image 2 is operator B AFTER the operation. Image 3 is observer A BEFORE the operation and "
        "Image 4 is observer A AFTER returning to the target. Focus only on the target cell at the CENTER of the "
        "colored ground cross. "
        f"The expected operation is {operation}; exactly {expected_count} block changes; expected transition: "
        f"{transition}. The formal target states are {expected_before} before and {expected_after} after. "
        "For each pair, output 1 only if the before image shows the expected before state AND the after image shows "
        "the expected after state, so the requested change is visibly demonstrated. A merely different camera view "
        "is not a state change. Ignore overall scene quality, viewpoint or framing differences, lighting, players, "
        "tools, HUD, particles, dropped items, and background outside the cross-center operation area. Output 0 if "
        "the cross center is offscreen, hidden, ambiguous, the count is wrong, or the expected transition is absent. "
        "Set operator_expected_change_occurred for Images 1-2 and "
        "returning_observer_expected_change_occurred for Images 3-4. "
        f"Return ONLY JSON with exactly {schema}."
    )


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    if set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"expected exactly {OUTPUT_KEYS}, got {tuple(value)}")
    return {key: binary(value[key], key) for key in OUTPUT_KEYS}


def score_case(frame_outputs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [validate_output(row) for row in frame_outputs]
    if len(rows) != 1:
        raise ValueError("shared-state persistence expects exactly one stable post-action check frame")
    questions = dict(rows[0])
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
        "aggregation": "single_check_frame_then_strict_product",
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
