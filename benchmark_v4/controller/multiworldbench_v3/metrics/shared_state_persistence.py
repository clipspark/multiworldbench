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
    block_id = expected_after if operation == "place" else expected_before
    block_name = str(case.get("block_name_zh", block_id))
    if operation == "place":
        transition = (
            f"the cross-center target changes from empty air to exactly one full {block_name} block "
            f"({block_id}); the placed cube must visibly occupy the center cell"
        )
    else:
        transition = (
            f"the one full one-meter {block_name} cube ({block_id}) at the cross center disappears, "
            "leaving the target cell empty/exposed. Judge the full cube, not whether any same-colored pixels "
            "remain: a small rotating or floating collectible item is merely a drop produced by destruction "
            "and MUST be ignored. A dropped item is much smaller than a 1x1x1 block, does not occupy the target "
            "cell, and does not need to disappear"
        )
    schema = "{" + ",".join(f'\"{key}\":0 or 1' for key in OUTPUT_KEYS) + "}"
    return (
        "Judge whether an expected Minecraft world-state change occurred by comparing BEFORE and AFTER images. "
        "There are two independent temporal comparisons from the same generated video. Image 1 is a labelled "
        "side-by-side comparison for operator B: its left panel is BEFORE and its right panel is AFTER the operation. "
        "Image 2 is the same labelled BEFORE/AFTER comparison for observer A after returning to the target. First "
        "apply a visibility gate independently to each comparison: the "
        "pair: the colored ground cross and its center operation cell must be clearly visible in BOTH images of that "
        "pair. If the cross or center is absent, offscreen, hidden, or ambiguous, that pair MUST receive 0 and you "
        "must not infer success from hallucinated disappearance. Focus only on the target cell at the CENTER of the "
        "colored ground cross. "
        f"The expected operation is {operation}; exactly {expected_count} block changes; expected transition: "
        f"{transition}. The formal target states are {expected_before} before and {expected_after} after. "
        "For each pair, output 1 only if the before image shows the expected before state AND the after image shows "
        "the expected after state, so the requested change is visibly demonstrated. A merely different camera view "
        "is not a state change. Ignore overall scene quality, viewpoint or framing differences, lighting, players, "
        "tools, HUD, particles, dropped items, and background outside the cross-center operation area. Output 0 if "
        "the count is wrong or the expected transition is absent. Set operator_expected_change_occurred from the "
        "left/right panels of Image 1 and returning_observer_expected_change_occurred from the left/right panels of "
        "Image 2. "
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
