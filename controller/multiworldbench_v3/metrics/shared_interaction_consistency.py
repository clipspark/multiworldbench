"""Shared interaction consistency with one strict score and two diagnostics."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary

OUTPUT_KEYS = (
    "actor_execution",
    "target_location_correct",
    "non_target_clean",
    "placement_material_correct",
    "observer_sync",
    "location_consistency",
)
OUTPUT_SCHEMA = {key: "0 or 1" for key in OUTPUT_KEYS}


def build_prompt(case: Mapping[str, Any]) -> str:
    """Build the fixed six-question VLM prompt for one case."""
    operation = str(case["operation"])
    roles = ", ".join(case["active_roles"])
    expected = ", ".join(
        f"{key}:{value}" for key, value in case["expected_after"].items()
    )
    material_rule = (
        "placement_material_correct: every placed target block is visually coarse dirt as specified. "
        if operation == "place"
        else "placement_material_correct: return 1 because material correctness is not applicable to destruction. "
    )
    return (
        "Judge one planned Minecraft shared-world interaction from four images. "
        "Images 1 and 2 are initial A and B; Images 3 and 4 are generated A and B "
        f"at the same post-action frame. The operation is {operation}; active role(s): {roles}; "
        f"expected final state: {expected}. "
        "actor_execution: all active roles visibly completed the intended operation. "
        "target_location_correct: every change is at the pre-aimed target. "
        "non_target_clean: no unrelated visible blocks were changed. "
        + material_rule
        + "observer_sync: the non-acting view, or both views when both act, shows the same resulting world state. "
        "location_consistency: both views place the result at the corresponding physical location. "
        "If evidence is hidden or unclear, answer 0. Return ONLY JSON with exactly "
        '{"actor_execution":0 or 1,"target_location_correct":0 or 1,'
        '"non_target_clean":0 or 1,"placement_material_correct":0 or 1,'
        '"observer_sync":0 or 1,"location_consistency":0 or 1}.'
    )


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    """Require the exact six-key binary VLM schema."""
    if set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"expected exactly {OUTPUT_KEYS}, got {tuple(value)}")
    return {key: binary(value[key], key) for key in OUTPUT_KEYS}


def score_case(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Score a case as the strict product of all required binary evidence."""
    operation = evidence.get("operation")
    components: dict[str, Any] = {key: evidence.get(key) for key in OUTPUT_KEYS}
    if operation == "destroy" and components["placement_material_correct"] is None:
        components["placement_material_correct"] = 1
    missing = [key for key, value in components.items() if value is None]
    if missing:
        return {
            "metric_id": "shared_interaction_consistency",
            "score": None,
            "score_0_100": None,
            "status": "not_evaluable",
            "missing": missing,
            "components": components,
        }
    values = {key: binary(value, key) for key, value in components.items()}
    interaction_correctness = (
        values["actor_execution"]
        * values["target_location_correct"]
        * values["non_target_clean"]
        * values["placement_material_correct"]
    )
    cross_view_synchronization = (
        values["actor_execution"]
        * values["observer_sync"]
        * values["location_consistency"]
    )
    score = interaction_correctness * values["observer_sync"] * values["location_consistency"]
    return {
        "metric_id": "shared_interaction_consistency",
        "score": score,
        "score_0_100": score * 100,
        "status": "binary",
        "components": values,
        "diagnostic_subscores": {
            "interaction_correctness": interaction_correctness,
            "cross_view_synchronization": cross_view_synchronization,
        },
        "aggregation": "strict_product",
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    eligible = [row for row in rows if row.get("score") is not None]
    fraction = arithmetic_mean([binary(row["score"], "score") for row in eligible]) if eligible else None
    diagnostics: dict[str, float | None] = {}
    for name in ("interaction_correctness", "cross_view_synchronization"):
        values = [
            binary(row["diagnostic_subscores"][name], name)
            for row in eligible
            if row.get("diagnostic_subscores", {}).get(name) is not None
        ]
        diagnostics[name] = arithmetic_mean(values) if values else None
    return {
        "metric_id": "shared_interaction_consistency",
        "score": fraction,
        "score_0_100": None if fraction is None else fraction * 100.0,
        "eligible_cases": len(eligible),
        "total_cases": len(rows),
        "coverage": len(eligible) / len(rows) if rows else 0.0,
        "diagnostic_subscores": diagnostics,
        "aggregation": "case_pass_rate_after_strict_product",
    }
