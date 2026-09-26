"""Causal interaction correctness with strict multiplicative gates."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary


def score_case(evidence: Mapping[str, Any]) -> dict[str, Any]:
    operation = evidence.get("operation")
    names = {
        "actor_execution": evidence.get("actor_execution"),
        "target_location_correct": evidence.get(
            "target_location_correct", evidence.get("location_consistency")
        ),
        "non_target_clean": evidence.get("non_target_clean"),
    }
    if names["non_target_clean"] is None and evidence.get("non_target_modification") is not None:
        names["non_target_clean"] = 1 - binary(
            evidence["non_target_modification"], "non_target_modification"
        )
    if operation == "place":
        names["placement_material_correct"] = evidence.get(
            "placement_material_correct",
            evidence.get("placement_material_consistency"),
        )
    else:
        names["placement_material_correct"] = 1
    missing = [key for key, value in names.items() if value is None]
    if missing:
        return {
            "metric_id": "interaction_fidelity",
            "score": None,
            "status": "not_evaluable",
            "missing": missing,
            "components": names,
        }
    components = {key: binary(value, key) for key, value in names.items()}
    score = 1
    for value in components.values():
        score *= value
    return {
        "metric_id": "interaction_fidelity",
        "score": score,
        "score_0_100": score * 100,
        "status": "binary",
        "components": components,
        "aggregation": "strict product",
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    eligible = [binary(row["score"], "score") for row in rows
                if row.get("score") is not None]
    fraction = arithmetic_mean(eligible) if eligible else None
    return {
        "metric_id": "interaction_fidelity",
        "score": fraction,
        "score_0_100": None if fraction is None else fraction * 100.0,
        "eligible_cases": len(eligible),
        "total_cases": len(rows),
        "coverage": len(eligible) / len(rows) if rows else 0.0,
    }
