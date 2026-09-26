"""Shared-state synchronization conditioned on actor success."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary


def score_case(evidence: Mapping[str, Any]) -> dict[str, Any]:
    actor_raw = evidence.get("actor_execution")
    if actor_raw is None:
        return {
            "metric_id": "shared_state_synchronization",
            "score": None,
            "status": "not_evaluable",
            "missing": ["actor_execution"],
        }
    actor = binary(actor_raw, "actor_execution")
    if actor == 0:
        return {
            "metric_id": "shared_state_synchronization",
            "score": 0,
            "score_0_100": 0,
            "status": "actor_failure_zero",
            "components": {
                "actor_success_gate": 0,
                "observer_sync_given_actor_success": None,
                "corresponding_target_location": None,
            },
        }
    observer = evidence.get("observer_sync")
    location = evidence.get("location_consistency")
    missing = [
        name for name, value in {
            "observer_sync": observer, "location_consistency": location
        }.items() if value is None
    ]
    if missing:
        return {
            "metric_id": "shared_state_synchronization",
            "score": None,
            "status": "not_evaluable",
            "missing": missing,
        }
    observer = binary(observer, "observer_sync")
    location = binary(location, "location_consistency")
    score = actor * observer * location
    return {
        "metric_id": "shared_state_synchronization",
        "score": score,
        "score_0_100": score * 100,
        "status": "binary",
        "components": {
            "actor_success_gate": actor,
            "observer_sync_given_actor_success": observer,
            "corresponding_target_location": location,
        },
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    eligible = [binary(row["score"], "score") for row in rows
                if row.get("score") is not None]
    fraction = arithmetic_mean(eligible) if eligible else None
    return {
        "metric_id": "shared_state_synchronization",
        "score": fraction,
        "score_0_100": None if fraction is None else fraction * 100.0,
        "eligible_cases": len(eligible),
        "total_cases": len(rows),
        "coverage": len(eligible) / len(rows) if rows else 0.0,
    }
