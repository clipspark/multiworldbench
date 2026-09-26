"""Action-grounded cross-view motion consistency."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import aggregate_numeric_cases, score_0_100


def score_case(model_root, alpha_actions, bravo_actions, *, case_id: str) -> dict[str, Any]:
    # The existing optical-flow implementation selects episodes from recorded
    # native actions and separates global observer motion from local actor motion.
    from ..engines.cross_view_flow import model_score
    diagnostics = model_score(
        model_root, alpha_actions, bravo_actions, case_id=case_id, strict=True
    )
    score = score_0_100(
        diagnostics["observed_self_agreement"], "observed_self_agreement"
    )
    return {
        "metric_id": "cross_view_motion_consistency",
        "score": score,
        "status": "diagnostic",
        "headline_component": "observed_self_agreement",
        "diagnostics": {
            "interference_isolation": diagnostics["interference_isolation"],
            "amplitude_parity": diagnostics["amplitude_parity"],
            "cross_player_score": diagnostics["cross_player_score"],
        },
        "directions": diagnostics["directions"],
        "action_selection": diagnostics["action_selection"],
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summary = aggregate_numeric_cases(rows)
    return {
        "metric_id": "cross_view_motion_consistency",
        "status": "diagnostic",
        "aggregation": "arithmetic mean of observed_self_agreement; diagnostics excluded",
        **summary,
    }
