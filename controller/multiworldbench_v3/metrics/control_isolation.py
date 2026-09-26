"""V3 control-isolation metric using native actions and DPVO pose trajectories."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .common import aggregate_numeric_cases, arithmetic_mean, score_0_100


def score_case(
    alpha_rows, alpha_poses, alpha_scale,
    bravo_rows, bravo_poses, bravo_scale,
    phase_pairs, *, fps: int = 20, response_floors=None,
    evidence_policy: str = "recorded_native",
) -> dict[str, Any]:
    # Import lazily so registry and case validation do not require the DPVO stack.
    from ..engines.control_isolation_pose import (
        score_control_isolation,
    )
    result = score_control_isolation(
        alpha_rows, alpha_poses, alpha_scale,
        bravo_rows, bravo_poses, bravo_scale,
        phase_pairs, fps=fps, response_floors=response_floors,
        evidence_policy=evidence_policy,
    )
    result["metric_id"] = "control_isolation"
    result["headline_rule"] = "weaker causal direction"
    return result


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Take the weaker role-swap direction within each independence family."""
    rows = list(rows)
    families: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        family = str(row.get("independence_group") or row.get("case_id"))
        families[family].append(score_0_100(row["score"], "score"))
    family_rows = [
        {"independence_group": family, "score": min(values), "case_scores": values}
        for family, values in sorted(families.items())
    ]
    summary = aggregate_numeric_cases(family_rows)
    return {
        "metric_id": "control_isolation",
        "aggregation": "minimum within role-swap family, arithmetic mean across families",
        "families": family_rows,
        **summary,
    }
