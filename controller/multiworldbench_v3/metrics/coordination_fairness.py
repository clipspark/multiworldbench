"""V3 action-conditioned coordination-fairness metric.

The metric asks whether Alpha and Bravo have similar response distributions
under the same navigation command class. It needs no shared coordinate system
and no A/B relative-pose estimate.
"""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .common import aggregate_numeric_cases, arithmetic_mean, score_0_100


FEATURES = (
    "travel_distance",
    "path_length",
    "normalized_path_deviation",
    "heading_change_deg",
)
ROLES = ("alpha", "bravo")
EPS = 1e-8


def canonical_action(command: str) -> str | None:
    """Map mirrored/parameterized native commands to a response class."""
    text = str(command).strip().upper().replace(" ", "")
    if text in ("", "-"):
        return None
    has_yaw = "YAW" in text
    tokens = set(re.findall(r"(?:^|\+)([WASD])(?=\+|$)", "+" + text))
    forward = "W" in tokens
    backward = "S" in tokens
    lateral = bool(tokens & {"A", "D"})
    if forward and has_yaw:
        return "forward_turn"
    if backward and has_yaw:
        return "backward_turn"
    if forward and lateral:
        return "forward_strafe"
    if backward and lateral:
        return "backward_strafe"
    if forward:
        return "forward"
    if backward:
        return "backward"
    if lateral:
        return "strafe"
    if has_yaw:
        return "turn"
    return text.lower()


def trajectory_response(poses, depth_scale: float) -> dict[str, float]:
    """Extract coordinate-free response features from one action episode."""
    import numpy as np

    value = np.asarray(poses, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4) or len(value) < 3:
        raise ValueError("poses must have shape (N>=3,4,4)")
    if not np.isfinite(value).all():
        raise ValueError("poses contain non-finite values")
    scale = float(depth_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale must be positive")

    position = value[:, :3, 3] / scale
    delta = position - position[0]
    ground_normal = value[0, :3, 1]
    ground_normal = ground_normal / max(float(np.linalg.norm(ground_normal)), EPS)
    planar = delta - np.outer(delta @ ground_normal, ground_normal)
    steps = np.linalg.norm(np.diff(planar, axis=0), axis=1)
    path_length = float(steps.sum())
    endpoint = planar[-1]
    travel_distance = float(np.linalg.norm(endpoint))
    if travel_distance > EPS and path_length > EPS:
        chord = endpoint / travel_distance
        off_chord = planar - np.outer(planar @ chord, chord)
        deviation = float(np.sqrt(np.mean(np.sum(off_chord * off_chord, axis=1))) / path_length)
    else:
        deviation = 0.0

    relative = value[0, :3, :3].T @ value[-1, :3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    heading_change = math.degrees(math.acos(cosine))
    return {
        "travel_distance": travel_distance,
        "path_length": path_length,
        "normalized_path_deviation": deviation,
        "heading_change_deg": heading_change,
    }


def _empirical_quantiles(values: Sequence[float], count: int) -> list[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("distribution samples must be finite non-negative values")
    if len(ordered) == 1:
        return ordered * count
    result = []
    for index in range(count):
        q = (index + 0.5) / count
        position = q * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        result.append(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
    return result


def distribution_similarity(alpha: Sequence[float], bravo: Sequence[float]) -> dict[str, float]:
    """Return normalized 1-D Wasserstein similarity in [0,100]."""
    if not alpha or not bravo:
        raise ValueError("both roles need at least one sample")
    count = max(32, len(alpha), len(bravo))
    aq = _empirical_quantiles(alpha, count)
    bq = _empirical_quantiles(bravo, count)
    wasserstein = arithmetic_mean([abs(a - b) for a, b in zip(aq, bq)])
    denominator = arithmetic_mean([abs(float(x)) for x in alpha]) + arithmetic_mean(
        [abs(float(x)) for x in bravo]
    )
    if denominator <= EPS:
        similarity = 100.0 if wasserstein <= EPS else 0.0
    else:
        similarity = 100.0 * (1.0 - min(wasserstein / denominator, 1.0))
    return {"score": similarity, "wasserstein": wasserstein, "normalizer": denominator}


def score_case(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Compare A/B feature distributions separately for every action class."""
    distributions = evidence.get("action_distributions")
    if not isinstance(distributions, Mapping) or not distributions:
        return {
            "metric_id": "coordination_fairness", "score": None,
            "status": "not_evaluable", "missing": ["action_distributions"],
            "ranking_ready": False,
        }

    action_rows = []
    missing_actions = []
    for action, roles in sorted(distributions.items()):
        if not isinstance(roles, Mapping):
            raise ValueError(f"invalid action distribution: {action}")
        role_samples = {role: roles.get(role) for role in ROLES}
        if any(not isinstance(role_samples[role], list) or not role_samples[role] for role in ROLES):
            missing_actions.append(str(action))
            continue
        feature_rows = {}
        for feature in FEATURES:
            values = {}
            for role in ROLES:
                try:
                    values[role] = [sample[feature] for sample in role_samples[role]]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"missing {feature} sample for {action}/{role}") from exc
            feature_rows[feature] = distribution_similarity(values["alpha"], values["bravo"])
        action_rows.append({
            "action": str(action),
            "score": arithmetic_mean([row["score"] for row in feature_rows.values()]),
            "sample_count": {role: len(role_samples[role]) for role in ROLES},
            "features": feature_rows,
        })

    if not action_rows:
        return {
            "metric_id": "coordination_fairness", "score": None,
            "status": "not_evaluable", "missing": ["matched_alpha_bravo_action_samples"],
            "missing_actions": missing_actions, "ranking_ready": False,
        }
    feature_scores = {
        feature: arithmetic_mean([row["features"][feature]["score"] for row in action_rows])
        for feature in FEATURES
    }
    return {
        "metric_id": "coordination_fairness",
        "score": arithmetic_mean([row["score"] for row in action_rows]),
        "status": "diagnostic_unranked", "ranking_ready": False,
        "components": {
            "feature_scores": feature_scores, "actions": action_rows,
            "missing_actions": missing_actions,
        },
        "formula": (
            "mean across action classes and response features of "
            "100 * (1 - normalized 1-D Wasserstein distance)"
        ),
        "relative_pose_required": False,
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Use the weaker member of every base/counter role-swap family."""
    rows = list(rows)
    families: dict[str, list[float]] = defaultdict(list)
    not_evaluable = []
    for row in rows:
        score = row.get("score")
        case_id = str(row.get("case_id", "unknown"))
        if score is None:
            not_evaluable.append(case_id)
            continue
        family = str(row.get("independence_group") or case_id)
        families[family].append(score_0_100(score, "score"))
    family_rows = [
        {"independence_group": family, "score": min(values), "case_scores": values}
        for family, values in sorted(families.items())
    ]
    return {
        "metric_id": "coordination_fairness", "status": "diagnostic_unranked",
        "ranking_ready": False,
        "aggregation": "minimum within role-swap family, arithmetic mean across families",
        "not_evaluable_cases": not_evaluable, "families": family_rows,
        "relative_pose_required": False,
        "definition": "A/B action-conditioned navigation-response distribution similarity",
        **aggregate_numeric_cases(family_rows),
    }
