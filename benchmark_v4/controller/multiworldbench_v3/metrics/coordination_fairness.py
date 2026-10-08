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
SCORING_VERSION = "fairness.v3.active_channels_directional_path_min_deviation_floor_0p1"
DEFAULT_NORMALIZATION_FLOORS = {
    "travel_distance": 0.5,
    "path_length": 0.5,
    "normalized_path_deviation": 0.1,
    "heading_change_deg": 10.0,
}


def normalization_floors(overrides=None):
    values = dict(DEFAULT_NORMALIZATION_FLOORS)
    if overrides is not None:
        if not isinstance(overrides, Mapping) or set(overrides) - set(values):
            raise ValueError("unknown normalization floor feature")
        values.update(overrides)
    for feature, value in values.items():
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{feature} normalization floor must be finite and positive")
        values[feature] = value
    return values


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


def trajectory_response(poses, depth_scale: float, *, ground_normal=None) -> dict[str, float]:
    """Extract response features; ground_normal is in the poses' world frame.

    Without a gravity/map normal, retain the full 3-D trajectory. Never infer
    ground from the camera down axis, which includes pitch/roll.
    """
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
    planar = delta
    if ground_normal is not None:
        normal = np.asarray(ground_normal, dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError("ground_normal must be a finite world-frame 3-vector")
        norm = float(np.linalg.norm(normal))
        if norm <= EPS:
            raise ValueError("ground_normal must be nonzero")
        normal = normal / norm
        planar = delta - np.outer(delta @ normal, normal)
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
    # atan2 avoids acos amplifying trace roundoff near identity.
    skew = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ]) * 0.5
    heading_change = math.degrees(math.atan2(float(np.linalg.norm(skew)), cosine))
    from ..engines.rotation import interpolate_rotations, rotation_magnitude
    relative_path = np.einsum("ij,njk->nik", value[0,:3,:3].T, value[:,:3,:3])
    profile = interpolate_rotations(np.linspace(0,1,len(value)), relative_path, np.linspace(0,1,20))
    angular_path = float(np.rad2deg(rotation_magnitude(np.einsum("nji,njk->nik",relative_path[:-1],relative_path[1:]))).sum())
    return {
        "rotation_profile": profile.tolist(),
        "angular_path_deg": angular_path,
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


def distribution_similarity(alpha: Sequence[float], bravo: Sequence[float], *, normalizer_floor: float = 0.5) -> dict[str, float]:
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
    floor = float(normalizer_floor)
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("normalizer_floor must be finite and positive")
    effective = max(denominator, floor)
    similarity = 100.0 * (1.0 - min(wasserstein / effective, 1.0))
    return {
        "score": similarity, "wasserstein": wasserstein,
        "normalizer": effective, "raw_normalizer": denominator,
        "normalizer_floor": floor,
    }


def rotation_distribution_similarity(alpha, bravo, floor):
    """Optimal assignment of whole SO(3) paths; sign and intermediate motion retained."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from ..engines.rotation import rotation_magnitude
    n=max(32,len(alpha),len(bravo))
    def expand(samples):
        order=sorted(samples,key=lambda x:x["angular_path_deg"])
        return [order[min(int((i+.5)*len(order)/n),len(order)-1)] for i in range(n)]
    aa,bb=expand(alpha),expand(bravo)
    costs=np.empty((n,n))
    for i,a in enumerate(aa):
        r=np.asarray(a["rotation_profile"],float)
        if r.shape!=(20,3,3) or not np.isfinite(r).all():raise ValueError("invalid rotation profile")
        for j,b in enumerate(bb):
            q=np.asarray(b["rotation_profile"],float)
            if q.shape!=(20,3,3) or not np.isfinite(q).all():raise ValueError("invalid rotation profile")
            distance=float(np.rad2deg(rotation_magnitude(np.einsum("nji,njk->nik",r,q))).mean())
            # Path length additionally catches overshoot/return with equal endpoints.
            denominator=max((a["angular_path_deg"]+b["angular_path_deg"])/2,floor)
            costs[i,j]=min((distance+abs(a["angular_path_deg"]-b["angular_path_deg"]))/denominator,1.)
    i,j=linear_sum_assignment(costs)
    return {"score":100*(1-float(costs[i,j].mean())),"comparison":"SO3 path geodesic plus angular path length; optimal distribution assignment"}

def score_case(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Score commanded channels only, with an explicit observed-response gate."""
    floors=normalization_floors(evidence.get("normalization_floors"))
    distributions=evidence.get("action_distributions")
    channels=evidence.get("action_channels")
    base={"metric_id":"coordination_fairness","ranking_ready":False,"scoring_version":SCORING_VERSION}
    if not isinstance(distributions,Mapping) or not distributions:
        return dict(base,score=None,status="not_evaluable",missing=["action_distributions"])
    if not isinstance(channels,Mapping):
        return dict(base,score=None,status="not_evaluable",missing=["actual_action_channels"])
    rows=[];missing=[]
    for action,roles in sorted(distributions.items()):
        active=channels.get(action)
        if not isinstance(active,Mapping) or set(active)!={"translation","rotation"} or any(type(v)is not bool for v in active.values()):
            missing.append({"action":action,"reason":"missing actual keyboard/camera channel evidence"});continue
        samples={r:roles.get(r) for r in ROLES}
        if any(not isinstance(samples[r],list) or not samples[r] for r in ROLES):
            missing.append({"action":action,"reason":"missing A/B response samples"});continue
        channel_scores={};features={};gates={}
        try:
            for channel in ("translation","rotation"):
                if not active[channel]:continue
                feature="path_length" if channel=="translation" else "angular_path_deg"
                for role in ROLES:
                    for sample in samples[role]:
                        amplitude=float(sample[feature]);threshold=float(sample["response_thresholds"][channel])
                        if not math.isfinite(amplitude) or not math.isfinite(threshold) or amplitude<0 or threshold<0:
                            raise ValueError("response amplitude/threshold must be finite and nonnegative")
                responded={r:[float(x[feature])>float(x["response_thresholds"][channel]) for x in samples[r]] for r in ROLES}
                gates[channel]=responded
                if not any(any(v) for v in responded.values()):
                    channel_scores[channel]=0.;continue
                if channel=="translation":
                    for f in FEATURES[:3]:
                        features[f]=distribution_similarity([x[f] for x in samples["alpha"]],[x[f] for x in samples["bravo"]],normalizer_floor=floors[f])
                    channel_scores[channel]=min(features[f]["score"] for f in FEATURES[:3])
                else:
                    features["rotation_path"]=rotation_distribution_similarity(samples["alpha"],samples["bravo"],floors["heading_change_deg"])
                    channel_scores[channel]=features["rotation_path"]["score"]
                # Nonresponse is an execution failure, including noise-identical failures.
                responsiveness=min(sum(v)/len(v) for v in responded.values())
                channel_scores[channel]*=responsiveness
        except (KeyError,TypeError,ValueError) as exc:
            missing.append({"action":action,"reason":"invalid/missing response evidence: "+str(exc)});continue
        if not channel_scores:
            missing.append({"action":action,"reason":"idle command has no scored channel"});continue
        rows.append({"action":action,"score":min(channel_scores.values()),
                     "active_channels":active,"channel_scores":channel_scores,"response_gate":gates,
                     "sample_count":{r:len(samples[r]) for r in ROLES},"features":features})
    if missing or not rows:
        return dict(base,score=None,status="not_evaluable",missing_actions=missing,components={"actions":rows},
                    reason="required matched action evidence incomplete; no selective averaging")
    keys={k for row in rows for k in row["features"]}
    return dict(base,score=arithmetic_mean([r["score"] for r in rows]),status="diagnostic_unranked",
                components={"actions":rows,"missing_actions":[],"feature_scores":{k:arithmetic_mean([r["features"][k]["score"] for r in rows if k in r["features"]]) for k in keys}},
                normalization_floors=floors,relative_pose_required=False,
                formula="mean actions(min active feature/channel similarities after minimum A/B response fraction gate); idle channels excluded",
                trajectory_geometry=evidence.get("trajectory_geometry","unspecified"))


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Use the weaker member of every base/counter role-swap family."""
    rows = list(rows)
    versions = {row.get("scoring_version", "legacy_or_unspecified") for row in rows if row.get("score") is not None}
    if len(versions) > 1:
        raise ValueError("cannot aggregate mixed fairness scoring versions")
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
        "scoring_version": next(iter(versions), SCORING_VERSION),
        "ranking_ready": False,
        "aggregation": "minimum within role-swap family, arithmetic mean across families",
        "not_evaluable_cases": not_evaluable, "families": family_rows,
        "relative_pose_required": False,
        "definition": "A/B action-conditioned navigation-response distribution similarity",
        **aggregate_numeric_cases(family_rows),
    }
