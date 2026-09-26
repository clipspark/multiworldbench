"""DPVO-pose control-isolation scoring selected by recorded native actions.

The scorer is independent of the pose estimator. It consumes hash-validated,
full-frame camera-to-world trajectories and actual native action rows. Planned
controller actions and turn names are never treated as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
from .rotation import interpolate_rotations, rotation_magnitude

from .navigation_core import EPS, parse_actions, pose_check, rotation_steps


CHANNELS = ("translation", "rotation")
SAMPLES = 20
DEFAULT_RESPONSE_FLOORS = {"translation": 1e-3, "rotation": 5e-2}


class IsolationNotEvaluable(ValueError):
    """Recorded actions do not contain the required isolation triplet."""


@dataclass(frozen=True)
class Window:
    turn_index: int
    action_range: tuple[int, int]
    idle_range: tuple[int, int]


def _validate_action_timeline(rows: list[dict], role: str, evidence_policy: str) -> None:
    last_epoch, last_tick = -np.inf, -np.inf
    for index, row in enumerate(rows):
        if evidence_policy == "recorded_native":
            if row.get("recorded_action_source") != "live_bot_state":
                raise ValueError(
                    f"{role}[{index}] is not a live_bot_state native-action record; "
                    "planned or reconstructed controls cannot score formal control isolation"
                )
        elif evidence_policy == "synthetic_diagnostic":
            if row.get("control_source") != "synthetic_planned_control":
                raise ValueError(f"{role}[{index}] lacks explicit synthetic-control provenance")
        else:
            raise ValueError(f"unknown action evidence policy: {evidence_policy}")
        epoch, epoch_ms, tick = row.get("epochTime"), row.get("epoch_ms"), row.get("minecraft_tick")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)
                   for x in (epoch, epoch_ms, tick)):
            raise ValueError(f"{role}[{index}] is missing a finite action timestamp/tick")
        if abs(float(epoch) * 1000.0 - float(epoch_ms)) > 1.0:
            raise ValueError(f"{role}[{index}] epochTime and epoch_ms disagree")
        if float(epoch) < last_epoch or float(tick) < last_tick:
            raise ValueError(f"{role} action timestamps/ticks are not monotonic")
        last_epoch, last_tick = float(epoch), float(tick)


def _range(value: Sequence[int], frames: int, label: str) -> tuple[int, int]:
    if len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError(f"{label} must be two integer frame indices")
    start, end = value
    if start < 0 or end > frames or end - start < 3:
        raise ValueError(f"{label} is outside the full pose/action timeline")
    return start, end


def _window_rows(rows: list[dict], bounds: tuple[int, int]) -> list[dict]:
    return rows[bounds[0]:bounds[1]]


def _as_action_rows(rows: list[dict]) -> list[dict]:
    """Add parse-only phase metadata without changing native action values."""
    return [{**row, "turn_index": 0, "phase": "ACTION"} for row in rows]


def _activity(rows: list[dict]) -> dict[str, bool]:
    moves, cameras = parse_actions(_as_action_rows(rows))
    return {
        "translation": bool(np.any(np.linalg.norm(moves, axis=1) > EPS)),
        "rotation": bool(np.any(np.abs(cameras) > EPS)),
    }


def _fully_idle(rows: list[dict]) -> bool:
    moves, cameras = parse_actions(_as_action_rows(rows))
    return bool(np.all(np.abs(moves) <= EPS) and np.all(np.abs(cameras) <= EPS))


def _signature(rows: list[dict]) -> dict[str, Any]:
    moves, cameras = parse_actions(_as_action_rows(rows))
    move_sum, camera_sum = moves.sum(axis=0), cameras.sum(axis=0)
    return {
        "active": _activity(rows),
        "move_direction": [int(np.sign(x)) for x in move_sum],
        "camera_direction": [int(np.sign(x)) for x in camera_sum],
        "move_samples": int(np.count_nonzero(np.linalg.norm(moves, axis=1) > EPS)),
        "camera_l1": float(np.abs(cameras).sum()),
    }


def _signatures_match(solo: Mapping[str, Any], joint: Mapping[str, Any]) -> bool:
    if solo["active"] != joint["active"]:
        return False
    if solo["active"]["translation"] and solo["move_direction"] != joint["move_direction"]:
        return False
    if solo["active"]["rotation"] and solo["camera_direction"] != joint["camera_direction"]:
        return False
    if solo["active"]["translation"]:
        ratio = joint["move_samples"] / max(solo["move_samples"], 1)
        if not 0.75 <= ratio <= 1.25:
            return False
    if solo["active"]["rotation"]:
        ratio = joint["camera_l1"] / max(solo["camera_l1"], EPS)
        if not 0.75 <= ratio <= 1.25:
            return False
    return True


def select_isolation_triplet(
    alpha_rows: list[dict],
    bravo_rows: list[dict],
    phase_pairs: Sequence[Mapping[str, Any]],
    *,
    evidence_policy: str = "recorded_native",
) -> dict[str, Any]:
    """Find A-only, B-only and joint windows solely from actual action rows."""
    if len(alpha_rows) != len(bravo_rows) or not alpha_rows:
        raise ValueError("Alpha/Bravo action timelines differ")
    _validate_action_timeline(alpha_rows, "alpha", evidence_policy)
    _validate_action_timeline(bravo_rows, "bravo", evidence_policy)
    parse_actions(_as_action_rows(alpha_rows))
    parse_actions(_as_action_rows(bravo_rows))
    windows = []
    for turn_index, pair in enumerate(phase_pairs):
        action_range = _range(pair["action_range"], len(alpha_rows), "action_range")
        idle_range = _range(pair["idle_range"], len(alpha_rows), "idle_range")
        alpha_action = _window_rows(alpha_rows, action_range)
        bravo_action = _window_rows(bravo_rows, action_range)
        if not (_fully_idle(_window_rows(alpha_rows, idle_range)) and
                _fully_idle(_window_rows(bravo_rows, idle_range))):
            raise ValueError("recovery IDLE contradicts recorded native input")
        windows.append({
            "window": Window(turn_index, action_range, idle_range),
            "alpha": _signature(alpha_action),
            "bravo": _signature(bravo_action),
        })

    alpha_solo = [x for x in windows if any(x["alpha"]["active"].values()) and
                  not any(x["bravo"]["active"].values())]
    bravo_solo = [x for x in windows if any(x["bravo"]["active"].values()) and
                  not any(x["alpha"]["active"].values())]
    joint = [x for x in windows if any(x["alpha"]["active"].values()) and
             any(x["bravo"]["active"].values())]
    if (len(alpha_solo), len(bravo_solo), len(joint)) != (1, 1, 1):
        raise IsolationNotEvaluable(
            "expected exactly one actual A-only, B-only and joint action window; "
            f"found {len(alpha_solo)}, {len(bravo_solo)}, {len(joint)}")
    alpha_solo, bravo_solo, joint = alpha_solo[0], bravo_solo[0], joint[0]
    if not _signatures_match(alpha_solo["alpha"], joint["alpha"]):
        raise IsolationNotEvaluable("A joint action does not match A solo native action")
    if not _signatures_match(bravo_solo["bravo"], joint["bravo"]):
        raise IsolationNotEvaluable("B joint action does not match B solo native action")
    channels = [
        channel for channel in CHANNELS
        if alpha_solo["alpha"]["active"][channel]
        and bravo_solo["bravo"]["active"][channel]
        and joint["alpha"]["active"][channel]
        and joint["bravo"]["active"][channel]
    ]
    if not channels:
        raise IsolationNotEvaluable("solo/joint native actions share no evaluated channel")
    return {
        "selection_method": (
            "recorded_native_actions_isolation_triplet_v1"
            if evidence_policy == "recorded_native"
            else "synthetic_planned_actions_isolation_triplet_diagnostic_v1"
        ),
        "evidence_policy": evidence_policy,
        "channels": channels,
        "alpha_solo": alpha_solo,
        "bravo_solo": bravo_solo,
        "joint": joint,
        "idle_ranges": [x["window"].idle_range for x in windows],
    }


def select_fixed_target_trials(
    alpha_rows: list[dict],
    bravo_rows: list[dict],
    phase_pairs: Sequence[Mapping[str, Any]],
    *,
    evidence_policy: str = "recorded_native",
) -> dict[str, Any]:
    """Find two target-only references bracketing one partner-variation trial.

    The target role must execute the same recorded native action in all three
    windows.  The partner is idle in both references and active only in the
    middle trial.  Turn names and planned case actions are deliberately ignored.
    """
    if len(alpha_rows) != len(bravo_rows) or not alpha_rows:
        raise ValueError("Alpha/Bravo action timelines differ")
    _validate_action_timeline(alpha_rows, "alpha", evidence_policy)
    _validate_action_timeline(bravo_rows, "bravo", evidence_policy)
    parse_actions(_as_action_rows(alpha_rows))
    parse_actions(_as_action_rows(bravo_rows))
    windows = []
    for turn_index, pair in enumerate(phase_pairs):
        action_range = _range(pair["action_range"], len(alpha_rows), "action_range")
        idle_range = _range(pair["idle_range"], len(alpha_rows), "idle_range")
        if not (_fully_idle(_window_rows(alpha_rows, idle_range)) and
                _fully_idle(_window_rows(bravo_rows, idle_range))):
            raise ValueError("recovery IDLE contradicts recorded native input")
        windows.append({
            "window": Window(turn_index, action_range, idle_range),
            "alpha": _signature(_window_rows(alpha_rows, action_range)),
            "bravo": _signature(_window_rows(bravo_rows, action_range)),
        })
    if len(windows) != 3:
        raise IsolationNotEvaluable(
            "fixed-target isolation requires exactly three actual action windows"
        )

    matches = []
    for target, partner in (("alpha", "bravo"), ("bravo", "alpha")):
        references = [
            item for item in windows
            if any(item[target]["active"].values())
            and not any(item[partner]["active"].values())
        ]
        treatments = [
            item for item in windows
            if any(item[target]["active"].values())
            and any(item[partner]["active"].values())
        ]
        if len(references) != 2 or len(treatments) != 1:
            continue
        references.sort(key=lambda item: item["window"].turn_index)
        treatment = treatments[0]
        if not (references[0]["window"].turn_index < treatment["window"].turn_index <
                references[1]["window"].turn_index):
            continue
        if not all(_signatures_match(references[0][target], item[target])
                   for item in (references[1], treatment)):
            continue
        channels = [
            channel for channel in CHANNELS
            if all(item[target]["active"][channel]
                   for item in (*references, treatment))
        ]
        if channels:
            matches.append((target, partner, references, treatment, channels))
    if len(matches) != 1:
        raise IsolationNotEvaluable(
            "expected one target role with two matching target-only references "
            "bracketing one partner-variation window"
        )
    target, partner, references, treatment, channels = matches[0]
    return {
        "selection_method": (
            "recorded_native_actions_fixed_target_partner_variation_v1"
            if evidence_policy == "recorded_native"
            else "synthetic_planned_actions_fixed_target_partner_variation_diagnostic_v1"
        ),
        "evidence_policy": evidence_policy,
        "target_role": target,
        "partner_role": partner,
        "channels": channels,
        "references": references,
        "treatment": treatment,
        "idle_ranges": [item["window"].idle_range for item in windows],
    }


def _motion_rate(poses: np.ndarray, scale: float, fps: int) -> dict[str, float]:
    poses = pose_check(poses)
    if not np.isfinite(scale) or scale <= 0 or fps <= 0:
        raise ValueError("positive depth scale and fps are required")
    seconds = max((len(poses) - 1) / fps, 1 / fps)
    translation = np.linalg.norm(np.diff(poses[:, :3, 3] / scale, axis=0), axis=1).sum()
    rotation = rotation_steps(poses[:, :3, :3]).sum()
    return {"translation": float(translation / seconds), "rotation": float(rotation / seconds)}


def _noise_floor(values: Sequence[float]) -> float:
    center = float(median(values))
    mad = float(median(abs(x - center) for x in values))
    return center + 3.0 * mad


def _progress(rows: list[dict], channel: str) -> tuple[np.ndarray, np.ndarray]:
    moves, cameras = parse_actions(_as_action_rows(rows))
    steps = (np.linalg.norm(moves, axis=1) if channel == "translation"
             else np.linalg.norm(cameras, axis=1))
    steps = steps.copy()
    steps[0] = 0.0
    progress = np.cumsum(steps)
    if progress[-1] <= EPS:
        progress = np.arange(len(rows), dtype=float)
    progress, indices = np.unique(progress, return_index=True)
    if len(progress) < 2:
        progress = np.arange(len(rows), dtype=float)
        indices = np.arange(len(rows))
    return progress, indices


def _relative(poses: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    poses = pose_check(poses)
    r0 = poses[0, :3, :3]
    positions = ((poses[:, :3, 3] - poses[0, :3, 3]) @ r0) / scale
    rotations = np.einsum("ij,njk->nik", r0.T, poses[:, :3, :3])
    return positions, rotations


def _resample_by_action(
    rows: list[dict], poses: np.ndarray, scale: float, channel: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions, rotations = _relative(poses, scale)
    progress, indices = _progress(rows, channel)
    target = np.linspace(progress[0], progress[-1], SAMPLES)
    pp = np.column_stack([
        np.interp(target, progress, positions[indices, axis]) for axis in range(3)
    ])
    rr = interpolate_rotations(progress, rotations[indices], target)
    return pp, rr


def _profile_error(
    solo_rows: list[dict], solo_poses: np.ndarray,
    joint_rows: list[dict], joint_poses: np.ndarray,
    scale: float, channel: str,
) -> float:
    sp, sr = _resample_by_action(solo_rows, solo_poses, scale, channel)
    jp, jr = _resample_by_action(joint_rows, joint_poses, scale, channel)
    if channel == "translation":
        denominator = np.linalg.norm(np.diff(sp, axis=0), axis=1).sum()
        return float(np.mean(np.linalg.norm(jp - sp, axis=1)) / max(denominator, EPS))
    difference = rotation_magnitude(np.einsum("nji,njk->nik", sr, jr))
    denominator = np.deg2rad(rotation_steps(sr).sum())
    return float(np.mean(difference) / max(denominator, EPS))


def _slice(poses: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
    return poses[bounds[0]:bounds[1]]


def _zero_direction(reason: str, channels: Sequence[str]) -> dict[str, Any]:
    return {
        "score": 0.0,
        "status": "model_failure_zero",
        "reason": reason,
        "channels": {
            channel: {
                "score": 0.0,
                "idle_leakage_score": 0.0,
                "active_invariance_score": 0.0,
            }
            for channel in channels
        },
    }


def _score_fallback_control_isolation(
    alpha_rows: list[dict], alpha_poses: np.ndarray, alpha_scale: float,
    bravo_rows: list[dict], bravo_poses: np.ndarray, bravo_scale: float,
    phase_pairs: Sequence[Mapping[str, Any]], *, fps: int = 20,
    response_floors: Mapping[str, float] | None = None,
    evidence_policy: str = "recorded_native",
) -> dict[str, Any]:
    """Return a numeric 0--100 isolation score and both causal directions."""
    alpha_poses, bravo_poses = pose_check(alpha_poses), pose_check(bravo_poses)
    if len(alpha_rows) != len(alpha_poses) or len(bravo_rows) != len(bravo_poses):
        raise ValueError("pose/action frame timeline mismatch")
    selected = select_isolation_triplet(
        alpha_rows, bravo_rows, phase_pairs, evidence_policy=evidence_policy
    )
    floors = {**DEFAULT_RESPONSE_FLOORS, **(response_floors or {})}
    roles = {
        "alpha": {"rows": alpha_rows, "poses": alpha_poses, "scale": alpha_scale},
        "bravo": {"rows": bravo_rows, "poses": bravo_poses, "scale": bravo_scale},
    }
    noise = {}
    for role, data in roles.items():
        idle_rates = [
            _motion_rate(_slice(data["poses"], bounds), data["scale"], fps)
            for bounds in selected["idle_ranges"]
        ]
        noise[role] = {
            channel: _noise_floor([item[channel] for item in idle_rates])
            for channel in CHANNELS
        }

    directions = {}
    for actor, observer, actor_key, observer_key in (
        ("alpha", "bravo", "alpha_solo", "bravo_solo"),
        ("bravo", "alpha", "bravo_solo", "alpha_solo"),
    ):
        actor_window = selected[actor_key]["window"].action_range
        observer_window = selected[observer_key]["window"].action_range
        joint_window = selected["joint"]["window"].action_range
        actor_data, observer_data = roles[actor], roles[observer]
        actor_rate = _motion_rate(_slice(actor_data["poses"], actor_window),
                                  actor_data["scale"], fps)
        observer_solo_rate = _motion_rate(_slice(observer_data["poses"], observer_window),
                                          observer_data["scale"], fps)
        leaked_rate = _motion_rate(_slice(observer_data["poses"], actor_window),
                                   observer_data["scale"], fps)
        channel_rows, failed = {}, None
        for channel in selected["channels"]:
            actor_response = max(actor_rate[channel] - noise[actor][channel], 0.0)
            observer_response = max(observer_solo_rate[channel] - noise[observer][channel], 0.0)
            if actor_response <= floors[channel] or observer_response <= floors[channel]:
                failed = (
                    f"{actor}/{observer} {channel} control input has no measurable "
                    "DPVO response above the idle noise floor"
                )
                break
            leakage = max(leaked_rate[channel] - noise[observer][channel], 0.0)
            leakage_ratio = leakage / max(observer_response, floors[channel])
            leakage_score = 100.0 * float(np.clip(1.0 - leakage_ratio, 0.0, 1.0))
            solo_rows = _window_rows(observer_data["rows"], observer_window)
            joint_rows = _window_rows(observer_data["rows"], joint_window)
            raw_error = _profile_error(
                solo_rows, _slice(observer_data["poses"], observer_window),
                joint_rows, _slice(observer_data["poses"], joint_window),
                observer_data["scale"], channel,
            )
            noise_ratio = noise[observer][channel] / max(observer_response, floors[channel])
            invariance_error = max(raw_error - noise_ratio, 0.0)
            invariance_score = 100.0 * float(np.clip(1.0 - invariance_error, 0.0, 1.0))
            channel_rows[channel] = {
                "score": (leakage_score + invariance_score) / 2.0,
                "idle_leakage_score": leakage_score,
                "active_invariance_score": invariance_score,
                "idle_leakage_ratio": leakage_ratio,
                "active_invariance_error": invariance_error,
                "actor_response": actor_response,
                "observer_solo_response": observer_response,
                "observer_idle_noise_floor": noise[observer][channel],
            }
        key = f"{actor}_to_{observer}"
        directions[key] = (_zero_direction(failed, selected["channels"]) if failed else {
            "score": float(np.mean([x["score"] for x in channel_rows.values()])),
            "status": "numeric",
            "channels": channel_rows,
        })

    score = min(item["score"] for item in directions.values())
    return {
        "score": float(np.clip(score, 0.0, 100.0)),
        "status": "numeric",
        "aggregation": "min(directions); arithmetic_mean(channel_subscores)",
        "selection": {
            "method": selected["selection_method"],
            "evidence_policy": selected["evidence_policy"],
            "channels": selected["channels"],
            "alpha_solo_turn": selected["alpha_solo"]["window"].turn_index,
            "bravo_solo_turn": selected["bravo_solo"]["window"].turn_index,
            "joint_turn": selected["joint"]["window"].turn_index,
        },
        "noise_floors": noise,
        "response_floors": floors,
        "directions": directions,
    }


def score_fixed_target_control_isolation(
    alpha_rows: list[dict], alpha_poses: np.ndarray, alpha_scale: float,
    bravo_rows: list[dict], bravo_poses: np.ndarray, bravo_scale: float,
    phase_pairs: Sequence[Mapping[str, Any]], *, fps: int = 20,
    response_floors: Mapping[str, float] | None = None,
    evidence_policy: str = "recorded_native",
) -> dict[str, Any]:
    """Score target-trajectory invariance while only the partner condition varies."""
    alpha_poses, bravo_poses = pose_check(alpha_poses), pose_check(bravo_poses)
    if len(alpha_rows) != len(alpha_poses) or len(bravo_rows) != len(bravo_poses):
        raise ValueError("pose/action frame timeline mismatch")
    selected = select_fixed_target_trials(
        alpha_rows, bravo_rows, phase_pairs, evidence_policy=evidence_policy
    )
    floors = {**DEFAULT_RESPONSE_FLOORS, **(response_floors or {})}
    roles = {
        "alpha": {"rows": alpha_rows, "poses": alpha_poses, "scale": alpha_scale},
        "bravo": {"rows": bravo_rows, "poses": bravo_poses, "scale": bravo_scale},
    }
    noise = {}
    for role, data in roles.items():
        idle_rates = [
            _motion_rate(_slice(data["poses"], bounds), data["scale"], fps)
            for bounds in selected["idle_ranges"]
        ]
        noise[role] = {
            channel: _noise_floor([item[channel] for item in idle_rates])
            for channel in CHANNELS
        }

    target, partner = selected["target_role"], selected["partner_role"]
    data = roles[target]
    reference_bounds = [item["window"].action_range for item in selected["references"]]
    treatment_bounds = selected["treatment"]["window"].action_range
    all_bounds = [*reference_bounds, treatment_bounds]
    rates = [
        _motion_rate(_slice(data["poses"], bounds), data["scale"], fps)
        for bounds in all_bounds
    ]
    channel_rows = {}
    failed = None
    for channel in selected["channels"]:
        responses = [max(item[channel] - noise[target][channel], 0.0) for item in rates]
        if any(response <= floors[channel] for response in responses):
            failed = (
                f"{target} {channel} target input has no measurable DPVO response "
                "above the idle noise floor in every repeated trial"
            )
            break
        reference_rows = [
            _window_rows(data["rows"], bounds) for bounds in reference_bounds
        ]
        reference_poses = [
            _slice(data["poses"], bounds) for bounds in reference_bounds
        ]
        treatment_rows = _window_rows(data["rows"], treatment_bounds)
        treatment_poses = _slice(data["poses"], treatment_bounds)
        baseline_error = _profile_error(
            reference_rows[0], reference_poses[0],
            reference_rows[1], reference_poses[1],
            data["scale"], channel,
        )
        condition_errors = [
            _profile_error(
                rows, poses, treatment_rows, treatment_poses,
                data["scale"], channel,
            )
            for rows, poses in zip(reference_rows, reference_poses)
        ]
        partner_condition_error = float(np.mean(condition_errors))
        excess_error = max(partner_condition_error - baseline_error, 0.0)
        score = 100.0 * float(np.clip(1.0 - excess_error, 0.0, 1.0))
        channel_rows[channel] = {
            "score": score,
            "trajectory_invariance_score": score,
            "baseline_repeatability_error": baseline_error,
            "partner_condition_error": partner_condition_error,
            "excess_interference_error": excess_error,
            "reference_responses": responses[:2],
            "treatment_response": responses[2],
            "target_idle_noise_floor": noise[target][channel],
        }

    direction_key = f"{partner}_to_{target}"
    if failed:
        direction = {
            "score": 0.0,
            "status": "model_failure_zero",
            "reason": failed,
            "channels": {
                channel: {
                    "score": 0.0,
                    "trajectory_invariance_score": 0.0,
                }
                for channel in selected["channels"]
            },
        }
    else:
        direction = {
            "score": float(np.mean([item["score"] for item in channel_rows.values()])),
            "status": "numeric",
            "channels": channel_rows,
        }
    return {
        "score": float(np.clip(direction["score"], 0.0, 100.0)),
        "status": "numeric",
        "design": "fixed_target_partner_variation_v1",
        "aggregation": (
            "arithmetic_mean(target_channels); paired role-swap case supplies "
            "the reverse causal direction"
        ),
        "selection": {
            "method": selected["selection_method"],
            "evidence_policy": selected["evidence_policy"],
            "target_role": target,
            "partner_role": partner,
            "channels": selected["channels"],
            "reference_turns": [
                item["window"].turn_index for item in selected["references"]
            ],
            "treatment_turn": selected["treatment"]["window"].turn_index,
        },
        "noise_floors": noise,
        "response_floors": floors,
        "directions": {direction_key: direction},
    }


def score_control_isolation(
    alpha_rows: list[dict], alpha_poses: np.ndarray, alpha_scale: float,
    bravo_rows: list[dict], bravo_poses: np.ndarray, bravo_scale: float,
    phase_pairs: Sequence[Mapping[str, Any]], *, fps: int = 20,
    response_floors: Mapping[str, float] | None = None,
    evidence_policy: str = "recorded_native",
) -> dict[str, Any]:
    """Dispatch to the fixed-target protocol or the fallback isolation triplet."""
    try:
        select_fixed_target_trials(
            alpha_rows, bravo_rows, phase_pairs, evidence_policy=evidence_policy
        )
    except IsolationNotEvaluable:
        return _score_fallback_control_isolation(
            alpha_rows, alpha_poses, alpha_scale,
            bravo_rows, bravo_poses, bravo_scale,
            phase_pairs, fps=fps, response_floors=response_floors,
            evidence_policy=evidence_policy,
        )
    return score_fixed_target_control_isolation(
        alpha_rows, alpha_poses, alpha_scale,
        bravo_rows, bravo_poses, bravo_scale,
        phase_pairs, fps=fps, response_floors=response_floors,
        evidence_policy=evidence_policy,
    )


def model_failure_zero(reason: str, channels: Sequence[str] = CHANNELS) -> dict[str, Any]:
    """Explicit zero only for a validated model output that is not trackable."""
    directions = {
        "alpha_to_bravo": _zero_direction(reason, channels),
        "bravo_to_alpha": _zero_direction(reason, channels),
    }
    return {
        "score": 0.0,
        "status": "model_failure_zero",
        "reason": reason,
        "directions": directions,
    }
