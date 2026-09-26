"""Action-grounded cross-player optical-flow diagnostics.

Actor/observer episodes are selected from recorded native actions; turn numbers
never carry actor semantics. This module reads videos and actions but modifies
neither.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .navigation_core import EPS, parse_actions, segments

CASE_ID = "MWB3-N-CV01A"
FLOW_NAMES = ("div", "dense", "global", "local_energy", "local_div")


@dataclass(frozen=True)
class _ActionInput:
    rows: list[dict[str, Any]]
    provenance: dict[str, Any]


def _canonical_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_actions(value: str | Path | list[dict[str, Any]], role: str) -> _ActionInput:
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"{role} actions are unreadable: {path}") from error
        source, source_sha256 = str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        rows, source, source_sha256 = value, "in_memory_rows", None
    # This is the formal native-action validation path used by navigation.
    parse_actions(rows)
    return _ActionInput(rows, {
        "source": source,
        "source_sha256": source_sha256,
        "canonical_rows_sha256": _canonical_digest(rows),
        "row_count": len(rows),
    })


def _timeline(rows: list[dict[str, Any]], strict: bool):
    parts = segments(rows, strict=strict)
    turns = {action.turn: (action, idle)
             for action, idle in zip(parts[::2], parts[1::2])}
    return parts, turns


def _fully_idle(moves, cameras, start: int, end: int) -> bool:
    return bool(np.all(np.abs(moves[start:end]) <= EPS) and
                np.all(np.abs(cameras[start:end]) <= EPS))


def _forward_only(moves, cameras, start: int, end: int) -> bool:
    chosen = moves[start:end]
    return bool(len(chosen) and np.all(np.abs(chosen[:, 0]) <= EPS) and
                np.all(np.abs(chosen[:, 1] - 1.0) <= EPS) and
                np.all(np.abs(cameras[start:end]) <= EPS))


def _direction(actor: str, observer: str, action, idle) -> dict[str, Any]:
    return {
        "actor": actor,
        "observer": observer,
        "turn_index": action.turn,
        "native_action": "forward",
        "actor_action": {
            "forward": True, "back": False, "left": False, "right": False,
            "camera": [0.0, 0.0], "all_action_rows_match": True,
        },
        "observer_action": "fully_idle",
        "action_range": [action.start, action.end],
        "idle_range": [idle.start, idle.end],
        # Flow i is video transition i -> i+1, so N frames yield N-1 flows.
        "action_flow_range": [action.start, action.end - 1],
        "idle_flow_range": [idle.start, idle.end - 1],
    }


def select_forward_swap(alpha_actions, bravo_actions, *, strict: bool = True) -> dict[str, Any]:
    """Select exactly one symmetric forward/idle role swap from actual actions."""
    alpha = _load_actions(alpha_actions, "alpha")
    bravo = _load_actions(bravo_actions, "bravo")
    aparts, aturns = _timeline(alpha.rows, strict)
    bparts, _ = _timeline(bravo.rows, strict)
    atimeline = [(p.turn, p.phase, p.start, p.end) for p in aparts]
    btimeline = [(p.turn, p.phase, p.start, p.end) for p in bparts]
    if atimeline != btimeline:
        raise ValueError("cross-player flow unevaluable: Alpha/Bravo phase timelines differ")

    amoves, acameras = parse_actions(alpha.rows)
    bmoves, bcameras = parse_actions(bravo.rows)
    alpha_actor, bravo_actor = [], []
    for turn in sorted(aturns):
        action, idle = aturns[turn]
        if (_forward_only(amoves, acameras, action.start, action.end) and
                _fully_idle(bmoves, bcameras, action.start, action.end)):
            alpha_actor.append(_direction("alpha", "bravo", action, idle))
        if (_forward_only(bmoves, bcameras, action.start, action.end) and
                _fully_idle(amoves, acameras, action.start, action.end)):
            bravo_actor.append(_direction("bravo", "alpha", action, idle))

    if len(alpha_actor) != 1 or len(bravo_actor) != 1:
        raise ValueError(
            "cross-player flow unevaluable: expected exactly one recorded "
            "Alpha-forward/Bravo-idle turn and one recorded "
            "Bravo-forward/Alpha-idle turn; found "
            f"{len(alpha_actor)} and {len(bravo_actor)}")
    if alpha_actor[0]["turn_index"] == bravo_actor[0]["turn_index"]:
        raise ValueError("cross-player flow unevaluable: actor roles did not swap across turns")
    alen = alpha_actor[0]["action_range"][1] - alpha_actor[0]["action_range"][0]
    blen = bravo_actor[0]["action_range"][1] - bravo_actor[0]["action_range"][0]
    if alen != blen:
        raise ValueError("cross-player flow unevaluable: swapped forward durations differ")
    return {
        "selection_method": "recorded_native_actions_v1",
        "pairing": "symmetric_forward_idle_role_swap",
        "action_inputs": {"alpha": alpha.provenance, "bravo": bravo.provenance},
        "directions": [alpha_actor[0], bravo_actor[0]],
    }


def smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("flow profile must be a nonempty 1D array")
    if type(width) is not int or width < 1:
        raise ValueError("smooth width must be a positive integer")
    return np.convolve(values, np.ones(width) / width, mode="same")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left, right = smooth(left), smooth(right)
    if left.shape != right.shape:
        raise ValueError("flow profiles must have equal length")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return (float(np.clip(np.dot(left, right) / denominator, 0, 1))
            if denominator > 1e-9 else 0.0)


def extract(path: str | Path, expected_frames: int) -> dict[str, np.ndarray]:
    """Extract global and local Farneback flow components from one view."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"video is unreadable: {path}")
    result = {name: [] for name in FLOW_NAMES}
    previous, frame_count = None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            height = frame.shape[0]
            gray = cv2.cvtColor(frame[int(.05*height):int(.72*height)],
                                cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (192, 112), interpolation=cv2.INTER_AREA)
            if previous is not None:
                optical = cv2.calcOpticalFlowFarneback(
                    previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
                fx, fy = optical[..., 0], optical[..., 1]
                xx = np.linspace(-1, 1, fx.shape[1], dtype=np.float32)[None, :]
                yy = np.linspace(-1, 1, fx.shape[0], dtype=np.float32)[:, None]
                dx, dy = float(np.median(fx)), float(np.median(fy))
                divergence = float(np.mean(fx*xx + fy*yy) / 2)
                dense = float(np.median(np.hypot(fx, fy)))
                global_motion = float(math.sqrt(dx*dx + dy*dy + divergence*divergence))
                # Remove camera translation/global expansion before local motion.
                rx, ry = fx - dx - 2*divergence*xx, fy - dy - 2*divergence*yy
                y0, y1 = int(.08*fx.shape[0]), int(.94*fx.shape[0])
                x0, x1 = int(.18*fx.shape[1]), int(.82*fx.shape[1])
                rx, ry = rx[y0:y1, x0:x1], ry[y0:y1, x0:x1]
                magnitude = np.hypot(rx, ry)
                threshold = np.percentile(magnitude, 90)
                mask = magnitude >= max(threshold, 1e-6)
                local_energy = float(np.mean(magnitude[mask])) if mask.any() else 0.0
                gy, gx = np.mgrid[0:magnitude.shape[0], 0:magnitude.shape[1]]
                weights, total = np.where(mask, magnitude, 0), float(np.where(mask, magnitude, 0).sum())
                if total > 1e-8:
                    cx = float((weights*gx).sum()/total)
                    cy = float((weights*gy).sum()/total)
                    radius = np.sqrt((gx-cx)**2 + (gy-cy)**2) + 1
                    local_div = float((weights*(rx*(gx-cx) + ry*(gy-cy))/radius).sum()/total)
                else:
                    local_div = 0.0
                for key, value in zip(FLOW_NAMES, (
                        divergence, dense, global_motion, local_energy, local_div)):
                    result[key].append(value)
            previous = gray
    finally:
        cap.release()
    if frame_count != expected_frames:
        raise ValueError(f"video/action frame mismatch: expected {expected_frames}, "
                         f"read {frame_count}: {path}")
    return {key: np.asarray(values, dtype=np.float64)
            for key, values in result.items()}


def _flow_slice(frame_range: Sequence[int]) -> slice:
    if len(frame_range) != 2:
        raise ValueError("frame range must be [start, end)")
    start, end = frame_range
    if type(start) is not int or type(end) is not int or start < 0 or end-start < 2:
        raise ValueError("flow frame range must contain at least two frames")
    return slice(start, end-1)


def positive_forward(view: Mapping[str, np.ndarray],
                     frame_range: Sequence[int]) -> np.ndarray:
    return np.maximum(np.asarray(view["div"])[_flow_slice(frame_range)], 0)


def excess(values: np.ndarray, action_range: Sequence[int],
           idle_range: Sequence[int]) -> float:
    action = float(np.mean(np.asarray(values)[_flow_slice(action_range)]))
    idle = float(np.mean(np.asarray(values)[_flow_slice(idle_range)]))
    return max(action-idle, 0.0)


def episode(actor: Mapping[str, np.ndarray], observer: Mapping[str, np.ndarray],
            action_range: Sequence[int], idle_range: Sequence[int]) -> dict[str, float]:
    """Score one action-selected actor/observer episode from flow arrays."""
    action_slice, idle_slice = _flow_slice(action_range), _flow_slice(idle_range)
    actor_response = max(
        float(np.mean(positive_forward(actor, action_range))) -
        float(np.mean(np.abs(np.asarray(actor["div"])[idle_slice]))), 0.0)
    observer_global_excess = excess(
        np.asarray(observer["global"]), action_range, idle_range)
    interference = (0.0 if actor_response <= 1e-8 else
                    100*(1-min(observer_global_excess/actor_response, 1)))

    actor_profile = positive_forward(actor, action_range)
    observer_profile = np.asarray(observer["local_energy"])[action_slice]
    observer_contrast = excess(
        np.asarray(observer["local_energy"]), action_range, idle_range)
    observer_action = float(np.mean(observer_profile))
    contrast_gate = observer_contrast/(observer_action+1e-8)
    speed_profile = cosine(actor_profile, observer_profile)
    actor_type = float(np.mean(
        actor_profile/(np.asarray(actor["dense"])[action_slice]+1e-4)))
    observed_type = float(np.mean(
        np.maximum(np.asarray(observer["local_div"])[action_slice], 0) /
        (observer_profile+1e-4)))
    type_agreement = float(np.clip(
        math.sqrt(max(actor_type, 0)*max(observed_type, 0)), 0, 1))
    cross_view = 100*(max(type_agreement, 0)*max(speed_profile, 0) *
                      max(contrast_gate, 0))**(1/3)
    return {
        "actor_forward_amplitude": actor_response,
        "observer_global_excess": observer_global_excess,
        "interference_isolation": interference,
        "motion_type_agreement": 100*type_agreement,
        "speed_profile_agreement": 100*speed_profile,
        "observer_action_idle_gate": 100*contrast_gate,
        "observed_self_agreement": cross_view,
    }


def model_score(root, alpha_actions, bravo_actions, *,
                case_id: str = CASE_ID, strict: bool = True) -> dict[str, Any]:
    """Score a model directory using episodes inferred from actual actions."""
    selection = select_forward_swap(alpha_actions, bravo_actions, strict=strict)
    row_count = selection["action_inputs"]["alpha"]["row_count"]
    root = Path(root)
    views = {
        "alpha": extract(root/case_id/"alpha_generated.mp4", row_count),
        "bravo": extract(root/case_id/"bravo_generated.mp4", row_count),
    }
    scored, amplitudes = {}, []
    for direction in selection["directions"]:
        key = f'{direction["actor"]}_moves_{direction["observer"]}_observes'
        result = episode(views[direction["actor"]], views[direction["observer"]],
                         direction["action_range"], direction["idle_range"])
        result["action_provenance"] = direction
        scored[key] = result
        amplitudes.append(result["actor_forward_amplitude"])
    amplitude = (100*min(amplitudes)/max(amplitudes)
                 if max(amplitudes) > 1e-8 else 0.0)
    isolation = float(np.mean([x["interference_isolation"] for x in scored.values()]))
    observed = float(np.mean([x["observed_self_agreement"] for x in scored.values()]))
    combined = (max(isolation, 0)*max(amplitude, 0)*max(observed, 0))**(1/3)
    return {
        "cross_player_score": combined,
        "interference_isolation": isolation,
        "amplitude_parity": amplitude,
        "observed_self_agreement": observed,
        "directions": scored,
        "action_selection": selection,
    }


def score_models(gamma, solaris, alpha_actions, bravo_actions, *,
                 case_id: str = CASE_ID) -> dict[str, Any]:
    models = {
        "Gamma World": model_score(gamma, alpha_actions, bravo_actions, case_id=case_id),
        "Solaris": model_score(solaris, alpha_actions, bravo_actions, case_id=case_id),
    }
    keys = ("cross_player_score", "interference_isolation",
            "amplitude_parity", "observed_self_agreement")
    return {
        "schema_version": "mwb.v3.motion.v1",
        "formal_score": False,
        "case_id": case_id,
        "name": "Cross-view motion diagnostics (action-grounded)",
        "metrics": {
            "interference_isolation": "When one player moves, penalize excess global camera motion in the idle observer; local avatar motion is excluded from the global statistic.",
            "amplitude_parity": "Ratio of weaker to stronger self-view response for the same recorded native action and duration.",
            "observed_self_agreement": "Geometric agreement of motion type, normalized speed profile, and ACTION/IDLE localization between views.",
            "cross_player_score": "Geometric mean of isolation, amplitude parity, and observed/self agreement.",
        },
        "models": models,
        "separation": {key: abs(models["Gamma World"][key]-models["Solaris"][key])
                       for key in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", required=True, type=Path)
    parser.add_argument("--solaris", required=True, type=Path)
    parser.add_argument("--alpha-actions", required=True, type=Path)
    parser.add_argument("--bravo-actions", required=True, type=Path)
    parser.add_argument("--case-id", default=CASE_ID)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = score_models(args.gamma, args.solaris, args.alpha_actions,
                           args.bravo_actions, case_id=args.case_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "separation": payload["separation"]}))


if __name__ == "__main__":
    main()
