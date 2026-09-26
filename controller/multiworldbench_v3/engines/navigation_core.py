"""Shared V3 navigation pose and native-action validation helpers.

Pose input uses OpenCV camera-to-world transforms.  This module deliberately
contains no metric aggregation: control isolation, coordination fairness, and
cross-view motion consistency each own their scoring definition.
"""

from dataclasses import dataclass
import math

import numpy as np

from .rotation import rotation_magnitude


EPS = 1e-12
KEYS = ("forward", "back", "left", "right", "jump", "sprint", "sneak")


def angles(rotations):
    return np.rad2deg(rotation_magnitude(rotations))


def rotation_steps(rotations):
    return angles(np.einsum("nji,njk->nik", rotations[:-1], rotations[1:]))


def pose_check(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3:
        raise ValueError("cam_c2w must have shape (N>=3,4,4)")
    if not np.isfinite(poses).all():
        raise ValueError("non-finite pose; tracking failure is not idle")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("invalid homogeneous pose")
    rotations = poses[:, :3, :3]
    if not np.allclose(
        np.einsum("nji,njk->nik", rotations, rotations), np.eye(3), atol=2e-3
    ):
        raise ValueError("non-orthogonal camera rotation")
    if not np.allclose(np.linalg.det(rotations), 1, atol=2e-3):
        raise ValueError("camera basis must be right-handed, not reflected")
    return poses


def parse_actions(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("actions must be a nonempty list of original rows")
    moves, cameras = [], []
    last_epoch = -math.inf
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("action"), dict):
            raise ValueError(f"missing recorded action at {index}; not idle")
        action = row["action"]
        if any(key not in action or type(action[key]) is not bool for key in KEYS):
            raise ValueError(f"missing/nonboolean native navigation key at {index}")
        if any(action[key] for key in ("jump", "sprint", "sneak")):
            raise ValueError(
                f"unsupported navigation geometry at {index}: jump/sprint/sneak"
            )
        if any(
            bool(value)
            for key, value in action.items()
            if key not in KEYS + ("camera",)
        ):
            raise ValueError(f"unsupported active input at {index}")
        camera = np.asarray(action.get("camera"), dtype=float)
        if camera.shape != (2,) or not np.isfinite(camera).all():
            raise ValueError(f"missing/non-finite camera at {index}")
        epoch = row.get("epochTime")
        if (
            not isinstance(epoch, (int, float))
            or not math.isfinite(epoch)
            or epoch < last_epoch
        ):
            raise ValueError(f"invalid/nonmonotonic action timestamp at {index}")
        last_epoch = epoch
        if type(row.get("turn_index")) is not int or row["turn_index"] < 0:
            raise ValueError(f"missing turn_index at {index}")
        if row.get("phase") not in ("ACTION", "IDLE"):
            raise ValueError(f"unknown phase at {index}")
        move = [
            int(action["right"]) - int(action["left"]),
            int(action["forward"]) - int(action["back"]),
        ]
        if row["phase"] == "IDLE" and (any(move) or np.any(camera != 0)):
            raise ValueError(f"IDLE contradicts recorded native input at {index}")
        moves.append(move)
        cameras.append(camera)
    return np.asarray(moves, dtype=float), np.asarray(cameras)


@dataclass(frozen=True)
class Segment:
    turn: int
    phase: str
    start: int
    end: int  # exclusive


def segments(rows, strict=True):
    parse_actions(rows)
    result = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or (
            rows[index]["turn_index"], rows[index]["phase"]
        ) != (rows[start]["turn_index"], rows[start]["phase"]):
            result.append(
                Segment(
                    rows[start]["turn_index"], rows[start]["phase"], start, index
                )
            )
            start = index
    expected = [
        (index, phase)
        for index in range(len(result) // 2)
        for phase in ("ACTION", "IDLE")
    ]
    if [(segment.turn, segment.phase) for segment in result] != expected:
        raise ValueError("each consecutive turn must contain ACTION then IDLE exactly once")
    for segment in result:
        length = segment.end - segment.start
        if length < 3 or (
            strict and length != (80 if segment.phase == "ACTION" else 20)
        ):
            raise ValueError(f"unexpected phase length: {segment}")
    return result
