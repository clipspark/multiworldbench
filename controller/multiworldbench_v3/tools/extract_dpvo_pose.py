#!/usr/bin/env python3
"""Extract a full-frame DPVO camera trajectory for one V3 video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch


@torch.no_grad()
def run_dpvo(cfg, network: str, video: str, calib: str):
    """Minimal DPVO runner without the demo-only plotting/evo dependency."""
    from dpvo.dpvo import DPVO
    from dpvo.stream import video_stream

    slam = None
    queue = Queue(maxsize=8)
    reader = Process(target=video_stream, args=(queue, video, calib, 1, 0))
    reader.start()
    intrinsics = None
    height = width = 0
    while True:
        timestamp, image, intrinsics_np = queue.get()
        if timestamp < 0:
            break
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics_np).cuda()
        if slam is None:
            _, height, width = image_tensor.shape
            slam = DPVO(cfg, network, ht=height, wd=width, viz=False)
        slam(timestamp, image_tensor, intrinsics)
    reader.join()
    if slam is None:
        raise RuntimeError(f"DPVO decoded no frames: {video}")
    return slam.terminate()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quaternion_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to rotation matrices."""
    values = np.asarray(quaternions, dtype=np.float64)
    values /= np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)
    x, y, z, w = values.T
    matrices = np.empty((len(values), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y*y + z*z)
    matrices[:, 0, 1] = 2 * (x*y - z*w)
    matrices[:, 0, 2] = 2 * (x*z + y*w)
    matrices[:, 1, 0] = 2 * (x*y + z*w)
    matrices[:, 1, 1] = 1 - 2 * (x*x + z*z)
    matrices[:, 1, 2] = 2 * (y*z - x*w)
    matrices[:, 2, 0] = 2 * (x*z - y*w)
    matrices[:, 2, 1] = 2 * (y*z + x*w)
    matrices[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return matrices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dpvo-root", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--horizontal-fov", type=float, default=70.0)
    args = parser.parse_args()
    video, output = args.video.resolve(), args.output.resolve()
    if output.is_file() and output.with_suffix(".json").is_file():
        print(json.dumps({"status": "cached", "output": str(output)}))
        return
    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if decoded != args.expected_frames:
        raise ValueError(f"expected {args.expected_frames} frames, found {decoded}: {video}")
    focal = width / (2.0 * math.tan(math.radians(args.horizontal_fov) / 2.0))
    intrinsics = np.array([focal, focal, width / 2.0, height / 2.0])
    sys.path.insert(0, str(args.dpvo_root))
    from dpvo.config import cfg
    cfg.merge_from_file(str(args.dpvo_root / "config/default.yaml"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(" ".join(str(value) for value in intrinsics) + "\n")
        calib = handle.name
    started = time.monotonic()
    try:
        poses, timestamps = run_dpvo(
            cfg, str(args.dpvo_root / "dpvo.pth"), str(video), calib)
    finally:
        os.unlink(calib)
    poses = np.asarray(poses, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if poses.shape != (args.expected_frames, 7):
        raise ValueError(f"unexpected DPVO pose shape {poses.shape}: {video}")
    cam_c2w = np.repeat(np.eye(4)[None], len(poses), axis=0)
    cam_c2w[:, :3, :3] = quaternion_matrix(poses[:, 3:7])
    cam_c2w[:, :3, 3] = poses[:, :3]
    if not np.isfinite(cam_c2w).all():
        raise ValueError(f"DPVO returned non-finite poses: {video}")
    np.savez_compressed(
        output, cam_c2w=cam_c2w,
        frame_indices=np.arange(len(poses), dtype=np.int64),
        timestamps=np.arange(len(poses), dtype=np.float64) / args.fps,
        depth_scale_median_patch=np.array(1.0),
        intrinsics_fullres_fx_fy_cx_cy=intrinsics,
        fps=np.array(args.fps),
    )
    metadata = {
        "status": "success", "decoded_frames": decoded,
        "returned_poses": len(poses), "frame_coverage_complete": True,
        "all_poses_finite": True, "pose_scale": 1.0,
        "pose_scale_semantics": "arbitrary_monocular_scale; ratios are intra-view",
        "video": str(video), "video_sha256": sha256(video),
        "trajectory_npz": str(output), "trajectory_sha256": sha256(output),
        "elapsed_seconds": time.monotonic() - started,
        "intrinsics_fullres_fx_fy_cx_cy": intrinsics.tolist(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
