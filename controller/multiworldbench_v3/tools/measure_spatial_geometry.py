#!/usr/bin/env python3
"""Measure V3 spatial-reasoning geometry from self-contained evidence frames.

This tool deliberately consumes only V3 assets and the current evaluation run.
SAM2 measures projected building area after a visually reviewed box prompt.  A
SIFT correspondence audit records whether camera pose can be recovered without
inventing an angle from independent monocular videos.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
RUN = Path(os.environ.get("MWB3_RUN", REPO / "MultiWorldBench/v3/results/latest")).resolve()
FRAME_ROOT = RUN / "evaluation/frames"
OUTPUT_ROOT = RUN / "evaluation/spatial_geometry"
SAM_ROOT = Path(os.environ.get(
    "MWB3_SAM_ROOT",
    "/data2/xuzhangbo/multihuman_motion/_bench/VMBench/Grounded-SAM-2",
))
SAM_CHECKPOINT = Path(os.environ.get(
    "MWB3_SAM_CHECKPOINT",
    "/data2/xuzhangbo/multihuman_motion/_dataset/OpenS2V-Nexus/sam2.1_hiera_large.pt",
))
CHECK_FRAMES = (80, 100, 120)

# Boxes and positive points are stored in the normalized 480x320 V3 evidence
# coordinate system.  They were reviewed against all three check frames.  This
# is evaluation evidence, not a dependency on an earlier benchmark version.
PROMPTS = {
    "gt": {"box": [160, 95, 300, 205], "points": [[205, 145], [181, 186], [284, 163]]},
    "solaris": {"box": [215, 112, 305, 195], "points": [[233, 155], [278, 162]]},
}


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def frame(model: str, role: str, number: int) -> Path:
    output = FRAME_ROOT / model / "spatial/MWB3-S-SR05" / f"{role}_{number:03d}.png"
    if output.is_file():
        return output
    if model == "gt":
        source = REPO / "MultiWorldBench/v3/assets/spatial/MWB3-S-SR05" / f"{role}_gt.mp4"
        crop = None
    elif model == "solaris":
        source = RUN / "models/solaris/spatial/MWB3-S-SR05" / f"{role}_generated.mp4"
        crop = None
    elif model == "gamma":
        source = RUN / "models/gamma/spatial/MWB3-S-SR05/generated.mp4"
        crop = "crop=iw/2:ih:0:0" if role == "alpha" else "crop=iw/2:ih:iw/2:0"
    else:
        raise KeyError(model)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = [f"select=eq(n\\,{number})"]
    if crop:
        filters.append(crop)
    filters.append("scale=480:320:flags=lanczos")
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", ",".join(filters), "-frames:v", "1", str(output),
    ], check=True)
    return output


def sam_measure(predictor: Any, model: str, number: int) -> dict[str, Any]:
    path = frame(model, "bravo", number)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prompt = PROMPTS[model]
    box = np.asarray(prompt["box"], dtype=np.float32)
    points = np.asarray(prompt["points"], dtype=np.float32)
    labels = np.ones(len(points), dtype=np.int32)
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            box=box,
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
    selected = int(np.argmax(scores))
    mask = masks[selected].astype(bool)
    stem = OUTPUT_ROOT / f"{model}_bravo_{number:03d}"
    cv2.imwrite(str(stem) + "_mask.png", mask.astype(np.uint8) * 255)
    overlay = rgb.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.asarray([255, 0, 180])).astype(np.uint8)
    cv2.rectangle(overlay, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (0, 255, 255), 1)
    cv2.imwrite(str(stem) + "_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {
        "pixels": int(mask.sum()),
        "confidence": float(scores[selected]),
        "box_xyxy": box.tolist(),
        "positive_points": points.tolist(),
        "box_source": "V3 normalized evidence; visually reviewed across frames 80, 100, 120",
    }


def sift_audit(model: str, role: str, number: int) -> dict[str, Any]:
    gt_path, pred_path = frame("gt", role, number), frame(model, role, number)
    gt, pred = cv2.imread(str(gt_path), 0), cv2.imread(str(pred_path), 0)
    if gt is None or pred is None:
        return {"status": "not_evaluable", "reason": "missing normalized evidence frame"}
    mask = None
    if role == "bravo":
        mask = np.zeros_like(gt)
        x1, y1, x2, y2 = PROMPTS["gt"]["box"]
        mask[y1:y2, x1:x2] = 255
    sift = cv2.SIFT_create(nfeatures=4000)
    kp_gt, desc_gt = sift.detectAndCompute(gt, mask)
    kp_pred, desc_pred = sift.detectAndCompute(pred, None)
    if desc_gt is None or desc_pred is None:
        return {"status": "not_evaluable", "matches": 0, "reason": "no SIFT descriptors"}
    pairs = cv2.BFMatcher().knnMatch(desc_gt, desc_pred, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.65 * p[1].distance]
    unique = {}
    for match in good:
        unique.setdefault(match.trainIdx, match)
    count = len(unique)
    if count < 10:
        return {
            "status": "not_evaluable",
            "matches": count,
            "required_matches": 10,
            "reason": "fewer than 10 unique distinctive correspondences for robust pose estimation",
        }
    src = np.float32([kp_gt[m.queryIdx].pt for m in unique.values()])
    dst = np.float32([kp_pred[m.trainIdx].pt for m in unique.values()])
    _, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    fraction = inlier_count / max(count, 1)
    # A homography can establish image agreement but cannot recover an
    # independent 3-D camera center from a mostly planar Minecraft facade.
    return {
        "status": "not_evaluable",
        "matches": count,
        "homography_inliers": inlier_count,
        "homography_inlier_fraction": fraction,
        "reason": "correspondences are planar/degenerate; no defensible independent 3-D camera center",
    }


def main() -> None:
    sys.path.insert(0, str(SAM_ROOT))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(SAM_CHECKPOINT),
        device="cuda",
    )
    predictor = SAM2ImagePredictor(model)
    output: dict[str, Any] = {
        "protocol": {
            "case_id": "MWB3-S-SR05",
            "frames": list(CHECK_FRAMES),
            "segmentation": "SAM2.1 Hiera Large with reviewed V3 evidence boxes and positive points",
            "area_score": "100 * min(predicted_pixels, gt_pixels) / max(predicted_pixels, gt_pixels); median over frames",
            "camera_pose": "SIFT ratio 0.65, unique matches >=10, homography audit; reject planar/insufficient evidence",
        },
        "solaris": {},
        "gamma": {},
    }
    solaris_scores = []
    solaris_frames = []
    for number in CHECK_FRAMES:
        gt = sam_measure(predictor, "gt", number)
        pred = sam_measure(predictor, "solaris", number)
        score = 100.0 * min(gt["pixels"], pred["pixels"]) / max(gt["pixels"], pred["pixels"])
        solaris_scores.append(score)
        solaris_frames.append({
            "frame": number,
            "gt": gt,
            "predicted": pred,
            "projected_area_score_0_100": score,
            "pose": {role: sift_audit("solaris", role, number) for role in ("alpha", "bravo")},
        })
    output["solaris"]["MWB3-S-SR05"] = {
        "projected_area_score_0_100": float(np.median(solaris_scores)),
        "camera_angle_score_0_100": None,
        "status": "area_evaluable_camera_angle_not_evaluable",
        "frames": solaris_frames,
        "camera_angle_reason": "neither role yielded a reliable independent non-coplanar camera pose",
    }
    # The Gamma B evidence does not contain a localizable copy of the target
    # facade.  Record this conflict instead of segmenting an unrelated distant
    # stone object and turning it into a fabricated area/angle score.
    output["gamma"]["MWB3-S-SR05"] = {
        "projected_area_score_0_100": None,
        "camera_angle_score_0_100": None,
        "status": "not_evaluable",
        "reason": "target building cannot be localized in generated B; SIFT audit has insufficient correspondences",
        "pose": {str(number): sift_audit("gamma", "bravo", number) for number in CHECK_FRAMES},
    }
    dump(RUN / "evaluation/spatial_geometry.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
