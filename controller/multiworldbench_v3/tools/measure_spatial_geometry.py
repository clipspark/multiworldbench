#!/usr/bin/env python3
"""Measure geometry for the shared V3 building-reveal case suite.

The tool uses only the current run and self-contained V3 assets. SAM prompts are
reviewed evidence stored outside the code so a box from one scene can never be
silently reused for another. Missing prompts or unreliable feature geometry are
reported as not evaluable.
"""

from __future__ import annotations

import json
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
PROMPT_PATH = Path(os.environ.get(
    "MWB3_GEOMETRY_PROMPTS", RUN / "evaluation/spatial_geometry_prompts.json"
)).resolve()
SAM_ROOT = Path(os.environ.get(
    "MWB3_SAM_ROOT", "/data2/xuzhangbo/multihuman_motion/_bench/VMBench/Grounded-SAM-2"
))
SAM_CHECKPOINT = Path(os.environ.get(
    "MWB3_SAM_CHECKPOINT",
    "/data2/xuzhangbo/multihuman_motion/_dataset/OpenS2V-Nexus/sam2.1_hiera_large.pt",
))
CHECK_FRAMES = (80, 100, 120)
CASE_IDS = tuple(f"MWB3-S-RV{index:02d}" for index in range(1, 7))
ROLE = "alpha"


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def frame(model: str, case_id: str, number: int) -> Path:
    output = FRAME_ROOT / model / "reveal" / case_id / f"{ROLE}_{number:03d}.png"
    if output.is_file():
        return output
    if model == "gt":
        source = REPO / "MultiWorldBench/v3/assets/reveal" / case_id / f"{ROLE}_gt.mp4"
        crop = None
    elif model == "solaris":
        source = RUN / "models/solaris/reveal" / case_id / f"{ROLE}_generated.mp4"
        crop = None
    elif model == "gamma":
        source = RUN / "models/gamma/reveal" / case_id / "generated.mp4"
        crop = "crop=iw/2:ih:0:0"
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


def prompt_for(prompts: dict[str, Any], case_id: str, model: str, number: int) -> dict[str, Any]:
    try:
        value = prompts["cases"][case_id][model][str(number)]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"missing reviewed SAM prompt: {case_id}/{model}/{number}") from exc
    if set(value) != {"box", "points"} or len(value["box"]) != 4 or not value["points"]:
        raise ValueError(f"invalid reviewed SAM prompt: {case_id}/{model}/{number}")
    return value


def sam_measure(predictor: Any, prompts: dict[str, Any], case_id: str,
                model: str, number: int) -> dict[str, Any]:
    path = frame(model, case_id, number)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prompt = prompt_for(prompts, case_id, model, number)
    box = np.asarray(prompt["box"], dtype=np.float32)
    points = np.asarray(prompt["points"], dtype=np.float32)
    labels = np.ones(len(points), dtype=np.int32)
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            box=box, point_coords=points, point_labels=labels, multimask_output=True,
        )
    selected = int(np.argmax(scores))
    mask = masks[selected].astype(bool)
    stem = OUTPUT_ROOT / case_id / f"{model}_{ROLE}_{number:03d}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(stem) + "_mask.png", mask.astype(np.uint8) * 255)
    overlay = rgb.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.asarray([255, 0, 180])).astype(np.uint8)
    cv2.rectangle(overlay, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (0, 255, 255), 1)
    cv2.imwrite(str(stem) + "_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {
        "pixels": int(mask.sum()), "confidence": float(scores[selected]),
        "box_xyxy": box.tolist(), "positive_points": points.tolist(),
        "box_source": str(PROMPT_PATH),
    }


def sift_audit(prompts: dict[str, Any], case_id: str, model: str, number: int) -> dict[str, Any]:
    gt_path, pred_path = frame("gt", case_id, number), frame(model, case_id, number)
    gt, pred = cv2.imread(str(gt_path), 0), cv2.imread(str(pred_path), 0)
    if gt is None or pred is None:
        return {"status": "not_evaluable", "reason": "missing normalized evidence frame"}
    mask = np.zeros_like(gt)
    x1, y1, x2, y2 = map(int, prompt_for(prompts, case_id, "gt", number)["box"])
    mask[y1:y2, x1:x2] = 255
    sift = cv2.SIFT_create(nfeatures=4000)
    kp_gt, desc_gt = sift.detectAndCompute(gt, mask)
    kp_pred, desc_pred = sift.detectAndCompute(pred, None)
    if desc_gt is None or desc_pred is None:
        return {"status": "not_evaluable", "matches": 0, "reason": "no SIFT descriptors"}
    pairs = cv2.BFMatcher().knnMatch(desc_gt, desc_pred, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.65 * pair[1].distance]
    unique = {match.trainIdx: match for match in good}
    count = len(unique)
    if count < 10:
        return {"status": "not_evaluable", "matches": count, "required_matches": 10,
                "reason": "fewer than 10 unique distinctive correspondences"}
    src = np.float32([kp_gt[item.queryIdx].pt for item in unique.values()])
    dst = np.float32([kp_pred[item.trainIdx].pt for item in unique.values()])
    _, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return {
        "status": "not_evaluable", "matches": count,
        "homography_inliers": inlier_count,
        "homography_inlier_fraction": inlier_count / max(count, 1),
        "reason": "planar correspondences do not recover a defensible independent 3-D camera angle",
    }


def score_case(predictor: Any, prompts: dict[str, Any], case_id: str,
               model: str) -> dict[str, Any]:
    rows, area_scores = [], []
    try:
        for number in CHECK_FRAMES:
            gt = sam_measure(predictor, prompts, case_id, "gt", number)
            predicted = sam_measure(predictor, prompts, case_id, model, number)
            score = 100.0 * min(gt["pixels"], predicted["pixels"]) / max(gt["pixels"], predicted["pixels"])
            area_scores.append(score)
            rows.append({
                "frame": number, "gt": gt, "predicted": predicted,
                "projected_area_score_0_100": score,
                "pose": sift_audit(prompts, case_id, model, number),
            })
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {"projected_area_score_0_100": None, "camera_angle_score_0_100": None,
                "status": "not_evaluable", "reason": str(exc)}
    return {
        "projected_area_score_0_100": float(np.median(area_scores)),
        "camera_angle_score_0_100": None,
        "status": "area_evaluable_camera_angle_not_evaluable",
        "frames": rows,
        "camera_angle_reason": "no reliable independent non-coplanar camera pose",
    }


def main() -> None:
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError(
            f"reviewed per-case SAM prompts are required: {PROMPT_PATH}; "
            "do not reuse coordinates from another case"
        )
    prompts = load(PROMPT_PATH)
    sys.path.insert(0, str(SAM_ROOT))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(SAM_CHECKPOINT), device="cuda")
    predictor = SAM2ImagePredictor(model)
    output: dict[str, Any] = {
        "protocol": {
            "case_suite": "building_reveal_shared", "case_ids": list(CASE_IDS),
            "role": ROLE, "frames": list(CHECK_FRAMES), "prompt_file": str(PROMPT_PATH),
            "segmentation": "SAM2.1 Hiera Large with reviewed per-case/per-frame prompts",
            "area_score": "100 * min(predicted_pixels, gt_pixels) / max(predicted_pixels, gt_pixels); median over frames",
            "camera_pose": "SIFT ratio 0.65, unique matches >=10; reject planar/insufficient geometry",
        },
        "solaris": {}, "gamma": {},
    }
    for candidate in ("solaris", "gamma"):
        for case_id in CASE_IDS:
            output[candidate][case_id] = score_case(predictor, prompts, case_id, candidate)
    dump(RUN / "evaluation/spatial_geometry.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
