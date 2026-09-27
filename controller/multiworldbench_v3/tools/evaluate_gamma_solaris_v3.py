#!/usr/bin/env python3
"""Evaluate Gamma-World and Solaris on the concrete, self-contained V3 cases.

The script has three resumable stages:
  prepare   extract the exact evidence frames and build immutable VLM requests;
  judge     run one deterministic Qwen shard on one GPU;
  aggregate validate outputs and call the V3 metric implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUN = Path(os.environ.get("MWB3_RUN", REPO / "MultiWorldBench/v3/results/latest")).resolve()
ASSETS = REPO / "MultiWorldBench/v3/assets"
QWEN = Path("/data2/xuzhangbo/models/Qwen2.5-VL-72B-Instruct-AWQ")
FRAMES = {"structural": [60, 80, 100], "spatial": [80, 100, 120], "race": [60, 80, 100], "interaction": [60, 70, 80]}
# Gamma-World's released VAE has statistics for at most 50 latent frames and
# its runner requires a latent-frame count divisible by three.  The nearest
# legal length is 48 latent / 189 RGB frames.  The native V4 trajectory uses
# frames 0..212, so Gamma receives an action-preserving temporal compression
# to frames 0..188.  Each tuple stores native initial/return and
# the phase-aligned Gamma initial/return frame.
REVISIT_PAIRS = [
    ("v0_revisit_1", 10, 90, 9, 80),
    ("v0_revisit_2", 10, 170, 9, 151),
    ("v1_revisit_1", 50, 132, 44, 117),
    ("v1_revisit_2", 50, 212, 44, 188),
]
SOLARIS_REVISIT_PAIRS = REVISIT_PAIRS

sys.path.insert(0, str(REPO))
from controller.multiworldbench_v3.metrics import (  # noqa: E402
    cross_view_structural_coherence as structural_metric,
    id_consistency as id_metric,
    shared_state_synchronization as sync_metric,
    race_condition_consistency as race_metric,
    spatial_reasoning as spatial_metric,
    style_consistency as style_metric,
)

SPATIAL_PROMPT = (
    "Judge the post-action state in two Minecraft images. Image 1 is B native ground truth and Image 2 is "
    "generated B at the same time. B was instructed to destroy the single foreground dirt block that initially "
    "occluded a distinctive stone-brick BUILDING with asymmetric wood attachments and a gold marker. Answer three "
    "binary questions. occluder_removed: is the foreground eye-level dirt occluder removed in generated B? "
    "target_revealed: is the target building visible and recognizable in generated B? same_building_gt: does generated "
    "B show the same physical building as GT B? For identity use geometry, openings, roofline, attached parts, and block "
    "layout; allow viewpoint, distance, lighting, and small texture differences. If unclear, answer 0. Return ONLY JSON "
    "with exactly {\"occluder_removed\":0 or 1,\"target_revealed\":0 or 1,\"same_building_gt\":0 or 1}."
)
SPATIAL_KEYS = ("occluder_removed", "target_revealed", "same_building_gt")


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(video: Path, frame: int, output: Path, crop: str | None = None) -> Path:
    if output.is_file():
        return output
    if not video.is_file():
        raise FileNotFoundError(video)
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = [f"select=eq(n\\,{frame})"]
    if crop:
        filters.append(crop)
    filters.append("scale=480:320:flags=lanczos")
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-vf", ",".join(filters), "-frames:v", "1", str(output),
    ], check=True)
    if not output.is_file():
        raise RuntimeError(f"failed to extract frame {frame} from {video}")
    return output


def image_half(image: Path, side: str, output: Path) -> Path:
    if output.is_file():
        return output
    from PIL import Image
    with Image.open(image).convert("RGB") as value:
        width, height = value.size
        box = (0, 0, width // 2, height) if side == "alpha" else (width // 2, 0, width, height)
        output.parent.mkdir(parents=True, exist_ok=True)
        value.crop(box).resize((480, 320), Image.Resampling.LANCZOS).save(output)
    return output


def gamma_video(group: str, case_id: str) -> Path:
    exact = RUN / "models" / "gamma" / group / case_id / "generated.mp4"
    if exact.is_file():
        return exact
    hits = list((RUN / "models" / "gamma" / group).glob(f"**/{case_id}/generated.mp4"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one Gamma output for {group}/{case_id}, found {hits}")
    return hits[0]


def gt_video(group: str, index: int, role: str) -> Path:
    case_id = {
        "structural": f"MWB3-S-ST{index + 1:02d}",
        "spatial": f"MWB3-S-SR{index + 1:02d}",
        "race": f"MWB3-SW-RC{index + 1:02d}",
        "revisit": f"MWB3-R-RV{index + 1:02d}",
    }[group]
    return ASSETS / group / case_id / f"{role}_gt.mp4"


def solaris_video(group: str, index: int, role: str) -> Path:
    case = {
        "interaction": f"MWB3-S-SS{index + 1:02d}",
        "structural": f"MWB3-S-ST{index + 1:02d}",
        "spatial": f"MWB3-S-SR{index + 1:02d}",
        "race": f"MWB3-SW-RC{index + 1:02d}",
        "revisit": f"MWB3-R-RV{index + 1:02d}",
    }[group]
    return RUN / "models" / "solaris" / group / case / f"{role}_generated.mp4"


def evidence_frame(model: str, group: str, index: int, role: str, frame: int) -> Path:
    case_id = {
        "structural": f"MWB3-S-ST{index + 1:02d}", "spatial": f"MWB3-S-SR{index + 1:02d}",
        "race": f"MWB3-SW-RC{index + 1:02d}", "revisit": f"MWB3-R-RV{index + 1:02d}",
        "interaction": f"MWB3-S-SS{index + 1:02d}",
    }[group]
    output = RUN / "evaluation" / "frames" / model / group / case_id / f"{role}_{frame:03d}.png"
    if model == "gt":
        return extract(gt_video(group, index, role), frame, output)
    if model == "solaris":
        return extract(solaris_video(group, index, role), frame, output)
    if model == "gamma":
        crop = "crop=iw/2:ih:0:0" if role == "alpha" else "crop=iw/2:ih:iw/2:0"
        return extract(gamma_video(group, case_id), frame, output, crop)
    raise KeyError(model)


def conditioning_frame(group: str, index: int, role: str) -> Path:
    case_id = f"MWB3-SW-RC{index + 1:02d}" if group == "race" else f"MWB3-S-SS{index + 1:02d}"
    source = ASSETS / group / case_id / "first_frame.png"
    return image_half(source, role, RUN / "evaluation" / "frames" / "conditioning" / group / case_id / f"{role}.png")


def request(request_id: str, metric: str, model: str, case_id: str, frame_or_pair: Any,
            images: list[Path], prompt: str, keys: tuple[str, ...], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "request_id": request_id, "metric": metric, "model": model, "case_id": case_id,
        "frame_or_pair": frame_or_pair, "images": [str(path) for path in images],
        "prompt": prompt, "keys": list(keys),
    }
    if extra:
        row.update(extra)
    return row


def interaction_prompt(case: dict[str, Any]) -> str:
    return sync_metric.build_prompt(case)


def prepare() -> None:
    if not (RUN / "models" / "gamma").is_dir():
        raise RuntimeError("Gamma inference directory is missing")
    requests: list[dict[str, Any]] = []
    # Cross-view building identity.
    for model in ("solaris", "gamma"):
        for index in range(6):
            case_id = f"MWB3-S-ST{index + 1:02d}"
            for frame in FRAMES["structural"]:
                images = [
                    evidence_frame("gt", "structural", index, "bravo", frame),
                    evidence_frame(model, "structural", index, "bravo", frame),
                    evidence_frame(model, "structural", index, "alpha", frame),
                ]
                requests.append(request(f"structural_{model}_{index+1:02d}_{frame}", "cross_view_structural_coherence", model, case_id, frame, images, structural_metric.PROMPT, structural_metric.OUTPUT_KEYS))
    # Spatial reveal gates and identity.
    for model in ("solaris", "gamma"):
        for index in range(6):
            case_id = f"MWB3-S-SR{index + 1:02d}"
            for frame in FRAMES["spatial"]:
                images = [evidence_frame("gt", "spatial", index, "bravo", frame), evidence_frame(model, "spatial", index, "bravo", frame)]
                requests.append(request(f"spatial_{model}_{index+1:02d}_{frame}", "spatial_reasoning", model, case_id, frame, images, SPATIAL_PROMPT, SPATIAL_KEYS))
    # Race conditions use initial views and candidate block references.
    race_cases = load(REPO / "MultiWorldBench/v3/cases/shared_world/race_condition_consistency.json")["cases"]
    for model in ("solaris", "gamma"):
        for index, case in enumerate(race_cases):
            for frame in FRAMES["race"]:
                refs = [REPO / "MultiWorldBench/v3" / case["block_reference_images"][key] for key in ("alpha", "bravo")]
                images = [conditioning_frame("race", index, "alpha"), conditioning_frame("race", index, "bravo"), evidence_frame(model, "race", index, "alpha", frame), evidence_frame(model, "race", index, "bravo", frame), *refs]
                requests.append(request(f"race_{model}_{index+1:02d}_{frame}", "race_condition_consistency", model, case["case_id"], frame, images, race_metric.PROMPT, race_metric.OUTPUT_KEYS))
    # Four revisit pairs exactly as configured in V3.
    for model in ("solaris", "gamma"):
        for index in range(6):
            case_id = f"MWB3-R-ID{index + 1:02d}"
            model_pairs = REVISIT_PAIRS if model == "gamma" else SOLARIS_REVISIT_PAIRS
            for pair_id, initial, returned, gamma_initial, gamma_returned in model_pairs:
                generated_initial = gamma_initial if model == "gamma" else initial
                generated_returned = gamma_returned if model == "gamma" else returned
                images = [evidence_frame("gt", "revisit", index, "bravo", initial),
                          evidence_frame("gt", "revisit", index, "bravo", returned),
                          evidence_frame(model, "revisit", index, "bravo", generated_initial),
                          evidence_frame(model, "revisit", index, "bravo", generated_returned)]
                requests.append(request(f"id_{model}_{index+1:02d}_{pair_id}", "id_consistency", model, case_id, pair_id, images, id_metric.PROMPT, id_metric.OUTPUT_KEYS,
                                        {"native_frames": [initial, returned], "generated_frames": [generated_initial, generated_returned]}))
    # GT-free interaction/shared-state pilot remains diagnostic for both models.
    interaction_cases = load(REPO / "MultiWorldBench/v3/cases/shared_world/shared_state_synchronization.json")["cases"]
    for model in ("solaris", "gamma"):
        for index, case in enumerate(interaction_cases):
            for frame in FRAMES["interaction"]:
                images = [conditioning_frame("interaction", index, "alpha"), conditioning_frame("interaction", index, "bravo"), evidence_frame(model, "interaction", index, "alpha", frame), evidence_frame(model, "interaction", index, "bravo", frame)]
                requests.append(request(f"interaction_{model}_{index+1:02d}_{frame}", "shared_state_synchronization_diagnostic", model, case["case_id"], frame, images, interaction_prompt(case), sync_metric.output_keys(case["operation"]), {"operation": case["operation"]}))
    for row in requests:
        row["image_sha256"] = [sha256(Path(path)) for path in row["images"]]
    dump(RUN / "evaluation" / "requests.json", requests)
    # Numeric style metric needs no VLM and is computed with the exact V3 implementation.
    style_rows: dict[str, list[dict[str, Any]]] = {"solaris": [], "gamma": []}
    for model in style_rows:
        for index in range(6):
            pairs = []
            model_pairs = REVISIT_PAIRS if model == "gamma" else SOLARIS_REVISIT_PAIRS
            for pair_id, initial, returned, gamma_initial, gamma_returned in model_pairs:
                generated_initial = gamma_initial if model == "gamma" else initial
                generated_returned = gamma_returned if model == "gamma" else returned
                pairs.append({
                    "pair_id": pair_id,
                    "gt_initial": evidence_frame("gt", "revisit", index, "bravo", initial),
                    "gt_return": evidence_frame("gt", "revisit", index, "bravo", returned),
                    "generated_initial": evidence_frame(model, "revisit", index, "bravo", generated_initial),
                    "generated_return": evidence_frame(model, "revisit", index, "bravo", generated_returned),
                })
            scored = style_metric.score_case(pairs)
            scored["case_id"] = f"MWB3-R-ST{index + 1:02d}"
            scored["evidence_pairs"] = len(model_pairs)
            scored["required_pairs"] = len(REVISIT_PAIRS)
            scored["evidence_coverage"] = len(model_pairs) / len(REVISIT_PAIRS)
            scored["strict_v3_evaluable"] = len(model_pairs) == len(REVISIT_PAIRS)
            style_rows[model].append(scored)
    style_output = {}
    for model, rows in style_rows.items():
        aggregate = style_metric.aggregate_cases(rows)
        aggregate["evidence_coverage"] = 1.0
        aggregate["strict_v3_evaluable"] = True
        style_output[model] = {"cases": rows, "aggregate": aggregate}
    dump(RUN / "evaluation" / "style_scores.json", style_output)
    dump(RUN / "evaluation" / "prepare_complete.json", {"requests": len(requests), "style_cases": 12})
    print(json.dumps({"requests": len(requests), "style_cases": 12}))


def parse_measurement(raw: str, keys: tuple[str, ...]) -> dict[str, int]:
    match = re.search(r"\{.*?\}", raw, re.S)
    if not match:
        raise ValueError("no JSON object")
    value = json.loads(match.group())
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"expected exactly {keys}, got {value}")
    if not all(type(value[key]) is int and value[key] in (0, 1) for key in keys):
        raise ValueError(f"all values must be integer 0/1: {value}")
    return {key: value[key] for key in keys}


def judge(shard_index: int, num_shards: int, max_memory_gib: int) -> None:
    import torch
    from qwen_vl_utils import process_vision_info
    sys.path.insert(0, "/data2/xuzhangbo/multihuman-animate/solaris/vlm_eval")
    from qwen_vlm_worker import QwenJudge
    requests = load(RUN / "evaluation" / "requests.json")
    selected = [row for index, row in enumerate(requests) if index % num_shards == shard_index]
    worker = QwenJudge(QWEN, 120, 384 * 28 * 28, "balanced", max_memory_gib, "sdpa")
    torch.manual_seed(12345)
    output_dir = RUN / "evaluation" / "vlm"
    completed = 0
    for row in selected:
        destination = output_dir / f"{row['request_id']}.json"
        if destination.is_file():
            try:
                if load(destination).get("measurement") is not None:
                    completed += 1
                    continue
            except Exception:
                pass
        content: list[dict[str, Any]] = [{"type": "text", "text": row["prompt"]}]
        for number, path in enumerate(row["images"], 1):
            content += [{"type": "text", "text": f"Image {number}"}, {"type": "image", "image": path}]
        messages = [{"role": "user", "content": content}]
        attempts = []
        measurement = None
        for attempt in range(2):
            start = time.time()
            chat = worker.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = worker.processor(text=[chat], images=images, videos=videos, padding=True, return_tensors="pt").to(worker.input_device)
            with torch.inference_mode():
                generated = worker.model.generate(**inputs, max_new_tokens=120, do_sample=False, use_cache=True)
            raw = worker.processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            error = None
            try:
                measurement = parse_measurement(raw, tuple(row["keys"]))
            except Exception as exc:
                error = repr(exc)
            attempts.append({"raw": raw, "error": error, "seconds": round(time.time() - start, 3)})
            del inputs, generated
            torch.cuda.empty_cache()
            if error is None:
                break
            messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": "Return only the requested JSON object with exactly the specified integer 0/1 fields."}]
        dump(destination, {"request": row, "attempts": attempts, "measurement": measurement})
        print(row["request_id"], measurement, flush=True)
        completed += 1
    dump(RUN / "evaluation" / f"judge_shard_{shard_index:02d}_of_{num_shards:02d}.json", {"completed": completed, "total": len(selected), "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")})


def majority(rows: list[dict[str, int]], key: str) -> int:
    return int(sum(row[key] for row in rows) * 2 >= len(rows))


def aggregate() -> None:
    requests = load(RUN / "evaluation" / "requests.json")
    by_id = {}
    missing = []
    for row in requests:
        path = RUN / "evaluation" / "vlm" / f"{row['request_id']}.json"
        if not path.is_file():
            missing.append(row["request_id"])
            continue
        value = load(path).get("measurement")
        if value is None:
            missing.append(row["request_id"])
        else:
            by_id[row["request_id"]] = value
    if missing:
        raise RuntimeError(f"missing/invalid VLM results ({len(missing)}): {missing[:10]}")
    output: dict[str, Any] = {"style_consistency": load(RUN / "evaluation" / "style_scores.json")}
    # Structural.
    output["cross_view_structural_coherence"] = {}
    for model in ("solaris", "gamma"):
        cases = []
        for index in range(6):
            frames = [by_id[f"structural_{model}_{index+1:02d}_{frame}"] for frame in FRAMES["structural"]]
            row = structural_metric.score_case(frames); row["case_id"] = f"MWB3-S-ST{index+1:02d}"; cases.append(row)
        output["cross_view_structural_coherence"][model] = {"cases": cases, "aggregate": structural_metric.aggregate_cases(cases)}
    # Spatial.  Optional geometry evidence is produced by the self-contained
    # SAM2/SIFT measurement tool.  Failed gates still have an exact score of
    # zero; passed gates remain explicitly not evaluable when a defensible
    # camera angle cannot be recovered.
    geometry_path = RUN / "evaluation" / "spatial_geometry.json"
    geometry = load(geometry_path) if geometry_path.is_file() else {}
    output["spatial_reasoning"] = {}
    for model in ("solaris", "gamma"):
        cases = []
        for index in range(6):
            frames = [by_id[f"spatial_{model}_{index+1:02d}_{frame}"] for frame in FRAMES["spatial"]]
            evidence = {key: majority(frames, key) for key in SPATIAL_KEYS}
            case_id = f"MWB3-S-SR{index+1:02d}"
            measured = geometry.get(model, {}).get(case_id, {})
            for key in ("projected_area_score_0_100", "camera_angle_score_0_100"):
                if measured.get(key) is not None:
                    evidence[key] = measured[key]
            row = spatial_metric.score_case(evidence); row["case_id"] = case_id; row["frames"] = frames
            if measured:
                row["geometry_evidence"] = measured
            cases.append(row)
        output["spatial_reasoning"][model] = {"cases": cases, "aggregate": spatial_metric.aggregate_cases(cases)}
    # Race.
    output["race_condition_consistency"] = {}
    for model in ("solaris", "gamma"):
        cases = []
        for index in range(6):
            frames = [by_id[f"race_{model}_{index+1:02d}_{frame}"] for frame in FRAMES["race"]]
            row = race_metric.score_case(frames); row["case_id"] = f"MWB3-SW-RC{index+1:02d}"; cases.append(row)
        output["race_condition_consistency"][model] = {"cases": cases, "aggregate": race_metric.aggregate_cases(cases)}
    # Revisit ID.
    output["id_consistency"] = {}
    for model in ("solaris", "gamma"):
        cases = []
        for index in range(6):
            model_pairs = REVISIT_PAIRS if model == "gamma" else SOLARIS_REVISIT_PAIRS
            pairs = [by_id[f"id_{model}_{index+1:02d}_{pair_id}"] for pair_id, *_ in model_pairs]
            row = id_metric.score_case(pairs); row["case_id"] = f"MWB3-R-ID{index+1:02d}"
            row["evidence_pairs"] = len(model_pairs); row["required_pairs"] = len(REVISIT_PAIRS)
            row["evidence_coverage"] = len(model_pairs) / len(REVISIT_PAIRS)
            row["strict_v3_evaluable"] = len(model_pairs) == len(REVISIT_PAIRS)
            cases.append(row)
        aggregate = id_metric.aggregate_cases(cases)
        aggregate["evidence_coverage"] = 1.0
        aggregate["strict_v3_evaluable"] = True
        output["id_consistency"][model] = {"cases": cases, "aggregate": aggregate}
    # Operation-specific shared-state scores: placement has four questions; destruction has three.
    interaction_config = load(REPO / "MultiWorldBench/v3/cases/shared_world/shared_state_synchronization.json")["cases"]
    output["shared_state_synchronization"] = {}
    for model in ("solaris", "gamma"):
        interaction_rows = []
        for index, case in enumerate(interaction_config):
            frames = [by_id[f"interaction_{model}_{index+1:02d}_{frame}"] for frame in FRAMES["interaction"]]
            keys = sync_metric.output_keys(case["operation"])
            voted = {key: majority(frames, key) for key in keys}
            row = sync_metric.score_case({"operation": case["operation"], **voted})
            row["case_id"] = case["case_id"]
            row["frames"] = frames
            interaction_rows.append(row)
        output["shared_state_synchronization"][model] = {
            "diagnostic_only": True,
            "cases": interaction_rows,
            "aggregate": sync_metric.aggregate_cases(interaction_rows),
        }
    # Navigation is computed separately because it uses DPVO and dense optical
    # flow rather than the VLM.  Merge the self-contained V3 results here so a
    # single comparison artifact covers the complete benchmark.
    for model in ("solaris", "gamma"):
        navigation = load(RUN / "evaluation" / "navigation" / f"{model}_scores.json")
        for metric in (
            "control_isolation", "coordination_fairness",
            "cross_view_motion_consistency",
        ):
            output.setdefault(metric, {})[model] = navigation[metric]
    dump(RUN / "comparison" / "scores.json", output)
    dump(RUN / "evaluation" / "complete.json", {"status": "complete", "vlm_requests": len(requests)})
    print(json.dumps({key: {model: value.get("aggregate", value) for model, value in models.items()} for key, models in output.items()}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "judge", "aggregate"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-memory-gib", type=int, default=43)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "judge":
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("invalid shard index")
        judge(args.shard_index, args.num_shards, args.max_memory_gib)
    else:
        aggregate()


if __name__ == "__main__":
    main()
