#!/usr/bin/env python3
"""Run the V3 race-condition VLM judge on generated A/B frames."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from qwen_vl_utils import process_vision_info


REPO = Path("/data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine")
QWEN_HELPER = Path("/data2/xuzhangbo/multihuman-animate/solaris/vlm_eval")
QWEN_MODEL = Path("/data2/xuzhangbo/models/Qwen2.5-VL-72B-Instruct-AWQ")
CHECK_FRAMES = (60, 80, 100)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(QWEN_HELPER))
from controller.multiworldbench_v3.metrics.race_condition_consistency import (  # noqa: E402
    PROMPT,
    aggregate_cases,
    score_case,
    validate_output,
)
from qwen_vlm_worker import QwenJudge  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_frame(video: Path, frame: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-vf", f"select=eq(n\\,{frame})", "-vsync", "0", "-frames:v", "1", str(output),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to extract frame {frame} from {video}")


def parse_measurement(raw: str) -> dict[str, int]:
    match = re.search(r"\{.*\}", raw, re.S)
    if match is None:
        raise ValueError("no JSON object in response")
    return validate_output(json.loads(match.group()))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: judge_race_condition.py RUN_ROOT")
    root = Path(sys.argv[1]).resolve()
    manifest = json.loads((root / "inputs" / "inference_manifest.json").read_text())
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)

    requests = []
    for case in manifest["cases"]:
        cid = case["case_id"]
        case_dir = output / cid
        initial_a = root / "inputs" / "cases" / cid / "alpha_first_frame.png"
        initial_b = root / "inputs" / "cases" / cid / "bravo_first_frame.png"
        reference_a = root / "inputs" / "cases" / cid / "alpha_block_reference.png"
        reference_b = root / "inputs" / "cases" / cid / "bravo_block_reference.png"
        for frame in CHECK_FRAMES:
            generated_a = case_dir / f"alpha_generated_{frame:03}.png"
            generated_b = case_dir / f"bravo_generated_{frame:03}.png"
            extract_frame(root / "solaris" / cid / "alpha_generated.mp4", frame, generated_a)
            extract_frame(root / "solaris" / cid / "bravo_generated.mp4", frame, generated_b)
            requests.append(
                {
                    "id": f"{cid}_{frame}",
                    "case_id": cid,
                    "frame": frame,
                    "images": [
                        str(initial_a), str(initial_b), str(generated_a), str(generated_b),
                        str(reference_a), str(reference_b),
                    ],
                    "prompt": PROMPT,
                }
            )
    (output / "requests.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2))

    torch.manual_seed(12345)
    judge = QwenJudge(QWEN_MODEL, 100, 384 * 28 * 28, "balanced", 43, "sdpa")
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "model": str(judge.model_path),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "seed": 12345,
                "do_sample": False,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            indent=2,
        )
    )

    for request in requests:
        destination = output / f"{request['id']}.json"
        content = [{"type": "text", "text": request["prompt"]}]
        for index, image in enumerate(request["images"], start=1):
            content.extend(
                [
                    {"type": "text", "text": f"Image {index}"},
                    {"type": "image", "image": image},
                ]
            )
        messages = [{"role": "user", "content": content}]
        attempts = []
        measurement = None
        for _ in range(2):
            started = time.time()
            chat = judge.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = judge.processor(
                text=[chat], images=images, videos=videos, padding=True, return_tensors="pt"
            ).to(judge.input_device)
            with torch.inference_mode():
                generated = judge.model.generate(
                    **inputs, max_new_tokens=100, do_sample=False, use_cache=True
                )
            raw = judge.processor.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            error = None
            try:
                measurement = parse_measurement(raw)
            except Exception as exc:  # retain malformed evidence and retry once
                error = repr(exc)
            attempts.append({"raw": raw, "error": error, "seconds": time.time() - started})
            del inputs, generated
            torch.cuda.empty_cache()
            if error is None:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            'Return exactly {"same_block_count":0 or 1,'
                            '"same_block_type":0 or 1,"matches_intended_block":0 or 1} '
                            'with integer values and no explanation.'
                        ),
                    },
                ]
            )
        if measurement is None:
            raise RuntimeError(f"invalid VLM response after retry: {request['id']}")
        destination.write_text(
            json.dumps(
                {
                    "request": request,
                    "image_sha256": [sha256(Path(path)) for path in request["images"]],
                    "attempts": attempts,
                    "measurement": measurement,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(request["id"], measurement, flush=True)

    cases = []
    for case in manifest["cases"]:
        cid = case["case_id"]
        per_frame = []
        values = []
        for frame in CHECK_FRAMES:
            evidence = json.loads((output / f"{cid}_{frame}.json").read_text())
            value = evidence["measurement"]
            values.append(value)
            per_frame.append({"frame": frame, **value})
        result = score_case(values)
        cases.append(
            {
                "case_id": cid,
                "visibility_class": case["visibility_class"],
                "intended_materials": case["intended_materials"],
                "questions": result["questions"],
                "score": result["score"],
                "score_0_100": result["score_0_100"],
                "frames": per_frame,
                "aggregation": result["aggregation"],
            }
        )
    aggregate = aggregate_cases(cases)
    scores = {
        "metric_id": "race_condition_consistency",
        "gt_collected": False,
        "questions": ["same_block_count", "same_block_type", "matches_intended_block"],
        "check_frames": list(CHECK_FRAMES),
        "case_aggregation": "majority per question, then minimum of all three questions",
        "cases": cases,
        "aggregate": aggregate,
    }
    (output / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2))
    (output / "complete.json").write_text(
        json.dumps({"status": "complete", "requests": len(requests)}, indent=2)
    )
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
