#!/usr/bin/env python3
"""Create Solaris datasets using only the self-contained V3 asset store."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


MINEFLAYER_BOOL_KEYS = (
    "forward", "back", "left", "right", "jump", "sprint", "sneak",
    "attack", "use", "mount", "dismount", "place_block", "place_entity", "mine",
    *(f"hotbar.{index}" for index in range(1, 10)),
)


def normalize_mineflayer_actions(rows: list[dict]) -> list[dict]:
    """Fill the complete Mineflayer schema expected by Solaris.

    V3 keeps source provenance fields, but every action payload is made explicit so
    Solaris can consume it without modifying or patching model code.
    """
    normalized = []
    for row in rows:
        item = dict(row)
        source = dict(item.get("action", {}))
        action = {key: bool(source.get(key, False)) for key in MINEFLAYER_BOOL_KEYS}
        camera = source.get("camera", [0.0, 0.0])
        if not isinstance(camera, list) or len(camera) != 2:
            raise ValueError(f"invalid Mineflayer camera action: {camera!r}")
        action["camera"] = [float(camera[0]), float(camera[1])]
        item["action"] = action
        normalized.append(item)
    return normalized


def still_video(image: Path, side: str, frames: int, output: Path) -> None:
    crop = "crop=iw/2:ih:0:0" if side == "alpha" else "crop=iw/2:ih:iw/2:0"
    duration = frames / 20.0
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-framerate", "20", "-i", str(image), "-t", str(duration),
        "-vf", f"{crop},scale=640:360:flags=lanczos", "-r", "20", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output),
    ], check=True)


def native_video(source: Path, frames: int, prefix_frames: int, output: Path) -> None:
    """Write an exact-length 20 Hz input and clone the initial frame prefix."""
    total = frames + prefix_frames
    filters = ["fps=20", f"trim=end_frame={frames}", "setpts=PTS-STARTPTS"]
    if prefix_frames:
        filters.append(f"tpad=start={prefix_frames}:start_mode=clone")
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", ",".join(filters), "-frames:v", str(total),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(output),
    ], check=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("interaction", "revisit", "navigation/control"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-glob", default="MWB3-*")
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--action-prefix-frames", type=int, default=0)
    parser.add_argument("--video-prefix-frames", type=int, default=1)
    args = parser.parse_args()
    source_root = repo / "MultiWorldBench/v3/assets" / args.group
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    test = output / "dataset/test"
    test.mkdir(parents=True)
    mapping = []
    cases = sorted(path for path in source_root.glob(args.case_glob) if path.is_dir())
    if not cases:
        raise RuntimeError(f"no cases matched {source_root / args.case_glob}")
    for index, case in enumerate(cases):
        row = {"index": index, "case_id": case.name, "frames": args.frames, "files_sha256": {}}
        for role, token in (("alpha", "Alpha"), ("bravo", "Bravo")):
            actions = json.loads((case / f"{role}_actions.json").read_text(encoding="utf-8"))
            if len(actions) < args.frames:
                raise ValueError(f"{case.name}/{role}: {len(actions)} actions < {args.frames}")
            actions = normalize_mineflayer_actions(actions[:args.frames])
            if args.action_prefix_frames:
                actions = [dict(actions[0]) for _ in range(args.action_prefix_frames)] + actions
            action_out = test / f"{index:02d}_{token}_instance_0.json"
            dump(action_out, actions)
            video_out = test / f"{index:02d}_{token}_instance_0_camera.mp4"
            native = case / f"{role}_gt.mp4"
            if native.is_file():
                native_video(native, args.frames, args.video_prefix_frames, video_out)
            else:
                still_video(case / "first_frame.png", role, args.frames + args.video_prefix_frames, video_out)
            row["files_sha256"][action_out.name] = sha256(action_out)
            row["files_sha256"][video_out.name] = sha256(video_out)
        mapping.append(row)
    dump(output / "mapping.json", {"schema_version": "mwb.v3.solaris_input.v1", "self_contained": True, "group": args.group, "frames": args.frames, "action_prefix_frames": args.action_prefix_frames, "video_prefix_frames": args.video_prefix_frames, "cases": mapping})
    config = """defaults:
  - shared
  - device: gpu
  - model: solaris
  - runner: inference
  - dataset@eval_datasets.eval_mwb3: eval_solid_sym_r2
  - _self_
eval_datasets:
  eval_mwb3:
    name: eval_mwb3
"""
    (output / "inference_mwb3.yaml").write_text(config, encoding="utf-8")
    print(json.dumps({"group": args.group, "cases": len(mapping), "frames": args.frames, "output": str(output)}))


if __name__ == "__main__":
    main()
