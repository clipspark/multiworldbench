#!/usr/bin/env python3
"""Extract generated A/B videos from a Solaris V3 inference run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="prepared Solaris input directory")
    parser.add_argument("--raw", required=True, type=Path, help="directory containing video_N_side_by_side.mp4")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    mapping = json.loads((args.input / "mapping.json").read_text(encoding="utf-8"))
    checks = []
    for row in mapping["cases"]:
        raw = args.raw / f"video_{row['index']}_side_by_side.mp4"
        stream = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(raw),
        ], text=True))["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        if width % 2 or height % 2:
            raise ValueError(f"unexpected Solaris layout {width}x{height}: {raw}")
        case_out = args.output / row["case_id"]
        case_out.mkdir(parents=True, exist_ok=True)
        for role, y in (("alpha", 0), ("bravo", height // 2)):
            target = case_out / f"{role}_generated.mp4"
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(raw), "-vf", f"crop={width//2}:{height//2}:{width//2}:{y}",
                "-frames:v", str(mapping["frames"]), "-an", "-c:v", "libx264", "-crf", "18", str(target),
            ], check=True)
            checks.append({"case_id": row["case_id"], "role": role, "frames": int(stream["nb_frames"]), "output": str(target)})
    result = {"status": "complete", "self_contained": True, "checks": checks}
    (args.output / "postprocess.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(mapping["cases"]), "videos": len(checks)}))


if __name__ == "__main__":
    main()
