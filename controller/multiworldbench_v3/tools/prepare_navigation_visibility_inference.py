#!/usr/bin/env python3
"""Validate the 180-case V3 navigation suite and build inference manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

REQUIRED_NATIVE = (
    ".complete", "alpha_gt.mp4", "bravo_gt.mp4", "alpha_actions.json",
    "bravo_actions.json", "first_frame.png", "action_left.json",
    "action_right.json", "prompt.txt",
)
REQUIRED_NO_GT = (
    ".complete", "alpha_actions.json", "bravo_actions.json", "first_frame.png",
    "action_left.json", "action_right.json", "prompt.txt", "provenance.json",
)
GAMMA_FILES = ("first_frame.png", "action_left.json", "action_right.json", "prompt.txt")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument(
        "--no-gt", action="store_true",
        help="validate planned-action inputs with a reused first frame and no GT videos",
    )
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    run = args.run_root.resolve()
    source = run / "native"
    spec_path = repo / "MultiWorldBench/v3/cases/navigation/navigation_visibility_suite.json"
    suite = json.loads(spec_path.read_text(encoding="utf-8"))
    cases = suite["cases"]
    ids = [row["case_id"] for row in cases]
    if len(ids) != 180 or len(set(ids)) != 180:
        raise ValueError(f"expected 180 unique cases, got {len(ids)}/{len(set(ids))}")
    for row in cases:
        case = source / row["case_id"]
        required = REQUIRED_NO_GT if args.no_gt else REQUIRED_NATIVE
        missing = [name for name in required if not (case / name).exists()]
        if missing:
            raise FileNotFoundError(f"{case}: missing {missing}")
        # Capture may have started before metadata-only suite regeneration.
        (case / "case_spec.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    prepared = run / "prepared"
    gamma = prepared / "gamma/navigation"
    if not (gamma / ".complete").is_file():
        if gamma.exists():
            shutil.rmtree(gamma)
        gamma.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gamma-navigation-", dir=gamma.parent) as temp_name:
            stage = Path(temp_name)
            records = []
            for case_id in ids:
                src = source / case_id
                dst = stage / case_id
                dst.mkdir()
                hashes = {}
                for name in GAMMA_FILES:
                    link_or_copy(src / name, dst / name)
                    hashes[name] = digest(dst / name)
                records.append({"case_id": case_id, "files_sha256": hashes})
            manifest = {
                "schema_version": "mwb.v3.gamma_navigation_input.v1",
                "self_contained": True,
                "case_count": len(records),
                "frames": 189,
                "gt_collected": not args.no_gt,
                "gt_used_for_scoring": False,
                "cases": records,
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (stage / ".complete").write_text("complete\n", encoding="utf-8")
            os.replace(stage, gamma)

    lists = prepared / f"solaris_shards_{args.shards}"
    if lists.exists():
        shutil.rmtree(lists)
    lists.mkdir(parents=True)
    buckets = [[] for _ in range(args.shards)]
    for index, case_id in enumerate(ids):
        buckets[index % args.shards].append(case_id)
    for index, bucket in enumerate(buckets):
        (lists / f"shard_{index:02d}.txt").write_text("\n".join(bucket) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "mwb.v3.navigation_inference_plan.v1",
        "case_count": len(ids),
        "shards": args.shards,
        "shard_sizes": [len(bucket) for bucket in buckets],
        "source": str(source),
        "gamma": str(gamma),
        "gt_collected": not args.no_gt,
        "gt_used_for_scoring": False,
    }
    (lists / "plan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
