#!/usr/bin/env python3
"""Materialize Gamma-World inputs from the self-contained V3 asset store."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


GROUPS = ("interaction", "structural", "spatial", "race", "revisit")
REQUIRED = ("first_frame.png", "action_left.json", "action_right.json", "prompt.txt")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=repo / "MultiWorldBench/v3/assets")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets, output = args.assets.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = {"schema_version": "mwb.v3.gamma_input.v2", "self_contained": True, "groups": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as name:
        stage = Path(name)
        for group in GROUPS:
            rows = []
            for source in sorted(path for path in (assets / group).glob("MWB3-*") if path.is_dir()):
                missing = [item for item in REQUIRED if not (source / item).is_file()]
                if missing:
                    raise FileNotFoundError(f"{source}: missing {missing}")
                target = stage / group / source.name
                target.mkdir(parents=True)
                hashes = {}
                for filename in REQUIRED:
                    shutil.copy2(source / filename, target / filename)
                    hashes[filename] = digest(target / filename)
                rows.append({"case_id": source.name, "files_sha256": hashes})
            if not rows:
                raise RuntimeError(f"no V3 assets for {group}")
            manifest["groups"][group] = {"case_count": len(rows), "cases": rows}
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(stage), str(output))
    print(json.dumps({key: value["case_count"] for key, value in manifest["groups"].items()}))


if __name__ == "__main__":
    main()
