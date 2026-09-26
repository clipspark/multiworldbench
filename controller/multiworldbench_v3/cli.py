"""Command-line entry point for standalone V3 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validate import validate


def main() -> None:
    parser = argparse.ArgumentParser(prog="multiworldbench-v3")
    subcommands = parser.add_subparsers(dest="command", required=True)
    check = subcommands.add_parser("validate")
    check.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
