#!/usr/bin/env python3
"""Build a no-GT navigation inference input set from V3 case specs and first frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

KEYS = (
    "inventory", "ESC", *(f"hotbar.{i}" for i in range(1, 10)),
    "forward", "back", "left", "right", "jump", "sneak", "sprint",
    "swapHands", "attack", "use", "pickItem", "drop",
)
KEY_INDEX = {name: index for index, name in enumerate(KEYS)}
BOOL_KEYS = (
    "forward", "back", "left", "right", "jump", "sprint", "sneak",
    "attack", "use", "mount", "dismount", "place_block", "place_entity", "mine",
)
MOVE_CONTROLS = {"forward", "back", "left", "right"}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def source_case_id(case: dict) -> str:
    code = case["case_id"].split("-")[2]
    scene = case["scene"]["scene_id"].rsplit("SC", 1)[1]
    family = case["action_template_id"][:2]
    # In one-way cases the two family directions intentionally use opposite
    # visibility directions from the captured source suite.
    template = "T06" if code == "OV" and family == "BF" else "T01"
    return f"MWB3-N-{code}-S{scene}-{template}"


def action_payload(action: dict, duration: int) -> dict:
    mineflayer = {key: False for key in BOOL_KEYS}
    gamma_keyboard = [0] * len(KEYS)
    gamma_camera = [0.0, 0.0]
    action_type = action["type"]
    if action_type == "MOVE":
        control = action["control"]
        if control not in MOVE_CONTROLS:
            raise ValueError(f"unsupported MOVE control: {control}")
        mineflayer[control] = True
        gamma_keyboard[KEY_INDEX[control]] = 1
    elif action_type == "LOOK":
        yaw_deg = float(action.get("yaw_delta_deg", 0.0)) / duration
        pitch_deg = float(action.get("pitch_delta_deg", 0.0)) / duration
        mineflayer["camera"] = [yaw_deg * 3.141592653589793 / 180.0, pitch_deg * 3.141592653589793 / 180.0]
        gamma_camera = [yaw_deg, pitch_deg]
    elif action_type == "IDLE":
        pass
    else:
        raise ValueError(f"unsupported action type: {action_type}")
    mineflayer.setdefault("camera", [0.0, 0.0])
    return {"mineflayer": mineflayer, "keyboard": gamma_keyboard, "camera": gamma_camera}


def build_role(case: dict, role: str, frames: int) -> tuple[list[dict], list[list[int]], list[list[float]]]:
    rows = []
    keyboard = [[0] * len(KEYS) for _ in range(frames)]
    camera = [[0.0, 0.0] for _ in range(frames)]
    phases = list(case["phase_ranges"])
    for frame in range(frames):
        turn_index = next((i for i, (start, end) in enumerate(phases) if int(start) <= frame < int(end)), None)
        if turn_index is None:
            action = {"action_id": "idle", "type": "IDLE"}
            phase = "initial_observation" if frame < int(case["initial_idle_frames"]) else "final_action_free_settle"
            turn_id = None
        else:
            action = case["turns"][turn_index][role]
            phase = "navigation_turn"
            turn_id = case["turns"][turn_index]["id"]
        duration = int(case["turns"][turn_index]["duration_ticks"]) if turn_index is not None else 1
        payload = action_payload(action, duration)
        keyboard[frame] = payload["keyboard"]
        camera[frame] = payload["camera"]
        rows.append({
            "frame_count": frame,
            "relativeTimeMs": round(frame * 1000 / int(case["fps"])),
            "phase": phase,
            "turn_id": turn_id,
            "planned_action": action,
            "recorded_action_source": "planned_v3_no_gt",
            "action": payload["mineflayer"],
        })
    return rows, keyboard, camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    spec_path = args.repo / "MultiWorldBench/v3/cases/navigation/navigation_visibility_suite.json"
    suite = json.loads(spec_path.read_text(encoding="utf-8"))
    cases = suite["cases"]
    root = args.run_root.resolve()
    native = root / "native"
    if native.exists():
        shutil.rmtree(native)
    native.mkdir(parents=True)
    records = []
    frames_seen = set()
    for case in cases:
        case_id = case["case_id"]
        frames = int(case["valid_num_frames"])
        frames_seen.add(frames)
        source_id = source_case_id(case)
        source = args.source_run / "native" / source_id / "first_frame.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = native / case_id
        target.mkdir()
        link_or_copy(source, target / "first_frame.png")
        (target / "prompt.txt").write_text("Two players navigate.\n", encoding="utf-8")
        dump(target / "case_spec.json", case)
        file_hashes = {"first_frame.png": digest(target / "first_frame.png")}
        for role, gamma_name in (("alpha", "action_left.json"), ("bravo", "action_right.json")):
            rows, keyboard, camera = build_role(case, role, frames)
            dump(target / f"{role}_actions.json", rows)
            dump(target / gamma_name, {"keyboard": keyboard, "camera": camera})
            file_hashes[f"{role}_actions.json"] = digest(target / f"{role}_actions.json")
            file_hashes[gamma_name] = digest(target / gamma_name)
        provenance = {
            "schema_version": "mwb.v3.navigation.no_gt_case.v1",
            "case_id": case_id,
            "no_gt_capture": True,
            "no_gt_trajectory": True,
            "condition_image_source": str(source),
            "condition_image_sha256": digest(source),
            "planned_actions_from": str(spec_path),
            "planned_actions_spec_sha256": digest(spec_path),
        }
        dump(target / "provenance.json", provenance)
        (target / ".complete").write_text("no_gt_input_complete\n", encoding="utf-8")
        records.append({**provenance, "files_sha256": file_hashes})
    if len(cases) != 180 or len(frames_seen) != 1:
        raise ValueError(f"invalid suite: cases={len(cases)}, frame_counts={frames_seen}")
    dump(native / "input_manifest.json", {
        "schema_version": "mwb.v3.navigation.no_gt_inputs.v1",
        "case_count": len(records),
        "frames": next(iter(frames_seen)),
        "gt_collected": False,
        "gt_used_for_scoring": False,
        "records": records,
    })
    print(json.dumps({"cases": len(records), "frames": next(iter(frames_seen)), "native": str(native), "gt_collected": False}))


if __name__ == "__main__":
    main()
