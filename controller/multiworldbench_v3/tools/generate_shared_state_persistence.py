#!/usr/bin/env python3
"""Generate the self-contained 20-case shared-state persistence suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCENES = [
    {"scene_id":"grass_oak","name_zh":"草地木构","floor":"minecraft:grass_block","cross":"minecraft:white_concrete","marker":"minecraft:red_concrete","geometry":[[-7,0,-6,-7,3,-6,"minecraft:oak_log"],[7,0,-6,7,3,-6,"minecraft:oak_log"],[-7,4,-6,7,4,-6,"minecraft:oak_log"],[-6,0,-6,-2,0,-6,"minecraft:mossy_cobblestone"],[2,0,-6,6,0,-6,"minecraft:mossy_cobblestone"]]},
    {"scene_id":"sandstone_court","name_zh":"砂岩庭院","floor":"minecraft:smooth_sandstone","cross":"minecraft:yellow_concrete","marker":"minecraft:blue_concrete","geometry":[[-4,0,-6,-4,3,-6,"minecraft:cut_sandstone"],[4,0,-6,4,3,-6,"minecraft:cut_sandstone"],[-4,4,-6,4,4,-6,"minecraft:smooth_sandstone"],[-9,0,-4,-7,1,-4,"minecraft:sandstone"],[7,0,-4,9,1,-4,"minecraft:sandstone"]]},
    {"scene_id":"stone_plaza","name_zh":"石砖广场","floor":"minecraft:stone_bricks","cross":"minecraft:cyan_concrete","marker":"minecraft:orange_concrete","geometry":[[-8,0,-6,-8,3,-6,"minecraft:polished_andesite"],[8,0,-6,8,3,-6,"minecraft:polished_andesite"],[-5,0,-7,-5,1,-7,"minecraft:chiseled_stone_bricks"],[5,0,-7,5,2,-7,"minecraft:chiseled_stone_bricks"],[-3,0,-6,3,0,-6,"minecraft:mossy_stone_bricks"]]},
    {"scene_id":"tuff_copper","name_zh":"凝灰铜庭","floor":"minecraft:tuff","cross":"minecraft:orange_concrete","marker":"minecraft:light_blue_concrete","geometry":[[-7,0,-6,-7,3,-6,"minecraft:oxidized_copper"],[7,0,-6,7,3,-6,"minecraft:cut_copper"],[-7,4,-6,7,4,-6,"minecraft:tuff_bricks"],[-4,0,-7,-2,1,-7,"minecraft:exposed_copper"],[2,0,-7,4,1,-7,"minecraft:weathered_copper"]]},
    {"scene_id":"snow_ice","name_zh":"冰雪庭院","floor":"minecraft:snow_block","cross":"minecraft:magenta_concrete","marker":"minecraft:lime_concrete","geometry":[[-7,0,-6,-7,3,-6,"minecraft:packed_ice"],[7,0,-6,7,3,-6,"minecraft:blue_ice"],[-7,4,-6,7,4,-6,"minecraft:spruce_log"],[-4,0,-7,-2,1,-7,"minecraft:packed_ice"],[2,0,-7,4,1,-7,"minecraft:blue_ice"]]},
]


def look_at(position: list[float], aim: list[float]) -> tuple[float, float]:
    dx = aim[0] - position[0]
    dy = aim[1] - (position[1] + 1.62)
    dz = aim[2] - position[2]
    return math.degrees(math.atan2(-dx, dz)), -math.degrees(math.atan2(dy, math.hypot(dx, dz)))


def timeline(operation: str, turn_sign: int) -> list[dict[str, object]]:
    operation_end = 61 if operation == "place" else 68
    action = "use_once" if operation == "place" else "hold_attack"
    return [
        {"range":[0,20],"phase":"joint_observe","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
        {"range":[20,40],"phase":"alpha_turns_away","alpha":{"action":"camera_yaw","total_delta_deg":110*turn_sign},"bravo":{"action":"idle"}},
        {"range":[40,60],"phase":"alpha_hidden_hold","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
        {"range":[60,operation_end],"phase":"bravo_operates","alpha":{"action":"idle"},"bravo":{"action":action,"button":"right" if operation == "place" else "left"}},
        {"range":[operation_end,90],"phase":"post_operation_settle","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
        {"range":[90,110],"phase":"bravo_turns_away","alpha":{"action":"idle"},"bravo":{"action":"camera_yaw","total_delta_deg":-110*turn_sign}},
        {"range":[110,130],"phase":"both_hidden_hold","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
        {"range":[130,150],"phase":"alpha_returns","alpha":{"action":"camera_yaw","total_delta_deg":-110*turn_sign},"bravo":{"action":"idle"}},
        {"range":[150,181],"phase":"alpha_reobserves","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
    ]


def build_case(index: int, scene: dict[str, object], operation: str, variant: int) -> dict[str, object]:
    tx, ty, tz = 9000 + (index - 1) * 48, 91, 0
    target = [tx, ty, tz]
    aim_y = ty if operation == "place" else ty + 0.5
    aim = [tx + 0.5, aim_y, tz + 0.5]
    positions = {
        "alpha": [tx - 1.5, ty, tz + 3.5],
        "bravo": [tx + 2.5, ty, tz + 3.5],
    }
    conditions = {}
    for role in ("alpha", "bravo"):
        yaw, pitch = look_at(positions[role], aim)
        conditions[role] = {
            "pose": {
                "position": positions[role],
                "native_yaw_deg": yaw,
                "native_pitch_deg": pitch,
                "aim_point": aim,
                "aim_role": "marked_target",
                "target_cell": target,
            }
        }
    turn_sign = 1 if variant == 0 else -1
    cid = f"MWB3-S-SP{index:03d}"
    return {
        "case_id": cid,
        "id": cid,
        "metrics": ["shared_state_persistence"],
        "scene": scene["scene_id"],
        "scene_name_zh": scene["name_zh"],
        "operation": operation,
        "variant": variant,
        "variant_description": "A turns right and B turns left" if variant == 0 else "A turns left and B turns right",
        "roles": {"operator":"bravo","returning_observer":"alpha"},
        "fps": 20,
        "num_frames": 181,
        "valid_num_frames": 181,
        "duration_seconds": 9.05,
        "round_count": 1,
        "initial_conditions": conditions,
        "target_cell": target,
        "expected_change_count": 1,
        "expected_before": {"target_cell":"minecraft:air" if operation == "place" else "minecraft:coarse_dirt"},
        "expected_after": {"target_cell":"minecraft:coarse_dirt" if operation == "place" else "minecraft:air"},
        "reference_marker": {
            "shape":"ground_cross",
            "center_cell":[tx,ty-1,tz],
            "center_material":scene["marker"],
            "arm_material":scene["cross"],
            "arm_radius_blocks":4,
        },
        "scene_spec": {
            "origin":[tx,ty,tz],
            "floor_material":scene["floor"],
            "cross_material":scene["cross"],
            "marker_material":scene["marker"],
            "geometry_relative":scene["geometry"],
        },
        "timeline": timeline(operation, turn_sign),
        "native_action_protocol": {
            "input":"X11_XTest",
            "camera":"relative_mouse_motion",
            "placement_button":"right",
            "destruction_button":"left",
            "only_bravo_changes_world":True,
            "player_positions_fixed":True,
        },
        "evidence_frames": {
            "joint_initial":10,
            "operator_post_action":80,
            "returning_observer_checks":[160,170,180],
        },
        "visibility_contract": {
            "both_see_target_initially":True,
            "alpha_target_hidden_range":[40,130],
            "bravo_target_hidden_range":[110,181],
            "both_target_hidden_range":[110,130],
            "alpha_returns_to_initial_yaw":True,
            "minimum_contiguous_hidden_frames":20,
        },
        "first_frame_reuse": {
            "allowed":True,
            "runtime_dependency":False,
            "policy":"copy matching shared-state scene still into this suite; the complete scene specification above remains authoritative",
        },
        "independence_group":f"MWB3-SP:{scene['scene_id']}:{operation}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    index = 0
    for scene in SCENES:
        for operation in ("place", "destroy"):
            for variant in (0, 1):
                index += 1
                cases.append(build_case(index, scene, operation, variant))
    suite = {
        "schema_version":"mwb.v3.shared_state_persistence.case_set.v1",
        "metric_id":"shared_state_persistence",
        "name_zh":"共享状态持久性",
        "case_count":len(cases),
        "formal_score":True,
        "gt_collected":False,
        "first_frames_materialized":False,
        "factorization":{"scenes":5,"operations":2,"turn_directions":2,"total":20},
        "design":"A and B jointly observe; A turns away; B changes one block; B turns away; A returns and verifies the persistent state",
        "score_protocol": {
            "questions":["operator_observed_expected_change","returning_observer_observed_same_change"],
            "check_frames":[160,170,180],
            "case_aggregation":"majority_per_question_then_strict_product",
            "suite_aggregation":"macro_mean_of_place_and_destroy_case_pass_rates",
        },
        "frame_range_convention":"half_open_[start,end)",
        "cases":cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output":str(args.output),"cases":len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
