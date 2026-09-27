#!/usr/bin/env python3
"""Generate the balanced 180-case V3 navigation joint-evaluation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCENES = [
    ("S03", "MWB3-N-SC03", "折返崖壁工坊", "switchback_cliff_workshop", 4192, [-25, 64, -16], [23, 64, -18], [1, 69, 8]),
    ("S05", "MWB3-N-SC05", "铸造厂高架庭院", "foundry_catwalk_yard", 4384, [-25, 64, -17], [24, 64, -16], [0, 69, 7]),
    ("S06", "MWB3-N-SC06", "双桥峡谷节点", "twin_bridge_canyon_hub", 4480, [-25, 64, -18], [24, 64, -16], [0, 68, 7]),
    ("S08", "MWB3-N-SC08", "岩林遗迹神社", "rocky_woodland_shrine", 4672, [-24, 64, -17], [23, 64, -17], [1, 67, 8]),
    ("S09", "MWB3-N-SC09", "干船坞造船场", "dry_dock_shipyard", 4768, [-25, 64, -17], [24, 64, -16], [0, 67, 8]),
    ("S10", "MWB3-N-SC10", "玄武岩晶簇洞厅", "basalt_crystal_cavern", 4864, [-23, 64, -18], [22, 64, -17], [0, 68, 7]),
]

# Five native intervention levels. Every case contains all five levels for each
# role. IDLE is an observed no-input control, not a synthetic trajectory.
ACTION_ORDER = ("idle", "forward", "back", "left", "look_left")
ACTIONS = {
    "idle": {"action_id": "idle", "type": "IDLE", "duration_ticks": 20},
    "forward": {"action_id": "forward", "type": "MOVE", "control": "forward", "distance": 1.2},
    "back": {"action_id": "back", "type": "MOVE", "control": "back", "distance": 1.0},
    "left": {"action_id": "left", "type": "MOVE", "control": "left", "distance": 1.0},
    "look_left": {"action_id": "look_left", "type": "LOOK", "yaw_delta_deg": 25, "pitch_delta_deg": 0},
}


def clone(value):
    return json.loads(json.dumps(value))


def rotate(values, shift):
    return list(values[shift:]) + list(values[:shift])


def templates():
    """Create two five-row cyclic intervention matrices."""
    result = []
    for family, fixed_role in (("AF", "alpha"), ("BF", "bravo")):
        variable_role = "bravo" if fixed_role == "alpha" else "alpha"
        for shift in range(5):
            fixed = list(ACTION_ORDER)
            variable = rotate(ACTION_ORDER, shift)
            schedules = {fixed_role: fixed, variable_role: variable}
            result.append({
                "id": f"{family}{shift + 1:02d}",
                "title": f"{fixed_role.upper()} 固定序列 / {variable_role.upper()} 循环偏移 {shift}",
                "fixed_role": fixed_role,
                "variable_role": variable_role,
                "latin_shift": shift,
                "alpha_action_ids": schedules["alpha"],
                "bravo_action_ids": schedules["bravo"],
                "counter_of": f"{'BF' if family == 'AF' else 'AF'}{shift + 1:02d}",
            })
    return result


def phase_ranges(round_count: int):
    start = 20
    return [[start + i * 20, start + (i + 1) * 20] for i in range(round_count)]


def visibility_for(prefix, family):
    if prefix == "MV":
        return {"alpha_sees_bravo": True, "bravo_sees_alpha": True}
    if prefix == "NV":
        return {"alpha_sees_bravo": False, "bravo_sees_alpha": False}
    return {
        "alpha_sees_bravo": family == "AF",
        "bravo_sees_alpha": family == "BF",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = []
    classes = [
        ("MV", "mutual_visible"),
        ("OV", "one_way_visible"),
        ("NV", "mutual_invisible"),
    ]
    action_templates = templates()
    for scene_key, scene_id, scene_name, slug, center_x, alpha, bravo, focus in SCENES:
        scene = {
            "scene_id": scene_id,
            "scene_name": scene_name,
            "slug": slug,
            "center_x": center_x,
            "alpha_offset": alpha,
            "bravo_offset": bravo,
            "focus_offset": focus,
            "geometry_rich": True,
        }
        for prefix, visibility_class in classes:
            suite_group = f"MWB3-N-{prefix}-{scene_key}-JOINT"
            for template in action_templates:
                family = template["id"][:2]
                case_id = f"MWB3-N-{prefix}-{scene_key}-{template['id']}"
                counter_case_id = f"MWB3-N-{prefix}-{scene_key}-{template['counter_of']}"
                turns = []
                for index, (alpha_id, bravo_id) in enumerate(
                    zip(template["alpha_action_ids"], template["bravo_action_ids"]), 1
                ):
                    turns.append({
                        "id": f"R{index:02d}",
                        "duration_ticks": 20,
                        "alpha": clone(ACTIONS[alpha_id]),
                        "bravo": clone(ACTIONS[bravo_id]),
                    })
                cases.append({
                    "case_id": case_id,
                    "id": case_id,
                    "scene": clone(scene),
                    "visibility_class": visibility_class,
                    "initial_visibility": visibility_for(prefix, family),
                    "counter_case_id": counter_case_id,
                    "counter_case_of": counter_case_id,
                    "role_swapped": family == "BF",
                    "independence_group": suite_group,
                    "intervention_group": f"{suite_group}-{family}",
                    "fixed_role": template["fixed_role"],
                    "variable_role": template["variable_role"],
                    "latin_shift": template["latin_shift"],
                    "action_template_id": template["id"],
                    "action_template_title": template["title"],
                    "round_count": len(turns),
                    "turns": turns,
                    "initial_idle_frames": 20,
                    "phase_ranges": phase_ranges(len(turns)),
                    "valid_num_frames": 189,
                    "fps": 20,
                    "metrics": [
                        "control_isolation",
                        "coordination_fairness",
                        "cross_view_motion_consistency",
                    ],
                })
    doc = {
        "schema_version": "mwb.v3.navigation_visibility_suite.v2",
        "benchmark_version": "3",
        "self_contained": True,
        "metric_ids": [
            "control_isolation",
            "coordination_fairness",
            "cross_view_motion_consistency",
        ],
        "case_count": 180,
        "description": "三类初始可见关系、六个富几何地图；每个地图使用十例双向平衡干预矩阵，同时支持控制隔离度与协同公平性。",
        "visibility_semantics": "由初始相机水平视锥决定；单向可见 AF/BF family 互为角色交换 counter。",
        "joint_metric_design": {
            "action_alphabet": list(ACTION_ORDER),
            "rounds_per_case": 5,
            "cases_per_scene_visibility": 10,
            "alpha_fixed_cases": 5,
            "bravo_fixed_cases": 5,
            "control_isolation": "在 AF family 内固定 A 的逐轮动作与路径、令 B 的动作按五阶循环矩阵变化；BF family 对称地固定 B。每个固定目标动作恰好配对全部五种另一角色动作。两个方向取较低分。",
            "coordination_fairness": "两种 family 和角色互换使 A/B 对每个精确动作拥有相同样本数；按动作比较相机响应分布。",
            "history_control": "每个 fixed-role family 内目标角色的完整五轮动作历史、初始位姿、地图和可见关系保持一致。",
        },
        "capture": {
            "fps": 20,
            "frames": 189,
            "resolution": [1280, 720],
            "initial_idle_frames": 20,
            "turn_ticks": 20,
        },
        "counts": {
            "total": 180,
            "visibility_classes": 3,
            "scenes_per_class": 6,
            "action_cases_per_scene": 10,
            "rounds_per_case": 5,
        },
        "scenes": [
            dict(
                zip(
                    ("key", "scene_id", "name", "slug", "center_x", "alpha_offset", "bravo_offset", "focus_offset"),
                    row,
                )
            )
            for row in SCENES
        ],
        "action_templates": action_templates,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
