#!/usr/bin/env python3
"""Generate the self-contained 180-case V3 navigation visibility suite."""

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

F = {"type": "MOVE", "control": "forward", "distance": 1.2}
B = {"type": "MOVE", "control": "back", "distance": 1.0}
L = {"type": "MOVE", "control": "left", "distance": 1.0}
R = {"type": "MOVE", "control": "right", "distance": 1.0}
YL = {"type": "LOOK", "yaw_delta_deg": 25, "pitch_delta_deg": 0}
YR = {"type": "LOOK", "yaw_delta_deg": -25, "pitch_delta_deg": 0}
FL = {"type": "MOVE_LOOK", "control": "forward", "distance": 1.2, "yaw_delta_deg": 20, "pitch_delta_deg": 0}
FR = {"type": "MOVE_LOOK", "control": "forward", "distance": 1.2, "yaw_delta_deg": -20, "pitch_delta_deg": 0}

BASE = [
    ("四轮基础折线", [F, L, YL, B], [F, R, YR, B]),
    ("五轮侧移回正", [L, F, FR, B, YL], [R, F, FL, B, YR]),
    ("六轮双向弧线", [F, YL, F, R, B, YR], [F, YR, F, L, B, YL]),
    ("七轮交错转向", [FL, L, B, YR, F, R, YL], [FR, R, B, YL, F, L, YR]),
    ("八轮完整组合", [F, L, YL, F, R, B, YR, F], [F, R, YR, F, L, B, YL, F]),
]


def clone(value):
    return json.loads(json.dumps(value))


def templates():
    result = []
    for index, (title, alpha, bravo) in enumerate(BASE, 1):
        result.append({"id": f"T{index:02d}", "title": title, "alpha": clone(alpha), "bravo": clone(bravo), "counter_of": f"T{index + 5:02d}"})
    for index, (title, alpha, bravo) in enumerate(BASE, 6):
        result.append({"id": f"T{index:02d}", "title": f"{title}：角色互换", "alpha": clone(bravo), "bravo": clone(alpha), "counter_of": f"T{index - 5:02d}"})
    return result


def phase_ranges(round_count: int):
    start = 20
    return [[start + i * 20, start + (i + 1) * 20] for i in range(round_count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = []
    classes = [
        ("MV", "mutual_visible", {"alpha_sees_bravo": True, "bravo_sees_alpha": True}),
        ("OV", "one_way_visible", None),
        ("NV", "mutual_invisible", {"alpha_sees_bravo": False, "bravo_sees_alpha": False}),
    ]
    actions = templates()
    for scene_key, scene_id, scene_name, slug, center_x, alpha, bravo, focus in SCENES:
        scene = {
            "scene_id": scene_id, "scene_name": scene_name, "slug": slug,
            "center_x": center_x, "alpha_offset": alpha, "bravo_offset": bravo,
            "focus_offset": focus, "geometry_rich": True,
        }
        for prefix, visibility_class, fixed_visibility in classes:
            for template in actions:
                case_id = f"MWB3-N-{prefix}-{scene_key}-{template['id']}"
                visibility = fixed_visibility
                counter_case_id = None
                if prefix == "OV":
                    alpha_sees = int(template["id"][1:]) <= 5
                    visibility = {"alpha_sees_bravo": alpha_sees, "bravo_sees_alpha": not alpha_sees}
                    counter_case_id = f"MWB3-N-OV-{scene_key}-{template['counter_of']}"
                template_number = int(template["id"][1:])
                pair_number = template_number if template_number <= 5 else template_number - 5
                independence_group = (
                    f"MWB3-N-OV-{scene_key}-P{pair_number:02d}"
                    if prefix == "OV" else f"MWB3-N-{prefix}-{scene_key}-{template['id']}"
                )
                turns = []
                for i, (a, b) in enumerate(zip(template["alpha"], template["bravo"]), 1):
                    turns.append({"id": f"R{i:02d}", "duration_ticks": 20, "alpha": clone(a), "bravo": clone(b)})
                cases.append({
                    "case_id": case_id,
                    "id": case_id,
                    "scene": clone(scene),
                    "visibility_class": visibility_class,
                    "initial_visibility": visibility,
                    "counter_case_id": counter_case_id,
                    "counter_case_of": counter_case_id,
                    "role_swapped": prefix == "OV" and template_number > 5,
                    "independence_group": independence_group,
                    "action_template_id": template["id"],
                    "action_template_title": template["title"],
                    "round_count": len(turns),
                    "turns": turns,
                    "initial_idle_frames": 20,
                    "phase_ranges": phase_ranges(len(turns)),
                    "valid_num_frames": 189,
                    "fps": 20,
                    "metrics": ["control_isolation", "coordination_fairness", "cross_view_motion_consistency"],
                })
    doc = {
        "schema_version": "mwb.v3.navigation_visibility_suite.v1",
        "benchmark_version": "3",
        "self_contained": True,
        "metric_ids": ["control_isolation", "coordination_fairness", "cross_view_motion_consistency"],
        "case_count": 180,
        "description": "三类初始可见关系、六个富几何地图、每图十个多轮双角色动作 case。",
        "visibility_semantics": "由初始相机水平视锥决定；单向可见 case 与角色互换 counter 成对。",
        "capture": {"fps": 20, "frames": 189, "resolution": [1280, 720], "initial_idle_frames": 20, "turn_ticks": 20},
        "counts": {"total": 180, "visibility_classes": 3, "scenes_per_class": 6, "action_cases_per_scene": 10},
        "scenes": [dict(zip(("key", "scene_id", "name", "slug", "center_x", "alpha_offset", "bravo_offset", "focus_offset"), row)) for row in SCENES],
        "action_templates": actions,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
