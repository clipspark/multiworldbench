"""Generate the self-contained 60-case V3 race-condition suite."""

from __future__ import annotations

import json
from pathlib import Path


SCENES = [
    {
        "id": "grass_oak_workshop",
        "name_zh": "草地橡木工坊",
        "floor_material": "grass_block",
        "cross_material": "yellow_concrete",
        "geometry": [
            {"from": [-9, 0, -9], "to": [9, 0, 9], "material": "grass_block"},
            {"from": [-8, 1, -8], "to": [-8, 4, -8], "material": "oak_log"},
            {"from": [8, 1, -8], "to": [8, 4, -8], "material": "birch_log"},
            {"from": [-8, 4, -8], "to": [8, 4, -8], "material": "oak_planks"},
            {"from": [-8, 1, 8], "to": [-8, 3, 8], "material": "mossy_cobblestone"},
            {"from": [8, 1, 8], "to": [8, 3, 8], "material": "cobblestone"},
            {"from": [-7, 1, -8], "to": [-5, 2, -8], "material": "bookshelf"},
            {"from": [5, 1, -8], "to": [7, 2, -8], "material": "bricks"},
        ],
    },
    {
        "id": "sandstone_courtyard",
        "name_zh": "砂岩庭院",
        "floor_material": "smooth_sandstone",
        "cross_material": "red_concrete",
        "geometry": [
            {"from": [-9, 0, -9], "to": [9, 0, 9], "material": "smooth_sandstone"},
            {"from": [-9, 1, -9], "to": [-9, 3, 9], "material": "cut_sandstone"},
            {"from": [9, 1, -9], "to": [9, 3, 9], "material": "sandstone"},
            {"from": [-7, 1, -8], "to": [-7, 5, -8], "material": "chiseled_sandstone"},
            {"from": [7, 1, -8], "to": [7, 5, -8], "material": "chiseled_sandstone"},
            {"from": [-7, 5, -8], "to": [7, 5, -8], "material": "smooth_sandstone"},
            {"from": [-4, 1, 8], "to": [-2, 2, 8], "material": "orange_terracotta"},
            {"from": [2, 1, 8], "to": [4, 3, 8], "material": "red_sandstone"},
        ],
    },
    {
        "id": "deepslate_amethyst_ruin",
        "name_zh": "深板岩紫晶遗迹",
        "floor_material": "stone_bricks",
        "cross_material": "lime_concrete",
        "geometry": [
            {"from": [-9, 0, -9], "to": [9, 0, 9], "material": "stone_bricks"},
            {"from": [-9, 1, -9], "to": [-6, 3, -9], "material": "deepslate_bricks"},
            {"from": [6, 1, -9], "to": [9, 4, -9], "material": "cracked_deepslate_bricks"},
            {"from": [-8, 1, 8], "to": [-8, 5, 8], "material": "polished_basalt"},
            {"from": [8, 1, 8], "to": [8, 5, 8], "material": "polished_basalt"},
            {"from": [-8, 5, 8], "to": [8, 5, 8], "material": "deepslate_tiles"},
            {"from": [-3, 1, -8], "to": [-1, 3, -8], "material": "amethyst_block"},
            {"from": [2, 1, -8], "to": [4, 2, -8], "material": "calcite"},
        ],
    },
    {
        "id": "tuff_copper_foundry",
        "name_zh": "凝灰岩铜工场",
        "floor_material": "tuff",
        "cross_material": "blue_concrete",
        "geometry": [
            {"from": [-9, 0, -9], "to": [9, 0, 9], "material": "tuff"},
            {"from": [-9, 1, -9], "to": [-9, 4, -6], "material": "cut_copper"},
            {"from": [9, 1, -9], "to": [9, 5, -6], "material": "exposed_cut_copper"},
            {"from": [-8, 1, 8], "to": [-8, 4, 8], "material": "oxidized_cut_copper"},
            {"from": [8, 1, 8], "to": [8, 4, 8], "material": "waxed_copper_block"},
            {"from": [-8, 4, 8], "to": [8, 4, 8], "material": "smooth_stone"},
            {"from": [-4, 1, -8], "to": [-2, 3, -8], "material": "copper_block"},
            {"from": [2, 1, -8], "to": [5, 2, -8], "material": "mud_bricks"},
        ],
    },
    {
        "id": "snowy_spruce_station",
        "name_zh": "雪地云杉驿站",
        "floor_material": "snow_block",
        "cross_material": "orange_concrete",
        "geometry": [
            {"from": [-9, 0, -9], "to": [9, 0, 9], "material": "snow_block"},
            {"from": [-9, 1, -9], "to": [-9, 4, -9], "material": "spruce_log"},
            {"from": [9, 1, -9], "to": [9, 4, -9], "material": "spruce_log"},
            {"from": [-9, 4, -9], "to": [9, 4, -9], "material": "spruce_planks"},
            {"from": [-8, 1, 8], "to": [-5, 3, 8], "material": "packed_ice"},
            {"from": [5, 1, 8], "to": [8, 4, 8], "material": "blue_ice"},
            {"from": [-4, 1, -8], "to": [-2, 2, -8], "material": "stripped_spruce_log"},
            {"from": [2, 1, -8], "to": [4, 3, -8], "material": "stone_bricks"},
        ],
    },
]

BLOCK_PAIRS = [
    {"id": "gold_vs_diamond", "alpha": "gold_block", "bravo": "diamond_block"},
    {"id": "red_vs_blue_wool", "alpha": "red_wool", "bravo": "blue_wool"},
    {"id": "emerald_vs_obsidian", "alpha": "emerald_block", "bravo": "obsidian"},
]

VISIBILITY = [
    {
        "id": "mutual",
        "name_zh": "A、B 相互可见",
        "alpha_sees_bravo": True,
        "bravo_sees_alpha": True,
        "alpha_offset": [-3.0, 0.0],
        "bravo_offset": [3.0, 0.0],
    },
    {
        "id": "alpha_sees_bravo",
        "name_zh": "仅 A 看见 B",
        "alpha_sees_bravo": True,
        "bravo_sees_alpha": False,
        "alpha_offset": [0.0, 4.0],
        "bravo_offset": [0.0, 2.2],
    },
    {
        "id": "bravo_sees_alpha",
        "name_zh": "仅 B 看见 A（counter）",
        "alpha_sees_bravo": False,
        "bravo_sees_alpha": True,
        "alpha_offset": [0.0, 2.2],
        "bravo_offset": [0.0, 4.0],
    },
    {
        "id": "mutual_invisible",
        "name_zh": "A、B 互相不可见",
        "alpha_sees_bravo": False,
        "bravo_sees_alpha": False,
        "alpha_offset": [2.598076211, 1.5],
        "bravo_offset": [2.598076211, -1.5],
    },
]


def generate() -> dict:
    cases = []
    lookup = {}
    number = 0
    for scene in SCENES:
        for pair in BLOCK_PAIRS:
            group = f"MWB3-SW-RC-G{len(lookup) // 4 + 1:02d}"
            for visibility in VISIBILITY:
                number += 1
                case_id = f"MWB3-SW-RC{number:03d}"
                col, row = (number - 1) % 10, (number - 1) // 10
                tx, ty, tz = 1000 + 24 * col, 91, 24 * row
                cx, cz = tx + 0.5, tz + 0.5
                alpha = [cx + visibility["alpha_offset"][0], ty, cz + visibility["alpha_offset"][1]]
                bravo = [cx + visibility["bravo_offset"][0], ty, cz + visibility["bravo_offset"][1]]
                case = {
                    "id": case_id,
                    "case_id": case_id,
                    "title": (
                        f"并发放置竞态 · {scene['name_zh']} · {visibility['name_zh']} · "
                        f"{pair['alpha']} vs {pair['bravo']}"
                    ),
                    "metrics": ["race_condition_consistency"],
                    "scene_id": scene["id"],
                    "block_pair_id": pair["id"],
                    "visibility_class": visibility["id"],
                    "expected_agent_visibility": {
                        "alpha_sees_bravo": visibility["alpha_sees_bravo"],
                        "bravo_sees_alpha": visibility["bravo_sees_alpha"],
                    },
                    "independence_group": group,
                    "world": {
                        "target_cell": [tx, ty, tz],
                        "support_cell": [tx, ty - 1, tz],
                        "alpha_position": alpha,
                        "bravo_position": bravo,
                    },
                    "intended_materials": {"alpha": pair["alpha"], "bravo": pair["bravo"]},
                    "simultaneous_action": {
                        "native_action": "use",
                        "active_frame_range": [21, 22],
                        "same_frames_for_both_roles": True,
                        "placements_per_role": 1,
                        "target": "same target_cell",
                    },
                    "num_frames": 101,
                    "fps": 20,
                    "gt_collected": False,
                    "capture_spec": {
                        "conditioning_only": True,
                        "initial_frames": {
                            "alpha": f"outputs/{case_id}/alpha_first_frame.png",
                            "bravo": f"outputs/{case_id}/bravo_first_frame.png",
                        },
                        "generated_videos": {
                            "alpha": f"outputs/{case_id}/alpha_generated.mp4",
                            "bravo": f"outputs/{case_id}/bravo_generated.mp4",
                        },
                    },
                    "block_reference_images": {
                        "alpha": f"assets/block_references/{pair['alpha']}.png",
                        "bravo": f"assets/block_references/{pair['bravo']}.png",
                    },
                }
                cases.append(case)
                lookup[(scene["id"], pair["id"], visibility["id"])] = case

    for scene in SCENES:
        for pair in BLOCK_PAIRS:
            alpha_case = lookup[(scene["id"], pair["id"], "alpha_sees_bravo")]
            bravo_case = lookup[(scene["id"], pair["id"], "bravo_sees_alpha")]
            alpha_case["counter_case_of"] = bravo_case["case_id"]
            bravo_case["counter_case_of"] = alpha_case["case_id"]

    return {
        "schema_version": "mwb.v3.case_set.v1",
        "benchmark_version": "3",
        "metric_id": "race_condition_consistency",
        "name": "竞态条件一致性",
        "description": (
            "A、B 在同一帧向同一位置各放置一次不同方块，比较生成双视角中最终方块数量、"
            "类型的一致性，并判断最终方块是否属于本例 A/B 指定的候选材质。"
        ),
        "gt_collected": False,
        "evaluation_uses_gt": False,
        "factor_design": {
            "visibility_classes": 4,
            "scenes": 5,
            "block_pairs": 3,
            "total_cases": 60,
            "cross_product": "4 visibility classes × 5 scenes × 3 A/B block pairs",
        },
        "visibility_catalog": [
            {key: value for key, value in row.items() if key not in {"alpha_offset", "bravo_offset"}}
            for row in VISIBILITY
        ],
        "scene_catalog": SCENES,
        "block_pair_catalog": BLOCK_PAIRS,
        "check_frames": [60, 80, 100],
        "vlm_questions": [
            {"id": "same_block_count", "text": "A、B 看到的目标位置方块数量是否一致？", "output": [0, 1]},
            {"id": "same_block_type", "text": "A、B 看到的目标位置方块类型是否一致？", "output": [0, 1]},
            {
                "id": "matches_intended_block",
                "text": "最终目标位置的方块是否与 A 或 B 本例指定的候选方块之一相同？",
                "output": [0, 1],
            },
        ],
        "frame_aggregation": "three-frame majority per question",
        "case_aggregation": "minimum of the three binary questions",
        "case_count": len(cases),
        "ranking_ready": False,
        "cases": cases,
    }


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[3] / "MultiWorldBench/v3/cases/shared_world/race_condition_consistency.json"
    destination.write_text(json.dumps(generate(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)
