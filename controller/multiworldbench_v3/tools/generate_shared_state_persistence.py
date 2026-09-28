#!/usr/bin/env python3
"""Generate the self-contained 50-case shared-state persistence suite."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

BLOCK_TYPES = [
    {"block_id":"minecraft:coarse_dirt","name_zh":"粗泥土","tool_id":"minecraft:diamond_shovel","tool_name_zh":"钻石铲"},
    {"block_id":"minecraft:oak_planks","name_zh":"橡木木板","tool_id":"minecraft:diamond_axe","tool_name_zh":"钻石斧"},
    {"block_id":"minecraft:red_wool","name_zh":"红色羊毛","tool_id":"minecraft:shears","tool_name_zh":"剪刀"},
    {"block_id":"minecraft:gold_block","name_zh":"金块","tool_id":"minecraft:diamond_pickaxe","tool_name_zh":"钻石镐"},
    {"block_id":"minecraft:diamond_block","name_zh":"钻石块","tool_id":"minecraft:diamond_pickaxe","tool_name_zh":"钻石镐"},
]
SCENES = [
    {"scene_id":"grass_oak","name_zh":"草地木构","floor":"minecraft:grass_block","cross":"minecraft:white_concrete","marker":"minecraft:red_concrete","geometry":[[-8,0,-7,-8,4,-7,"minecraft:oak_log"],[8,0,-7,8,4,-7,"minecraft:oak_log"],[-8,5,-7,8,5,-7,"minecraft:oak_log"],[-7,0,-7,-3,1,-7,"minecraft:mossy_cobblestone"],[3,0,-7,7,2,-7,"minecraft:oak_planks"],[-10,0,-2,-9,3,-2,"minecraft:birch_log"],[9,0,-3,10,2,-3,"minecraft:moss_block"]]},
    {"scene_id":"sandstone_court","name_zh":"砂岩庭院","floor":"minecraft:smooth_sandstone","cross":"minecraft:cyan_concrete","marker":"minecraft:black_concrete","geometry":[[-6,0,-7,-6,5,-7,"minecraft:cut_sandstone"],[6,0,-7,6,5,-7,"minecraft:cut_sandstone"],[-6,6,-7,6,6,-7,"minecraft:smooth_sandstone"],[-10,0,-4,-8,2,-4,"minecraft:chiseled_sandstone"],[8,0,-4,10,3,-4,"minecraft:sandstone"],[-3,0,-9,3,1,-9,"minecraft:smooth_red_sandstone"]]},
    {"scene_id":"deepslate_ruin","name_zh":"深板岩遗迹","floor":"minecraft:polished_deepslate","cross":"minecraft:lime_concrete","marker":"minecraft:yellow_concrete","geometry":[[-8,0,-7,-8,5,-7,"minecraft:polished_basalt"],[8,0,-7,8,5,-7,"minecraft:polished_basalt"],[-8,6,-7,8,6,-7,"minecraft:deepslate_bricks"],[-5,0,-8,-2,2,-8,"minecraft:deepslate_tiles"],[2,0,-8,5,3,-8,"minecraft:cracked_deepslate_bricks"],[-10,0,-2,-9,3,-2,"minecraft:amethyst_block"],[9,0,-2,10,2,-2,"minecraft:calcite"]]},
    {"scene_id":"tuff_copper","name_zh":"凝灰铜庭","floor":"minecraft:tuff","cross":"minecraft:orange_concrete","marker":"minecraft:light_blue_concrete","geometry":[[-8,0,-7,-8,5,-7,"minecraft:oxidized_copper"],[8,0,-7,8,5,-7,"minecraft:cut_copper"],[-8,6,-7,8,6,-7,"minecraft:tuff_bricks"],[-6,0,-8,-3,2,-8,"minecraft:exposed_copper"],[3,0,-8,6,3,-8,"minecraft:weathered_copper"],[-10,0,-2,-9,3,-2,"minecraft:chiseled_tuff"],[9,0,-3,10,2,-3,"minecraft:copper_grate"]]},
    {"scene_id":"snow_spruce","name_zh":"冰雪云杉庭","floor":"minecraft:snow_block","cross":"minecraft:magenta_concrete","marker":"minecraft:lime_concrete","geometry":[[-8,0,-7,-8,5,-7,"minecraft:packed_ice"],[8,0,-7,8,5,-7,"minecraft:blue_ice"],[-8,6,-7,8,6,-7,"minecraft:spruce_log"],[-6,0,-8,-3,2,-8,"minecraft:spruce_planks"],[3,0,-8,6,3,-8,"minecraft:packed_ice"],[-10,0,-2,-9,3,-2,"minecraft:blue_ice"],[9,0,-3,10,2,-3,"minecraft:spruce_log"]]},
]

def look_at(position, aim):
    dx, dy, dz = aim[0]-position[0], aim[1]-(position[1]+1.62), aim[2]-position[2]
    return math.degrees(math.atan2(-dx,dz)), -math.degrees(math.atan2(dy,math.hypot(dx,dz)))

def timeline(operation, turn_sign):
    action = "use_once" if operation == "place" else "hold_attack"
    operation_end = 56 if operation == "place" else 95
    rows = [
      {"range":[0,20],"phase":"joint_observe","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
      {"range":[20,40],"phase":"alpha_turns_away","alpha":{"action":"camera_yaw","total_delta_deg":110*turn_sign},"bravo":{"action":"idle"}},
      {"range":[40,55],"phase":"alpha_hidden_hold","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
      {"range":[55,operation_end],"phase":"bravo_operates","alpha":{"action":"idle"},"bravo":{"action":action,"button":"right" if operation=="place" else "left"}},
    ]
    if operation_end < 95:
        rows.append({"range":[operation_end,95],"phase":"post_operation_settle","alpha":{"action":"idle"},"bravo":{"action":"idle"}})
    rows += [
      {"range":[95,105],"phase":"post_operation_observe","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
      {"range":[105,125],"phase":"bravo_turns_away","alpha":{"action":"idle"},"bravo":{"action":"camera_yaw","total_delta_deg":-110*turn_sign}},
      {"range":[125,145],"phase":"both_hidden_hold","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
      {"range":[145,165],"phase":"alpha_returns","alpha":{"action":"camera_yaw","total_delta_deg":-110*turn_sign},"bravo":{"action":"idle"}},
      {"range":[165,181],"phase":"alpha_reobserves","alpha":{"action":"idle"},"bravo":{"action":"idle"}},
    ]
    return rows


def build_case(index, scene, block, operation, turn_sign):
    tx,ty,tz=13000+(index-1)*48,91,0; target=[tx,ty,tz]
    aim=[tx+.5,ty if operation=="place" else ty+.5,tz+.5]
    # Alpha keeps the approved near observer view. Bravo starts inside vanilla
    # interaction reach so the operation is executed without camera/position drift.
    positions={"alpha":[tx-2.6,ty,tz+6.6],"bravo":[tx+1.6,ty,tz+4.0]}; conditions={}
    for role in ("alpha","bravo"):
        yaw,pitch=look_at(positions[role],aim)
        conditions[role]={"pose":{"position":positions[role],"native_yaw_deg":yaw,"native_pitch_deg":pitch,"aim_point":aim,"aim_role":"marked_target","target_cell":target}}
    cid=f"MWB3-S-SP{index:03d}"; before="minecraft:air" if operation=="place" else block["block_id"]; after=block["block_id"] if operation=="place" else "minecraft:air"
    return {
      "case_id":cid,"id":cid,"metrics":["shared_state_persistence"],"scene":scene["scene_id"],"scene_name_zh":scene["name_zh"],
      "block_type":block["block_id"],"block_name_zh":block["name_zh"],"tool_type":block["tool_id"],"tool_name_zh":block["tool_name_zh"],"operation":operation,
      "turn_direction":"right" if turn_sign>0 else "left","roles":{"operator":"bravo","returning_observer":"alpha"},"fps":20,"num_frames":181,"valid_num_frames":181,"duration_seconds":9.05,"round_count":1,
      "capture_warmup_ms":5000,"initial_conditions":conditions,"target_cell":target,"expected_change_count":1,
      "expected_before":{"target_cell":before},"expected_after":{"target_cell":after},
      "reference_marker":{"shape":"ground_cross","center_cell":[tx,ty-1,tz],"center_material":scene["marker"],"arm_material":scene["cross"],"arm_radius_blocks":4},
      "scene_spec":{"origin":[tx,ty,tz],"floor_material":scene["floor"],"cross_material":scene["cross"],"marker_material":scene["marker"],"geometry_relative":scene["geometry"]},
      "timeline":timeline(operation,turn_sign),
      "native_action_protocol":{"input":"X11_XTest","camera":"relative_mouse_motion","placement_button":"right","destruction_button":"left","only_bravo_changes_world":True,"player_positions_fixed":True},
      "evidence_frames":{"joint_initial":18,"operator_post_action":100,"returning_observer_checks":[180]},
      "visibility_contract":{"both_see_target_initially":True,"alpha_target_hidden_range":[40,145],"bravo_target_hidden_range":[125,181],"both_target_hidden_range":[125,145],"alpha_returns_to_initial_yaw":True,"minimum_contiguous_hidden_frames":20},
      "independence_group":f"MWB3-SP:{scene['scene_id']}:{block['block_id']}",
    }

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();cases=[];index=0
    for si,scene in enumerate(SCENES):
      for bi,block in enumerate(BLOCK_TYPES):
        turn_sign=1 if (si+bi)%2==0 else -1
        for operation in ("place","destroy"):
          index+=1;cases.append(build_case(index,scene,block,operation,turn_sign))
    suite={"schema_version":"mwb.v3.shared_state_persistence.case_set.v2","metric_id":"shared_state_persistence","name_zh":"共享状态持久性","case_count":len(cases),"formal_score":True,"gt_collected":False,"first_frames_materialized":False,
      "factorization":{"scenes":5,"block_types":5,"operations":2,"total":50},
      "design":"A and B jointly observe the marked center; A turns away; B places or destroys one configured block; B turns away; A returns and verifies persistence",
      "score_protocol":{"questions":["operator_expected_change_occurred","returning_observer_expected_change_occurred"],"check_frames":[180],"case_aggregation":"single_check_frame_then_strict_product","suite_aggregation":"macro_mean_of_place_and_destroy_case_pass_rates"},
      "frame_range_convention":"half_open_[start,end)","block_types":BLOCK_TYPES,"cases":cases}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(suite,ensure_ascii=False,indent=2)+"\n");print(json.dumps({"output":str(a.output),"cases":len(cases)},ensure_ascii=False))
if __name__=="__main__":main()
