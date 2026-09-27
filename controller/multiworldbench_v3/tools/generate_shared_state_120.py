#!/usr/bin/env python3
"""Generate the 120-case V3 shared-state synchronization suite."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

SELECTED = [
    {"source_candidate":"MWB3-SS-SC01","scene_id":"grass_oak","name_zh":"草地木构","floor":"minecraft:grass_block","cross":"minecraft:white_concrete","marker":"minecraft:red_concrete","geometry":[[-7,0,-6,-7,3,-6,"minecraft:oak_log"],[7,0,-6,7,3,-6,"minecraft:oak_log"],[-7,4,-6,7,4,-6,"minecraft:oak_log"],[-6,0,-6,-2,0,-6,"minecraft:mossy_cobblestone"],[2,0,-6,6,0,-6,"minecraft:mossy_cobblestone"]]},
    {"source_candidate":"MWB3-SS-SC02","scene_id":"sandstone_court","name_zh":"砂岩庭院","floor":"minecraft:smooth_sandstone","cross":"minecraft:yellow_concrete","marker":"minecraft:blue_concrete","geometry":[[-4,0,-6,-4,3,-6,"minecraft:cut_sandstone"],[4,0,-6,4,3,-6,"minecraft:cut_sandstone"],[-4,4,-6,4,4,-6,"minecraft:smooth_sandstone"],[-9,0,-4,-7,1,-4,"minecraft:sandstone"],[7,0,-4,9,1,-4,"minecraft:sandstone"]]},
    {"source_candidate":"MWB3-SS-SC03","scene_id":"stone_plaza","name_zh":"石砖广场","floor":"minecraft:stone_bricks","cross":"minecraft:cyan_concrete","marker":"minecraft:orange_concrete","geometry":[[-8,0,-6,-8,3,-6,"minecraft:polished_andesite"],[8,0,-6,8,3,-6,"minecraft:polished_andesite"],[-5,0,-7,-5,1,-7,"minecraft:chiseled_stone_bricks"],[5,0,-7,5,2,-7,"minecraft:chiseled_stone_bricks"],[-3,0,-6,3,0,-6,"minecraft:mossy_stone_bricks"]]},
    {"source_candidate":"MWB3-SS-SC05","scene_id":"tuff_copper","name_zh":"凝灰铜庭","floor":"minecraft:tuff","cross":"minecraft:orange_concrete","marker":"minecraft:light_blue_concrete","geometry":[[-7,0,-6,-7,3,-6,"minecraft:oxidized_copper"],[7,0,-6,7,3,-6,"minecraft:cut_copper"],[-7,4,-6,7,4,-6,"minecraft:tuff_bricks"],[-4,0,-7,-2,1,-7,"minecraft:exposed_copper"],[2,0,-7,4,1,-7,"minecraft:weathered_copper"]]},
    {"source_candidate":"MWB3-SS-SC07","scene_id":"snow_ice","name_zh":"冰雪庭院","floor":"minecraft:snow_block","cross":"minecraft:magenta_concrete","marker":"minecraft:lime_concrete","geometry":[[-7,0,-6,-7,3,-6,"minecraft:packed_ice"],[7,0,-6,7,3,-6,"minecraft:blue_ice"],[-7,4,-6,7,4,-6,"minecraft:spruce_log"],[-4,0,-7,-2,1,-7,"minecraft:packed_ice"],[2,0,-7,4,1,-7,"minecraft:blue_ice"]]},
]
VIS = ("mutual", "one_way", "none")
INTERACTIONS = (
    ("single_place", "place", "single"),
    ("single_destroy", "destroy", "single"),
    ("dual_place", "place", "dual"),
    ("dual_destroy", "destroy", "dual"),
)

def look_at(position, aim):
    dx=aim[0]-position[0]; dy=aim[1]-(position[1]+1.62); dz=aim[2]-position[2]
    return math.degrees(math.atan2(-dx,dz)), -math.degrees(math.atan2(dy,math.hypot(dx,dz)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    tx,ty,tz=760,91,0; cases=[]; index=0
    for scene_i,scene in enumerate(SELECTED,1):
      for visibility in VIS:
       for interaction,operation,actors in INTERACTIONS:
        for variant in range(2):
          index += 1; cid=f"MWB3-S-SS{index:03d}"
          if visibility=='mutual':
            positions={'alpha':[tx-1.5,ty,tz+3.5], 'bravo':[tx+2.5,ty,tz+3.5]}; visibility_direction='bidirectional'
          elif visibility=='one_way' and variant==0:
            positions={'alpha':[tx-1.5,ty,tz+4.0], 'bravo':[tx+2.5,ty,tz+2.5]}; visibility_direction='alpha_sees_bravo'
          elif visibility=='one_way':
            positions={'alpha':[tx-1.5,ty,tz+2.5], 'bravo':[tx+2.5,ty,tz+4.0]}; visibility_direction='bravo_sees_alpha'
          else:
            positions={'alpha':[tx-1.5,ty,tz+3.5], 'bravo':[tx+2.5,ty,tz+3.5]}; visibility_direction='neither'
          if actors=='single':
            active=['alpha'] if variant==0 else ['bravo']; targets={'alpha':[tx,ty,tz], 'bravo':[tx,ty,tz]}; target_cells=[[tx,ty,tz]]
          else:
            active=['alpha','bravo'];
            targets={'alpha':[tx-1,ty,tz], 'bravo':[tx+1,ty,tz]}
            target_cells=[[tx-1,ty,tz],[tx+1,ty,tz]]
          poses={}
          for role in ('alpha','bravo'):
            target=targets[role]; aim_y=ty if operation=='place' else ty+0.5
            aim=[target[0]+.5,aim_y,target[2]+.5]; yaw,pitch=look_at(positions[role],aim)
            poses[role]={'pose':{'position':positions[role],'native_yaw_deg':yaw,'native_pitch_deg':pitch,'aim_point':aim,'aim_role':'marked_target','active':role in active,'target_cell':target,'support_cell':[target[0],ty-1,target[2]]}}
          expected_count=len(target_cells)
          cases.append({
            'case_id':cid,'id':cid,'metrics':['shared_state_synchronization'],'scene':scene['scene_id'],'scene_name_zh':scene['name_zh'],'source_candidate':scene['source_candidate'],
            'visibility_class':visibility,'visibility_direction':visibility_direction,'interaction_type':interaction,'operation':operation,'actor_mode':actors,'variant':variant,
            'active_roles':active,'fps':20,'num_frames':81,'valid_num_frames':81,'duration_seconds':4.05,'round_count':1,
            'initial_conditions':poses,'target_cell':target_cells[0],'target_cells':target_cells,'expected_change_count':expected_count,
            'expected_after':{'marked_targets':'minecraft:coarse_dirt' if operation=='place' else 'minecraft:air'},
            'reference_marker':{'shape':'ground_cross','center_cell':[tx,ty-1,tz],'target_support_cells':[[x,ty-1,z] for x,_,z in target_cells], 'center_material':scene['marker'],'arm_material':scene['cross'],'arm_radius_blocks':4},
            'scene_spec':{'origin':[tx,ty,tz],'floor_material':scene['floor'],'cross_material':scene['cross'],'marker_material':scene['marker'],'geometry_relative':scene['geometry'],'visibility_occluder':({'relative':[0,0,3,0,2,3],'material':'minecraft:polished_basalt'} if visibility=='none' else None)},
            'action_protocol':{'native_input':'X11_XTest','start_frame':10,'end_frame_exclusive':11 if operation=='place' else 18,'button':'right' if operation=='place' else 'left'},
            'independence_group':f"MWB3-SS:{scene['scene_id']}:{visibility}:{interaction}",
          })
    suite={'schema_version':'mwb.v3.shared_state.case_set.v2','case_count':len(cases),'formal_score':True,'gt_collected':False,
      'selection':{'method':'fixed_balanced_sample','selected_candidates':[s['source_candidate'] for s in SELECTED],'seed':20260927},
      'factorization':{'scenes':5,'visibility_classes':3,'interaction_types':4,'variants_per_condition':2,'total':120},
      'visibility_classes':{'mutual':'A and B mutually visible','one_way':'variant 0 A sees B; variant 1 B sees A','none':'A and B mutually invisible'},
      'interaction_types':['single_place','single_destroy','dual_place','dual_destroy'],
      'action_evidence_policy':'native_x11_xtest_gt_with_versioned_model_actions','cases':cases}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(suite,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'cases':len(cases),'scenes':suite['selection']['selected_candidates']},ensure_ascii=False))
if __name__=='__main__': main()
