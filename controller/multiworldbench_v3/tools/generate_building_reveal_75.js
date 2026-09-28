'use strict';

const fs = require('fs');
const path = require('path');
const {candidates, aimPose, builder, base} = require('./capture_building_reveal_expansion_stills.js');

const ANGLES = [20, 30, 40];
const CHECK_FRAMES = [80, 100, 120];
const ACTOR_RADIUS = 21.85;
const OBSERVER_RADIUS = 26;
const CENTER_Z = 2.5;
const ALPHA_Z = CENTER_Z - ACTOR_RADIUS;
const OCCLUDER_Z = -19;
const BUILDING_FOCUS_Y = 69;

function normalize2(x, z) {
  const n = Math.hypot(x, z);
  return [x / n, z / n];
}

function observerFocus(bravo, alpha, building) {
  const toActor = normalize2(alpha[0] - bravo[0], alpha[2] - bravo[2]);
  const toBuilding = normalize2(building[0] - bravo[0], building[2] - bravo[2]);
  const bisector = normalize2(toActor[0] + toBuilding[0], toActor[1] + toBuilding[1]);
  return [bravo[0] + bisector[0] * 25, 67.5, bravo[2] + bisector[1] * 25];
}
const scenes = candidates.map(({scene_id, scene, ground, sceneBuild}) => ({scene_id, scene, ground, sceneBuild}));
const buildings = candidates.map(({building_id, building, buildingBuild}) => ({building_id, building, buildingBuild}));

async function commandsFor(cx, scene, building, alpha, bravo) {
  const commands = [];
  const send = async command => { commands.push(command); return ''; };
  const b = builder(send, cx);
  await base(b, scene.ground);
  await scene.sceneBuild(b);
  await building.buildingBuild(b);
  // Keep both stationary cameras on deterministic ground. Some rich scenes
  // contain rails or docks at the nominal orbit coordinates; a small local
  // landing pad prevents collision push without changing game code.
  for (const position of [alpha, bravo]) {
    const px=Math.floor(position[0]-cx), pz=Math.floor(position[2]);
    await b.fill(px-1,63,pz-1,px+1,63,pz+1,scene.ground);
    await b.fill(px-1,64,pz-1,px+1,66,pz+1,'air');
  }
  // One nearby dirt block hides the building before the action. There is no
  // surrounding aperture wall: removing this single block opens the complete
  // view rather than a one-block slit.
  await send(`setblock ${cx} 65 ${OCCLUDER_Z} minecraft:dirt`);
  return commands;
}

async function main() {
  const repo = path.resolve(__dirname, '../../..');
  const output = path.join(repo, 'MultiWorldBench/v3/cases/shared_world/building_reveal_shared.json');
  const cases = [];
  let ordinal = 0;
  for (let si=0; si<scenes.length; si++) {
    for (let bi=0; bi<buildings.length; bi++) {
      for (const angle of ANGLES) {
        const scene=scenes[si], building=buildings[bi];
        const cx=7200 + ordinal*80;
        const alpha=[cx+0.5,64,ALPHA_Z];
        const focus=[cx+0.5,BUILDING_FOCUS_Y,CENTER_Z];
        const rad=angle*Math.PI/180;
        const observerRadius=OBSERVER_RADIUS;
        const bravo=[cx+0.5+observerRadius*Math.sin(rad),64,CENTER_Z-observerRadius*Math.cos(rad)];
        const bravoFocus=observerFocus(bravo,alpha,focus);
        const alphaAim=aimPose(alpha,focus), bravoAim=aimPose(bravo,bravoFocus);
        const caseId=`MWB3-S-RV-S${String(si+1).padStart(2,'0')}-B${String(bi+1).padStart(2,'0')}-A${angle}`;
        cases.push({
          case_id:caseId,id:caseId,
          metrics:['cross_view_structural_coherence','spatial_reasoning'],
          independence_group:`MWB3-S-RV-S${String(si+1).padStart(2,'0')}-B${String(bi+1).padStart(2,'0')}`,
          scene_id:scene.scene_id,scene_name:scene.scene,
          building_id:building.building_id,building_name:building.building,
          relative_agent_angle_deg:angle,actor_radius:ACTOR_RADIUS,
          observer_radius:observerRadius,world_x_offset:cx,
          setup_commands:await commandsFor(cx,scene,building,alpha,bravo),
          initial_poses:{
            alpha:{role:'actor',position:alpha,native_yaw_deg:alphaAim.yaw,native_pitch_deg:alphaAim.pitch,focus},
            bravo:{role:'stationary_observer',position:bravo,native_yaw_deg:bravoAim.yaw,native_pitch_deg:bravoAim.pitch,focus:bravoFocus}
          },
          visibility_contract:{
            before_action:{alpha_sees_building:false,bravo_sees_building:true,bravo_sees_alpha:true,bravo_sees_building_and_alpha_simultaneously:true},
            after_action:{alpha_sees_building:true,alpha_visible_building_fraction:[0.60,0.80],bravo_sees_building:true,bravo_sees_alpha:true,bravo_sees_building_and_alpha_simultaneously:true}
          },
          reveal:{
            actor:'alpha',occluder:{material:'minecraft:dirt',position:[cx,65,OCCLUDER_Z],count:1},
            reveal_visibility_target:{minimum_fraction:0.60,preferred_range:[0.60,0.80],basis:'whole_building_visible_in_native_gt_alpha'},
            required_native_action:'hold_attack',action_frames:[21,60]
          },
          recording:{fps:20,frames:129,check_frames:CHECK_FRAMES},
          action_timeline:[
            {frames:[0,20],alpha:'idle',bravo:'idle'},
            {frames:[21,60],alpha:'attack_hold',bravo:'idle'},
            {frames:[61,128],alpha:'idle_after_reveal',bravo:'idle'}
          ],
          camera_angle_tolerance_deg:30,
          output_directory:`outputs/shared_world/building_reveal/${caseId}`
        });
        ordinal++;
      }
    }
  }
  const suite={
    schema_version:'mwb.v3.shared_building_reveal.v4',benchmark_version:'3',
    metric_ids:['cross_view_structural_coherence','spatial_reasoning'],
    status:'designed_not_collected',ranking_ready:false,self_contained:true,external_case_dependencies:[],
    case_count:cases.length,factorization:{scenes:5,buildings:5,relative_agent_angles_deg:ANGLES,total:75},
    description:'A is initially blocked by one near-camera dirt block and destroys it with native attack input. The unobstructed GT view must expose at least 60% of the whole building. Stationary observer B stays 26 blocks from the building and simultaneously sees A and the building throughout. The same fully crossed suite supplies both metrics.',
    check_frames:CHECK_FRAMES,
    common_preconditions:[
      {id:'occluder_removed',text:'A 是否成功移除遮挡建筑的泥土块？',values:[0,1]},
      {id:'target_revealed',text:'移除遮挡后，A 是否能看到并辨认建筑？',values:[0,1]}
    ],
    vlm_questions:[
      {id:'same_building_gt',text:'生成 A 与 GT A 中是不是同一个建筑？',values:[0,1]},
      {id:'same_building_ab',text:'生成 A 与生成 B 中是不是同一个建筑？',values:[0,1]}
    ],
    metric_aggregation:{
      cross_view_structural_coherence:'公共前提通过后，两个建筑一致性问题各做 80/100/120 三帧多数票，case 分数取较低值。',
      spatial_reasoning:'复用公共前提和跨视角 same_building_gt；不重复进行建筑身份 VLM 判断。通过后取投影面积一致性与相机角度一致性的较低值。'
    },
    scenes:scenes.map(({scene_id,scene,ground})=>({scene_id,name:scene,ground_material:`minecraft:${ground}`})),
    buildings:buildings.map(({building_id,building})=>({building_id,name:building})),cases
  };
  fs.writeFileSync(output,JSON.stringify(suite,null,2)+'\n');
  const manifestPath=path.join(repo,'MultiWorldBench/v3/manifest.json');
  const manifest=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
  for(const metric of manifest.metrics){
    if(['cross_view_structural_coherence','spatial_reasoning'].includes(metric.id)){
      metric.case_count=75;metric.status='implemented_uncollected';
      metric.suite_definition='5 geometry-rich scenes × 5 distinctive buildings × 3 relative agent angles (20°, 30°, 40°); A reveals, stationary B at radius 26 simultaneously sees A and building';
    }
  }
  fs.writeFileSync(manifestPath,JSON.stringify(manifest,null,2)+'\n');
  console.log(JSON.stringify({output,cases:cases.length,first:cases[0].case_id,last:cases[cases.length-1].case_id}));
}
main().catch(error=>{console.error(error.stack||error);process.exit(1);});
