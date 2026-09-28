'use strict';

const fs = require('fs');
const path = require('path');
const {candidates, aimPose, builder, base} = require('./capture_building_reveal_expansion_stills.js');

const ANGLES = [20, 30, 40, 50];
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

async function commandsFor(cx, scene, building, alpha, bravo) {
  const commands = [];
  const send = async command => { commands.push(command); return ''; };
  const b = builder(send, cx);
  await base(b, scene.ground);
  await scene.sceneBuild(b);
  await building.buildingBuild(b);
  for (const position of [alpha, bravo]) {
    const px = Math.floor(position[0] - cx), pz = Math.floor(position[2]);
    await b.fill(px - 1, 63, pz - 1, px + 1, 63, pz + 1, scene.ground);
    await b.fill(px - 1, 64, pz - 1, px + 1, 66, pz + 1, 'air');
  }
  await send(`setblock ${cx} 65 ${OCCLUDER_Z} minecraft:dirt`);
  return commands;
}

async function main() {
  const repo = path.resolve(__dirname, '../../..');
  const output = path.join(repo, 'MultiWorldBench/v3/designs/building_reveal_observer_pilot.json');
  const scene = candidates[0];
  const building = candidates[0];
  const cases = [];
  for (let i = 0; i < ANGLES.length; i++) {
    const angle = ANGLES[i];
    const cx = 15000;
    const alpha = [cx + 0.5, 64, ALPHA_Z];
    const buildingFocus = [cx + 0.5, BUILDING_FOCUS_Y, CENTER_Z];
    const rad = angle * Math.PI / 180;
    const bravo = [cx + 0.5 + OBSERVER_RADIUS * Math.sin(rad), 64, CENTER_Z - OBSERVER_RADIUS * Math.cos(rad)];
    const bravoFocus = observerFocus(bravo, alpha, buildingFocus);
    const alphaAim = aimPose(alpha, buildingFocus);
    const bravoAim = aimPose(bravo, bravoFocus);
    const caseId = `MWB3-S-RV-PILOT-S01-B01-A${angle}`;
    cases.push({
      case_id: caseId, id: caseId,
      metrics: ['cross_view_structural_coherence', 'spatial_reasoning'],
      pilot: true,
      scene_id: scene.scene_id, scene_name: scene.scene,
      building_id: building.building_id, building_name: building.building,
      relative_agent_angle_deg: angle,
      actor_radius: ACTOR_RADIUS,
      observer_radius: OBSERVER_RADIUS,
      world_x_offset: cx,
      setup_commands: await commandsFor(cx, scene, building, alpha, bravo),
      initial_poses: {
        alpha: {role: 'actor', position: alpha, native_yaw_deg: alphaAim.yaw, native_pitch_deg: alphaAim.pitch, focus: buildingFocus},
        bravo: {role: 'stationary_observer', position: bravo, native_yaw_deg: bravoAim.yaw, native_pitch_deg: bravoAim.pitch, focus: bravoFocus}
      },
      visibility_contract: {
        before_action: {alpha_sees_building: false, bravo_sees_building: true, bravo_sees_alpha: true, bravo_sees_building_and_alpha_simultaneously: true},
        after_action: {alpha_sees_building: true, alpha_visible_building_fraction: [0.60, 0.80], bravo_sees_building: true, bravo_sees_alpha: true, bravo_sees_building_and_alpha_simultaneously: true}
      },
      reveal: {
        actor: 'alpha',
        occluder: {material: 'minecraft:dirt', position: [cx, 65, OCCLUDER_Z], count: 1},
        reveal_visibility_target: {minimum_fraction: 0.60, preferred_range: [0.60, 0.80], basis: 'whole_building_visible_in_native_gt_alpha'},
        required_native_action: 'hold_attack', action_frames: [21, 60]
      },
      recording: {fps: 20, frames: 129, check_frames: CHECK_FRAMES},
      action_timeline: [
        {frames: [0, 20], alpha: 'idle', bravo: 'idle'},
        {frames: [21, 60], alpha: 'attack_hold', bravo: 'idle'},
        {frames: [61, 128], alpha: 'idle_after_reveal', bravo: 'idle'}
      ],
      camera_angle_tolerance_deg: 30,
      output_directory: `outputs/shared_world/building_reveal_pilot/${caseId}`
    });
  }
  const suite = {
    schema_version: 'mwb.v3.shared_building_reveal.observer_pilot.v1',
    benchmark_version: '3', status: 'pilot_not_collected', self_contained: true,
    case_count: cases.length,
    factorization: {scenes: 1, buildings: 1, relative_agent_angles_deg: ANGLES, total: cases.length},
    description: 'Observer-layout pilot. Stationary B is farther from the target and aims at the angular bisector so that the building and occluded actor A are simultaneously visible.',
    common_visibility_requirement: 'B must simultaneously show the building and A in the same native GT frame before and after reveal.',
    cases
  };
  fs.writeFileSync(output, JSON.stringify(suite, null, 2) + '\n');
  console.log(JSON.stringify({output, cases: cases.map(c => ({id:c.case_id, angle:c.relative_agent_angle_deg, bravo:c.initial_poses.bravo.position, yaw:c.initial_poses.bravo.native_yaw_deg}))}, null, 2));
}
main().catch(error => { console.error(error.stack || error); process.exit(1); });
