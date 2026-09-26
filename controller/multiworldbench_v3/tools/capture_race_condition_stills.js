'use strict';
// V3 conditioning stills only. No native interaction trajectory or GT is recorded.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const { Rcon } = require('rcon-client');

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sha256 = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const angleDeg = (a, b) => {
  const dot = a[0] * b[0] + a[1] * b[1];
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  return Math.acos(Math.max(-1, Math.min(1, dot / (na * nb)))) * 180 / Math.PI;
};

async function main() {
  if (!process.env.RCON_PASSWORD) {
    throw new Error("RCON_PASSWORD must be set in the environment");
  }
  const repo = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  if (fs.existsSync(output)) throw new Error(`Refusing to overwrite ${output}`);
  const suite = JSON.parse(fs.readFileSync(path.join(repo, 'MultiWorldBench/v3/cases/shared_world/race_condition_consistency.json')));
  fs.mkdirSync(output, { recursive: true });
  const rcon = await Rcon.connect({ host: '127.0.0.1', port: 25721, password: process.env.RCON_PASSWORD, timeout: 10000 });
  const log = path.join(output, 'initialization.jsonl');
  const send = async (command) => {
    const response = await rcon.send(command);
    fs.appendFileSync(log, `${JSON.stringify({ epoch_ms: Date.now(), command, response })}\n`);
    if (/unknown or incomplete|incorrect argument|not a valid|no player was found|position is not loaded|too many blocks/i.test(String(response))) {
      throw new Error(`${command} -> ${response}`);
    }
    return response;
  };
  try {
    const online = await rcon.send('list');
    if (!online.includes('CameraAlpha') || !online.includes('CameraBravo')) throw new Error(`camera clients unavailable: ${online}`);
    await send('forceload add 690 -8 770 8');
    await send('difficulty peaceful');
    await send('gamerule doDaylightCycle false');
    await send('gamerule doWeatherCycle false');
    await send('weather clear');
    await send('time set noon');
    await send('fill 690 90 -8 770 90 8 minecraft:grass_block');
    await send('fill 690 91 -8 770 98 8 minecraft:air');
    await rcon.send('team add mwb3race');
    await send('team modify mwb3race collisionRule never');
    await send('team modify mwb3race nametagVisibility never');
    await send('team join mwb3race CameraAlpha');
    await send('team join mwb3race CameraBravo');

    const captures = [];
    for (const spec of suite.cases) {
      const caseDir = path.join(output, spec.case_id);
      fs.mkdirSync(caseDir);
      const [tx, ty, tz] = spec.world.target_cell;
      await send(`fill ${tx - 5} 91 ${tz - 5} ${tx + 5} 97 ${tz + 5} minecraft:air`);
      await send(`setblock ${tx} ${ty - 1} ${tz} minecraft:smooth_quartz`);
      await send(`setblock ${tx} ${ty} ${tz} minecraft:air`);
      // Low floor markers identify the contested support without adding objects above it.
      for (const [dx, dz] of [[-1,-1],[-1,1],[1,-1],[1,1]]) {
        await send(`setblock ${tx + dx} ${ty - 1} ${tz + dz} minecraft:yellow_concrete`);
      }
      const poses = {};
      for (const [role, name] of [['alpha', 'CameraAlpha'], ['bravo', 'CameraBravo']]) {
        const position = spec.world[`${role}_position`];
        const aim = [tx + 0.5, ty, tz + 0.5];
        const delta = [aim[0] - position[0], aim[1] - (position[1] + 1.62), aim[2] - position[2]];
        const reach = Math.hypot(...delta);
        if (reach > 4.5) throw new Error(`${spec.case_id} ${role} target is out of reach: ${reach}`);
        const yaw = Math.atan2(-delta[0], delta[2]) * 180 / Math.PI;
        const pitch = -Math.atan2(delta[1], Math.hypot(delta[0], delta[2])) * 180 / Math.PI;
        await send(`gamemode survival ${name}`);
        await send(`clear ${name}`);
        await send(`item replace entity ${name} weapon.mainhand with minecraft:${spec.intended_materials[role]} 64`);
        await send(`tp ${name} ${position.join(' ')} ${yaw} ${pitch}`);
        poses[role] = { position, aim, native_yaw_deg: yaw, native_pitch_deg: pitch, aim_distance_blocks: reach };
      }
      await wait(3500);
      const images = {};
      for (const [role, name, display] of [['alpha','CameraAlpha',':130.0'], ['bravo','CameraBravo',':131.0']]) {
        const image = path.join(caseDir, `${role}_first_frame.png`);
        execFileSync('ffmpeg', ['-nostdin','-v','error','-f','x11grab','-video_size','1280x720','-i',display,'-frames:v','1',image]);
        images[role] = {
          image,
          sha256: sha256(image),
          actual_pose: await send(`data get entity ${name} Pos`),
          actual_rotation: await send(`data get entity ${name} Rotation`),
          selected_item: await send(`data get entity ${name} SelectedItem`),
        };
      }
      const center = [tx + 0.5, tz + 0.5];
      const pa = spec.world.alpha_position, pb = spec.world.bravo_position;
      const alphaOtherAngle = angleDeg([center[0]-pa[0],center[1]-pa[2]],[pb[0]-pa[0],pb[2]-pa[2]]);
      const bravoOtherAngle = angleDeg([center[0]-pb[0],center[1]-pb[2]],[pa[0]-pb[0],pa[2]-pb[2]]);
      const geometricVisibility = {
        alpha_sees_bravo: alphaOtherAngle <= 51,
        bravo_sees_alpha: bravoOtherAngle <= 51,
        alpha_other_angle_deg: alphaOtherAngle,
        bravo_other_angle_deg: bravoOtherAngle,
      };
      if (geometricVisibility.alpha_sees_bravo !== spec.expected_agent_visibility.alpha_sees_bravo ||
          geometricVisibility.bravo_sees_alpha !== spec.expected_agent_visibility.bravo_sees_alpha) {
        throw new Error(`${spec.case_id} visibility geometry mismatch: ${JSON.stringify(geometricVisibility)}`);
      }
      const row = { case_id: spec.case_id, visibility_class: spec.visibility_class, poses, images, geometric_visibility: geometricVisibility, intended_materials: spec.intended_materials, target_cell: spec.world.target_cell };
      fs.writeFileSync(path.join(caseDir, 'capture.json'), JSON.stringify(row, null, 2));
      captures.push(row);
      console.log(`${spec.case_id} ${spec.visibility_class}`);
    }
    fs.writeFileSync(path.join(output, 'capture_manifest.json'), JSON.stringify({
      protocol: 'V3 race-condition conditioning stills only',
      gt_collected: false,
      native_interaction_executed: false,
      resolution: [1280,720],
      horizontal_fov_deg_approx: 102,
      case_count: captures.length,
      captures,
    }, null, 2));
  } finally {
    await rcon.end().catch(() => {});
  }
}
main().catch((error) => { console.error(error.stack || error.message); process.exit(1); });
