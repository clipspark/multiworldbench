'use strict';

// Record native Minecraft GT for the six V3 marked-cross shared-state cases.
// RCON is used only for scene initialization and state validation. The formal
// interaction is executed through native X11/XTest mouse input and every
// frame records the actual button state returned by the input bridge.
const fs = require('fs');
const path = require('path');
const { spawn, execFileSync } = require('child_process');
const readline = require('readline');
const { Rcon } = require('rcon-client');

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const nowMs = () => Date.now();

function minecraftWindow(display) {
  const out = execFileSync('/usr/bin/xwininfo', ['-root', '-tree'], {
    env: { ...process.env, DISPLAY: display }, encoding: 'utf8',
  });
  const match = out.match(/\n\s+(0x[0-9a-f]+)\s+"Minecraft/);
  if (!match) throw new Error(`Minecraft window not found on ${display}`);
  return match[1];
}

function startX11Sender(repo, display, windowId) {
  const child = spawn('python3', [
    path.join(repo, 'controller/style_x11_native_action_server.py'), display, windowId,
  ], { stdio: ['pipe', 'pipe', 'inherit'] });
  const lines = readline.createInterface({ input: child.stdout });
  const pending = [];
  lines.on('line', line => {
    const item = pending.shift();
    if (!item) return;
    try { item.resolve(JSON.parse(line)); } catch (error) { item.reject(error); }
  });
  const sendRaw = item => new Promise((resolve, reject) => {
    pending.push({ resolve, reject });
    child.stdin.write(JSON.stringify(item) + '\n');
  });
  return {
    ready: sendRaw({ focus: true }),
    focus: () => sendRaw({ focus: true }),
    releaseAll: () => sendRaw({ release_all: true }),
    send: heldButtons => sendRaw({ dx: 0, dy: 0, held_keys: [], held_buttons: heldButtons }),
    close: () => { child.stdin.end(); child.kill('SIGTERM'); },
  };
}

function startVideo(display, output, frames) {
  const child = spawn('ffmpeg', [
    '-nostdin', '-hide_banner', '-loglevel', 'warning', '-y',
    '-use_wallclock_as_timestamps', '1', '-f', 'x11grab',
    '-video_size', '1280x720', '-framerate', '20', '-i', display,
    '-copyts', '-vsync', '0', '-enc_time_base', '1:1000',
    '-frames:v', String(frames), '-c:v', 'libx264', '-threads', '2',
    '-preset', 'ultrafast', '-crf', '18', '-bf', '0', '-pix_fmt', 'yuv420p', output,
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', chunk => { stderr += chunk.toString(); });
  const startedEpochMs = nowMs();
  const done = new Promise((resolve, reject) => child.once('close', code => {
    if (code === 0) resolve();
    else reject(new Error(`ffmpeg ${display} exited ${code}: ${stderr}`));
  }));
  return { startedEpochMs, done };
}

function normalizeVideo(input, output) {
  execFileSync('ffmpeg', [
    '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', input,
    '-vf', 'scale=640:360:flags=lanczos,crop=640:352:0:4,fps=20,trim=end_frame=81,setpts=PTS-STARTPTS',
    '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
    '-pix_fmt', 'yuv420p', output,
  ]);
}

function actionRow(frame, epochMs, operation, active, nativeResult) {
  const attack = active && operation === 'destroy' && frame >= 10 && frame < 18;
  const use = active && operation === 'place' && frame === 10;
  return {
    frame_index: frame,
    epochTime: epochMs / 1000,
    epoch_ms: epochMs,
    phase: frame < 10 ? 'PRE_ACTION' : (attack || use ? 'ACTION' : 'POST_ACTION'),
    primitive: attack ? 'ATTACK_HOLD' : (use ? 'USE_ONCE' : '-'),
    control_source: 'native_x11_xtest',
    action: {
      forward: false, back: false, left: false, right: false,
      jump: false, sprint: false, sneak: false,
      attack, use, mine: attack, place_block: use,
      camera: [0.0, 0.0],
    },
    native_input: {
      device: 'X11_XTest', held_buttons: nativeResult.held_buttons || [],
      sent_epoch_ms: nativeResult.sent_epoch_ms,
    },
  };
}

async function main() {
  if (!process.env.RCON_PASSWORD) throw new Error('RCON_PASSWORD must be set');
  const repo = path.resolve(process.argv[2] || '.');
  const run = path.resolve(process.argv[3] || '');
  if (!run) throw new Error('usage: node capture_shared_state_gt.js REPO RUN');
  const gtRoot = path.join(run, 'models/gt/interaction');
  if (fs.existsSync(gtRoot)) throw new Error(`Refusing overwrite: ${gtRoot}`);
  fs.mkdirSync(gtRoot, { recursive: true });
  const suite = JSON.parse(fs.readFileSync(path.join(
    repo, 'MultiWorldBench/v3/cases/shared_world/shared_state_synchronization.json'
  ), 'utf8'));
  const rcon = await Rcon.connect({
    host: '127.0.0.1', port: 25721, password: process.env.RCON_PASSWORD, timeout: 30000,
  });
  const windows = { alpha: minecraftWindow(':130'), bravo: minecraftWindow(':131') };
  const senders = {
    alpha: startX11Sender(repo, ':130', windows.alpha),
    bravo: startX11Sender(repo, ':131', windows.bravo),
  };
  await Promise.all([senders.alpha.ready, senders.bravo.ready]);
  const manifest = {
    schema_version: 'mwb.v3.shared_state.native_gt.v1',
    protocol: 'native Minecraft X11/XTest interaction; RCON initialization and validation only',
    fps: 20, frames_per_case: 81, cases: [],
  };
  try {
    const online = await rcon.send('list');
    if (!online.includes('CameraAlpha') || !online.includes('CameraBravo')) {
      throw new Error(`camera clients unavailable: ${online}`);
    }
    for (const spec of suite.cases) {
      const cid = spec.case_id;
      const caseDir = path.join(gtRoot, cid);
      fs.mkdirSync(caseDir, { recursive: true });
      const initializationLog = path.join(caseDir, 'initialization_rcon.jsonl');
      const validationLog = path.join(caseDir, 'validation_rcon.jsonl');
      const sendInit = async command => {
        const response = await rcon.send(command);
        fs.appendFileSync(initializationLog, JSON.stringify({ epoch_ms: nowMs(), command, response }) + '\n');
        if (/unknown or incomplete|incorrect argument|not a valid|no player was found|position is not loaded|too many blocks/i.test(String(response))) {
          throw new Error(`${command} -> ${response}`);
        }
        return response;
      };
      const sendValidate = async command => {
        const response = await rcon.send(command);
        fs.appendFileSync(validationLog, JSON.stringify({ epoch_ms: nowMs(), command, response }) + '\n');
        return response;
      };
      const [tx, ty, tz] = spec.target_cell;
      await sendInit(`forceload add ${tx - 12} ${tz - 12} ${tx + 12} ${tz + 12}`);
      await sendInit('difficulty peaceful');
      await sendInit('gamerule doDaylightCycle false');
      await sendInit('gamerule doWeatherCycle false');
      await sendInit('weather clear');
      await sendInit('time set noon');
      await rcon.send('team add mwb3sync');
      await sendInit('team modify mwb3sync collisionRule never');
      await sendInit('team modify mwb3sync nametagVisibility never');
      await sendInit('team join mwb3sync CameraAlpha');
      await sendInit('team join mwb3sync CameraBravo');
      await sendInit(`fill ${tx - 8} ${ty - 1} ${tz - 7} ${tx + 8} ${ty - 1} ${tz + 9} minecraft:grass_block`);
      await sendInit(`fill ${tx - 8} ${ty} ${tz - 7} ${tx + 8} ${ty + 7} ${tz + 9} minecraft:air`);
      await sendInit(`execute positioned ${tx} ${ty} ${tz} run kill @e[type=minecraft:item,distance=..32]`);
      await sendInit(`fill ${tx - 4} ${ty - 1} ${tz} ${tx + 4} ${ty - 1} ${tz} minecraft:white_concrete`);
      await sendInit(`fill ${tx} ${ty - 1} ${tz - 4} ${tx} ${ty - 1} ${tz + 4} minecraft:white_concrete`);
      await sendInit(`setblock ${tx} ${ty - 1} ${tz} minecraft:red_concrete`);
      const initialBlock = spec.operation === 'destroy' ? 'coarse_dirt' : 'air';
      await sendInit(`setblock ${tx} ${ty} ${tz} minecraft:${initialBlock}`);
      for (const [role, name] of [['alpha', 'CameraAlpha'], ['bravo', 'CameraBravo']]) {
        const pose = spec.initial_conditions[role].pose;
        await sendInit(`gamemode survival ${name}`);
        await sendInit(`clear ${name}`);
        if (pose.active && spec.operation === 'place') {
          await sendInit(`item replace entity ${name} weapon.mainhand with minecraft:coarse_dirt 1`);
        } else if (pose.active && spec.operation === 'destroy') {
          await sendInit(`item replace entity ${name} weapon.mainhand with minecraft:diamond_shovel 1`);
        }
        await sendInit(`tp ${name} ${pose.position.join(' ')} ${pose.native_yaw_deg} ${pose.native_pitch_deg}`);
      }
      // Refresh native mouse capture after all teleports. focus() emits a
      // harmless left click, so reset the target and held item immediately
      // afterwards before the formal recording starts.
      await Promise.all([senders.alpha.focus(), senders.bravo.focus()]);
      await sleep(150);
      await Promise.all([senders.alpha.releaseAll(), senders.bravo.releaseAll()]);
      await sendInit(`setblock ${tx} ${ty - 1} ${tz} minecraft:red_concrete`);
      await sendInit(`setblock ${tx} ${ty} ${tz} minecraft:${initialBlock}`);
      for (const [role, name] of [['alpha', 'CameraAlpha'], ['bravo', 'CameraBravo']]) {
        const pose = spec.initial_conditions[role].pose;
        await sendInit(`clear ${name}`);
        if (pose.active && spec.operation === 'place') {
          await sendInit(`item replace entity ${name} weapon.mainhand with minecraft:coarse_dirt 1`);
        } else if (pose.active && spec.operation === 'destroy') {
          await sendInit(`item replace entity ${name} weapon.mainhand with minecraft:diamond_shovel 1`);
        }
        await sendInit(`tp ${name} ${pose.position.join(' ')} ${pose.native_yaw_deg} ${pose.native_pitch_deg}`);
      }
      await Promise.all([senders.alpha.send([]), senders.bravo.send([])]);
      await sleep(1500);
      const rawA = path.join(caseDir, 'alpha_raw.mp4');
      const rawB = path.join(caseDir, 'bravo_raw.mp4');
      const videoA = startVideo(':130.0', rawA, 100);
      const videoB = startVideo(':131.0', rawB, 100);
      const actionStartEpochMs = nowMs() + 250;
      const actions = { alpha: [], bravo: [] };
      for (let frame = 0; frame < 81; frame += 1) {
        const targetEpochMs = actionStartEpochMs + frame * 50;
        const remaining = targetEpochMs - nowMs();
        if (remaining > 0) await sleep(remaining);
        const roleButtons = role => {
          if (!spec.active_roles.includes(role)) return [];
          if (spec.operation === 'place' && frame === 10) return [3];
          if (spec.operation === 'destroy' && frame >= 10 && frame < 18) return [1];
          return [];
        };
        const epochMs = nowMs();
        const [aResult, bResult] = await Promise.all([
          senders.alpha.send(roleButtons('alpha')),
          senders.bravo.send(roleButtons('bravo')),
        ]);
        actions.alpha.push(actionRow(frame, epochMs, spec.operation, spec.active_roles.includes('alpha'), aResult));
        actions.bravo.push(actionRow(frame, epochMs, spec.operation, spec.active_roles.includes('bravo'), bResult));
      }
      await Promise.all([senders.alpha.send([]), senders.bravo.send([])]);
      await Promise.all([videoA.done, videoB.done]);
      const finalExpected = spec.operation === 'place' ? 'coarse_dirt' : 'air';
      const targetResponse = await sendValidate(`execute if block ${tx} ${ty} ${tz} minecraft:${finalExpected} run time query gametime`);
      const supportResponse = await sendValidate(`execute if block ${tx} ${ty - 1} ${tz} minecraft:red_concrete run time query gametime`);
      const targetValid = Boolean(String(targetResponse).trim());
      const supportValid = Boolean(String(supportResponse).trim());
      let coarseDirtCount = 0;
      for (let x = tx - 2; x <= tx + 2; x += 1) {
        for (let y = ty; y <= ty + 2; y += 1) {
          for (let z = tz - 2; z <= tz + 2; z += 1) {
            const response = await sendValidate(`execute if block ${x} ${y} ${z} minecraft:coarse_dirt run time query gametime`);
            if (String(response).trim()) coarseDirtCount += 1;
          }
        }
      }
      const expectedCoarseDirtCount = spec.operation === 'place' ? 1 : 0;
      const exactChangeValid = coarseDirtCount === expectedCoarseDirtCount;
      if (!targetValid || !supportValid || !exactChangeValid) {
        throw new Error(`${cid} state validation failed: target=${targetResponse} support=${supportResponse} coarse_dirt_count=${coarseDirtCount}`);
      }
      fs.writeFileSync(path.join(caseDir, 'alpha_actions.json'), JSON.stringify(actions.alpha, null, 2) + '\n');
      fs.writeFileSync(path.join(caseDir, 'bravo_actions.json'), JSON.stringify(actions.bravo, null, 2) + '\n');
      normalizeVideo(rawA, path.join(caseDir, 'alpha_gt.mp4'));
      normalizeVideo(rawB, path.join(caseDir, 'bravo_gt.mp4'));
      fs.unlinkSync(rawA);
      fs.unlinkSync(rawB);
      const caseManifest = {
        case_id: cid, operation: spec.operation, active_roles: spec.active_roles,
        target_cell: spec.target_cell, expected_change_count: 1,
        native_action: spec.operation === 'place'
          ? { button: 'right', button_id: 3, start_frame: 10, end_frame_exclusive: 11 }
          : { button: 'left', button_id: 1, start_frame: 10, end_frame_exclusive: 18 },
        recording: { fps: 20, frames: 81, action_start_epoch_ms: actionStartEpochMs,
          alpha_video_start_epoch_ms: videoA.startedEpochMs, bravo_video_start_epoch_ms: videoB.startedEpochMs },
        validation: { target_expected: `minecraft:${finalExpected}`, target_valid: targetValid,
          support_expected: 'minecraft:red_concrete', support_valid: supportValid,
          observed_coarse_dirt_count: coarseDirtCount,
          expected_coarse_dirt_count: expectedCoarseDirtCount,
          exact_changed_block_count: 1,
          exact_change_valid: exactChangeValid,
          valid: targetValid && supportValid && exactChangeValid },
      };
      fs.writeFileSync(path.join(caseDir, 'case_manifest.json'), JSON.stringify(caseManifest, null, 2) + '\n');
      manifest.cases.push(caseManifest);
      console.log(`${cid} ${spec.operation} valid=${caseManifest.validation.valid}`);
    }
    fs.writeFileSync(path.join(gtRoot, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
    fs.writeFileSync(path.join(run, '.gt_capture_complete'), JSON.stringify({
      case_count: manifest.cases.length, all_valid: manifest.cases.every(c => c.validation.valid),
    }, null, 2) + '\n');
  } finally {
    senders.alpha.close();
    senders.bravo.close();
    await rcon.end().catch(() => {});
  }
}

main().catch(error => { console.error(error.stack || error.message); process.exit(1); });
