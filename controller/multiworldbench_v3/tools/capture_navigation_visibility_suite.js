"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { spawn, execFileSync } = require("child_process");
const mineflayer = require("mineflayer");
const { Rcon } = require("rcon-client");

const { executeNativeAction, poseFromEntity, round } = require("./navigation_native_actions");
const { scenes, builder } = require("./capture_navigation_rich_scene_stills");

const ROLE_NAMES = { A: "Alpha", B: "Bravo" };
const CAMERA_NAMES = { A: "CameraAlpha", B: "CameraBravo" };
const DISPLAYS = { A: ":130.0", B: ":131.0" };
const CONTROL_KEYS = ["forward", "back", "left", "right", "jump", "sprint", "sneak"];
const ACTION_KEYS = [
  "inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
  "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9", "forward", "back", "left", "right",
  "jump", "sneak", "sprint", "swapHands", "attack", "use", "pickItem", "drop",
];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sha256 = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");

function writeJson(file, value) {
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

function writeJsonl(file, rows) {
  fs.writeFileSync(file, rows.map((row) => JSON.stringify(row)).join("\n") + (rows.length ? "\n" : ""));
}

function normalizeAngle(value) {
  return ((value + 540) % 360) - 180;
}

function bearing(from, to) {
  return Math.atan2(-(to[0] - from[0]), to[2] - from[2]) * 180 / Math.PI;
}

function aimAt(from, to) {
  const eyeY = from[1] + 1.62;
  return {
    yaw: bearing(from, to),
    pitch: -Math.atan2(to[1] + 1.62 - eyeY, Math.hypot(to[0] - from[0], to[2] - from[2])) * 180 / Math.PI,
  };
}

function initialPoses(caseSpec) {
  const scene = caseSpec.scene;
  const a = [scene.center_x + scene.alpha_offset[0], scene.alpha_offset[1], scene.alpha_offset[2]];
  const b = [scene.center_x + scene.bravo_offset[0], scene.bravo_offset[1], scene.bravo_offset[2]];
  const focus = [scene.center_x + scene.focus_offset[0], scene.focus_offset[1], scene.focus_offset[2]];
  const aFocus = aimAt(a, focus);
  const bFocus = aimAt(b, focus);
  const visibility = caseSpec.initial_visibility;
  return {
    A: { position: a, yaw: aFocus.yaw + (visibility.alpha_sees_bravo ? 0 : 40), pitch: aFocus.pitch },
    B: { position: b, yaw: bFocus.yaw + (visibility.bravo_sees_alpha ? 0 : -40), pitch: bFocus.pitch },
  };
}

function visibilityCheck(caseSpec, poses) {
  const yawAB = bearing(poses.A.position, poses.B.position);
  const yawBA = bearing(poses.B.position, poses.A.position);
  const observed = {
    alpha_sees_bravo: Math.abs(normalizeAngle(poses.A.yaw - yawAB)) <= 55,
    bravo_sees_alpha: Math.abs(normalizeAngle(poses.B.yaw - yawBA)) <= 55,
  };
  const expected = caseSpec.initial_visibility;
  if (observed.alpha_sees_bravo !== expected.alpha_sees_bravo || observed.bravo_sees_alpha !== expected.bravo_sees_alpha) {
    throw new Error(`initial visibility mismatch ${caseSpec.id}: ${JSON.stringify({ expected, observed })}`);
  }
  return {
    semantics: "initial horizontal camera frustum, conservative half FOV 55 degrees",
    expected,
    observed,
    angular_offsets_deg: {
      alpha_to_bravo: round(Math.abs(normalizeAngle(poses.A.yaw - yawAB))),
      bravo_to_alpha: round(Math.abs(normalizeAngle(poses.B.yaw - yawBA))),
    },
  };
}

function stopControls(bot) {
  if (!bot) return;
  if (typeof bot.clearControlStates === "function") bot.clearControlStates();
  else for (const key of CONTROL_KEYS) bot.setControlState(key, false);
}

function liveControls(bot) {
  return Object.fromEntries(CONTROL_KEYS.map((key) => [key, Boolean(bot.controlState[key])]));
}

function angularDelta(current, previous) {
  return ((current - previous + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
}

function createRecorder(bots, totalFrames, fps) {
  const frames = { A: [], B: [] };
  const previous = {};
  let phase = "initial_observation";
  let planned = null;
  let turnId = null;
  let doneResolve;
  const done = new Promise((resolve) => { doneResolve = resolve; });
  const startEpochMs = Date.now();
  const take = () => {
    const epochMs = Date.now();
    for (const role of ["A", "B"]) {
      const bot = bots[role];
      const yaw = Number(bot.entity.yaw);
      const pitch = Number(bot.entity.pitch);
      const prior = previous[role];
      const action = {
        ...liveControls(bot), attack: false, use: false, mount: false, dismount: false,
        place_block: false, place_entity: false, mine: false,
        camera: prior ? [angularDelta(yaw, prior.yaw), pitch - prior.pitch] : [0, 0],
      };
      frames[role].push({
        frame_count: frames[role].length,
        epoch_ms: epochMs,
        epochTime: epochMs / 1000,
        relativeTimeMs: epochMs - startEpochMs,
        minecraft_tick: Number(bot.time.age),
        phase,
        turn_id: turnId,
        planned_action: planned ? planned[role] : null,
        recorded_action_source: "live_bot_state",
        action,
        ...poseFromEntity(bot.entity),
      });
      previous[role] = { yaw, pitch };
    }
    if (frames.A.length >= totalFrames) {
      clearInterval(timer);
      doneResolve();
    }
  };
  take();
  const timer = setInterval(take, 1000 / fps);
  return {
    frames, done, startEpochMs,
    setInitial() { phase = "initial_observation"; planned = null; turnId = null; },
    setTurn(turn) { phase = "navigation_turn"; planned = { A: turn.alpha, B: turn.bravo }; turnId = turn.id; },
    setSettle() { phase = "final_action_free_settle"; planned = null; turnId = null; },
  };
}

function gammaActions(rows) {
  const keyboard = [];
  const camera = [];
  for (const row of rows) {
    const vector = Array(23).fill(0);
    for (const key of ["forward", "back", "left", "right", "jump", "sneak", "sprint"]) {
      if (row.action[key]) vector[ACTION_KEYS.indexOf(key)] = 1;
    }
    keyboard.push(vector);
    camera.push(row.action.camera.map((value) => value * 180 / Math.PI));
  }
  return { keyboard, camera };
}

function spawnFfmpeg(display, file, frames, fps) {
  const args = [
    "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-thread_queue_size", "128",
    "-f", "x11grab", "-video_size", "1280x720", "-framerate", String(fps), "-i", display,
    "-frames:v", String(frames), "-c:v", "libx264", "-threads", "2", "-preset", "ultrafast",
    "-crf", "18", "-bf", "0", "-pix_fmt", "yuv420p", file,
  ];
  const child = spawn("ffmpeg", args, { stdio: ["ignore", "ignore", "pipe"] });
  let stderr = "";
  child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
  const done = new Promise((resolve, reject) => child.on("close", (code) => code === 0 ? resolve() : reject(new Error(`ffmpeg ${display} exit ${code}: ${stderr}`))));
  return { child, done };
}

function frameCount(file) {
  return Number(execFileSync("ffprobe", ["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", file], { encoding: "utf8" }).trim());
}

function createFirstFrame(alphaVideo, bravoVideo, output) {
  execFileSync("ffmpeg", ["-nostdin", "-v", "error", "-y", "-i", alphaVideo, "-i", bravoVideo,
    "-filter_complex", "[0:v]select=eq(n\\,0),scale=480:320[a];[1:v]select=eq(n\\,0),scale=480:320[b];[a][b]hstack=inputs=2",
    "-frames:v", "1", output]);
}

async function joinBot(role, deathState) {
  return new Promise((resolve, reject) => {
    const bot = mineflayer.createBot({ username: ROLE_NAMES[role], host: "127.0.0.1", port: 25720, version: "1.21", checkTimeoutInterval: 120000 });
    const fail = (error) => reject(error);
    bot.once("error", fail);
    bot.once("spawn", () => {
      bot.removeListener("error", fail);
      bot.on("error", (error) => console.error(`[${role}] ${error.message}`));
      bot.on("kicked", (reason) => console.error(`[${role}] kicked ${reason}`));
      bot.on("death", () => { deathState[role] = true; });
      resolve(bot);
    });
  });
}

async function quitBot(bot) {
  if (!bot) return;
  stopControls(bot);
  await Promise.race([new Promise((resolve) => { bot.once("end", resolve); bot.quit(); }), sleep(3000)]);
}

async function buildScenes(rcon, outputRoot, selectedSceneIds) {
  const log = [];
  const send = async (command) => {
    const response = await rcon.send(command);
    log.push({ epoch_ms: Date.now(), command, response: String(response) });
    if (/unknown or incomplete|incorrect argument|expected integer|not a valid|too many blocks/i.test(String(response))) throw new Error(`${command} -> ${response}`);
    return response;
  };
  for (const spec of scenes.filter((item) => selectedSceneIds.has(item.id))) {
    const index = scenes.findIndex((item) => item.id === spec.id);
    const cx = 4000 + index * 96;
    await send(`forceload add ${cx - 32} -28 ${cx + 32} 28`);
    await spec.build(builder(send, cx));
  }
  writeJson(path.join(outputRoot, "scene_initialization.json"), { policy: "before_all_capture", scenes: [...selectedSceneIds], commands: log });
}

async function runCase(rcon, bots, deathState, caseSpec, outputRoot, specPath) {
  const caseDir = path.join(outputRoot, caseSpec.id);
  if (fs.existsSync(path.join(caseDir, ".complete"))) {
    console.log("CASE_SKIP", caseSpec.id);
    return;
  }
  fs.rmSync(caseDir, { recursive: true, force: true });
  fs.mkdirSync(caseDir, { recursive: true });
  writeJson(path.join(caseDir, "case_spec.json"), caseSpec);
  const initialization = { case_id: caseSpec.id, policy: "pre_capture_only", source_spec: specPath, commands: [], rcon_during_capture: false };
  const send = async (stage, command) => {
    const response = await rcon.send(command);
    initialization.commands.push({ stage, command, epoch_ms: Date.now(), response: String(response) });
    return response;
  };
  await send("pairing_stop", "episode stop");
  for (const bot of Object.values(bots)) stopControls(bot);
  deathState.A = false; deathState.B = false;
  const poses = initialPoses(caseSpec);
  for (const role of ["A", "B"]) {
    const pose = poses[role];
    await send("player_reset", `tp ${ROLE_NAMES[role]} ${pose.position.join(" ")} ${pose.yaw} ${pose.pitch}`);
    await send("player_reset", `gamemode survival ${ROLE_NAMES[role]}`);
    await send("player_reset", `effect clear ${ROLE_NAMES[role]}`);
    await send("player_reset", `clear ${ROLE_NAMES[role]}`);
  }
  await send("survival_pairing", "episode start Alpha CameraAlpha technoblade.png Bravo CameraBravo test.png");
  await Promise.all([bots.A.waitForTicks(12), bots.B.waitForTicks(12)]);
  const actualPoses = { A: poseFromEntity(bots.A.entity), B: poseFromEntity(bots.B.entity) };
  const check = visibilityCheck(caseSpec, poses);
  check.planned_poses = poses;
  check.actual_controller_poses = actualPoses;
  check.checked_epoch_ms = Date.now();
  writeJson(path.join(caseDir, "initial_view_check.json"), check);
  initialization.completed_epoch_ms = Date.now();
  initialization.capture_started_epoch_ms = Date.now();
  writeJson(path.join(caseDir, "initialization.json"), initialization);

  const alphaVideo = path.join(caseDir, "alpha_gt.mp4");
  const bravoVideo = path.join(caseDir, "bravo_gt.mp4");
  const ffmpeg = [spawnFfmpeg(DISPLAYS.A, alphaVideo, caseSpec.valid_num_frames, caseSpec.fps), spawnFfmpeg(DISPLAYS.B, bravoVideo, caseSpec.valid_num_frames, caseSpec.fps)];
  const recorder = createRecorder(bots, caseSpec.valid_num_frames, caseSpec.fps);
  const gt = [];
  await Promise.all([bots.A.waitForTicks(caseSpec.initial_idle_frames), bots.B.waitForTicks(caseSpec.initial_idle_frames)]);
  for (const [turnIndex, turn] of caseSpec.turns.entries()) {
    recorder.setTurn(turn);
    const startEpochMs = Date.now() + 10;
    const startTick = Math.max(Number(bots.A.time.age), Number(bots.B.time.age));
    const [aResult, bResult] = await Promise.all([
      executeNativeAction({ bot: bots.A, action: turn.alpha, startTick, startEpochMs, turnTicks: turn.duration_ticks }),
      executeNativeAction({ bot: bots.B, action: turn.bravo, startTick, startEpochMs, turnTicks: turn.duration_ticks }),
    ]);
    if (!["completed", "idle"].includes(aResult.status) || !["completed", "idle"].includes(bResult.status)) {
      throw new Error(`native action failed ${caseSpec.id}/${turn.id}: ${aResult.status},${bResult.status}`);
    }
    gt.push({
      benchmark: "MultiWorldBench-v3-navigation-visibility", case_id: caseSpec.id,
      visibility_class: caseSpec.visibility_class, action_template_id: caseSpec.action_template_id,
      turn_id: turn.id, turn_index: turnIndex, planned_actions: { A: turn.alpha, B: turn.bravo },
      action_results: { A: aResult, B: bResult },
      start_skew_ms: Math.abs(aResult.actual_start_epoch_ms - bResult.actual_start_epoch_ms),
      observed_end_poses: { A: poseFromEntity(bots.A.entity), B: poseFromEntity(bots.B.entity) },
    });
  }
  recorder.setSettle();
  stopControls(bots.A); stopControls(bots.B);
  await Promise.all([recorder.done, ...ffmpeg.map((item) => item.done)]);
  if (deathState.A || deathState.B) throw new Error(`player death ${caseSpec.id}`);
  if (recorder.frames.A.length !== caseSpec.valid_num_frames || recorder.frames.B.length !== caseSpec.valid_num_frames) throw new Error("action frame count mismatch");
  if (frameCount(alphaVideo) !== caseSpec.valid_num_frames || frameCount(bravoVideo) !== caseSpec.valid_num_frames) throw new Error("video frame count mismatch");
  writeJson(path.join(caseDir, "alpha_actions.json"), recorder.frames.A);
  writeJson(path.join(caseDir, "bravo_actions.json"), recorder.frames.B);
  writeJsonl(path.join(caseDir, "GT_state.jsonl"), gt);
  writeJson(path.join(caseDir, "action_left.json"), gammaActions(recorder.frames.A));
  writeJson(path.join(caseDir, "action_right.json"), gammaActions(recorder.frames.B));
  fs.writeFileSync(path.join(caseDir, "prompt.txt"), "Minecraft gameplay.\n");
  createFirstFrame(alphaVideo, bravoVideo, path.join(caseDir, "first_frame.png"));
  const files = ["alpha_gt.mp4", "bravo_gt.mp4", "alpha_actions.json", "bravo_actions.json", "GT_state.jsonl", "action_left.json", "action_right.json", "first_frame.png"];
  const episodeInfo = {
    case_id: caseSpec.id, frame_count: caseSpec.valid_num_frames, fps: caseSpec.fps,
    per_tick_trajectory_turns: gt.length, capture_start_epoch_ms: recorder.startEpochMs,
    capture_end_epoch_ms: Date.now(), actual_action_source: "live control state plus consumed physics tick trajectories",
    camera_mode: "survival_mirror_hud", rcon_during_capture: false, hashes: Object.fromEntries(files.map((name) => [name, sha256(path.join(caseDir, name))])),
  };
  writeJson(path.join(caseDir, "episode_info.json"), episodeInfo);
  fs.writeFileSync(path.join(caseDir, ".complete"), `${episodeInfo.capture_end_epoch_ms}\n`);
  console.log("CASE_OK", caseSpec.id, `${caseSpec.round_count} rounds`);
}

async function main() {
  const outputRoot = path.resolve(process.env.OUTPUT_DIR || "");
  if (!process.env.OUTPUT_DIR) throw new Error("OUTPUT_DIR is required");
  fs.mkdirSync(outputRoot, { recursive: true });
  const specPath = path.resolve(process.env.MWB3_NAV_SPEC_PATH || path.join(__dirname, "../../../MultiWorldBench/v3/cases/navigation/navigation_visibility_suite.json"));
  const document = JSON.parse(fs.readFileSync(specPath, "utf8"));
  let selected = document.cases;
  if (process.env.MWB3_NAV_CASES) {
    const ids = new Set(process.env.MWB3_NAV_CASES.split(",").map((item) => item.trim()).filter(Boolean));
    selected = document.cases.filter((item) => ids.has(item.id));
    if (selected.length !== ids.size) throw new Error(`unknown selected cases; requested ${ids.size}, found ${selected.length}`);
  }
  const deathState = { A: false, B: false };
  const bots = {};
  let rcon;
  try {
    bots.A = await joinBot("A", deathState);
    bots.B = await joinBot("B", deathState);
    rcon = await Rcon.connect({ host: "127.0.0.1", port: 25721, password: "research", timeout: 30000 });
    const globalCommands = [
      "difficulty peaceful", "gamerule doDaylightCycle false", "gamerule doWeatherCycle false", "gamerule keepInventory true",
      "gamerule fallDamage false", "weather clear", "time set noon", "kill @e[type=!player]",
      "team add mwb3nav", "team modify mwb3nav collisionRule never", "team modify mwb3nav nametagVisibility never",
      "team join mwb3nav Alpha", "team join mwb3nav Bravo", "team join mwb3nav CameraAlpha", "team join mwb3nav CameraBravo",
    ];
    const globalLog = [];
    for (const command of globalCommands) {
      const response = await rcon.send(command);
      globalLog.push({ command, response: String(response), epoch_ms: Date.now() });
    }
    writeJson(path.join(outputRoot, "global_initialization.json"), { commands: globalLog });
    if (process.env.MWB3_BUILD_SCENES === "1" && !fs.existsSync(path.join(outputRoot, "scene_initialization.json"))) {
      await buildScenes(rcon, outputRoot, new Set(selected.map((item) => item.scene.scene_id)));
    }
    for (const caseSpec of selected) {
      let completed = false;
      let lastError;
      for (let attempt = 1; attempt <= 3 && !completed; attempt += 1) {
        try {
          await runCase(rcon, bots, deathState, caseSpec, outputRoot, specPath);
          completed = true;
        } catch (error) {
          lastError = error;
          console.error("CASE_RETRY", caseSpec.id, "attempt=" + attempt, error.message);
          stopControls(bots.A); stopControls(bots.B);
          await rcon.send("episode stop").catch(() => {});
          // A failed action may leave fixed-length ffmpeg recorders alive. Let them
          // terminate before replacing the incomplete case directory.
          await new Promise((resolve) => setTimeout(resolve, 11000));
        }
      }
      if (!completed) throw lastError;
    }
    const complete = fs.readdirSync(outputRoot).filter((name) => fs.existsSync(path.join(outputRoot, name, ".complete"))).length;
    writeJson(path.join(outputRoot, "capture_summary.json"), { selected: selected.length, complete, finished_epoch_ms: Date.now() });
  } finally {
    stopControls(bots.A); stopControls(bots.B);
    if (rcon) { await rcon.send("episode stop").catch(() => {}); await rcon.end().catch(() => {}); }
    await Promise.all([quitBot(bots.A), quitBot(bots.B)]);
  }
}

if (require.main === module) main().catch((error) => { console.error(error.stack || error.message); process.exit(1); });

module.exports = { gammaActions, initialPoses, visibilityCheck };
